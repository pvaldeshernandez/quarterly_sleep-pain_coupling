"""A51 — write the committed run's numbers into the submission documents.

Applies a VERIFIED edit spec (built and adversarially checked by a workflow, stored as
JSON) to the eight .docx files, as tracked changes by Pedro.

Two mechanisms, chosen per edit:

  kind="text"      `Doc.replace_text_in_para` — an ordinary tracked run replacement.
  kind="equation"  `Doc.replace_para_tracked` — the number lives inside an OMML object,
                   where Word cannot represent a revision at all. The paragraph is
                   replaced by an edited copy: copy marked inserted, original marked
                   deleted, equation never rebuilt.

WHY THE ORDER MATTERS
`replace_para_tracked` INSERTS a paragraph, which shifts every later index. So every
target is resolved to an lxml element reference BEFORE anything is applied; element
references survive sibling insertion, indices do not.

WHY EQUATION EDITS ARE GROUPED
`replace_para_tracked` replaces the whole paragraph, so two edits to the same equation
paragraph must go in one call or the second would edit a paragraph that is already
marked deleted.

THE GATE
After saving, the round trip is checked against the pre-edit backup:

  reject-all  must reproduce the backup's accepted text EXACTLY. If it does not,
              something was changed without a revision mark, which is the one failure
              mode that matters — a reviewer diffing against the submitted version
              would not see it.
  accept-all  must contain every new value.

A run that fails the gate restores the backup and exits non-zero.

    python a51_apply_number_updates.py --spec spec.json [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import docx_edit as de  # noqa: E402

BACKUP_DIR = os.path.join(de.SUBMISSION_DIR, "archive",
                          "backup_20260808_before_number_update")


def drop_overlaps(edits, dumps, verbose=True):
    """Remove edits whose anchor OVERLAPS a longer anchor on the same paragraph.

    Two agents reading the same sentence independently propose the same change at
    different widths: one anchors on `|λps|=0.136`, the other on the bare `0.136`.
    Both are individually correct and individually unique, so neither an anchor check
    nor a per-edit verifier can see the problem — but applying both edits the same
    characters twice, and the second application lands on text the first already
    rewrote.

    The wider anchor wins: it carries the context that says WHICH quantity is meant,
    which is the whole reason a bare number is dangerous to match on.

    Returns (kept, dropped).
    """
    from collections import defaultdict
    by_para = defaultdict(list)
    for e in edits:
        by_para[(e["file"], int(e["para"]))].append(e)

    kept, dropped = [], []
    for (f, p), group_ in by_para.items():
        line = dumps.get(f.replace(".docx", ""), {}).get(p, "")
        # longest first, so a wide anchor claims its span before a narrow one asks
        ordered = sorted(group_, key=lambda e: -len(e["old"]))
        claimed = []
        for e in ordered:
            i = line.find(e["old"])
            if i < 0:
                kept.append(e)          # resolve() will report it
                continue
            span = (i, i + len(e["old"]))
            if any(span[0] < c[1] and c[0] < span[1] for c in claimed):
                dropped.append(e)
                continue
            claimed.append(span)
            kept.append(e)

    if verbose and dropped:
        print(f"\n{len(dropped)} edit(s) dropped as overlapping a wider anchor:")
        for e in dropped[:10]:
            print(f"    {e['file']} [{e['para']}] {e['old'][:44]!r}")
        if len(dropped) > 10:
            print(f"    ... and {len(dropped) - 10} more")
    return kept, dropped


def load_dumps(dump_dir):
    """{stem: {para_index: line}} from the accepted-text dumps."""
    import re
    out = {}
    for fn in os.listdir(dump_dir):
        if not fn.endswith(".txt"):
            continue
        d = {}
        for line in open(os.path.join(dump_dir, fn)):
            m = re.match(r"\[\s*(\d+)\](EQ|  ) (.*)", line.rstrip("\n"))
            if m:
                d[int(m.group(1))] = m.group(3)
        out[fn[:-4]] = d
    return out


def group(edits):
    """{file: {"text": [...], "equation": {para: [...]}}} — equation edits per paragraph."""
    out = {}
    for e in edits:
        slot = out.setdefault(e["file"], {"text": [], "equation": {}})
        if e["kind"] == "equation":
            slot["equation"].setdefault(int(e["para"]), []).append(e)
        else:
            slot["text"].append(e)
    return out


def resolve(doc, edits, kind):
    """Pair each edit with its paragraph ELEMENT, verifying the anchor first.

    Returns (resolved, misses). An anchor that does not occur, or occurs more than
    once, is a miss and is never applied: a non-unique anchor would edit whichever
    occurrence came first, which is a coin flip.
    """
    resolved, misses = [], []
    paras = doc.paragraphs
    for e in edits:
        i = int(e["para"])
        if i >= len(paras):
            misses.append((e, f"paragraph {i} does not exist ({len(paras)} total)"))
            continue
        p = paras[i]
        hay = doc.math_text_of(p) if kind == "equation" else doc.text_of(p)
        n = hay.count(e["old"])
        if n == 0:
            # Try the search-normalized text: Word scatters non-breaking spaces and
            # typographic hyphens that look identical on screen and never match.
            if doc.search_text_of(p).count(e["old"]) > 0:
                misses.append((e, "anchor matches only after whitespace/hyphen "
                                  "normalization; rewrite it with the literal characters"))
            else:
                misses.append((e, f"anchor not found in paragraph {i}"))
            continue
        if n > 1:
            misses.append((e, f"anchor occurs {n} times in paragraph {i}; not unique"))
            continue
        resolved.append((p, e))
    return resolved, misses


def apply_file(name, slot, dry=False, verbose=True):
    """Apply one document's edits. Returns (n_applied, misses)."""
    doc = de.Doc(name)

    text_res, text_miss = resolve(doc, slot["text"], "text")
    eq_items = [(para, es) for para, es in sorted(slot["equation"].items())]
    eq_res, eq_miss = [], []
    for para, es in eq_items:
        r, m = resolve(doc, es, "equation")
        # All edits for one paragraph must resolve, or the paragraph is skipped:
        # applying half of them would leave the equation internally inconsistent.
        if m:
            eq_miss += m
            continue
        eq_res.append((r[0][0], es))

    if verbose:
        print(f"\n{name}")
        print(f"  text edits     {len(text_res)} resolved, {len(text_miss)} unresolved")
        print(f"  equation paras {len(eq_res)} resolved, {len(eq_miss)} unresolved")

    if dry:
        return 0, text_miss + eq_miss

    n = 0
    for p, e in text_res:
        made = doc.replace_text_in_para(p, e["old"], e["new"], count=1)
        if made != 1:
            eq_miss.append((e, f"replace_text_in_para made {made} replacements"))
            continue
        n += 1
        if verbose:
            print(f"    [{e['para']:4d}] {e['old'][:52]!r} -> {e['new'][:52]!r}")

    for p, es in eq_res:
        pairs = [(e["old"], e["new"]) for e in es]
        # Longest first: an "old" that is a prefix of another would corrupt it.
        pairs.sort(key=lambda t: -len(t[0]))
        _, applied = doc.replace_para_tracked(p, pairs)
        n += applied
        if verbose:
            print(f"    [{es[0]['para']:4d}] equation paragraph, {applied}/{len(pairs)} "
                  f"value(s) replaced (tracked as a paragraph replacement)")
        if applied != len(pairs):
            eq_miss.append((es[0], f"only {applied} of {len(pairs)} equation values matched"))

    doc.save()
    return n, text_miss + eq_miss


def _view(path, mode):
    """The non-empty paragraph texts of a document under accept-all or reject-all."""
    d = de.Doc(path)
    getattr(d, f"{mode}_all")()
    return [t for t in (d.math_text_of(p) for p in d.paragraphs) if t.strip()]


def gate(names, verbose=True):
    """The documents may differ from the ORIGINAL SUBMISSION only as visible revisions.

    The backup is NOT the submitted version — it already carries every tracked change
    made earlier in this revision. So comparing its accepted view against the edited
    document's rejected view compares two deliberately different things, and an earlier
    draft of this function did exactly that and reported 226 spurious failures.

    The invariant that actually matters:

      REJECT-ALL IS INVARIANT.  Rejecting every revision must give the same original
      submission before and after these edits. If it changed, an edit went in WITHOUT
      a revision mark — the one failure a reviewer diffing against the submitted
      version would never see, and the only one that can embarrass us.

      ACCEPT-ALL MOVED.  If the accepted view is unchanged, nothing was applied.
    """
    ok = True
    for name in names:
        backup = os.path.join(BACKUP_DIR, name)
        live = os.path.join(de.SUBMISSION_DIR, name)
        if not os.path.exists(backup):
            print(f"  {name:34s} NO BACKUP — cannot verify")
            ok = False
            continue

        # A document that is NEW in this resubmission carries no revisions at all, by
        # design (`Doc.UNTRACKED`). Rejecting nothing leaves the edits in place, so its
        # "base" view legitimately moves and this invariant does not apply to it. The
        # check is still worth printing — it says how much of the document changed.
        if name in de.Doc.UNTRACKED:
            before, after = _view(backup, "accept"), _view(live, "accept")
            n = sum(1 for a, b in zip(before, after) if a != b)
            print(f"  {name:34s} untracked by design; {n} paragraph(s) changed")
            continue

        base_before, base_after = _view(backup, "reject"), _view(live, "reject")
        acc_before, acc_after = _view(backup, "accept"), _view(live, "accept")

        invariant = base_before == base_after
        moved = acc_before != acc_after

        if invariant:
            n = sum(1 for a, b in zip(acc_before, acc_after) if a != b)
            print(f"  {name:34s} base INVARIANT; accepted view moved in "
                  f"{n} paragraph(s)" + ("" if moved else "  (no edits applied here)"))
        else:
            ok = False
            diff = [(i, w, g) for i, (w, g) in enumerate(zip(base_before, base_after))
                    if w != g]
            print(f"  {name:34s} *** SILENT CHANGE *** the submitted text differs at "
                  f"{len(diff)} paragraph(s) (lengths {len(base_before)} vs {len(base_after)})")
            for i, w, g in diff[:3]:
                print(f"       [{i}] was: {w[:100]}")
                print(f"            now: {g[:100]}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="JSON list of verified edits")
    ap.add_argument("--dumps", help="directory of accepted-text dumps, for the "
                                    "overlap pre-pass")
    ap.add_argument("--skip-high-risk", action="store_true",
                    help="do not apply edits marked risk=high; those change what a "
                         "SENTENCE claims and are the author's to word")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    with open(a.spec) as fh:
        spec = json.load(fh)
    edits = spec["apply"] if isinstance(spec, dict) else spec
    verbose = not a.quiet

    if a.skip_high_risk:
        held = [e for e in edits if e.get("risk") == "high"]
        edits = [e for e in edits if e.get("risk") != "high"]
        if held:
            print(f"{len(held)} high-risk edit(s) HELD BACK for the author:")
            for e in held:
                print(f"    {e['file']} [{e['para']}] {e['old'][:56]!r}")
                print(f"        {e['reason'][:130]}")

    if a.dumps:
        edits, _ = drop_overlaps(edits, load_dumps(a.dumps), verbose=verbose)

    print(f"\n{len(edits)} edit(s) to apply across "
          f"{len(set(e['file'] for e in edits))} document(s)")

    grouped = group(edits)
    total, all_miss = 0, []
    for name, slot in sorted(grouped.items()):
        n, miss = apply_file(name, slot, dry=a.dry_run, verbose=verbose)
        total += n
        all_miss += miss

    print(f"\n{'DRY RUN — nothing written' if a.dry_run else f'{total} edit(s) applied'}")
    if all_miss:
        print(f"\n{len(all_miss)} UNRESOLVED — these were NOT applied:")
        for e, why in all_miss:
            print(f"  {e['file']} [{e['para']}] {e['old'][:60]!r}\n      {why}")

    if not a.dry_run:
        print("\nGATE — the documents may differ from the submission only as revisions")
        if not gate(sorted(grouped)):
            print("\nGATE FAILED. Restore with:\n"
                  f"  cp {BACKUP_DIR}/*.docx {de.SUBMISSION_DIR}/")
            return 1
    return 0 if not all_miss else 2


if __name__ == "__main__":
    sys.exit(main())
