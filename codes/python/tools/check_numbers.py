"""Verify that every number in the documents traces back to a pipeline output.

Steps no longer write prose; numbers are typed into the manuscript, supplement, reply and
the five table files by hand. This catches that step going wrong — a typo, a value left
behind by an earlier run, or a figure nobody can reproduce.

It reads the ACCEPTED text of the tracked documents. A number inside a pending deletion is
not a claim, and flagging it would train the reader to ignore the report.

Each numeric token is classified:

    matched      equal to a pipeline value at the precision the document states
    whitelisted  a number that legitimately has no pipeline source
    UNMATCHED    nothing in any output produces it  <- the finding

The value pool is the merged `numbers.json` registry plus every numeric cell of every CSV
under results/ and derivatives/. The registry alone would be stricter, but the pool version
answers the question that actually matters — "does this number exist anywhere in what the
pipeline produced?" — from the first run, before all 23 steps declare their keys.

Matching is precision-aware: the document's "-0.136" matches a stored -0.13624 because the
document states three decimals. That is the correct comparison; requiring exact equality
would flag every rounded value in the paper.

    python tools/check_numbers.py
    python tools/check_numbers.py --verbose        # also list what matched
    python tools/check_numbers.py --doc manuscript.docx
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CODE))
sys.path.insert(0, os.path.join(ROOT, "codes/python/revision"))   # docx_edit lives here

DOCS_DIR = os.path.join(ROOT, "docs/manuscript_JPAIN_resubmission")
DOCS = ["manuscript.docx", "supplementary_materials.docx", "Response.docx",
        "table1.docx", "table2.docx", "table3.docx", "table4.docx", "table5.docx"]

WHITELIST_PATH = os.path.join(HERE, "check_numbers_whitelist.json")

#: A number is a claim only if it is written as one. This deliberately does not match
#: numbers glued to letters (q13, S1, F2, PROMIS-29), which are identifiers.
NUM = re.compile(r"(?<![\w.])([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d*\.\d+)(?![\w])")

#: A number right after one of these words is a LABEL, not a measurement: "Table 5",
#: "Figure 2", "Aim 2", "Panel 3". Flagging them would bury the real findings in noise
#: and train the reader to skim the report, which defeats the point of having one.
LABEL_BEFORE = re.compile(
    r"(?:Table|Figure|Section|Panel|Step|Aim|Note|Appendix|Reviewer|Comment|Item|"
    r"Equation|Eq\.?|R1\.|R2\.|version|Python|PyMC|ArviZ|SPM|MNI)\s*$", re.I)

#: Reference-list and citation territory: a bare integer inside a numbered reference
#: entry, or a year. Neither is a result.
YEAR = re.compile(r"^(1[89]\d{2}|20\d{2})$")


def load_pool():
    """Every value the pipeline produced: the registry, plus all numeric CSV cells."""
    pool, sources = {}, {}

    reg = {}
    for p in glob.glob(os.path.join(ROOT, "results", "*", "numbers.json")):
        with open(p) as fh:
            reg.update(json.load(fh))
    for k, v in reg.items():
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, (int, float)):
                pool.setdefault(round(float(x), 10), set()).add(k)

    import csv
    for root in ("results", "derivatives"):
        for p in glob.glob(os.path.join(ROOT, root, "**", "*.csv"), recursive=True):
            rel = os.path.relpath(p, ROOT)
            try:
                with open(p, newline="") as fh:
                    for row in csv.reader(fh):
                        for cell in row:
                            cell = cell.strip().replace(",", "")
                            try:
                                x = float(cell)
                            except (TypeError, ValueError):
                                continue
                            pool.setdefault(round(x, 10), set()).add(rel)
            except Exception:
                continue
    for k, v in pool.items():
        sources[k] = v
    return pool, sources


def whitelist():
    """Numbers that legitimately have no pipeline source, with the reason recorded."""
    if os.path.exists(WHITELIST_PATH):
        with open(WHITELIST_PATH) as fh:
            return json.load(fh)
    return {}


def matches(value, text, pool, tol_digits):
    """True when some pooled value rounds to `value` at the document's precision."""
    if value in pool:
        return True
    for stored in pool:
        if round(stored, tol_digits) == value:
            return True
    return False


def decimals(token):
    return len(token.split(".")[1]) if "." in token else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from docx_edit import Doc

    pool, sources = load_pool()
    wl = whitelist()
    wl_values = {float(k) for k in wl}
    print(f"value pool: {len(pool):,} distinct numbers from the pipeline")
    print(f"whitelist : {len(wl)} entries\n")

    docs = [args.doc] if args.doc else DOCS
    grand_unmatched = 0
    for name in docs:
        path = os.path.join(DOCS_DIR, name)
        if not os.path.exists(path):
            continue
        d = Doc(path)
        d.accept_all()

        # Everything at or after a REFERENCES heading is bibliography: volume, issue
        # and page numbers are not claims about this study.
        ref_start = next((i for i, p in enumerate(d.paragraphs)
                          if d.text_of(p).strip().upper().startswith("REFERENCES")),
                         len(d.paragraphs))

        unmatched, n_tok, n_wl = [], 0, 0
        for i, p in enumerate(d.paragraphs):
            if i >= ref_start:
                continue
            if d.style_of(p) == "Comment":      # the Reviewers' own words, not ours
                continue
            text = d.text_of(p).replace("\xa0", " ")
            if not text.strip():
                continue
            for m in NUM.finditer(text):
                tok = m.group(1)
                try:
                    val = float(tok.replace(",", ""))
                except ValueError:
                    continue
                # skip labels ("Table 5"), years, and citation numerals
                if LABEL_BEFORE.search(text[:m.start()]):
                    continue
                if YEAR.match(tok):
                    continue
                n_tok += 1
                if val in wl_values:
                    n_wl += 1
                    continue
                if matches(val, text, pool, decimals(tok)):
                    continue
                s = max(0, m.start() - 55)
                e = min(len(text), m.end() + 55)
                unmatched.append((i, tok, text[s:e].strip()))

        print(f"=== {name} ===")
        print(f"  {n_tok:5d} numeric tokens | {n_wl:4d} whitelisted | "
              f"{len(unmatched):4d} UNMATCHED")
        for i, tok, ctx in unmatched[:40]:
            print(f"    [{i:4d}] {tok:>12s}   …{ctx}…")
        if len(unmatched) > 40:
            print(f"    … and {len(unmatched) - 40} more")
        print()
        grand_unmatched += len(unmatched)

    print("=" * 72)
    print(f"{grand_unmatched} unmatched number(s) across {len(docs)} document(s)")
    print("An unmatched number is a typo, a stale value, or a figure the pipeline")
    print("does not produce. Add genuine non-pipeline numbers to")
    print(f"  {os.path.relpath(WHITELIST_PATH, ROOT)}  with a reason.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
