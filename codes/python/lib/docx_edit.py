"""Surgical, tracked-change-safe editing of the submission's Word documents.

The three documents (manuscript, supplement, response) carry Pedro's own tracked
changes and, in the manuscript, live Mendeley citations.  Three invariants govern
every edit made through this module:

1.  **Only `word/document.xml` is ever rewritten.**  Every other part of the package
    -- `customXml/`, `docProps/custom.xml` (where Mendeley keeps the document id, user
    id and citation style), styles, numbering, media, rels -- is copied through byte
    for byte in its original zip order.

2.  **Mendeley citations are untouchable.**  Mendeley Cite stores each citation as a
    content control -- a `w:sdt` whose `w:tag` begins `MENDELEY_CITATION_` -- and not
    as a field code, so grepping for `ADDIN CSL_CITATION` finds nothing and proves
    nothing.  `manuscript.docx` holds 103 of these.  Rewriting text inside one breaks
    the link back to the library, so `replace_text_in_para` raises rather than edit
    there.  The supplement has none; its citations really are plain text.

3.  **Every revision is authored by Pedro**, using the exact author string already
    present in the documents, so Word folds new edits into his existing revision set
    and renders them in a single colour rather than as a second reviewer.

The second invariant has a subtlety that matters visually.  Editing text that already
sits inside one of Pedro's `w:ins` runs must NOT produce a nested `w:del`/`w:ins`
pair -- Word draws that as a change-on-a-change.  Such text is rewritten in place,
with no new markup, because it is already marked as his insertion.  Only untracked
original text gets the delete/insert treatment.  `replace_text_in_para` decides which
case applies by looking at the run's ancestry.

Usage:
    from docx_edit import Doc
    d = Doc("manuscript.docx")
    for i, p in enumerate(d.paragraphs):
        print(i, d.style_of(p), d.text_of(p))
    d.replace_text_in_para(p, "Note S1", "Section S8")
    d.save()
"""
import os
import re
import shutil
import zipfile

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W16DU = "http://schemas.microsoft.com/office/word/2023/wordml/word16du"
OMML = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W}

SUBMISSION_DIR = ("/orange/cruzalmeida/pvaldeshernandez/Sleep-Pain_Coupling/UPLOAD2"
                  "/docs/manuscript_JPAIN_resubmission")

# The author string already used by every revision in the documents.  Must match
# character for character (note: no space after the comma) or Word treats the edit as
# coming from a different person and colours it separately.
AUTHOR = "Valdes Hernandez,Pedro Antonio"
DATE = "2026-08-06T02:00:00Z"
DATE_UTC = "2026-08-06T06:00:00Z"


def q(tag):
    return f"{{{W}}}{tag}"


class MendeleyGuard(Exception):
    """Raised when an edit would land inside a Mendeley citation content control."""




class Doc:
    """One .docx, opened for surgical editing.

    `track` decides whether edits are recorded as revisions. The manuscript and the
    supplement are revisions of documents the reviewers already have, so their edits
    must be tracked. `Response.docx` is written fresh for the resubmission -- nobody is
    comparing it against a previous version -- so it is edited in place, with no markup.
    """

    #: files that are new in this resubmission, and so carry no revision marks
    UNTRACKED = {"Response.docx"}

    def __init__(self, path, track=None):
        self.path = path if os.path.isabs(path) else os.path.join(SUBMISSION_DIR, path)
        if track is None:
            track = os.path.basename(self.path) not in self.UNTRACKED
        self.track = track
        with zipfile.ZipFile(self.path) as z:
            self.xml = z.read("word/document.xml")
        self.tree = etree.fromstring(self.xml)
        self._next_id = self._max_revision_id() + 1
        # Auxiliary package parts loaded on demand and rewritten by `save` only if
        # touched. Everything absent from here is still copied byte for byte.
        self._parts = {}
        # Binary parts (figures) queued for replacement by `save`.
        self._media = {}

    # ---------------------------------------------------------------- reading

    @property
    def paragraphs(self):
        """Every paragraph in document order, including those inside textboxes."""
        return self.tree.findall(f".//{q('p')}")

    def math_text_of(self, p):
        """`text_of` with Word's equation objects rendered too.

        This manuscript stores most of its coefficients as OMML equations (173 of them),
        whose characters live in `m:t`, not `w:t`. `text_of` therefore renders them as
        nothing at all, which is invisible and dangerous: a numeric audit that reads only
        `text_of` cannot see the very numbers it is auditing. Use this for checking
        numbers; use `text_of` for anchoring edits, since an anchor that spans an equation
        cannot be edited as text anyway.
        """
        out = []
        for el in p.iter(q("t"), q("noBreakHyphen"), q("tab"),
                         f"{{{OMML}}}t"):
            if el.tag == q("noBreakHyphen"):
                out.append("-")
            elif el.tag == q("tab"):
                out.append("\t")
            else:
                out.append(el.text or "")
        return "".join(out)

    def text_of(self, p):
        """The accepted view of a paragraph: insertions kept, deletions dropped.

        Deleted text lives in `w:delText`, never in `w:t`, so collecting `w:t` in
        document order is exactly the text a reader sees with changes accepted.
        This is why python-docx is unusable here -- its `.text` skips `w:ins`.

        `w:noBreakHyphen` is its own element, not a character inside `w:t`, so it has
        to be rendered explicitly or "pain-to-sleep" comes back as "paintosleep" and
        every anchor containing a hyphen silently fails to match.
        """
        out = []
        for el in p.iter(q("t"), q("noBreakHyphen"), q("tab")):
            if el.tag == q("t"):
                out.append(el.text or "")
            elif el.tag == q("noBreakHyphen"):
                out.append("-")
            else:
                out.append("\t")
        return "".join(out)

    def search_text_of(self, p):
        """`text_of` with the invisible variants folded to their ASCII equivalents.

        Word scatters non-breaking spaces and typographic hyphens through this
        manuscript -- "Figure\\u00a0S4" looks exactly like "Figure S4" on screen but
        does not match it. Search against this; edit against `text_of`.
        """
        return (self.text_of(p)
                .replace(" ", " ").replace(" ", " ").replace(" ", " ")
                .replace("‑", "-").replace("–", "-").replace("—", "-"))

    def raw_text_of(self, p):
        """The text with deletions still visible, for diagnosing revision state."""
        out = []
        for t in p.iter(q("t"), q("delText")):
            out.append(t.text or "")
        return "".join(out)

    def style_of(self, p):
        s = p.find(f"{q('pPr')}/{q('pStyle')}")
        return s.get(q("val")) if s is not None else ""

    def is_inserted(self, node):
        """True when the node sits inside one of Pedro's pending insertions."""
        for anc in node.iterancestors():
            if anc.tag == q("ins"):
                return True
        return False

    def in_mendeley(self, node):
        """True when the node sits inside a Mendeley citation content control.

        Mendeley Cite wraps every citation in a `w:sdt` tagged `MENDELEY_CITATION_...`.
        The rendered citation text is regenerated from that control, so editing it by
        hand desynchronises the document from the library.
        """
        for anc in node.iterancestors():
            if anc.tag == q("sdt"):
                tag = anc.find(f"{q('sdtPr')}/{q('tag')}")
                if tag is not None and (tag.get(q("val")) or "").startswith("MENDELEY"):
                    return True
        return False

    # ---------------------------------------------------------------- editing

    def _max_revision_id(self):
        ids = [0]
        for el in self.tree.iter():
            if el.tag in (q("ins"), q("del")):
                v = el.get(q("id"))
                if v and v.isdigit():
                    ids.append(int(v))
        return max(ids)

    def _rev_id(self):
        self._next_id += 1
        return str(self._next_id)

    def _rev_attrs(self, el):
        el.set(q("id"), self._rev_id())
        el.set(q("author"), AUTHOR)
        el.set(q("date"), DATE)
        el.set(f"{{{W16DU}}}dateUtc", DATE_UTC)
        return el

    def _el(self, tag, **attrs):
        e = etree.SubElement(etree.Element("x"), q(tag))
        for k, v in attrs.items():
            e.set(q(k), v)
        return e

    def _clone_run(self, run, text):
        """A copy of `run` carrying `text`, so formatting (size, italic) is preserved."""
        new = etree.fromstring(etree.tostring(run))
        for child in list(new):
            if child.tag != q("rPr"):
                new.remove(child)
        t = etree.SubElement(new, q("t"))
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = text
        return new

    def replace_text_in_para(self, p, old, new, count=0):
        """Replace `old` with `new` inside one paragraph, as a revision by Pedro.

        Returns the number of replacements made.  Handles the common case where the
        target text is split across several runs by first locating it in the
        paragraph's concatenated text, then rewriting only the runs it touches.
        """
        made, start = 0, 0
        while True:
            # Re-collect every pass: a replacement swaps runs out of the tree, so a
            # list captured earlier would hold detached elements.
            runs = [r for r in p.findall(f".//{q('r')}")
                    if r.find(q("t")) is not None]
            spans, pos = [], 0
            for r in runs:
                n = len(r.find(q("t")).text or "")
                spans.append((pos, pos + n, r))
                pos += n
            whole = "".join((r.find(q("t")).text or "") for r in runs)
            # Resume past the text just written, so a replacement that contains the
            # search string (\"Table S1\" -> \"Section S5, Table S1\") cannot re-match.
            idx = whole.find(old, start)
            if idx < 0 or (count and made >= count):
                break
            start = idx + len(new)
            end = idx + len(old)
            touched = [(s, e, r) for s, e, r in spans if s < end and e > idx]
            if any(self.in_mendeley(r) for _, _, r in touched):
                raise MendeleyGuard(
                    f"refusing to edit {old!r}: it lies inside a Mendeley citation")
            self._apply_span(p, touched, idx, end, new)
            made += 1
        return made

    def _del_run(self, run, text):
        """A copy of `run` whose text is marked deleted (`w:delText`, not `w:t`)."""
        new = etree.fromstring(etree.tostring(run))
        for child in list(new):
            if child.tag != q("rPr"):
                new.remove(child)
        dt = etree.SubElement(new, q("delText"))
        dt.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        dt.text = text
        return new

    def _apply_span(self, p, touched, idx, end, new):
        """Rewrite the runs covering [idx, end) so that region reads `new`.

        Each run is treated according to its own revision state, because a target
        string routinely straddles both -- the note headings, for instance, have an
        untracked "Note S", then a digit Pedro inserted, then untracked ": Title".

        * A run already inside one of Pedro's pending insertions is edited **in
          place**.  Removing text he has inserted but not yet accepted leaves nothing
          behind, exactly as Word does; wrapping it in `w:del` instead would draw a
          deletion on top of an insertion in a second colour.
        * An untracked run has its old text wrapped in `w:del`.

        The replacement text is emitted once, at the first touched run -- inside the
        existing `w:ins` if that run is already an insertion, otherwise as a new one.
        """
        first_s, _, first_r = touched[0]
        head = (first_r.find(q("t")).text or "")[:idx - first_s]
        last_s, _, last_r = touched[-1]
        tail = (last_r.find(q("t")).text or "")[end - last_s:]

        for _, _, r in touched:
            seg = r.find(q("t")).text or ""
            keep_head = head if r is first_r else ""
            keep_tail = tail if r is last_r else ""
            stop = len(seg) - len(keep_tail) if keep_tail else len(seg)
            gone = seg[len(keep_head):stop]
            put = new if r is first_r else ""

            if not self.track or self.is_inserted(r):
                t = r.find(q("t"))
                t.text = keep_head + put + keep_tail
                t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                continue

            parent = r.getparent()
            cursor = list(parent).index(r)
            parent.remove(r)
            pieces = []
            if keep_head:
                pieces.append(self._clone_run(r, keep_head))
            if put:
                ins = self._rev_attrs(self._el("ins"))
                ins.append(self._clone_run(r, put))
                pieces.append(ins)
            if gone:
                dele = self._rev_attrs(self._el("del"))
                dele.append(self._del_run(r, gone))
                pieces.append(dele)
            if keep_tail:
                pieces.append(self._clone_run(r, keep_tail))
            for offset, piece in enumerate(pieces):
                parent.insert(cursor + offset, piece)

    def accept_all(self):
        """Accept every revision: unwrap `w:ins`, delete `w:del`.

        For a document that is new in this resubmission, revision marks are noise --
        there is no earlier version to compare against, so the reader should just see
        the text. Returns the number of revisions resolved. The accepted view of the
        document is unchanged by construction, which is worth asserting after calling.
        """
        n = 0
        for el in list(self.tree.iter(q("ins"))):
            parent = el.getparent()
            at = list(parent).index(el)
            for offset, child in enumerate(list(el)):
                parent.insert(at + offset, child)
            parent.remove(el)
            n += 1
        for el in list(self.tree.iter(q("del"))):
            el.getparent().remove(el)
            n += 1
        return n

    def reject_all(self):
        """Reject every revision: drop `w:ins` content, restore `w:del` content.

        The inverse of `accept_all`, and the only way to answer the question that
        actually matters for a resubmission: does what we are sending still contain,
        as tracked changes, everything that differs from the version the reviewers
        already have? If rejecting reproduces the submitted text, every difference is
        visible in the reviewing pane. If it does not, something was changed silently.

        `w:delText` carries deleted characters and must be read back as `w:t`, since
        the whole point is to reconstitute text a reader can see.
        """
        n = 0
        for el in list(self.tree.iter(q("ins"))):
            el.getparent().remove(el)
            n += 1
        for el in list(self.tree.iter(q("del"))):
            parent = el.getparent()
            at = list(parent).index(el)
            for offset, child in enumerate(list(el)):
                for dt in child.iter(q("delText")):
                    dt.tag = q("t")
                parent.insert(at + offset, child)
            parent.remove(el)
            n += 1
        # A paragraph whose MARK was inserted must also vanish; one whose mark was
        # deleted must come back. Those live in w:pPr/w:rPr and are handled above by
        # the ins/del sweep, which already removed or restored them.
        return n

    def run_spans(self, p):
        """`[(start, end, run)]` over the accepted text, in document order.

        The offsets line up with `text_of`, so a match found there maps straight back
        to the runs carrying it.  Runs inside `w:del` hold `w:delText`, contribute no
        characters, and therefore come back as empty spans -- the same convention
        `text_of` uses.
        """
        out, pos = [], 0
        for r in p.findall(f".//{q('r')}"):
            n = 0
            for el in r:
                if el.tag == q("t"):
                    n += len(el.text or "")
                elif el.tag in (q("noBreakHyphen"), q("tab")):
                    n += 1
            out.append((pos, pos + n, r))
            pos += n
        return out

    def _trim_run(self, run, lo, hi):
        """Keep only characters [lo, hi) of `run`, dropping the children outside."""
        pos = 0
        for el in list(run):
            if el.tag == q("t"):
                n = len(el.text or "")
            elif el.tag in (q("noBreakHyphen"), q("tab")):
                n = 1
            else:
                continue          # rPr and other zero-width children stay put
            s, e = pos, pos + n
            pos = e
            keep_s, keep_e = max(s, lo), min(e, hi)
            if keep_s >= keep_e:
                run.remove(el)
            elif el.tag == q("t"):
                el.text = (el.text or "")[keep_s - s:keep_e - s]
                el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

    def _split_run(self, run, offset, length):
        """Split `run` after `offset` characters, leaving both halves in the tree."""
        right = etree.fromstring(etree.tostring(run))
        run.addnext(right)
        self._trim_run(run, 0, offset)
        self._trim_run(right, offset, length)
        return run, right

    def cover_runs(self, p, start, end):
        """Split runs at both boundaries and return those covering [start, end).

        After this the range is spanned by whole runs, so a property can be set on
        each without touching a neighbouring character.
        """
        for cut in (end, start):        # trailing edge first: it cannot move `start`
            for s, e, r in self.run_spans(p):
                if s < cut < e:
                    self._split_run(r, cut - s, e - s)
                    break
        return [r for s, e, r in self.run_spans(p) if s < end and e > start and e > s]

    # CT_RPr is a sequence, not a choice: a property inserted out of order is a schema
    # violation and Word refuses the file. Only the prefix up to `b` is needed here.
    RPR_ORDER = ("ins", "del", "moveFrom", "moveTo", "rStyle", "rFonts", "b", "bCs",
                 "i", "iCs", "caps", "smallCaps", "strike", "dstrike", "outline",
                 "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish",
                 "webHidden", "color", "spacing", "w", "kern", "position", "sz",
                 "szCs", "highlight", "u", "effect", "bdr", "shd", "fitText",
                 "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout",
                 "specVanish", "oMath", "rPrChange")

    def _insert_ordered(self, rPr, el):
        rank = self.RPR_ORDER.index(el.tag.split("}")[1])
        for child in rPr:
            name = child.tag.split("}")[1]
            if name in self.RPR_ORDER and self.RPR_ORDER.index(name) > rank:
                child.addprevious(el)
                return
        rPr.append(el)

    def set_bold(self, p, start, end):
        """Make [start, end) bold, as a formatting revision by Pedro.

        Word records a formatting change as `w:rPrChange` carrying the run's previous
        properties, which is what makes it show up as his change rather than as text
        that was always bold.  In the untracked document the property is simply set.
        Runs that are already bold are left alone, so the pass is idempotent.
        """
        n = 0
        for r in self.cover_runs(p, start, end):
            if self.in_mendeley(r):
                raise MendeleyGuard(f"refusing to bold [{start}, {end}): inside a citation")
            rPr = r.find(q("rPr"))
            if rPr is None:
                rPr = etree.Element(q("rPr"))
                r.insert(0, rPr)
            b = rPr.find(q("b"))
            if b is not None and b.get(q("val")) not in ("0", "false"):
                continue
            was = etree.fromstring(etree.tostring(rPr))
            for stale in was.findall(q("rPrChange")):
                was.remove(stale)
            if b is None:
                self._insert_ordered(rPr, etree.Element(q("b")))
            else:
                b.attrib.pop(q("val"), None)
            if self.track:
                for stale in rPr.findall(q("rPrChange")):
                    rPr.remove(stale)
                change = self._el("rPrChange", id=self._rev_id(), author=AUTHOR, date=DATE)
                change.append(was)
                rPr.append(change)      # w:rPrChange is last in CT_RPr
            n += 1
        return n

    def apply_char_style(self, p, start, end, style):
        """Give the accepted-text range [start, end) the character style `style`.

        Runs are split at the two boundaries so the range is covered by whole runs,
        then each gets a `w:rStyle`.  Direct formatting already on a run -- the
        different font on an abbreviation inside a quotation, say -- is preserved;
        only the style reference is added.  Returns the number of runs styled.

        Word records this as formatting, not as content, so it is invisible to a
        text comparison: the accepted text is unchanged by construction.
        """
        n = 0
        for r in self.cover_runs(p, start, end):
            if self.in_mendeley(r):
                raise MendeleyGuard(
                    f"refusing to style [{start}, {end}): it lies inside a citation")
            rPr = r.find(q("rPr"))
            if rPr is None:
                rPr = etree.Element(q("rPr"))
                r.insert(0, rPr)
            rs = rPr.find(q("rStyle"))
            if rs is None:
                rs = etree.Element(q("rStyle"))
                rPr.insert(0, rs)       # w:rStyle is the first child of w:rPr
            if rs.get(q("val")) != style:
                rs.set(q("val"), style)
                n += 1
        return n

    def clear_char_style(self, r, style):
        """Drop `style` from a run, and the `w:rPr` with it if nothing else is left."""
        rPr = r.find(q("rPr"))
        rs = rPr.find(q("rStyle")) if rPr is not None else None
        if rs is None or rs.get(q("val")) != style:
            return False
        rPr.remove(rs)
        if len(rPr) == 0:
            r.remove(rPr)
        return True

    def move_blocks(self, start, stop, before):
        """Move body-level children [start, stop) so they sit before index `before`.

        Word would record this as a `w:moveFrom`/`w:moveTo` pair, but the block being
        moved contains a table, and malformed move markup is the one thing Word
        refuses to open.  The move is therefore made silently; the tracked renumbering
        of the headings on either side is what makes the new order legible.
        """
        body = self.tree.find(q("body"))
        children = list(body)
        block = children[start:stop]
        # Resolve the destination to an element BEFORE removing anything: `before` is an
        # index into the pre-removal body, and a forward move would shift it by the size
        # of the block. Holding the element instead makes the direction irrelevant.
        anchor = children[before] if before < len(children) else None
        for el in block:
            body.remove(el)
        at = list(body).index(anchor) if anchor is not None else len(body)
        for offset, el in enumerate(block):
            body.insert(at + offset, el)
        return len(block)

    def prepend_to_para(self, p, text):
        """Put `text` at the very start of a paragraph as an insertion by Pedro."""
        first = None
        for r in p.findall(f".//{q('r')}"):
            if r.find(q("t")) is not None:
                first = r
                break
        if first is None:
            return False
        if not self.track or self.is_inserted(first):
            t = first.find(q("t"))
            t.text = text + (t.text or "")
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            return True
        ins = self._rev_attrs(self._el("ins"))
        ins.append(self._clone_run(first, text))
        parent = first.getparent()
        parent.insert(list(parent).index(first), ins)
        return True

    def math_runs(self, p):
        """Every OMML text node in the paragraph, in order: [(start, end, m:t node)]."""
        M = f"{{{OMML}}}"
        spans, pos = [], 0
        for t in p.iter(f"{M}t"):
            n = len(t.text or "")
            spans.append((pos, pos + n, t))
            pos += n
        return spans

    def replace_math_text_in_para(self, p, old, new, count=0):
        """Replace `old` with `new` inside a paragraph's EQUATIONS.

        More than half the manuscript's numbers live in OMML rather than in `w:t`
        (311 against 253), so `replace_text_in_para` cannot reach them: it walks
        `w:t` and an equation stores its characters in `m:t`.

        IMPORTANT LIMITATION, and the reason every caller must log what it did: this
        edits the text node in place and is therefore NOT a tracked change. Word does
        not represent revisions inside an equation the way it does in a paragraph, and
        fabricating `m:del`/`m:ins` markup is one of the reliable ways to make Word
        refuse the file. A number changed here will not appear in the reviewing pane,
        so the change log is the only record.

        Handles a match spanning several `m:t` nodes, which happens when Word splits a
        number across runs after an edit.
        """
        made, start = 0, 0
        while True:
            spans = self.math_runs(p)
            whole = "".join((t.text or "") for _, _, t in spans)
            idx = whole.find(old, start)
            if idx < 0 or (count and made >= count):
                break
            end = idx + len(old)
            start = idx + len(new)
            touched = [(s, e, t) for s, e, t in spans if s < end and e > idx]
            first_s, _, first_t = touched[0]
            last_s, _, last_t = touched[-1]
            head = (first_t.text or "")[:idx - first_s]
            tail = (last_t.text or "")[end - last_s:]
            for _, _, t in touched:
                t.text = ""
            first_t.text = head + new
            if last_t is not first_t:
                last_t.text = tail
            else:
                first_t.text = head + new + tail
            made += 1
        return made

    # ------------------------------------------------- equations -> plain text

    #: An equation is a REPORTED VALUE when it is a short left-hand side, one
    #: relation, and a purely numeric right-hand side: "lambda_ps = -0.141",
    #: "p < 0.001", "F(2,226) = 16.56". The model equations do not match, because
    #: their right-hand sides are expressions rather than numbers -- which is the
    #: distinction that keeps the specification typeset as mathematics.
    #: `\u2248` (approximately) counts as a relation -- the old Supplementary Notes
    #: report values as "lambda_1 = ~0.128" -- and the left-hand side may be EMPTY,
    #: because those sections also write a bare "~25" mid-sentence.
    REPORTED = re.compile(
        r"^\s*(.{0,26}?)\s*([=<>\u2264\u2265\u2248])\s*([+-]?\d[\d.,]*)\s*$")

    def _omml_runs(self, node, template, sub=False, sup=False):
        """Flatten an OMML subtree into `w:r` runs, preserving sub/superscripts."""
        M = f"{{{OMML}}}"
        out = []
        for child in node:
            tag = etree.QName(child).localname
            if tag == "r":
                txt = "".join(t.text or "" for t in child.iter(f"{M}t"))
                if txt:
                    out.append(self._text_run(txt, template, sub, sup))
            elif tag == "sSub":
                base = child.find(f"{M}e")
                sb = child.find(f"{M}sub")
                out += self._omml_runs(base, template, sub, sup)
                out += self._omml_runs(sb, template, True, False)
            elif tag == "sSup":
                base = child.find(f"{M}e")
                sp = child.find(f"{M}sup")
                out += self._omml_runs(base, template, sub, sup)
                out += self._omml_runs(sp, template, False, True)
            elif tag == "d":                      # delimiter: (...)
                out.append(self._text_run("(", template, sub, sup))
                for e in child.findall(f"{M}e"):
                    out += self._omml_runs(e, template, sub, sup)
                out.append(self._text_run(")", template, sub, sup))
            elif tag in ("e", "acc", "num", "den", "sub", "sup"):
                out += self._omml_runs(child, template, sub, sup)
        return out

    def _text_run(self, text, template, sub=False, sup=False):
        """A `w:r` carrying `text`, cloning `template`'s formatting."""
        r = etree.Element(q("r"))
        if template is not None:
            src = template.find(q("rPr"))
            if src is not None:
                r.append(etree.fromstring(etree.tostring(src)))
        if sub or sup:
            rPr = r.find(q("rPr"))
            if rPr is None:
                rPr = etree.Element(q("rPr"))
                r.insert(0, rPr)
            # strip the Cambria Math face the equation carried; this is prose now
            for f in rPr.findall(q("rFonts")):
                rPr.remove(f)
            va = etree.SubElement(rPr, q("vertAlign"))
            va.set(q("val"), "subscript" if sub else "superscript")
        else:
            rPr = r.find(q("rPr"))
            if rPr is not None:
                for f in rPr.findall(q("rFonts")):
                    rPr.remove(f)
        t = etree.SubElement(r, q("t"))
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = text
        return r

    def equations_to_text(self, p, only_reported=True):
        """Replace this paragraph's equations with equivalent plain-text runs.

        Word stores an equation's characters in `m:t`, which `replace_text_in_para`
        cannot reach and which Word will not record a tracked revision inside. Any
        number left in an equation is therefore both hard to edit and invisible in the
        reviewing pane when it changes. Converting the reported values to text
        homogenises them with the rest of the prose and makes every later correction a
        normal tracked change.

        The VISIBLE TEXT IS UNCHANGED -- the same characters, with subscripts preserved
        as `w:vertAlign` -- so this is a representation change, not an edit to content.

        `only_reported` restricts the conversion to equations matching `REPORTED`, so
        the model specification stays typeset as mathematics.

        Returns the list of converted strings.
        """
        M = f"{{{OMML}}}"
        done = []
        for om in list(p.iter(f"{M}oMath")):
            txt = "".join(t.text or "" for t in om.iter(f"{M}t"))
            if only_reported and not self.REPORTED.match(txt):
                continue
            # Superscripts alone are NOT a reason to stay typeset: `_omml_runs` renders
            # them as real w:vertAlign runs, so "V_vox = 1.5^3 = 3.375" reads identically
            # as text. STRUCTURE is the reason. A fraction, a radical, a summation or a
            # matrix cannot be linearised without losing meaning -- "e^(-2.05/13) = ~0.85"
            # would read as e minus 2.05 divided by 13 -- so anything carrying one stays
            # an equation. Only flat symbol/sub/superscript sequences convert.
            if only_reported and any(om.find(f".//{M}{k}") is not None
                                     for k in ("f", "nary", "rad", "func",
                                               "limLow", "limUpp", "m", "bar", "acc")):
                continue
            template = None
            for r in om.iter(f"{M}r"):
                template = r
                break
            runs = self._omml_runs(om, template)
            if not runs:
                continue
            parent = om.getparent()
            at = list(parent).index(om)
            # An oMath sits inside a w:p or an m:oMathPara; either way, splice the runs
            # in where it stood so surrounding text keeps its order.
            for offset, r in enumerate(runs):
                parent.insert(at + offset, r)
            parent.remove(om)
            done.append(txt)
        return done

    # ------------------------------------------------------- paragraph deletion

    #: `w:pPr` is a sequence; `w:rPr` sits near its end, before only these two.
    PPR_TAIL = ("sectPr", "pPrChange")

    def _para_mark_rpr(self, p):
        """The paragraph mark's `w:rPr`, created in the right place if absent.

        The paragraph MARK carries its own run properties, and that is where Word
        records whether the mark itself was inserted or deleted -- which is what makes
        a whole paragraph appear as added or removed rather than merely emptied.
        """
        pPr = p.find(q("pPr"))
        if pPr is None:
            pPr = etree.Element(q("pPr"))
            p.insert(0, pPr)
        rPr = pPr.find(q("rPr"))
        if rPr is None:
            rPr = etree.Element(q("rPr"))
            at = len(pPr)
            for i, child in enumerate(pPr):
                if etree.QName(child).localname in self.PPR_TAIL:
                    at = i
                    break
            pPr.insert(at, rPr)
        return rPr

    def _mark_para_mark(self, p, tag):
        """Mark the paragraph mark as inserted or deleted (`tag` is 'ins' or 'del').

        In `CT_ParaRPr` the revision elements come FIRST, unlike `CT_RPr` where they do
        not exist at all -- inserting them anywhere else makes Word declare the file
        corrupt.
        """
        rPr = self._para_mark_rpr(p)
        for existing in rPr.findall(q(tag)):
            rPr.remove(existing)
        rPr.insert(0, self._rev_attrs(self._el(tag)))

    def delete_para(self, p):
        """Delete a whole paragraph as a revision by Pedro.

        The paragraph is not removed from the XML: Word shows a tracked deletion by
        keeping the text and marking it, so removing the element outright would make
        the paragraph vanish from the "All Markup" view and be indistinguishable from
        text that was never there. Runs Pedro inserted but has not accepted ARE removed
        outright, because deleting an unaccepted insertion leaves nothing behind.

        Returns the number of runs affected.
        """
        n = 0
        for r in list(p.findall(f".//{q('r')}")):
            if r.find(q("t")) is None:
                continue
            if self.in_mendeley(r):
                raise MendeleyGuard(
                    "refusing to delete a paragraph containing a Mendeley citation")
            if self.is_inserted(r):
                ins = next((a for a in r.iterancestors() if a.tag == q("ins")), None)
                target = ins if ins is not None and len(ins) == 1 else r
                target.getparent().remove(target)
                n += 1
                continue
            text = r.find(q("t")).text or ""
            parent = r.getparent()
            at = list(parent).index(r)
            parent.remove(r)
            dele = self._rev_attrs(self._el("del"))
            dele.append(self._del_run(r, text))
            parent.insert(at, dele)
            n += 1
        n += self._wrap_math(p, "del")
        self._mark_para_mark(p, "del")
        return n

    def _wrap_math(self, p, tag):
        """Wrap the paragraph's top-level equations in `w:ins` or `w:del`.

        Equation content lives in `m:t`, never in `w:t`, so the run loops above walk
        straight past a paragraph that is nothing but an equation and mark nothing at
        all. Section S14's worked derivations are exactly that, and without this a
        "deleted" equation paragraph would survive both accept and reject unchanged --
        i.e. an edit that looks tracked and is not.

        `w:ins`/`w:del` are valid parents of `m:oMathPara`/`m:oMath` in the paragraph
        content model, and both `accept_all` and `reject_all` already treat them
        structurally (unwrap / drop the whole subtree), so no separate handling is
        needed there.

        Returns the number of equations wrapped.
        """
        n = 0
        for child in list(p):
            if child.tag not in (f"{{{OMML}}}oMathPara", f"{{{OMML}}}oMath"):
                continue
            at = list(p).index(child)
            p.remove(child)
            wrap = self._rev_attrs(self._el(tag))
            wrap.append(child)
            p.insert(at, wrap)
            n += 1
        return n

    def _replace_across(self, node, tag, old, new):
        """Replace `old` with `new` in the CONCATENATION of `node`'s `tag` elements.

        The text of an equation, and often of a sentence, is split across many elements
        for reasons of formatting that have nothing to do with meaning. Matching each
        element separately therefore misses any anchor that crosses a boundary.

        The replacement goes into the FIRST element the match touches; the remainder of
        the match is deleted from the elements after it. That keeps the surrounding
        formatting of the first element, which is what a reader sees.

        Returns True if a replacement was made.
        """
        els = node.findall(f".//{tag}")
        if not els:
            return False
        spans, pos = [], 0
        for el in els:
            n = len(el.text or "")
            spans.append((pos, pos + n, el))
            pos += n
        whole = "".join(el.text or "" for el in els)

        idx = whole.find(old)
        if idx < 0:
            return False
        end = idx + len(old)

        first = True
        for s, e, el in spans:
            if e <= idx or s >= end:
                continue
            text = el.text or ""
            lo, hi = max(idx, s) - s, min(end, e) - s
            if first:
                el.text = text[:lo] + new + text[hi:]
                first = False
            else:
                el.text = text[:lo] + text[hi:]
        return True

    def replace_para_tracked(self, p, edits, mark_runs=True):
        """Replace a paragraph with an edited copy, as a tracked change.

        The problem this solves: a number typeset inside an equation object cannot be
        revised in place. Word represents revisions with `w:ins` / `w:del` around RUNS,
        and OMML text lives in `m:t`, outside that machinery — so
        `replace_math_text_in_para` silently changes the value with no revision mark,
        and a reviewer comparing against the submitted version sees nothing.

        So instead of editing the equation, the whole PARAGRAPH is replaced:

          1. deep-copy the paragraph, equation structure and all;
          2. apply `edits` to the copy's math text and to its ordinary runs;
          3. mark every run of the copy as an insertion;
          4. mark the original as a deletion, via `delete_para`.

        Word renders that as struck-through old paragraph followed by underlined new
        one — a visible, acceptable, rejectable revision — while the equation itself is
        never rebuilt, so nothing about its typesetting can be lost. This is the only
        way to change a number inside an equation and still have it tracked.

        Parameters
        ----------
        p : the paragraph element to replace.
        edits : sequence of (old, new) string pairs, applied in order to the COPY.
            Each is applied to the OMML text and to the ordinary runs, so a value that
            appears in both places is handled once. Longest-first is NOT assumed —
            order the pairs yourself when one old string is a prefix of another.
        mark_runs : bool, default True
            Mark the copy's runs as inserted. False leaves the copy unmarked, which is
            only correct when the caller is going to mark it some other way.

        Returns
        -------
        (new_para, n_applied) : the inserted paragraph element, and how many of the
        `edits` actually matched. A pair that matched NOTHING is reported by the count,
        never silently ignored — an edit that did not apply means the document does not
        say what the caller thought it said.
        """
        import copy as _copy

        if self.in_mendeley(p):
            raise MendeleyGuard(
                "refusing to replace a paragraph containing a Mendeley citation")

        new = _copy.deepcopy(p)

        n_applied = 0
        for old, repl in edits:
            # OMML first: that is where the equation values live. SPAN-AWARE, because
            # Word splits an equation into one m:t per typographic atom -- "ln",
            # "(0.104)-", "ln", "(0.005)" -- so an anchor like "ln(0.104)-ln(0.005)"
            # exists only in the CONCATENATION and matches no single element. A
            # per-element replace silently applies the easy half of an edit and leaves
            # an equation internally inconsistent, which is worse than not editing it.
            hit = self._replace_across(new, f"{{{OMML}}}t", old, repl)
            # ordinary runs, for values sitting in the prose of the same paragraph
            hit |= self._replace_across(new, q("t"), old, repl)
            n_applied += int(hit)

        if mark_runs:
            for r in list(new.findall(f".//{q('r')}")):
                if r.find(q("t")) is None:
                    continue
                if next((a for a in r.iterancestors() if a.tag == q("ins")),
                        None) is not None:
                    continue
                parent = r.getparent()
                at = list(parent).index(r)
                parent.remove(r)
                ins = self._rev_attrs(self._el("ins"))
                ins.append(r)
                parent.insert(at, ins)
            self._wrap_math(new, "ins")
            self._mark_para_mark(new, "ins")

        p.addnext(new)
        self.delete_para(p)
        return new, n_applied

    # ------------------------------------------------------------ table markup

    def mark_table_inserted(self, tbl):
        """Mark every row of a table as inserted by Pedro.

        A table added during the revision must show as an insertion, or a reviewer
        reading with Track Changes on sees it as though it had always been there. Word
        needs this in two places at once: `w:trPr/w:ins` on each row, so the row itself
        is an insertion, and `w:ins` around the runs plus the paragraph marks inside
        the cells, so the content is too. Marking only the rows leaves the text looking
        unchanged inside a new row.

        Rows and runs already marked are left alone, so this is safe to re-run and safe
        on a table that was only partly marked.

        Returns (rows_marked, runs_wrapped).
        """
        rows = runs = 0
        for tr in tbl.findall(q("tr")):
            trPr = tr.find(q("trPr"))
            if trPr is None:
                trPr = etree.Element(q("trPr"))
                # CT_Row is (tblPrEx?, trPr?, cells...), so trPr goes first UNLESS the
                # row carries table-property exceptions, which precede it. Some rows in
                # this document do; putting trPr before a tblPrEx makes Word reject the
                # file, and it would only show up on a row that happened to have one.
                at = 1 if tr.find(q("tblPrEx")) is not None else 0
                tr.insert(at, trPr)
            if trPr.find(q("ins")) is None:
                trPr.append(self._rev_attrs(self._el("ins")))
                rows += 1
            for p in tr.iter(q("p")):
                for r in list(p.findall(f".//{q('r')}")):
                    if r.find(q("t")) is None or self.is_inserted(r):
                        continue
                    parent = r.getparent()
                    at = list(parent).index(r)
                    parent.remove(r)
                    ins = self._rev_attrs(self._el("ins"))
                    ins.append(r)
                    parent.insert(at, ins)
                    runs += 1
                if self._para_mark_rpr(p).find(q("ins")) is None:
                    self._mark_para_mark(p, "ins")
        return rows, runs

    # ---------------------------------------------------------------- comments

    COMMENT_PARTS = ("word/comments.xml", "word/commentsExtended.xml",
                     "word/commentsIds.xml", "word/commentsExtensible.xml")

    def _part(self, name):
        """Parse an auxiliary part once, and mark it for rewriting on save."""
        if name not in self._parts:
            with zipfile.ZipFile(self.path) as z:
                if name not in z.namelist():
                    return None
                self._parts[name] = etree.fromstring(z.read(name))
        return self._parts[name]

    def comments(self):
        """Every Word comment: id, author, date, its text, and the text it anchors to.

        Returned in document order of the anchor, which is the order Pedro sees them in
        the reviewing pane.
        """
        root = self._part("word/comments.xml")
        if root is None:
            return []
        bodies = {}
        for c in root.findall(q("comment")):
            bodies[c.get(q("id"))] = {
                "id": c.get(q("id")), "author": c.get(q("author")),
                "date": (c.get(q("date")) or "")[:16],
                "text": " ".join((t.text or "") for t in c.iter(q("t"))).strip(),
                "anchor": "",
            }
        # The anchor is the text BETWEEN commentRangeStart and commentRangeEnd -- what the
        # author actually selected before typing the comment. Reporting the paragraph that
        # merely contains the start marker is wrong whenever a selection begins mid-
        # paragraph or spans several, and it silently points at the wrong text.
        open_ids, sel = set(), {cid: [] for cid in bodies}
        for el in self.tree.iter():
            if el.tag == q("commentRangeStart"):
                open_ids.add(el.get(q("id")))
            elif el.tag == q("commentRangeEnd"):
                open_ids.discard(el.get(q("id")))
            elif open_ids:
                if el.tag in (q("t"), f"{{{OMML}}}t"):
                    for cid in open_ids:
                        if cid in sel:
                            sel[cid].append(el.text or "")
                elif el.tag == q("p"):
                    for cid in open_ids:
                        if cid in sel:
                            sel[cid].append(" ")
        for cid, chunks in sel.items():
            bodies[cid]["anchor"] = "".join(chunks).strip()
            # A collapsed range selects nothing; report where it sits instead.
            if not bodies[cid]["anchor"]:
                for p in self.paragraphs:
                    if any(e.get(q("id")) == cid for e in p.iter(q("commentRangeStart"))):
                        bodies[cid]["anchor"] = f"[insertion point in] {self.text_of(p)}"
                        break
        return list(bodies.values())

    def delete_comment(self, cid):
        """Remove a comment completely: its body, its anchors, and its metadata.

        Word keeps a comment in up to four parts. Deleting it from `comments.xml` alone
        leaves orphaned `commentRangeStart`/`End`/`Reference` marks in the body and stale
        rows in `commentsExtended`/`commentsIds`, which is one of the few states that makes
        Word refuse to open the file.
        """
        cid = str(cid)
        root = self._part("word/comments.xml")
        if root is None:
            return False
        W14 = "{http://schemas.microsoft.com/office/word/2010/wordml}"
        # Collect the comment's own paragraph ids BEFORE removing it: they are the key
        # commentsExtended and commentsIds use, and once the element is gone so are they.
        para_ids = set()
        gone = False
        for c in root.findall(q("comment")):
            if c.get(q("id")) != cid:
                continue
            for cp in c.iter(q("p")):
                pid = cp.get(f"{W14}paraId")
                if pid:
                    para_ids.add(pid)
            root.remove(c)
            gone = True
        if not gone:
            return False
        for tag in ("commentRangeStart", "commentRangeEnd"):
            for e in list(self.tree.iter(q(tag))):
                if e.get(q("id")) == cid:
                    e.getparent().remove(e)
        for r in list(self.tree.iter(q("r"))):
            ref = r.find(q("commentReference"))
            if ref is not None and ref.get(q("id")) == cid:
                r.getparent().remove(r)

        W15 = "{http://schemas.microsoft.com/office/word/2012/wordml}"
        W16 = "{http://schemas.microsoft.com/office/word/2016/wordml/cid}"
        ext = self._part("word/commentsExtended.xml")
        if ext is not None:
            for e in list(ext):
                if e.get(f"{W15}paraId") in para_ids:
                    ext.remove(e)
        ids = self._part("word/commentsIds.xml")
        if ids is not None:
            for e in list(ids):
                if e.get(f"{W16}paraId") in para_ids:
                    ids.remove(e)
        return True

    # ------------------------------------------------------------------- media

    def image_parts_for(self, p):
        """The `word/media/...` parts every image in paragraph `p` draws from.

        A figure is routinely stored twice -- once under `mc:Choice` as a DrawingML blip
        and once under `mc:Fallback` as a VML image -- pointing at two separate copies of
        the same bytes. Replacing only the one Word happens to render leaves the other to
        surface in a different Word version, so callers need both.
        """
        A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
        R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
        V = "{urn:schemas-microsoft-com:vml}"
        rels = self._part("word/_rels/document.xml.rels")
        target = {r.get("Id"): r.get("Target") for r in rels}
        rids = [b.get(f"{R}embed") for b in p.iter(f"{A}blip")]
        rids += [d.get(f"{R}id") for d in p.iter(f"{V}imagedata")]
        out = []
        for rid in rids:
            t = target.get(rid)
            if t and not t.startswith("word/"):
                t = "word/" + t
            if t and t not in out:
                out.append(t)
        return out

    def replace_media(self, part_name, source_path):
        """Swap the bytes of a packaged media part for those of a file on disk.

        Queued rather than applied, so a caller can replace several images and still get
        one atomic `save()`.
        """
        with open(source_path, "rb") as fh:
            self._media[part_name] = fh.read()
        return len(self._media[part_name])

    # ---------------------------------------------------------------- writing

    def save(self, out=None):
        """Write the package back, replacing `word/document.xml` and any touched part.

        Every entry not in `self._parts` is copied verbatim, in its original order, so
        Mendeley's `docProps/custom.xml` and the `customXml/` parts stay bit-identical.
        """
        out = out or self.path
        rewritten = {"word/document.xml": self.tree}
        rewritten.update(self._parts)
        blobs = {name: etree.tostring(t, xml_declaration=True, encoding="UTF-8",
                                      standalone=True)
                 for name, t in rewritten.items()}
        blobs.update(getattr(self, "_media", {}))
        tmp = out + ".tmp"
        with zipfile.ZipFile(self.path) as zin, \
                zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = blobs.get(item.filename) or zin.read(item.filename)
                zout.writestr(item, data)
        shutil.move(tmp, out)
        return out


REF_PATTERN = re.compile(
    r"(Supplementary Materials?|Supplemental Materials?|Supplementary|Supplemental"
    r"|Notes? S\d+(?:\s*(?:and|,|&)\s*S\d+)*"
    r"|Tables? S\d+(?:\s*(?:and|,|&)\s*S\d+)*"
    r"|Figures? S\d+(?:\s*(?:and|,|&)\s*S\d+)*"
    r"|Sections? S\d+(?:\s*(?:and|,|&)\s*S\d+)*)")


def scan(path):
    """Every paragraph that mentions the supplement, for building the edit plan."""
    d = Doc(path)
    hits = []
    for i, p in enumerate(d.paragraphs):
        txt = d.text_of(p)
        found = REF_PATTERN.findall(txt)
        if found:
            hits.append({"i": i, "style": d.style_of(p), "refs": found, "text": txt})
    return d, hits
