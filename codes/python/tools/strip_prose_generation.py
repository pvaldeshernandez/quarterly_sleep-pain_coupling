"""Remove narrative generation from every pipeline step (Decision 3).

Steps compute numbers, tables and figures. They no longer write manuscript prose.

The reason is not tidiness. The generated paragraphs had already gone stale in exactly the
way this arrangement guarantees: `step08_text.md` still emitted "Convergence was adequate:
maximum R-hat = 1.01; all effective sample sizes > 7,000" and "4 chains x 2,000 posterior
draws" — the placeholder a reviewer caught and that the documents no longer contain. Re-running
the pipeline would have REVERTED the corrections. Because Pedro authors the prose himself,
keeping the generators honest would mean back-porting every wording change from Word into
Python forever.

Function spans come from the AST, not from a line-range guess: `generate_text_paragraphs` is
followed by different code in each of the twelve files, and a regex that stops at the next
`def` would eat decorators and trailing comments in some of them.

`generate_text_numbers` is deliberately KEPT. Despite the name it writes a CSV of computed
quantities, which is the numbers registry's ancestor, not prose.

    python tools/strip_prose_generation.py --dry-run
    python tools/strip_prose_generation.py
"""
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CODE))

TARGET = "generate_text_paragraphs"


def strip_file(path, dry):
    """Delete the generator definition and every call to it. Returns (lines, calls)."""
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    lines = src.splitlines(keepends=True)

    tree = ast.parse(src)
    spans = [(n.lineno - 1, n.end_lineno) for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == TARGET]
    if not spans:
        return 0, 0

    # Blank out the definition, then drop calls, then collapse the run of blank lines
    # the deletion leaves between two top-level definitions.
    removed = 0
    for lo, hi in sorted(spans, reverse=True):
        removed += hi - lo
        del lines[lo:hi]

    out, calls = [], 0
    for ln in lines:
        if re.match(rf"\s*{TARGET}\s*\(", ln):
            calls += 1
            continue
        out.append(ln)

    text = "".join(out)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    ast.parse(text)                       # never write a file that will not import

    if not dry:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return removed, calls


def main():
    dry = "--dry-run" in sys.argv
    tot_l = tot_c = tot_f = 0
    for name in sorted(os.listdir(CODE)):
        if not re.match(r"step\d{2}_.+\.py$", name):
            continue
        path = os.path.join(CODE, name)
        with open(path, encoding="utf-8") as fh:
            if TARGET not in fh.read():
                continue
        lines, calls = strip_file(path, dry)
        if lines:
            print(f"  {name:42s} -{lines:4d} lines, -{calls} call(s)")
            tot_l += lines
            tot_c += calls
            tot_f += 1
    print(f"\n{tot_f} file(s), {tot_l} lines of prose generation removed, {tot_c} call sites")
    if dry:
        print("dry run -- nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
