#!/usr/bin/env python3
"""Diff the manuscript/supplement against code-generated text outputs.

Classifies each paragraph in manuscript_pain.md and
supplementary_materials.md as one of:

  MATCH        sim >= 0.92 to some code-generated paragraph
  CLOSE        0.70 <= sim < 0.92 (code emits similar text but wording drifted)
  UNMATCHED    sim < 0.70 and paragraph cites numbers -> code should emit
  NARRATIVE    sim < 0.70 but paragraph does not cite computed numbers ->
               manually authored, will not be regenerated

Also lists code-generated paragraphs that have no manuscript match
(ORPHAN-CODE), which means the generator is emitting stale text the
document no longer uses.

Usage:
    python verify_code_vs_docs.py [--verbose]

Writes a report to results/verify_docs_report.md.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DOCS = os.path.join(ROOT, "docs")
RESULTS = os.path.join(ROOT, "results")

MANUSCRIPT = os.path.join(DOCS, "manuscript_pain.md")
SUPPLEMENT = os.path.join(DOCS, "supplementary_materials.md")

CODE_TEXT_FILES = [
    os.path.join(RESULTS, "step01_factor_analysis", "step01_text.md"),
    os.path.join(RESULTS, "step02_contrast_validation", "step02_text.md"),
    os.path.join(RESULTS, "step03_varx_data", "step03_text.md"),
    os.path.join(RESULTS, "step04_coupling_model", "step04_text.md"),
    os.path.join(RESULTS, "step05_contrast_moderation", "step05_text.md"),
    os.path.join(RESULTS, "step08_sp_moderation", "step08_text.md"),
    os.path.join(RESULTS, "step09_sp_jn", "step09_text.md"),
    os.path.join(RESULTS, "step11_ps_moderation", "step11_text.md"),
    os.path.join(
        RESULTS, "supplementary_materials", "step04_supp_text.md"
    ),
    os.path.join(
        RESULTS, "supplementary_materials", "step13_text.md"
    ),
]

REPORT = os.path.join(RESULTS, "verify_docs_report.md")

# A paragraph is "code-owned" only if it cites a COMPUTED model output.
# Strong signals of computed results (vs. method parameters).
RESULT_CLAIM_RE = re.compile(
    r"95% CrI"
    r"|\\hat\{|\\widehat\{"
    r"|P\(<0|P\(\\hat"
    r"|\\tau_\{|\\lambda_\{|\\gamma_\{|\\delta_\{|\\rho[^a-z]|\\omega_\{|\\phi_\{"
    r"|credibly (negative|different|negative|positive)"
    r"|\bLOO\b|\\Delta LOO|\\Delta/SE"
    r"|\brhat\b|\bR-hat\b"
    r"|sign-concordance|concordance"
    r"|exact sign test"
)

# Method/parameter patterns that explicitly do NOT mean the paragraph
# is code-owned. MNI coordinates, pulse sequence parameters, etc.
METHOD_ONLY_HINTS = (
    r"MNI\s*[±\-+]?\d",          # MNI coordinate
    r"TR\s*=\s*\d|TE\s*=\s*\d",   # fMRI timing
    r"mm\s*isotropic",            # voxel size
    r"FWHM|DARTEL|SPM12",
    r"Bartlett factor scores",
    r"Horn's parallel analysis",
    r"Lipinski|Veber",
)
METHOD_ONLY_RE = re.compile("|".join(METHOD_ONLY_HINTS))

# Paragraphs that DEFINE variables ("Here X is ...", "where X is ...") are
# methods-prose, not code-owned.
DEFINITION_RE = re.compile(
    r"^(Here|where|with)\s+\$|\bare the (coupling|population|random|conditional|between-person) "
    r"|\bis the (atlas|conditional|population-average|within-person|Johnson)"
    r"|\bis a free conditioning"
    r"|classical frequentist|no single standard error"
    r"|quadratic equation"
    r"|nested models were fit"
    r"|Cholesky decomposition"
    r"|To assess the evidence|To characterize the continuous",
    re.I,
)

# Paragraphs that start with these tokens are structurally never expected to
# be code-generated (e.g., abstract lead, author affil, reference list).
NARRATIVE_STARTS = (
    "Sleep and pain are bidirectionally",  # abstract opener
    "Sleep disturbance",                    # intro opener
    "Pedro A.",
    "Department of",
    "College of",
    "Email:",
    "Phone:",
    "Knee-body pain contrast",              # keywords
    "The data that support",                # data availability
    "We are grateful",                      # acknowledgements
    "This work was supported",              # funding
    "PAVH:",                                # CRediT
)


@dataclass
class Paragraph:
    source: str        # file path
    kind: str          # 'prose', 'caption', 'note', 'table', 'heading'
    text: str          # raw text
    normalized: str    # number-stripped lower-case
    cites_numbers: bool


def normalize(s: str) -> str:
    s = re.sub(r"\$[^$]*\$", " ", s)              # strip inline math
    s = re.sub(r"\$\$[^$]*\$\$", " ", s)           # strip display math
    s = re.sub(r"[*_\\`~]", " ", s)               # strip markdown
    s = re.sub(r"\([\d\s,.\-–+]+\)", " ", s)      # drop numeric parens
    s = re.sub(r"[+\-]?\d+\.\d+", "#", s)         # decimal numbers
    s = re.sub(r"\d+", "#", s)                    # integers
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # Drop fenced code blocks and pure tables; tables handled separately
    cleaned = []
    for p in paras:
        if p.startswith("```"):
            continue
        cleaned.append(p)
    return cleaned


def classify_kind(p: str) -> str:
    if p.startswith("#"):
        return "heading"
    if p.startswith("!["):
        return "figure_link"
    if p.startswith("**Table"):
        return "caption"
    if p.startswith("**Figure"):
        return "caption"
    if p.startswith("**Note.**") or re.match(r"^\*?Note[s]?[\.:]", p):
        return "note"
    if p.startswith("|") or p.startswith("$$"):
        return "table_or_equation"
    if p.startswith("- ") or re.match(r"^\d+\. ", p):
        # enumerated / bullet — treat as prose but tag
        return "list"
    return "prose"


def cites_numbers(p: str) -> bool:
    """True if paragraph reports *computed* results (not method parameters)."""
    if re.match(r"^\d+\.\s+[A-Z]", p):  # reference list entry
        return False
    # Pure math block (starts and ends with $, no sentence-like prose)
    if p.startswith("$") and p.count(".") <= 1 and len(p) < 300:
        return False
    if METHOD_ONLY_RE.search(p) and not RESULT_CLAIM_RE.search(p):
        return False
    if DEFINITION_RE.search(p) and not re.search(r"\b(credibly|95% CrI|credible)", p):
        return False
    # Note S2 derivation paragraphs: contain lambda/phi as variables but
    # not as computed estimates (no \hat{...} or \widehat{...}).
    if "\\lambda" in p and "\\hat{" not in p and "\\widehat{" not in p:
        # Unless it explicitly quotes credible intervals or p-values
        if not re.search(r"95% CrI|credibly|\\approx\s*[+\-]?\d", p):
            return False
    return bool(RESULT_CLAIM_RE.search(p))


def load_paragraphs(path: str) -> list[Paragraph]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        text = f.read()
    out: list[Paragraph] = []
    for p in split_paragraphs(text):
        kind = classify_kind(p)
        out.append(
            Paragraph(
                source=path,
                kind=kind,
                text=p,
                normalized=normalize(p),
                cites_numbers=cites_numbers(p),
            )
        )
    return out


def is_narrative_only(p: Paragraph) -> bool:
    """True if paragraph is creative prose not backed by code outputs."""
    if p.kind in ("heading", "figure_link", "table_or_equation"):
        return True
    if not p.cites_numbers:
        return True
    for marker in NARRATIVE_STARTS:
        if p.text.startswith(marker):
            return True
    # Reference list item
    if re.match(r"^\d+\.\s+[A-Z]\.\s", p.text):
        return True
    return False


def best_match(
    target: Paragraph, candidates: Iterable[Paragraph]
) -> tuple[float, Optional[Paragraph]]:
    best = (0.0, None)
    tn = target.normalized
    if len(tn) < 30:
        return best
    for c in candidates:
        if len(c.normalized) < 30:
            continue
        r = SequenceMatcher(None, tn, c.normalized).ratio()
        if r > best[0]:
            best = (r, c)
    return best


def short(s: str, n: int = 160) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def run_verification(verbose: bool = False) -> int:
    """Return 0 if no unmatched reportable paragraphs; 1 otherwise."""
    manuscript = load_paragraphs(MANUSCRIPT)
    supplement = load_paragraphs(SUPPLEMENT)

    code_paras: list[Paragraph] = []
    for cf in CODE_TEXT_FILES:
        code_paras.extend(load_paragraphs(cf))

    # For code orphan detection: track which code paragraphs were matched
    code_used = [False] * len(code_paras)

    report_lines = ["# Document-vs-code verification report", ""]

    def process(label: str, paras: list[Paragraph]) -> dict[str, int]:
        counts = {"MATCH": 0, "CLOSE": 0, "UNMATCHED": 0, "NARRATIVE": 0}
        report_lines.append(f"## {label}")
        report_lines.append("")
        for p in paras:
            if p.kind in ("heading", "figure_link", "table_or_equation"):
                continue
            if is_narrative_only(p):
                counts["NARRATIVE"] += 1
                continue
            sim, cand = best_match(p, code_paras)
            if cand is not None:
                # mark code paragraph as used
                idx = code_paras.index(cand)
                code_used[idx] = True
            if sim >= 0.92:
                counts["MATCH"] += 1
                if verbose:
                    report_lines.append(
                        f"- MATCH   ({sim:.2f}) {short(p.text)}"
                    )
            elif sim >= 0.70:
                counts["CLOSE"] += 1
                report_lines.append(
                    f"- CLOSE   ({sim:.2f})"
                )
                report_lines.append(f"  - DOC : {short(p.text)}")
                code_src = os.path.relpath(cand.source, ROOT) if cand else "?"
                report_lines.append(
                    f"  - CODE: ({code_src}) "
                    f"{short(cand.text) if cand else '—'}"
                )
            else:
                counts["UNMATCHED"] += 1
                report_lines.append("- UNMATCHED")
                report_lines.append(f"  - DOC : {short(p.text)}")
                if cand:
                    code_src = os.path.relpath(cand.source, ROOT)
                    report_lines.append(
                        f"  - BEST CODE ({sim:.2f}, {code_src}): "
                        f"{short(cand.text)}"
                    )
                else:
                    report_lines.append("  - BEST CODE: (no candidate)")
        report_lines.append("")
        report_lines.append(
            f"**Totals:** MATCH={counts['MATCH']}, CLOSE={counts['CLOSE']}, "
            f"UNMATCHED={counts['UNMATCHED']}, NARRATIVE={counts['NARRATIVE']}"
        )
        report_lines.append("")
        return counts

    ms_counts = process("manuscript_pain.md", manuscript)
    sp_counts = process("supplementary_materials.md", supplement)

    # Orphan code paragraphs
    report_lines.append("## Code paragraphs with no document match")
    report_lines.append("")
    orphan_count = 0
    for used, p in zip(code_used, code_paras):
        if used:
            continue
        if p.kind in ("heading", "figure_link", "table_or_equation"):
            continue
        if not p.cites_numbers:
            continue
        if len(p.normalized) < 60:
            continue
        orphan_count += 1
        src = os.path.relpath(p.source, ROOT)
        report_lines.append(f"- ORPHAN ({src}): {short(p.text)}")
    if orphan_count == 0:
        report_lines.append("*(none)*")
    report_lines.append("")

    # Summary
    report_lines.insert(
        2,
        f"**Manuscript:** {ms_counts}  \n"
        f"**Supplement:** {sp_counts}  \n"
        f"**Orphan code paragraphs:** {orphan_count}\n",
    )

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("\n".join(report_lines))

    n_unmatched = ms_counts["UNMATCHED"] + sp_counts["UNMATCHED"]
    print(f"Manuscript: {ms_counts}")
    print(f"Supplement: {sp_counts}")
    print(f"Orphan code paragraphs: {orphan_count}")
    print(f"Report: {REPORT}")
    return 0 if n_unmatched == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    sys.exit(run_verification(verbose=args.verbose))


if __name__ == "__main__":
    main()
