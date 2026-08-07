#!/usr/bin/env python3
"""Verify code-generated text matches the documents.

For every paragraph the code emits in step*_text.md, find the closest
matching paragraph in manuscript_pain.md and supplementary_materials.md
and classify as:

  MATCH      sim >= 0.92  (identical wording; only numbers may differ)
  CLOSE      0.70 <= sim < 0.92  (similar wording, drift worth checking)
  ORPHAN     sim < 0.70  (code emits text that's not in any doc;
                          either you removed it from the doc, or the
                          doc edit drifted and the code needs updating)

Usage:
    python verify_code_vs_docs.py [--verbose]

Writes a report to derivatives/verify_docs_report.md and exits 0 if
every code paragraph is MATCH/CLOSE, 1 otherwise.
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

# Internal-use report; lives under derivatives/, not user-facing results.
REPORT = os.path.join(ROOT, "derivatives", "verify_docs_report.md")

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
    """One-way check: for every code-generated paragraph, find its closest
    match in the documents. Return 0 if all are MATCH or CLOSE, 1 otherwise.
    """
    # Pool all document paragraphs (manuscript + supplement) as candidates.
    doc_paras: list[Paragraph] = []
    for path in (MANUSCRIPT, SUPPLEMENT):
        doc_paras.extend(load_paragraphs(path))

    code_paras: list[Paragraph] = []
    for cf in CODE_TEXT_FILES:
        code_paras.extend(load_paragraphs(cf))

    counts = {"MATCH": 0, "CLOSE": 0, "ORPHAN": 0}
    rows: list[tuple[str, float, Paragraph, Optional[Paragraph]]] = []

    for cp in code_paras:
        # Skip headings and structural markers — they're not prose.
        if cp.kind in ("heading", "figure_link", "table_or_equation"):
            continue
        if len(cp.normalized) < 40:
            continue
        sim, doc_match = best_match(cp, doc_paras)
        if sim >= 0.92:
            counts["MATCH"] += 1
            rows.append(("MATCH", sim, cp, doc_match))
        elif sim >= 0.70:
            counts["CLOSE"] += 1
            rows.append(("CLOSE", sim, cp, doc_match))
        else:
            counts["ORPHAN"] += 1
            rows.append(("ORPHAN", sim, cp, doc_match))

    report_lines = [
        "# Code-output verification report",
        "",
        "Each code-generated paragraph is matched against the closest "
        "document paragraph.",
        "",
        f"**Totals:** MATCH={counts['MATCH']}, CLOSE={counts['CLOSE']}, "
        f"ORPHAN={counts['ORPHAN']}",
        "",
    ]

    for label in ("ORPHAN", "CLOSE", "MATCH"):
        items = [r for r in rows if r[0] == label]
        if not items:
            continue
        if label == "MATCH" and not verbose:
            continue
        report_lines.append(f"## {label} ({len(items)})")
        report_lines.append("")
        for _, sim, cp, dm in items:
            code_src = os.path.relpath(cp.source, ROOT)
            report_lines.append(f"- sim={sim:.2f}")
            report_lines.append(
                f"  - CODE ({code_src}): {short(cp.text)}"
            )
            if dm is not None:
                doc_src = os.path.relpath(dm.source, ROOT)
                report_lines.append(
                    f"  - DOC  ({doc_src}): {short(dm.text)}"
                )
            else:
                report_lines.append("  - DOC : (no candidate)")
        report_lines.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("\n".join(report_lines))

    print(
        f"Code output: MATCH={counts['MATCH']}, "
        f"CLOSE={counts['CLOSE']}, ORPHAN={counts['ORPHAN']}"
    )
    print(f"Report: {REPORT}")
    return 0 if counts["ORPHAN"] == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    sys.exit(run_verification(verbose=args.verbose))


if __name__ == "__main__":
    main()
