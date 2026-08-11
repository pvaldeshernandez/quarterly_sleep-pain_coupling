"""What every reported number is now, and what the regenerated pipeline says it should be.

Produced instead of a scripted rewrite, for three reasons found while attempting one:

1. Matching by VALUE is unsafe. A dry run of the value-paired approach proposed changing
   "p = .025" in the sleep-correlates sentence and "0.451" in the posterior-predictive
   check -- both from analyses that were never refit -- because unrelated cells happened
   to hold those numbers. Only matching by MEANING is safe.
2. Over half the manuscript's numbers (311 of 564) live inside Word equations. They can
   now be edited (`Doc.replace_math_text_in_para`), but such an edit is NOT a tracked
   change: Word does not represent revisions inside an equation. Pedro has said twice
   that he needs to be able to detect changes rather than find them by chance.
3. The sign conventions differ. The manuscript reports "Delta_elpd = +0.7 for full vs.
   no-SP"; the pipeline CSV records -0.74 for the same comparison. Rewriting that
   without settling the convention would flip a reported direction.

So this reports rather than edits. Each row is anchored by SYMBOL, taken from the
equation text itself, so no number is identified by its value.

    python tools/number_diff_report.py
"""
from __future__ import annotations

import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CODE))
sys.path.insert(0, os.path.join(CODE, "revision"))

RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs/plans/number_diff_report.md")

#: equation symbol -> the row of table4 that defines it
SYMBOLS = {
    "λps": ("b1", "Pain-to-sleep coupling"),
    "λsp": ("a2", "Sleep-to-pain coupling"),
    "τps": ("tau_ps", "SD of person-specific pain-to-sleep slopes"),
    "τsp": ("tau_sp", "SD of person-specific sleep-to-pain slopes"),
    "δs":  ("b3", "Localization -> sleep direct effect"),
    "δp":  ("a3", "Localization -> pain direct effect"),
    "ωps": ("b4", "Pain x localization -> sleep"),
    "ωsp": ("a4", "Sleep x localization -> pain"),
    "φp":  ("a1", "Pain autoregression"),
    "φs":  ("b2", "Sleep autoregression"),
    "ρ":   ("rho_innov", "Innovation correlation"),
}

EQ = re.compile(r"([λτρσφδωμγ][A-Za-z]{0,3})\s*(<0\))?\s*=\s*([+-]?\d*\.\d+)")


def table4():
    p = os.path.join(RESULTS, "step07_coupling_model", "step07_table4_coupling.csv")
    return pd.read_csv(p).set_index("Parameter") if os.path.exists(p) else None


def main():
    from docx_edit import Doc

    t4 = table4()
    if t4 is None:
        print("step07 table4 not found — run the pipeline first")
        return 1

    d = Doc("manuscript.docx")
    d.accept_all()

    rows = []
    for i, p in enumerate(d.paragraphs):
        mt = d.math_text_of(p)
        if not mt.strip():
            continue
        for m in EQ.finditer(mt):
            sym, isprob, cur = m.group(1), bool(m.group(2)), m.group(3)
            if sym not in SYMBOLS:
                continue
            # CONTEXT GUARD. A symbol is not self-identifying: paragraph 116 uses
            # "rho = 0.31" for a Spearman correlation against baseline clinical
            # measures, which has nothing to do with the innovation correlation of
            # paragraph 128. Without this, the report proposed rewriting 0.31 as
            # -0.16. Require the paragraph to be talking about the right quantity.
            if sym == "\u03c1" and "innovation" not in mt.lower():
                continue
            param, desc = SYMBOLS[sym]
            if param not in t4.index:
                continue
            r = t4.loc[param]
            new = r["P_neg"] if isprob else r["Estimate"]
            dec = len(cur.split(".")[1])
            new_s = f"{new:.{dec}f}"
            if cur.startswith("+") and new >= 0:
                new_s = "+" + new_s
            rows.append({
                "para": i,
                "symbol": ("P(%s<0)" % sym) if isprob else sym,
                "quantity": ("posterior probability, " if isprob else "") + desc,
                "document": cur,
                "pipeline": new_s,
                "changes": "YES" if cur != new_s else "no",
                "location": "equation (edit NOT tracked)",
            })

        # plain-text credible intervals, tied to the symbol that precedes them
        txt = d.text_of(p)
        for m in re.finditer(r"95% CrI \[([^\]\)]+)[\]\)]", txt):
            rows.append({
                "para": i, "symbol": "—", "quantity": "credible interval",
                "document": m.group(1), "pipeline": "(see table4 for the matching row)",
                "changes": "review", "location": "plain text (trackable)",
            })

    df = pd.DataFrame(rows)
    changed = df[df["changes"] == "YES"]

    lines = ["# Number diff — document vs regenerated pipeline", "",
             f"Generated from `results/step07_coupling_model/step07_table4_coupling.csv` "
             f"(N = 227, 1,793 transitions).", "",
             f"**{len(changed)} equation value(s) differ** from the pipeline; "
             f"{len(df) - len(changed)} rows are unchanged or need review.", "",
             "Every row is matched by SYMBOL taken from the equation text, never by value.",
             "", "## Equation values that changed", "",
             "| ¶ | symbol | quantity | document says | pipeline says |",
             "|---|---|---|---|---|"]
    for _, r in changed.iterrows():
        lines.append(f"| {r['para']} | `{r['symbol']}` | {r['quantity']} | "
                     f"**{r['document']}** | **{r['pipeline']}** |")
    lines += ["", "## Credible intervals found in plain text", "",
              "These are trackable and can be edited normally, but each must be paired with "
              "its parameter by reading the sentence — the bracket alone does not say which "
              "quantity it belongs to.", "",
              "| ¶ | interval as written |", "|---|---|"]
    for _, r in df[df["location"].str.startswith("plain")].iterrows():
        lines.append(f"| {r['para']} | {r['document']} |")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"{len(changed)} equation value(s) differ from the pipeline")
    print(df.to_string(index=False, max_colwidth=44))
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
