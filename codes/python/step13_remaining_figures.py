"""
Step 13 — Generate person-level coupling figures (Figures 2 and 3).
======================================================================

Input:
  derivatives/step4/step4_person_coupling.csv   — per-person lambda estimates
  results/step4/step4_table4_coupling.csv       — population parameters

Output (results/step13/):
  step13_figure2_ps_coupling.png   — Figure 2: Pain-to-Sleep
  step13_figure3_sp_coupling.png   — Figure 3: Sleep-to-Pain

Author: Pedro Valdes-Hernandez (with Claude Opus 4.6)
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DERIV_DIR = os.path.join(ROOT, "derivatives")
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step13")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

OUT_FIG2 = os.path.join(STEP_RESULTS_DIR, "step13_figure2_ps_coupling.png")
OUT_FIG3 = os.path.join(STEP_RESULTS_DIR, "step13_figure3_sp_coupling.png")


# =====================================================================
# Figures 2 & 3 — Person-level coupling
# =====================================================================

def generate_coupling_figure(person_df, pop_mean, pop_ci_lo, pop_ci_hi,
                             col_mean, col_ci_lo, col_ci_hi, col_prob,
                             direction_label, out_path):
    """Draw a 2-panel coupling figure (boxstrip + forest)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = person_df.sort_values(col_mean).reset_index(drop=True)
    n = len(df)
    means = df[col_mean].values
    lo = df[col_ci_lo].values
    hi = df[col_ci_hi].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8),
                                    gridspec_kw={"width_ratios": [1, 2]})

    # Panel A: boxstrip
    colors = ["#1565C0" if m < 0 else "#D32F2F" for m in means]
    ax1.scatter(np.random.normal(0, 0.08, n), means, c=colors,
                s=20, alpha=0.5, zorder=2)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
    ax1.scatter([0], [pop_mean], marker="D", color="#757575", s=100,
                zorder=5, label="Population mean")
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_xticks([])
    ax1.set_ylabel(f"Person-specific {direction_label} coupling", fontsize=12)
    ax1.set_title("(A)", fontsize=14, fontweight="bold", loc="left")
    ax1.legend(fontsize=10, loc="upper right")

    # Panel B: forest
    for i in range(n):
        color = "#1565C0" if means[i] < 0 else "#D32F2F"
        ax2.plot([lo[i], hi[i]], [i, i], color=color, linewidth=0.8, alpha=0.6)
    ax2.scatter(means, range(n), c=colors, s=8, zorder=3, alpha=0.7)
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
    ax2.axvline(pop_mean, color="#757575", linewidth=1.5, linestyle="--",
                alpha=0.7, label=f"Population mean = {pop_mean:.3f}")
    ax2.set_yticks([])
    ax2.set_xlabel(f"{direction_label} coupling slope", fontsize=12)
    ax2.set_ylabel(f"Participant (N = {n})", fontsize=12)
    ax2.set_title("(B)", fontsize=14, fontweight="bold", loc="left")
    ax2.legend(fontsize=10, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_figures_2_3(verbose=True):
    """Generate Figures 2 (PS) and 3 (SP)."""
    person = pd.read_csv(os.path.join(DERIV_DIR, "step4", "step4_person_coupling.csv"))
    table4 = pd.read_csv(os.path.join(RESULTS_DIR, "step4", "step4_table4_coupling.csv"))

    ps_row = table4[table4["Parameter"] == "b1"].iloc[0]
    sp_row = table4[table4["Parameter"] == "a2"].iloc[0]

    if verbose:
        print("  Figure 2: Pain-to-Sleep person-level coupling")
    generate_coupling_figure(
        person,
        pop_mean=ps_row["Estimate"],
        pop_ci_lo=ps_row["CrI_lo"],
        pop_ci_hi=ps_row["CrI_hi"],
        col_mean="beta_ps_mean", col_ci_lo="beta_ps_ci_lo",
        col_ci_hi="beta_ps_ci_hi", col_prob="beta_ps_prob_neg",
        direction_label="Pain -> Sleep",
        out_path=OUT_FIG2,
    )
    if verbose:
        print(f"    Saved: {OUT_FIG2}")

    if verbose:
        print("  Figure 3: Sleep-to-Pain person-level coupling")
    generate_coupling_figure(
        person,
        pop_mean=sp_row["Estimate"],
        pop_ci_lo=sp_row["CrI_lo"],
        pop_ci_hi=sp_row["CrI_hi"],
        col_mean="beta_sp_mean", col_ci_lo="beta_sp_ci_lo",
        col_ci_hi="beta_sp_ci_hi", col_prob="beta_sp_prob_neg",
        direction_label="Sleep -> Pain",
        out_path=OUT_FIG3,
    )
    if verbose:
        print(f"    Saved: {OUT_FIG3}")


# =====================================================================
# Main
# =====================================================================

def run_step13(verbose=True):
    if verbose:
        print("=" * 70)
        print("STEP 13 — Person-level coupling figures (2 and 3)")
        print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    generate_figures_2_3(verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 13 COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 13 — Person-level coupling figures."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_step13(verbose=not args.quiet)


if __name__ == "__main__":
    main()
