"""
Step 11 — Generate remaining figures (2, 3, S1, S2).
======================================================================

Input:
  derivatives/step3_person_coupling.csv      — Figures 2, 3
  derivatives/step3_posterior_draws.npz       — Figures 2, 3 (pop mean)
  results/step3_table4_coupling.csv           — Figures 2, 3 (pop params)
  data/step0_extracted_long.csv               — Figure S1 (endorsements)
  derivatives/step2_processed_long.csv        — Figure S2
  data/original/participants_wideformat.xlsx  — Figure S2

Output (results/):
  step13_figure2_ps_coupling.png   — Figure 2
  step13_figure3_sp_coupling.png   — Figure 3
  step13_figure_s1_endorsement.png — Figure S1
  step13_figure_s2_convergent.png  — Figure S2

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
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
DERIV_DIR = os.path.join(ROOT, "derivatives")
RESULTS_DIR = os.path.join(ROOT, "results")

OUT_FIG2 = os.path.join(RESULTS_DIR, "step13_figure2_ps_coupling.png")
OUT_FIG3 = os.path.join(RESULTS_DIR, "step13_figure3_sp_coupling.png")
OUT_FIG_S1 = os.path.join(RESULTS_DIR, "step13_figure_s1_endorsement.png")
OUT_FIG_S2 = os.path.join(RESULTS_DIR, "step13_figure_s2_convergent.png")


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
    prob_neg = df[col_prob].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8),
                                    gridspec_kw={"width_ratios": [1, 2]})

    # Panel A: boxstrip
    colors = ["#1565C0" if m < 0 else "#D32F2F" for m in means]
    ax1.scatter(np.random.normal(0, 0.08, n), means, c=colors,
                s=20, alpha=0.5, zorder=2)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
    ax1.scatter([0], [pop_mean], marker="D", color="#757575", s=100,
                zorder=5, label=f"Population mean")
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
    person = pd.read_csv(os.path.join(DERIV_DIR, "step3_person_coupling.csv"))
    table4 = pd.read_csv(os.path.join(RESULTS_DIR, "step3_table4_coupling.csv"))

    # Get population means
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
        direction_label="Pain → Sleep",
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
        direction_label="Sleep → Pain",
        out_path=OUT_FIG3,
    )
    if verbose:
        print(f"    Saved: {OUT_FIG3}")


# =====================================================================
# Figure S1 — Factor endorsement validation
# =====================================================================

def generate_figure_s1(verbose=True):
    """Figure S1: body-map endorsement ANOVA + point-biserial correlations.

    Shows that the pain localization contrast factor correlates with
    body-area endorsement patterns in the expected way: positive
    correlation with knee endorsement, negative with non-knee areas.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    if verbose:
        print("  Figure S1: Factor endorsement validation")

    extracted = pd.read_csv(os.path.join(DATA_DIR, "step0_extracted_long.csv"))
    processed = pd.read_csv(os.path.join(DERIV_DIR, "step2_processed_long.csv"))

    # Compute person-mean contrast from the processed data
    ki = processed.groupby("ID")["contrast_factor"].mean().reset_index()
    ki.columns = ["ID", "K_i"]

    # Get baseline endorsements (quarter 0)
    baseline = extracted[extracted["quarter"] == 0].copy()
    baseline["ID"] = baseline["ID"].astype(str)
    ki["ID"] = ki["ID"].astype(str)

    df = ki.merge(baseline, on="ID", how="inner")

    # Body-area endorsement columns
    area_cols = [f"phq_pain_areas___{i}__s1" for i in range(1, 14)]
    available_areas = [c for c in area_cols if c in df.columns]

    if len(available_areas) < 5:
        if verbose:
            print("    SKIP: insufficient body-area endorsement columns")
        return

    # Area labels (from dictionary order: 1..13)
    area_labels = [
        "Hands", "Arms", "Shoulders", "Neck", "Head/Face/Jaw",
        "Chest", "Stomach", "Pelvis", "Upper Back", "Lower Back",
        "Knees", "Legs", "Feet/Ankles",
    ]

    # Point-biserial correlations
    rpb_results = []
    for i, col in enumerate(available_areas):
        x = df[col].fillna(0).astype(int)
        y = df["K_i"]
        both = x.notna() & y.notna()
        if both.sum() < 10:
            continue
        r, p = sp_stats.pointbiserialr(x[both], y[both])
        label = area_labels[i] if i < len(area_labels) else f"Area {i+1}"
        rpb_results.append({"area": label, "r_pb": r, "p": p, "idx": i})

    if not rpb_results:
        if verbose:
            print("    SKIP: no valid point-biserial correlations")
        return

    rpb_df = pd.DataFrame(rpb_results).sort_values("r_pb", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#1565C0" if r < 0 else "#D32F2F" for r in rpb_df["r_pb"]]
    bars = ax.barh(range(len(rpb_df)), rpb_df["r_pb"].values, color=colors, alpha=0.7)

    ax.set_yticks(range(len(rpb_df)))
    ax.set_yticklabels(rpb_df["area"].values, fontsize=11)
    ax.set_xlabel("Point-biserial correlation with mean contrast ($\\bar{K}_i$)",
                  fontsize=12)
    ax.axvline(0, color="black", linewidth=0.8)

    # Mark significant ones
    for i, (_, row) in enumerate(rpb_df.iterrows()):
        if row["p"] < 0.05:
            ax.text(row["r_pb"] + 0.01 * np.sign(row["r_pb"]),
                    i, "*", fontsize=14, ha="center", va="center",
                    fontweight="bold")

    ax.set_title("Point-biserial correlations: body-area endorsement vs. "
                 "pain localization contrast", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG_S1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    Saved: {OUT_FIG_S1}")


# =====================================================================
# Figure S2 — Convergent validity scatter
# =====================================================================

def generate_figure_s2(verbose=True):
    """Figure S2: convergent validity of the contrast factor vs baseline
    clinical measures.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    if verbose:
        print("  Figure S2: Convergent validity scatter")

    processed = pd.read_csv(os.path.join(DERIV_DIR, "step2_processed_long.csv"))
    wide_path = os.path.join(DATA_DIR, "original", "participants_wideformat.xlsx")
    if not os.path.exists(wide_path):
        if verbose:
            print("    SKIP: participants_wideformat.xlsx not found")
        return

    wide = pd.read_excel(wide_path)
    wide["ID"] = wide["ID"].astype(str)

    # Person-mean contrast
    ki = processed.groupby("ID")["contrast_factor"].mean().reset_index()
    ki.columns = ["ID", "K_i"]
    ki["ID"] = ki["ID"].astype(str)

    df = ki.merge(wide, on="ID", how="inner")

    panels = [
        ("phq_knee_pain_days__s1", "PHQ knee pain days per week"),
        ("phq_percent_pain__s1", "PHQ % waking day in knee pain"),
        ("womac_pain__s1", "WOMAC Pain"),
        ("total_womac__s1", "WOMAC Total"),
        ("womac_phys_function__s1", "WOMAC Physical Function"),
        ("womac_stiffness__s1", "WOMAC Stiffness"),
        ("qst_knee_pain_rating__s1", "Knee pain rating"),
    ]
    available_panels = [(c, l) for c, l in panels if c in df.columns]

    if not available_panels:
        if verbose:
            print("    SKIP: no clinical columns found")
        return

    n_panels = len(available_panels)
    ncols = 4
    nrows = max(1, (n_panels + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i, (col, label) in enumerate(available_panels):
        ax = axes[i]
        tmp = df[["K_i", col]].dropna()
        if len(tmp) < 5:
            ax.set_visible(False)
            continue
        x, y = tmp["K_i"].values, tmp[col].values
        n = len(x)
        r, p = sp_stats.pearsonr(x, y)

        ax.scatter(x, y, alpha=0.35, s=18, color="steelblue", edgecolor="none")
        slope, intercept = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 200)
        ax.plot(xline, intercept + slope * xline, color="firebrick", linewidth=2)

        pstr = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.05, 0.95, f"r = {r:.3f}\n{pstr}\nN = {n}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                          alpha=0.85, edgecolor="gray"))
        ax.set_xlabel("Person-mean contrast ($\\bar{K}_i$)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.tick_params(labelsize=9)

    for j in range(len(available_panels), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Convergent validity: person-mean contrast vs baseline "
                 "clinical measures", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_FIG_S2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    Saved: {OUT_FIG_S2}")


# =====================================================================
# Main
# =====================================================================

def run_step11(verbose=True):
    if verbose:
        print("=" * 70)
        print("STEP 11 — Remaining figures (2, 3, S1, S2)")
        print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    generate_figures_2_3(verbose)
    generate_figure_s1(verbose)
    generate_figure_s2(verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 11 COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 11 — generate remaining figures."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_step11(verbose=not args.quiet)


if __name__ == "__main__":
    main()
