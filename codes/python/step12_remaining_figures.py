"""
Step 12 — Generate remaining figures (2, 3, S1, S2) and convergent
          validity text numbers.
======================================================================

Input:
  derivatives/step3_person_coupling.csv      — Figures 2, 3
  derivatives/step3_posterior_draws.npz       — Figures 2, 3 (pop mean)
  results/step3_table4_coupling.csv           — Figures 2, 3 (pop params)
  data/step0_extracted_long.csv               — Figure S1 (endorsements)
  derivatives/step2_processed_long.csv        — Figure S2 + text numbers
  data/original/participants_wideformat.xlsx  — Figure S2 + text numbers

Output (results/step12/):
  step12_figure2_ps_coupling.png   — Figure 2
  step12_figure3_sp_coupling.png   — Figure 3
  step12_figure_s1_endorsement.png — Figure S1
  step12_figure_s2_convergent.png  — Figure S2
  step12_text_numbers.csv          — ANOVA, Tukey, point-biserial,
                                     clinical correlations

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
DATA_DIR = os.path.join(ROOT, "data")
DERIV_DIR = os.path.join(ROOT, "derivatives")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step12")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step12")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

OUT_FIG2 = os.path.join(STEP_RESULTS_DIR, "step12_figure2_ps_coupling.png")
OUT_FIG3 = os.path.join(STEP_RESULTS_DIR, "step12_figure3_sp_coupling.png")
OUT_FIG_S1 = os.path.join(STEP_RESULTS_DIR, "step12_figure_s1_endorsement.png")
OUT_FIG_S2 = os.path.join(STEP_RESULTS_DIR, "step12_figure_s2_convergent.png")
OUT_TEXT_CSV = os.path.join(STEP_RESULTS_DIR, "step12_text_numbers.csv")


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
    person = pd.read_csv(os.path.join(DERIV_DIR, "step3", "step3_person_coupling.csv"))
    table4 = pd.read_csv(os.path.join(RESULTS_DIR, "step3", "step3_table4_coupling.csv"))

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
    processed = pd.read_csv(os.path.join(DERIV_DIR, "step2", "step2_processed_long.csv"))

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

    processed = pd.read_csv(os.path.join(DERIV_DIR, "step2", "step2_processed_long.csv"))
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
# Convergent validity text numbers
# =====================================================================

def generate_text_numbers(verbose=True):
    """Compute and save all convergent validity statistics.

    Outputs step12_text_numbers.csv with:
      - ANOVA: F-statistic, df, p-value, group means/SDs
      - Tukey post-hoc: pairwise p-values
      - Point-biserial correlations for 13 PHQ body areas (with FDR)
      - Pearson/Spearman correlations with clinical measures
    """
    from scipy import stats as sp_stats
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.multitest import multipletests

    if verbose:
        print("  Text numbers: convergent validity statistics")

    processed = pd.read_csv(os.path.join(DERIV_DIR, "step2", "step2_processed_long.csv"))
    extracted = pd.read_csv(os.path.join(DATA_DIR, "step0_extracted_long.csv"))
    wide_path = os.path.join(DATA_DIR, "original", "participants_wideformat.xlsx")

    text_rows = []

    def _t(metric, value, note=""):
        text_rows.append({"metric": metric, "value": str(value), "note": note})

    # --- Person-mean contrast ---
    ki = processed.groupby("ID")["contrast_factor"].mean().reset_index()
    ki.columns = ["ID", "K_i"]
    ki["ID"] = ki["ID"].astype(str)

    # --- ANOVA by pain distribution group ---
    # Pain distribution determined from baseline (quarter 0) PHQ endorsements
    baseline = extracted[extracted["quarter"] == 0].copy()
    baseline["ID"] = baseline["ID"].astype(str)

    # PHQ area columns
    area_cols_base = [f"phq_pain_areas___{i}__s1" for i in range(1, 14)]
    available_area_cols = [c for c in area_cols_base if c in baseline.columns]

    knee_col = "phq_pain_areas___11__s1"  # knee is area 11 in PHQ

    if available_area_cols and knee_col in available_area_cols:
        bdf = baseline.merge(ki, on="ID", how="inner")
        # Count non-knee areas endorsed (areas 1-10, 12-13)
        non_knee_cols = [c for c in available_area_cols if c != knee_col]
        bdf["n_other_areas"] = bdf[non_knee_cols].fillna(0).sum(axis=1)
        bdf["knee_endorsed"] = bdf[knee_col].fillna(0).astype(int)

        # Classify into 3 groups
        def classify(row):
            if row["knee_endorsed"] == 1 and row["n_other_areas"] == 0:
                return "knee_only"
            elif row["knee_endorsed"] == 1 and row["n_other_areas"] > 0:
                return "knee_plus"
            else:
                return "no_knee"

        bdf["pain_group"] = bdf.apply(classify, axis=1)
        bdf = bdf.dropna(subset=["K_i", "pain_group"])

        groups = bdf.groupby("pain_group")["K_i"]
        group_labels = ["knee_only", "knee_plus", "no_knee"]
        group_data = [bdf[bdf["pain_group"] == g]["K_i"].dropna().values
                      for g in group_labels]
        n_groups = [len(g) for g in group_data]

        _t("anova_n_knee_only", n_groups[0])
        _t("anova_n_knee_plus", n_groups[1])
        _t("anova_n_no_knee", n_groups[2])

        if all(n > 1 for n in n_groups):
            f_stat, p_anova = sp_stats.f_oneway(*group_data)
            df_between = 2
            df_within = sum(n_groups) - 3
            _t("anova_F", f"{f_stat:.2f}")
            _t("anova_df", f"({df_between},{df_within})")
            _t("anova_p", f"{p_anova:.3f}" if p_anova >= 0.001 else "<0.001")

            for g, data in zip(group_labels, group_data):
                _t(f"anova_mean_{g}", f"{np.mean(data):.2f}")
                _t(f"anova_sd_{g}", f"{np.std(data, ddof=1):.2f}")
                _t(f"anova_n_{g}", str(len(data)))

            if verbose:
                print(f"    ANOVA: F({df_between},{df_within})={f_stat:.2f}, "
                      f"p={'<0.001' if p_anova < 0.001 else f'{p_anova:.3f}'}")

            # Tukey post-hoc
            all_vals = np.concatenate(group_data)
            all_labels = np.concatenate([[g] * n for g, n in
                                          zip(group_labels, n_groups)])
            tukey = pairwise_tukeyhsd(all_vals, all_labels, alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:],
                                     columns=tukey._results_table.data[0])
            for _, trow in tukey_df.iterrows():
                pair = f"{trow['group1']}_vs_{trow['group2']}"
                _t(f"tukey_p_{pair}", f"{trow['p-adj']:.3f}"
                   if float(trow['p-adj']) >= 0.001 else "<0.001")
                _t(f"tukey_meandiff_{pair}", f"{trow['meandiff']:.3f}")
                if verbose:
                    print(f"    Tukey {pair}: p={trow['p-adj']:.3f}")

    # --- Point-biserial correlations ---
    area_labels = [
        "Hands", "Arms", "Shoulders", "Neck", "Head/Face/Jaw",
        "Chest", "Stomach", "Pelvis", "Upper Back", "Lower Back",
        "Knees", "Legs", "Feet/Ankles",
    ]

    if available_area_cols and "bdf" in dir():
        pb_rs = []
        pb_ps = []
        pb_labels = []
        for i, col in enumerate(available_area_cols):
            x = bdf[col].fillna(0).astype(int)
            y = bdf["K_i"]
            both = x.notna() & y.notna()
            if both.sum() < 10:
                continue
            r, p = sp_stats.pointbiserialr(x[both], y[both])
            label = area_labels[i] if i < len(area_labels) else f"Area {i+1}"
            pb_rs.append(r)
            pb_ps.append(p)
            pb_labels.append(label)

        if pb_ps:
            _, pb_fdr, _, _ = multipletests(pb_ps, method="fdr_bh")
            for lbl, r, p, q in zip(pb_labels, pb_rs, pb_ps, pb_fdr):
                key = lbl.lower().replace("/", "_").replace(" ", "_")
                _t(f"rpb_{key}_r", f"{r:.3f}")
                _t(f"rpb_{key}_p", f"{p:.3f}" if p >= 0.001 else "<0.001")
                _t(f"rpb_{key}_fdr", f"{q:.3f}" if q >= 0.001 else "<0.001")
                if verbose:
                    sig = "*" if p < 0.05 else ""
                    fdr_sig = "†" if q < 0.05 else ""
                    print(f"    {lbl}: r={r:.3f}, p={p:.3f}{sig}, FDR={q:.3f}{fdr_sig}")

    # --- Clinical measure correlations ---
    wide_path = os.path.join(DATA_DIR, "original", "participants_wideformat.xlsx")
    if os.path.exists(wide_path):
        wide = pd.read_excel(wide_path)
        wide["ID"] = wide["ID"].astype(str)
        df_clin = ki.merge(wide, on="ID", how="inner")

        pearson_panels = [
            ("phq_knee_pain_days__s1", "phq_knee_pain_days"),
            ("phq_percent_pain__s1", "phq_percent_pain"),
            ("womac_pain__s1", "womac_pain"),
            ("total_womac__s1", "womac_total"),
            ("womac_phys_function__s1", "womac_phys_function"),
            ("womac_stiffness__s1", "womac_stiffness"),
            ("qst_knee_pain_rating__s1", "qst_knee_pain_rating"),
        ]

        for col, key in pearson_panels:
            if col not in df_clin.columns:
                continue
            tmp = df_clin[["K_i", col]].dropna()
            if len(tmp) < 5:
                continue
            r, p = sp_stats.pearsonr(tmp["K_i"], tmp[col])
            _t(f"pearson_{key}_r", f"{r:.3f}")
            _t(f"pearson_{key}_p", f"{p:.3f}" if p >= 0.001 else "<0.001")
            _t(f"pearson_{key}_n", str(len(tmp)))
            if verbose:
                print(f"    Pearson {key}: r={r:.3f}, p={'<0.001' if p<0.001 else f'{p:.3f}'}, N={len(tmp)}")

        # Spearman for KL grade (ordinal)
        kl_col = next((c for c in ["kl_grade__s1", "kl_grade_s1", "klg__s1",
                                    "KL_grade__s1", "kellgren_lawrence__s1"]
                       if c in df_clin.columns), None)
        if kl_col:
            tmp = df_clin[["K_i", kl_col]].dropna()
            if len(tmp) >= 5:
                rho, p = sp_stats.spearmanr(tmp["K_i"], tmp[kl_col])
                _t("spearman_kl_grade_rho", f"{rho:.3f}")
                _t("spearman_kl_grade_p",
                   f"{p:.3f}" if p >= 0.001 else "<0.001")
                _t("spearman_kl_grade_n", str(len(tmp)))
                if verbose:
                    print(f"    Spearman KL grade: rho={rho:.3f}, "
                          f"p={'<0.001' if p<0.001 else f'{p:.3f}'}, N={len(tmp)}")
        else:
            if verbose:
                print("    KL grade column not found — skipping Spearman")

    text_df = pd.DataFrame(text_rows)
    text_df.to_csv(OUT_TEXT_CSV, index=False)
    if verbose:
        print(f"    Saved: {OUT_TEXT_CSV}")


# =====================================================================
# Main
# =====================================================================

def run_step12(verbose=True):
    if verbose:
        print("=" * 70)
        print("STEP 12 — Remaining figures (2, 3, S1, S2) + text numbers")
        print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    generate_figures_2_3(verbose)
    generate_figure_s1(verbose)
    generate_figure_s2(verbose)
    generate_text_numbers(verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 12 COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 12 — generate remaining figures."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_step12(verbose=not args.quiet)


if __name__ == "__main__":
    main()
