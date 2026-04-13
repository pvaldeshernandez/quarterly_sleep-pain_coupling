"""
Step 12 — Generate all supplementary material outputs.
======================================================================

Reads derivatives from previous steps and produces all supplementary
tables, figures, and text numbers.

Output (results/):
  step12_table_s1_fmri_arousal.csv     — Table S1 fMRI panel
  step12_table_s1_vbm_arousal.csv      — Table S1 VBM panel
  step12_vbm_sign_concordance.csv      — VBM 5/5 sign test
  step12_figure_s1_endorsement.png     — Figure S1
  step12_figure_s2_convergent.png      — Figure S2
  step12_figure_s3_jn_localization_sp.png — Figure S3
  step12_figure_s5_krause_jn.png       — Figure S5
  step12_figure_s7_fmri_arousal_jn.png — Figure S7
  step12_figure_s8_vbm_arousal_jn.png  — Figure S8

Author: Pedro Valdes-Hernandez (with Claude Opus 4.6)
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
DERIV_DIR = os.path.join(ROOT, "derivatives")
RESULTS_DIR = os.path.join(ROOT, "results")

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)


# =====================================================================
# Shared JN panel drawing
# =====================================================================

def _draw_jn_panel(ax, jn_grid_df, title):
    """Draw one JN panel from a grid DataFrame."""
    x = jn_grid_df["x"].values
    mean = jn_grid_df["mean"].values
    lo = jn_grid_df["ci_lo"].values
    hi = jn_grid_df["ci_hi"].values

    credible = (lo > 0) | (hi < 0)
    for i in range(len(x) - 1):
        if credible[i]:
            ax.axvspan(x[i], x[i + 1], color="#C8E6C9", alpha=0.4, linewidth=0)

    ax.plot(x, mean, color="#1565C0", linewidth=2)
    ax.plot(x, lo, "--", color="#1565C0", linewidth=1, alpha=0.7)
    ax.plot(x, hi, "--", color="#1565C0", linewidth=1, alpha=0.7)
    ax.fill_between(x, lo, hi, color="#BBDEFB", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Moderator (z)", fontsize=9)
    ax.tick_params(labelsize=8)


def _draw_jn_figure(jn_grid_df, title, ylabel, out_path):
    """Draw a single-panel JN figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_jn_panel(ax, jn_grid_df, title)
    ax.set_ylabel(ylabel, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_multi_jn(jn_grid_df, roi_col, label_map, suptitle, ylabel, out_path):
    """Draw a multi-panel JN figure (2-column grid)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rois = jn_grid_df[roi_col].unique()
    n = len(rois)
    ncols = 2
    nrows = max(1, (n + 1) // 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i, roi in enumerate(rois):
        sub = jn_grid_df[jn_grid_df[roi_col] == roi]
        label = label_map.get(roi, roi)
        _draw_jn_panel(axes[i], sub, label)
        axes[i].set_ylabel(ylabel, fontsize=9)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Figure S1 — Factor endorsement validation
# =====================================================================

def generate_figure_s1(out_path, verbose=True):
    from scipy import stats as sp_stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if verbose:
        print("  Figure S1: Factor endorsement validation")

    extracted = pd.read_csv(os.path.join(DATA_DIR, "step0_extracted_long.csv"))
    processed = pd.read_csv(os.path.join(DERIV_DIR, "step2_processed_long.csv"))

    ki = processed.groupby("ID")["contrast_factor"].mean().reset_index()
    ki.columns = ["ID", "K_i"]

    baseline = extracted[extracted["quarter"] == 0].copy()
    baseline["ID"] = baseline["ID"].astype(str)
    ki["ID"] = ki["ID"].astype(str)
    df = ki.merge(baseline, on="ID", how="inner")

    area_cols = [f"phq_pain_areas___{i}__s1" for i in range(1, 14)]
    available = [c for c in area_cols if c in df.columns]

    area_labels = [
        "Hands", "Arms", "Shoulders", "Neck", "Head/Face/Jaw",
        "Chest", "Stomach", "Pelvis", "Upper Back", "Lower Back",
        "Knees", "Legs", "Feet/Ankles",
    ]

    rpb_results = []
    for i, col in enumerate(available):
        x = df[col].fillna(0).astype(int)
        y = df["K_i"]
        both = x.notna() & y.notna()
        if both.sum() < 10:
            continue
        r, p = sp_stats.pointbiserialr(x[both], y[both])
        label = area_labels[i] if i < len(area_labels) else f"Area {i+1}"
        rpb_results.append({"area": label, "r_pb": r, "p": p})

    if not rpb_results:
        if verbose:
            print("    SKIP: no valid correlations")
        return

    rpb_df = pd.DataFrame(rpb_results).sort_values("r_pb", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#1565C0" if r < 0 else "#D32F2F" for r in rpb_df["r_pb"]]
    ax.barh(range(len(rpb_df)), rpb_df["r_pb"].values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(rpb_df)))
    ax.set_yticklabels(rpb_df["area"].values, fontsize=11)
    ax.set_xlabel("Point-biserial r with mean contrast", fontsize=12)
    ax.axvline(0, color="black", linewidth=0.8)
    for i, (_, row) in enumerate(rpb_df.iterrows()):
        if row["p"] < 0.05:
            ax.text(row["r_pb"] + 0.01 * np.sign(row["r_pb"]),
                    i, "*", fontsize=14, ha="center", va="center",
                    fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    Saved: {out_path}")


# =====================================================================
# Figure S2 — Convergent validity scatter
# =====================================================================

def generate_figure_s2(out_path, verbose=True):
    from scipy import stats as sp_stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if verbose:
        print("  Figure S2: Convergent validity scatter")

    processed = pd.read_csv(os.path.join(DERIV_DIR, "step2_processed_long.csv"))
    wide_path = os.path.join(DATA_DIR, "original", "participants_wideformat.xlsx")
    if not os.path.exists(wide_path):
        if verbose:
            print("    SKIP: wideformat xlsx not found")
        return

    wide = pd.read_excel(wide_path)
    wide["ID"] = wide["ID"].astype(str)

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
    available = [(c, l) for c, l in panels if c in df.columns]
    if not available:
        if verbose:
            print("    SKIP: no clinical columns")
        return

    ncols = 4
    nrows = max(1, (len(available) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.ravel() if len(available) > 1 else [axes]

    for i, (col, label) in enumerate(available):
        ax = axes[i]
        tmp = df[["K_i", col]].dropna()
        if len(tmp) < 5:
            ax.set_visible(False)
            continue
        x, y = tmp["K_i"].values, tmp[col].values
        r, p = sp_stats.pearsonr(x, y)
        ax.scatter(x, y, alpha=0.35, s=18, color="steelblue", edgecolor="none")
        slope, intercept = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 200)
        ax.plot(xline, intercept + slope * xline, color="firebrick", linewidth=2)
        pstr = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.05, 0.95, f"r = {r:.3f}\n{pstr}\nN = {len(tmp)}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                          alpha=0.85, edgecolor="gray"))
        ax.set_xlabel("Person-mean contrast", fontsize=10)
        ax.set_ylabel(label, fontsize=10)

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Convergent validity: contrast vs baseline clinical measures",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    Saved: {out_path}")


# =====================================================================
# Main
# =====================================================================

def run_step12(verbose=True):
    if verbose:
        print("=" * 70)
        print("STEP 12 — Supplementary materials")
        print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Table S1: copy from Step 10 derivatives to results ----
    for src_name, dst_name in [
        ("step10_fmri_arousal_moderation.csv", "step12_table_s1_fmri_arousal.csv"),
        ("step10_vbm_arousal_moderation.csv", "step12_table_s1_vbm_arousal.csv"),
        ("step10_vbm_sign_concordance.csv", "step12_vbm_sign_concordance.csv"),
    ]:
        src = os.path.join(DERIV_DIR, src_name)
        dst = os.path.join(RESULTS_DIR, dst_name)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)
            if verbose:
                print(f"  Saved: {dst}")

    # ---- Figure S1 ----
    generate_figure_s1(
        os.path.join(RESULTS_DIR, "step12_figure_s1_endorsement.png"), verbose
    )

    # ---- Figure S2 ----
    generate_figure_s2(
        os.path.join(RESULTS_DIR, "step12_figure_s2_convergent.png"), verbose
    )

    # ---- Figure S3: SP localization JN (null) ----
    jn_loc = os.path.join(DERIV_DIR, "step4_jn_localization_results.csv")
    if os.path.exists(jn_loc):
        jn_df = pd.read_csv(jn_loc)
        sp_grid = jn_df[jn_df["direction"] == "SP"]
        if len(sp_grid) > 0:
            out_s3 = os.path.join(RESULTS_DIR, "step12_figure_s3_jn_localization_sp.png")
            _draw_jn_figure(sp_grid, "Sleep \u2192 Pain (null)",
                            "Conditional \u03bb_sp", out_s3)
            if verbose:
                print(f"  Saved Figure S3: {out_s3}")

    # ---- Figure S5: non-sig Krause ROIs 2x2 ----
    sp_jn_path = os.path.join(DERIV_DIR, "step8_jn_sp_results.csv")
    sp_table_path = os.path.join(RESULTS_DIR, "step7_table5_sp_moderation.csv")
    if os.path.exists(sp_jn_path) and os.path.exists(sp_table_path):
        sp_jn = pd.read_csv(sp_jn_path)
        sp_table = pd.read_csv(sp_table_path)
        s5_rois = ["Right_S1", "Right_Middle_Insula",
                    "Left_Thalamus", "Left_Anterior_Insula"]
        s5_grid = sp_jn[sp_jn["ROI"].isin(s5_rois)]
        if len(s5_grid) > 0:
            label_map = dict(zip(sp_table["ROI"], sp_table["Label"]))
            out_s5 = os.path.join(RESULTS_DIR, "step12_figure_s5_krause_jn.png")
            _draw_multi_jn(s5_grid, "ROI", label_map,
                           "Non-significant Krause ROIs (SP moderation)",
                           "Conditional \u03bb_sp", out_s5)
            if verbose:
                print(f"  Saved Figure S5: {out_s5}")

    # ---- Figures S7, S8: arousal JN panels ----
    for modality, jn_file, table_file, fig_name, suptitle, ylabel in [
        ("fMRI", "step11_jn_ps_fmri_results.csv",
         "step10_fmri_arousal_moderation.csv",
         "step12_figure_s7_fmri_arousal_jn.png",
         "Pain-to-Sleep arousal moderation (fMRI BOLD)",
         "Conditional \u03bb_ps"),
        ("VBM", "step11_jn_ps_vbm_results.csv",
         "step10_vbm_arousal_moderation.csv",
         "step12_figure_s8_vbm_arousal_jn.png",
         "Pain-to-Sleep arousal moderation (VBM GM volume)",
         "Conditional \u03bb_ps"),
    ]:
        jn_path = os.path.join(DERIV_DIR, jn_file)
        tbl_path = os.path.join(DERIV_DIR, table_file)
        if os.path.exists(jn_path) and os.path.exists(tbl_path):
            jn_df = pd.read_csv(jn_path)
            tbl = pd.read_csv(tbl_path)
            label_map = dict(zip(tbl["ROI"], tbl["Label"]))
            out = os.path.join(RESULTS_DIR, fig_name)
            _draw_multi_jn(jn_df, "ROI", label_map, suptitle, ylabel, out)
            if verbose:
                print(f"  Saved {fig_name}: {out}")

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 12 COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 12 — generate all supplementary material outputs."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_step12(verbose=not args.quiet)


if __name__ == "__main__":
    main()
