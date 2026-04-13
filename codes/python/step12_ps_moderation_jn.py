"""
Step 11 — Johnson-Neyman analysis for PS arousal moderation ROIs.
======================================================================

Input:  derivatives/step10_ps_fmri_posterior_draws.npz
        derivatives/step10_ps_vbm_posterior_draws.npz
        results/step9_table_s1_fmri_arousal.csv
        results/step9_table_s1_vbm_arousal.csv
Output:
  derivatives/
    step11_jn_ps_fmri_results.csv
    step11_jn_ps_vbm_results.csv
  results/
    step11_figure_s7_fmri_arousal_jn.png   — Figure S7
    step11_figure_s8_vbm_arousal_jn.png    — Figure S8
    step11_text_numbers.csv

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
ROOT = os.path.dirname(os.path.dirname(HERE))
DERIV_DIR = os.path.join(ROOT, "derivatives")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step12")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step12")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

IN_FMRI_DRAWS = os.path.join(DERIV_DIR, "step11", "step11_ps_fmri_posterior_draws.npz")
IN_VBM_DRAWS = os.path.join(DERIV_DIR, "step11", "step11_ps_vbm_posterior_draws.npz")
IN_FMRI_TABLE = os.path.join(RESULTS_DIR, "step11", "step11_table_s1_fmri_arousal.csv")
IN_VBM_TABLE = os.path.join(RESULTS_DIR, "step11", "step11_table_s1_vbm_arousal.csv")

OUT_FMRI_JN = os.path.join(STEP_DERIV_DIR, "step12_jn_ps_fmri_results.csv")
OUT_VBM_JN = os.path.join(STEP_DERIV_DIR, "step12_jn_ps_vbm_results.csv")
OUT_FIG_S7 = os.path.join(STEP_RESULTS_DIR, "step12_figure_s7_fmri_arousal_jn.png")
OUT_FIG_S8 = os.path.join(STEP_RESULTS_DIR, "step12_figure_s8_vbm_arousal_jn.png")
OUT_TEXT_CSV = os.path.join(STEP_RESULTS_DIR, "step12_text_numbers.csv")


def draw_jn_panel(ax, jn, title):
    """Draw one JN panel."""
    x_grid = jn["x_grid"]
    mean_curve = jn["post_mean"]
    lo_curve = jn["ci_lo"]
    hi_curve = jn["ci_hi"]

    credible = (lo_curve > 0) | (hi_curve < 0)
    for i in range(len(x_grid) - 1):
        if credible[i]:
            ax.axvspan(x_grid[i], x_grid[i + 1],
                       color="#C8E6C9", alpha=0.4, linewidth=0)

    ax.plot(x_grid, mean_curve, color="#1565C0", linewidth=2)
    ax.plot(x_grid, lo_curve, "--", color="#1565C0", linewidth=1, alpha=0.7)
    ax.plot(x_grid, hi_curve, "--", color="#1565C0", linewidth=1, alpha=0.7)
    ax.fill_between(x_grid, lo_curve, hi_curve, color="#BBDEFB", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    bds = jn["jn_boundaries"]
    for bd in bds:
        ax.axvline(bd, color="#D32F2F", linewidth=1.5, linestyle=":", alpha=0.8)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Moderator (z)", fontsize=9)
    ax.set_ylabel("Conditional λ_ps", fontsize=9)
    ax.tick_params(labelsize=8)


def run_jn_for_modality(modality_name, draws_path, table_path, verbose=True):
    """Run JN for all ROIs in one modality. Returns JN grid rows + text rows."""
    from coupling_model import compute_jn_curve

    d = np.load(draws_path)
    table = pd.read_csv(table_path)

    jn_rows = []
    text_rows = []
    jn_results = {}

    for _, row in table.iterrows():
        roi_name = row["ROI"]
        b1_key = f"{roi_name}_b1_draws"
        gamma_key = f"{roi_name}_gamma_ps_draws"
        x_key = f"{roi_name}_X_vals"
        mean_key = f"{roi_name}_raw_mean"
        sd_key = f"{roi_name}_raw_sd"

        if b1_key not in d:
            continue

        b1_draws = d[b1_key]
        gamma_ps_draws = d[gamma_key]
        X_vals = d[x_key]
        raw_mean = float(d[mean_key][0])
        raw_sd = float(d[sd_key][0])

        jn = compute_jn_curve(b1_draws, gamma_ps_draws, X_vals,
                              raw_mean=raw_mean, raw_sd=raw_sd,
                              clip_pct=(1, 99))
        jn_results[roi_name] = jn

        bds = jn["jn_boundaries"]
        boundary = float(bds[0]) if len(bds) > 0 else None

        if verbose:
            print(f"    {row['Label']} ({modality_name}):")
            if boundary is not None:
                print(f"      JN boundary: z={boundary:.3f}")
            else:
                print(f"      JN boundary: none")

        # Text numbers
        if boundary is not None:
            text_rows.append({"metric": f"jn_ps_{modality_name}_{roi_name}_boundary_z",
                              "value": f"{boundary:.3f}"})
        else:
            text_rows.append({"metric": f"jn_ps_{modality_name}_{roi_name}_boundary",
                              "value": "none"})

        # JN grid
        for i, x in enumerate(jn["x_grid"]):
            jn_rows.append({
                "ROI": roi_name, "modality": modality_name,
                "x": float(x),
                "mean": float(jn["post_mean"][i]),
                "ci_lo": float(jn["ci_lo"][i]),
                "ci_hi": float(jn["ci_hi"][i]),
            })

    return jn_rows, text_rows, jn_results, table


def run_step12(verbose=True):
    if verbose:
        print("=" * 70)
        print("STEP 10 — PS arousal moderation Johnson-Neyman analysis")
        print("=" * 70)

    os.makedirs(DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_text = []

    # ---- fMRI BOLD ----
    if verbose:
        print("\n  fMRI BOLD JN:")
    fmri_jn_rows, fmri_text, fmri_jn_results, fmri_table = run_jn_for_modality(
        "fMRI", IN_FMRI_DRAWS, IN_FMRI_TABLE, verbose
    )
    pd.DataFrame(fmri_jn_rows).to_csv(OUT_FMRI_JN, index=False)
    all_text.extend(fmri_text)

    # Figure S7: fMRI arousal JN panels
    rois_fmri = list(fmri_jn_results.keys())
    if rois_fmri:
        ncols = 2
        nrows = (len(rois_fmri) + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
        axes = axes.ravel() if len(rois_fmri) > 1 else [axes]
        for i, roi_name in enumerate(rois_fmri):
            row = fmri_table[fmri_table["ROI"] == roi_name].iloc[0]
            draw_jn_panel(axes[i], fmri_jn_results[roi_name],
                          f"{row['Label']} (fMRI BOLD)")
        for j in range(len(rois_fmri), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Pain-to-Sleep arousal moderation (fMRI BOLD)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT_FIG_S7, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  Saved Figure S7: {OUT_FIG_S7}")

    # ---- VBM GM volume ----
    if verbose:
        print("\n  VBM GM volume JN:")
    vbm_jn_rows, vbm_text, vbm_jn_results, vbm_table = run_jn_for_modality(
        "VBM", IN_VBM_DRAWS, IN_VBM_TABLE, verbose
    )
    pd.DataFrame(vbm_jn_rows).to_csv(OUT_VBM_JN, index=False)
    all_text.extend(vbm_text)

    # Figure S8: VBM arousal JN panels
    rois_vbm = list(vbm_jn_results.keys())
    if rois_vbm:
        ncols = 2
        nrows = (len(rois_vbm) + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
        axes = axes.ravel() if len(rois_vbm) > 1 else [axes]
        for i, roi_name in enumerate(rois_vbm):
            row = vbm_table[vbm_table["ROI"] == roi_name].iloc[0]
            draw_jn_panel(axes[i], vbm_jn_results[roi_name],
                          f"{row['Label']} (VBM GM volume)")
        for j in range(len(rois_vbm), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Pain-to-Sleep arousal moderation (VBM GM volume)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT_FIG_S8, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  Saved Figure S8: {OUT_FIG_S8}")

    # Save text numbers
    pd.DataFrame(all_text).to_csv(OUT_TEXT_CSV, index=False)
    if verbose:
        print(f"  Saved text numbers: {OUT_TEXT_CSV}")
        print("\n" + "=" * 70)
        print("STEP 10 COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 11 — PS arousal moderation JN analysis."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_step12(verbose=not args.quiet)


if __name__ == "__main__":
    main()
