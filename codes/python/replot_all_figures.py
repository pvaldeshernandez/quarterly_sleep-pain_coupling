#!/usr/bin/env python3
"""
Replot all manuscript figures from saved posterior draws — no model refitting.

Each step's own plotting code is used, saving figures to results/stepN/ as
the canonical location. Figures are then copied to figures/ for markdown rendering.

Figure 6 (ACC JN) uses the Right dACC/MCC from the current step8/step9 run.

Usage:
    python replot_all_figures.py
    python replot_all_figures.py --quiet

Author: Pedro Valdes-Hernandez
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(os.path.dirname(HERE))
DERIV   = os.path.join(ROOT, "derivatives")
RES     = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
ARCHIVE_GT = os.path.join(ROOT, "archive", "old_paper", "results.groundtruth")
LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

os.makedirs(FIGURES, exist_ok=True)


def copy_to_figures(src, dest_name, verbose=True):
    dest = os.path.join(FIGURES, dest_name)
    shutil.copy2(src, dest)
    if verbose:
        print(f"    -> figures/{dest_name}")


def run_step4_figures(verbose=True):
    """Figures 2 & 3 — coupling boxstrip + forest."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patheffects as patheffects
    from matplotlib.lines import Line2D

    if verbose:
        print("\n--- Figures 2 & 3: coupling boxstrip + forest ---")

    person_df = pd.read_csv(os.path.join(DERIV, "step4", "step4_person_coupling.csv"))
    table4    = pd.read_csv(os.path.join(RES,   "step4", "step4_table4_coupling.csv"))

    def get_param(key):
        row = table4[table4["Parameter"] == key].iloc[0]
        return float(row["Estimate"]), float(row["CrI_lo"]), float(row["CrI_hi"]), float(row["P_neg"])

    b1_mean, b1_lo, b1_hi, b1_pneg = get_param("b1")
    a2_mean, a2_lo, a2_hi, a2_pneg = get_param("a2")

    STEP_RES = os.path.join(RES, "step4")
    os.makedirs(STEP_RES, exist_ok=True)

    for direction, pop_mean, pop_lo, pop_hi, prob_neg, col_mean, col_lo, col_hi, fname in [
        ("Pain -> Sleep", b1_mean, b1_lo, b1_hi, b1_pneg,
         "beta_ps_mean", "beta_ps_ci_lo", "beta_ps_ci_hi",
         "step4_figure2_ps_coupling.png"),
        ("Sleep -> Pain", a2_mean, a2_lo, a2_hi, a2_pneg,
         "beta_sp_mean", "beta_sp_ci_lo", "beta_sp_ci_hi",
         "step4_figure3_sp_coupling.png"),
    ]:
        df   = person_df.sort_values(col_mean).reset_index(drop=True)
        n    = len(df)
        vals = df[col_mean].values
        lo   = df[col_lo].values
        hi   = df[col_hi].values

        fig = plt.figure(figsize=(18, 8))
        gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.12)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])

        ax_a.boxplot(vals, vert=False, widths=0.5, patch_artist=True,
                     boxprops=dict(facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=1.5),
                     medianprops=dict(color="#D32F2F", linewidth=2.5),
                     whiskerprops=dict(color="#1565C0", linewidth=1.2),
                     capprops=dict(color="#1565C0", linewidth=1.2),
                     flierprops=dict(marker="", markersize=0))

        rng   = np.random.default_rng(42)
        jitter = rng.uniform(0.7, 1.3, n)
        dot_colors  = ["#42A5F5" if x < 0 else "#EF5350" for x in vals]
        edge_colors = ["#0D47A1"  if x < 0 else "#B71C1C"  for x in vals]
        sc = ax_a.scatter(vals, jitter, c=dot_colors, alpha=0.7, s=80, zorder=3,
                          marker="o", edgecolors=edge_colors, linewidths=0.8)
        sc.set_path_effects([patheffects.withSimplePatchShadow(
            offset=(0.5, -0.5), shadow_rgbFace="#555555", alpha=0.25)])

        ax_a.axvline(0, color="black", linewidth=1, linestyle="-", zorder=2, alpha=0.5)
        mean_val = float(np.mean(vals))
        ax_a.plot(mean_val, 1.0, marker="D", color="#757575", markersize=12,
                  zorder=5, markeredgecolor="white", markeredgewidth=1.2)

        lam = "\u03bb"; arrow = "\u2192"
        parts = direction.split("->")
        title_a = f"A.  {parts[0].strip()} {arrow} {parts[1].strip()} coupling"
        ax_a.set_title(title_a, fontsize=20, fontweight="bold", loc="left", pad=10)
        ax_a.set_xlabel(f"{lam} (posterior mean)", fontsize=18)
        ax_a.set_yticks([]); ax_a.tick_params(axis="x", labelsize=16)

        pop_text = (f"Population {lam} = {pop_mean:.3f}\n"
                    f"[{pop_lo:.3f}, {pop_hi:.3f}]\n"
                    f"P({lam} < 0) = {prob_neg:.3f}")
        ax_a.text(0.02, 0.98, pop_text, transform=ax_a.transAxes,
                  fontsize=14, va="top", ha="left", color="black", fontweight="bold")

        med = float(np.median(vals)); q25, q75 = np.percentile(vals, [25, 75])
        stats_text = (f"N = {n}\nmean = {mean_val:.3f}\nmedian = {med:.3f}\n"
                      f"IQR = [{q25:.3f}, {q75:.3f}]")
        ax_a.text(0.98, 0.98, stats_text, transform=ax_a.transAxes,
                  fontsize=13, va="top", ha="right", color="#333333",
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            edgecolor="#CCCCCC", alpha=0.9))
        ax_a.legend(handles=[
            Line2D([0],[0], color="#D32F2F", linewidth=2.5, label="Median"),
            Line2D([0],[0], marker="D", color="#757575", markersize=10,
                   markeredgecolor="white", markeredgewidth=0.8, linestyle="", label="Mean"),
            Line2D([0],[0], marker="o", color="#42A5F5", markersize=8,
                   markeredgecolor="#0D47A1", markeredgewidth=0.8, linestyle="",
                   label=f"Individual ({lam} < 0)"),
            Line2D([0],[0], marker="o", color="#EF5350", markersize=8,
                   markeredgecolor="#B71C1C", markeredgewidth=0.8, linestyle="",
                   label=f"Individual ({lam} > 0)"),
        ], loc="lower right", fontsize=13, framealpha=0.9, edgecolor="#CCCCCC")

        for k in range(n):
            color = "#1565C0" if vals[k] < 0 else "#D32F2F"
            excl  = (lo[k] > 0) or (hi[k] < 0)
            ax_b.plot([lo[k], hi[k]], [k, k], color=color,
                      linewidth=1.8 if excl else 0.7, alpha=0.7 if excl else 0.4, zorder=2)
            ax_b.plot(vals[k], k, marker="|", color=color,
                      markersize=3.5, alpha=0.8, zorder=3, markeredgewidth=0.9)

        ax_b.axvline(0,        color="black",   linewidth=0.8, linestyle="-", alpha=0.5, zorder=1)
        ax_b.axvline(mean_val, color="#757575",  linewidth=2,   linestyle="--", alpha=0.7, zorder=4)
        title_b = f"B.  {parts[0].strip()} {arrow} {parts[1].strip()} coupling"
        ax_b.set_title(title_b, fontsize=20, fontweight="bold", loc="left", pad=10)
        ax_b.set_xlabel(f"{lam} (posterior mean \u00b1 95% CrI)", fontsize=18)
        ax_b.set_yticks([]); ax_b.set_ylim(-1, n+1); ax_b.tick_params(axis="x", labelsize=16)

        fig.canvas.draw()
        pos_a = ax_a.get_position(); pos_b = ax_b.get_position()
        top = max(pos_a.y1, pos_b.y1); bottom = min(pos_a.y0, pos_b.y0)
        ax_a.set_position([pos_a.x0, bottom, pos_a.width, top-bottom])
        ax_b.set_position([pos_b.x0, bottom, pos_b.width, top-bottom])

        out = os.path.join(STEP_RES, fname)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  Saved: results/step4/{fname}")

    fig_map = {
        "step4_figure2_ps_coupling.png": "figure2.png",
        "step4_figure3_sp_coupling.png": "figure3.png",
    }
    for src_name, dest_name in fig_map.items():
        copy_to_figures(os.path.join(STEP_RES, src_name), dest_name, verbose)


def run_step5_figures(verbose=True):
    """Figures 4 & S3 — contrast moderation JN."""
    if verbose:
        print("\n--- Figures 4 & S3: contrast moderation JN ---")
    # Import step5's own run function — it saves to results/step5/
    import step5_contrast_moderation as s5
    s5.run_step5(verbose=verbose)

    copy_to_figures(s5.OUT_FIG4,    "figure4.png",              verbose)
    copy_to_figures(s5.OUT_FIG_S3,  "figure_s2_jn_contrast_sp.png", verbose)


def run_step9_figures(verbose=True):
    """Figures 5, 6, S5 — SP fMRI ROI JN."""
    if verbose:
        print("\n--- Figures 5, 6, S5: SP fMRI ROI JN ---")

    import step9_sp_moderation_jn as s9
    s9.run_step9(verbose=verbose)

    copy_to_figures(os.path.join(RES, "step9", "step9_figure5_jn_nacc.png"),     "figure5.png",  verbose)
    copy_to_figures(os.path.join(RES, "step9", "step9_figure6_jn_acc.png"),       "figure6.png",  verbose)
    copy_to_figures(os.path.join(RES, "step9", "step9_figure_s5_krause_jn.png"), "figure_s5_krause_sp_jn_merged.png", verbose)


def run_step12_figures(verbose=True):
    """Figures S7 & S8 — PS arousal JN."""
    if verbose:
        print("\n--- Figures S7 & S8: PS arousal JN ---")
    import step12_ps_moderation_jn as s12
    s12.run_step12(verbose=verbose)

    copy_to_figures(s12.OUT_FIG_S7, "figure_s9_fmri_arousal_jn_merged.png", verbose)
    copy_to_figures(s12.OUT_FIG_S8, "figure_s10_vbm_arousal_jn_merged.png", verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Replot all manuscript figures from saved draws."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    run_step4_figures(verbose)
    run_step5_figures(verbose)
    run_step9_figures(verbose)
    run_step12_figures(verbose)

    if verbose:
        print(f"\nAll figures saved to results/stepN/ and copied to figures/")


if __name__ == "__main__":
    main()
