#!/usr/bin/env python3
"""
Replot all manuscript figures from saved posterior draws — no model refitting.

Each step's own plotting code is used, saving figures to results/stepN/ as
the canonical location. Figures are then copied to figures/ for markdown rendering.

Special case: Figure 6 (bilateral ACC) uses Right dACC/MCC draws from the
archive groundtruth (the original run) plus Left dACC/MCC from the current step8.

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
    """Figures 5, 6, S5 — SP fMRI ROI JN.

    Figure 6 is bilateral ACC: Right dACC/MCC from archive groundtruth +
    Left dACC/MCC from current step8 draws.
    """
    if verbose:
        print("\n--- Figures 5, 6, S5: SP fMRI ROI JN ---")

    from coupling_model import compute_jn_curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as patheffects
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    import step9_sp_moderation_jn as s9
    s9.run_step9(verbose=verbose)

    # Figure 6 needs special treatment: rebuild with Right ACC from archive
    if verbose:
        print("  Rebuilding Figure 6 with Right dACC/MCC from archive groundtruth...")

    d8 = np.load(os.path.join(DERIV, "step8", "step8_sp_posterior_draws.npz"))
    d_gt = np.load(os.path.join(ARCHIVE_GT, "acc_posterior_draws.npz"))

    draw_jn = s9.draw_jn_panel

    def _roi_jn_from_d(d, roi_name, table5):
        a2d  = d[f"{roi_name}_a2_draws"]
        gd   = d[f"{roi_name}_gamma_sp_draws"]
        Xv   = d[f"{roi_name}_X_vals"]
        rm   = float(d[f"{roi_name}_raw_mean"])
        rs   = float(d[f"{roi_name}_raw_sd"])
        u_key = f"{roi_name}_u_sp_mean"
        u_mean = d[u_key] if u_key in d else None
        jn = compute_jn_curve(a2d, gd, Xv, raw_mean=rm, raw_sd=rs, clip_pct=(1, 99))
        raw_vals = Xv * rs + rm
        q1 = np.percentile(raw_vals, 25); q3 = np.percentile(raw_vals, 75)
        iqr = q3 - q1; med = np.median(raw_vals)
        low_r = q1 - 1.5*iqr; high_r = q3 + 1.5*iqr
        slopes = {}
        for lbl, xr in [("low", low_r), ("median", float(med)), ("high", high_r)]:
            xz = (xr - rm) / rs
            cond = a2d + gd * xz
            slopes[lbl] = {
                "beta":  float(np.mean(cond)),
                "ci_lo": float(np.percentile(cond, 2.5)),
                "ci_hi": float(np.percentile(cond, 97.5)),
                "sig":   (float(np.percentile(cond, 97.5)) < 0) or
                         (float(np.percentile(cond, 2.5)) > 0),
                "x_val": xr,
            }
        fence_raws = [low_r, float(med), high_r]
        person_y = a2d.mean() + gd.mean() * Xv + (u_mean if u_mean is not None else 0)
        dots = {"x_raw": raw_vals, "y": person_y}
        try:
            import pandas as pd
            lbl = table5[table5["ROI"] == roi_name].iloc[0]["Label"]
        except Exception:
            lbl = roi_name
        return jn, slopes, fence_raws, dots, lbl

    def _level_labels(fence_raws):
        return [
            f"Q1\u22121.5\u00b7IQR\n({fence_raws[0]:.3f})",
            f"Median\n({fence_raws[1]:.3f})",
            f"Q3+1.5\u00b7IQR\n({fence_raws[2]:.3f})",
        ]

    import pandas as pd
    table5 = pd.read_csv(os.path.join(RES, "step8", "step8_table5_sp_moderation.csv"))

    # Right dACC/MCC from archive groundtruth
    a2_gt = d_gt["a2_draws"]; g_gt = d_gt["gamma_sp_draws"]
    X_gt  = d_gt["X_vals"]
    rm_gt = float(d_gt["acc_mean"]); rs_gt = float(d_gt["acc_sd"])
    u_gt  = d_gt["u_sp_mean"] if "u_sp_mean" in d_gt else None

    jn_right = compute_jn_curve(a2_gt, g_gt, X_gt,
                                raw_mean=rm_gt, raw_sd=rs_gt, clip_pct=(1, 99))
    raw_gt   = X_gt * rs_gt + rm_gt
    q1 = np.percentile(raw_gt, 25); q3 = np.percentile(raw_gt, 75)
    iqr = q3 - q1; med = np.median(raw_gt)
    low_r = q1 - 1.5*iqr; high_r = q3 + 1.5*iqr
    sl_right = {}
    for lbl, xr in [("low", low_r), ("median", float(med)), ("high", high_r)]:
        xz = (xr - rm_gt) / rs_gt
        cond = a2_gt + g_gt * xz
        sl_right[lbl] = {
            "beta":  float(np.mean(cond)), "ci_lo": float(np.percentile(cond, 2.5)),
            "ci_hi": float(np.percentile(cond, 97.5)),
            "sig":   (float(np.percentile(cond, 97.5)) < 0) or
                     (float(np.percentile(cond, 2.5)) > 0),
            "x_val": xr,
        }
    fr_right = [low_r, float(med), high_r]
    py_right  = a2_gt.mean() + g_gt.mean() * X_gt + (u_gt if u_gt is not None else 0)
    dots_right = {"x_raw": raw_gt, "y": py_right}

    # Left dACC/MCC from step8
    jn_left, sl_left, fr_left, dots_left, lbl_left = _roi_jn_from_d(
        d8, "Left_dACC_MCC", table5)

    fig6, axes6 = plt.subplots(2, 1, figsize=(12.8, 8.05 * 2))

    draw_jn(axes6[0], jn_right, "Sleep \u2192 Pain", sl_right,
            _level_labels(fr_right), fr_right,
            xlabel="Right dACC/MCC BOLD activation (mean contrast)",
            legend_loc="lower right", info_loc="upper left",
            person_dots=dots_right)
    axes6[0].set_xlim(jn_right["x_grid"][0], jn_right["x_grid"][-1])

    draw_jn(axes6[1], jn_left, "Sleep \u2192 Pain", sl_left,
            _level_labels(fr_left), fr_left,
            xlabel=f"{lbl_left} BOLD activation (mean contrast)",
            legend_loc="lower right", info_loc="upper left",
            person_dots=dots_left)
    axes6[1].set_xlim(jn_left["x_grid"][0], jn_left["x_grid"][-1])

    fig6.tight_layout()
    out_fig6 = os.path.join(RES, "step9", "step9_figure6_jn_acc.png")
    fig6.savefig(out_fig6, dpi=300, bbox_inches="tight")
    plt.close(fig6)
    if verbose:
        print(f"  Saved: results/step9/step9_figure6_jn_acc.png (bilateral ACC)")

    copy_to_figures(os.path.join(RES, "step9", "step9_figure5_jn_nacc.png"),   "figure5.png",  verbose)
    copy_to_figures(out_fig6,                                                   "figure6.png",  verbose)
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
