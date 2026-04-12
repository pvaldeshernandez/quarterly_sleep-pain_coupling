"""
Step 7 — Johnson-Neyman analysis for SP moderation ROIs.
======================================================================

Input:  derivatives/step8_sp_posterior_draws.npz
        results/step8_table5_sp_moderation.csv
Output:
  derivatives/
    step9_jn_sp_results.csv           — full JN grids per ROI
  results/
    step9_figure5_jn_nacc.png         — Figure 5: Left NAcc JN
    step9_figure6_jn_acc.png          — Figure 6: ACC JN
    step9_figure_s5_krause_jn.png     — Figure S5: 4 non-sig Krause JN
    step9_text_numbers.csv            — JN boundaries, % sample, slopes

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
DERIV_DIR = os.path.join(ROOT, "derivatives")
RESULTS_DIR = os.path.join(ROOT, "results")

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

IN_DRAWS_NPZ = os.path.join(DERIV_DIR, "step8_sp_posterior_draws.npz")
IN_TABLE5_CSV = os.path.join(RESULTS_DIR, "step8_table5_sp_moderation.csv")

OUT_JN_CSV = os.path.join(DERIV_DIR, "step9_jn_sp_results.csv")
OUT_FIG5 = os.path.join(RESULTS_DIR, "step9_figure5_jn_nacc.png")
OUT_FIG6 = os.path.join(RESULTS_DIR, "step9_figure6_jn_acc.png")
OUT_FIG_S5 = os.path.join(RESULTS_DIR, "step9_figure_s5_krause_jn.png")
OUT_TEXT_CSV = os.path.join(RESULTS_DIR, "step9_text_numbers.csv")

# ROIs that get individual JN figures
MAIN_FIGURE_ROIS = {
    "Left_NAcc": ("Figure 5", OUT_FIG5),
    "Right_dACC_MCC": ("Figure 6", OUT_FIG6),
}
# Non-significant Krause ROIs for the S5 2x2 merge
S5_ROIS = ["Right_S1", "Right_Middle_Insula", "Left_Thalamus", "Left_Anterior_Insula"]


def draw_jn_panel(ax, jn, title, simple_slopes):
    """Draw one JN panel on a matplotlib axes."""
    x_grid = jn["x_grid"]
    mean_curve = jn["post_mean"]
    lo_curve = jn["ci_lo"]
    hi_curve = jn["ci_hi"]

    credible = (lo_curve > 0) | (hi_curve < 0)
    for i in range(len(x_grid) - 1):
        if credible[i]:
            ax.axvspan(x_grid[i], x_grid[i + 1],
                       color="#C8E6C9", alpha=0.4, linewidth=0)

    ax.plot(x_grid, mean_curve, color="#1565C0", linewidth=2.5)
    ax.plot(x_grid, lo_curve, color="#1565C0", linewidth=1,
            linestyle="--", alpha=0.7)
    ax.plot(x_grid, hi_curve, color="#1565C0", linewidth=1,
            linestyle="--", alpha=0.7)
    ax.fill_between(x_grid, lo_curve, hi_curve, color="#BBDEFB", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    bds = jn["jn_boundaries"]
    if len(bds) > 0:
        ax.axvline(bds[0], color="#D32F2F", linewidth=1.5,
                   linestyle=":", alpha=0.8)

    colors = ["#E53935", "#FB8C00", "#43A047"]
    for i, ss in enumerate(simple_slopes):
        ax.errorbar(ss["x_val"], ss["mean"],
                    yerr=[[ss["mean"] - ss["ci_lo"]],
                          [ss["ci_hi"] - ss["mean"]]],
                    fmt="D", markersize=8, color=colors[i % len(colors)],
                    elinewidth=2, capsize=5, capthick=1.5, zorder=5)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Moderator (z)", fontsize=10)
    ax.set_ylabel("Conditional λ_sp", fontsize=10)
    ax.tick_params(labelsize=9)


def run_step7(verbose=True):
    from coupling_model import compute_jn_curve

    if verbose:
        print("=" * 70)
        print("STEP 7 — SP moderation Johnson-Neyman analysis")
        print("=" * 70)

    os.makedirs(DERIV_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    d = np.load(IN_DRAWS_NPZ)
    table5 = pd.read_csv(IN_TABLE5_CSV)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jn_rows = []
    text_rows = []
    jn_results = {}

    def _t(metric, value, note=""):
        text_rows.append({"metric": metric, "value": str(value), "note": note})

    for _, row in table5.iterrows():
        roi_name = row["ROI"]
        label = row["Label"]

        a2_key = f"{roi_name}_a2_draws"
        gamma_key = f"{roi_name}_gamma_sp_draws"
        x_key = f"{roi_name}_X_vals"
        mean_key = f"{roi_name}_raw_mean"
        sd_key = f"{roi_name}_raw_sd"

        if a2_key not in d:
            continue

        a2_draws = d[a2_key]
        gamma_sp_draws = d[gamma_key]
        X_vals = d[x_key]
        raw_mean = float(d[mean_key][0])
        raw_sd = float(d[sd_key][0])

        jn = compute_jn_curve(a2_draws, gamma_sp_draws, X_vals,
                              raw_mean=raw_mean, raw_sd=raw_sd,
                              clip_pct=(1, 99))
        jn_results[roi_name] = jn

        # Simple slopes at z = -2, 0, +2
        slopes = []
        for z_label, z_val in [("z=-2", -2.0), ("z=0", 0.0), ("z=+2", 2.0)]:
            cond = a2_draws + gamma_sp_draws * z_val
            slopes.append({
                "label": z_label, "x_val": z_val,
                "mean": float(np.mean(cond)),
                "ci_lo": float(np.percentile(cond, 2.5)),
                "ci_hi": float(np.percentile(cond, 97.5)),
            })

        bds = jn["jn_boundaries"]
        boundary = float(bds[0]) if len(bds) > 0 else None

        if verbose:
            print(f"\n  {label}:")
            if boundary is not None:
                raw_bd = boundary * raw_sd + raw_mean
                pct_below = float((X_vals < boundary).mean() * 100)
                print(f"    JN boundary: z={boundary:.3f} (raw={raw_bd:.3f}), "
                      f"{pct_below:.1f}% below")
            else:
                print(f"    JN boundary: none")
            for ss in slopes:
                sig = "*" if (ss["ci_lo"] > 0 or ss["ci_hi"] < 0) else ""
                print(f"    {ss['label']}: coupling={ss['mean']:.4f} "
                      f"[{ss['ci_lo']:.4f}, {ss['ci_hi']:.4f}]{sig}")

        # Text numbers
        if boundary is not None:
            raw_bd = boundary * raw_sd + raw_mean
            pct_below = float((X_vals < boundary).mean() * 100)
            _t(f"jn_sp_{roi_name}_boundary_z", f"{boundary:.3f}")
            _t(f"jn_sp_{roi_name}_boundary_raw", f"{raw_bd:.4f}")
            _t(f"jn_sp_{roi_name}_pct_below", f"{pct_below:.1f}")
        else:
            _t(f"jn_sp_{roi_name}_boundary", "none")
        for ss in slopes:
            _t(f"slope_sp_{roi_name}_{ss['label']}", f"{ss['mean']:.4f}")
            _t(f"slope_sp_{roi_name}_{ss['label']}_ci",
               f"[{ss['ci_lo']:.4f}, {ss['ci_hi']:.4f}]")

        # JN grid
        for i, x in enumerate(jn["x_grid"]):
            jn_rows.append({
                "ROI": roi_name, "x": float(x),
                "mean": float(jn["post_mean"][i]),
                "ci_lo": float(jn["ci_lo"][i]),
                "ci_hi": float(jn["ci_hi"][i]),
            })

    # Save JN grid
    pd.DataFrame(jn_rows).to_csv(OUT_JN_CSV, index=False)
    if verbose:
        print(f"\n  Saved JN grid: {OUT_JN_CSV}")

    # --- Main figures (5, 6): individual JN panels ---
    for roi_name, (fig_label, out_path) in MAIN_FIGURE_ROIS.items():
        if roi_name not in jn_results:
            continue
        jn = jn_results[roi_name]
        a2_draws = d[f"{roi_name}_a2_draws"]
        gamma_draws = d[f"{roi_name}_gamma_sp_draws"]
        slopes = []
        for z_label, z_val in [("z=-2", -2.0), ("z=0", 0.0), ("z=+2", 2.0)]:
            cond = a2_draws + gamma_draws * z_val
            slopes.append({
                "label": z_label, "x_val": z_val,
                "mean": float(np.mean(cond)),
                "ci_lo": float(np.percentile(cond, 2.5)),
                "ci_hi": float(np.percentile(cond, 97.5)),
            })
        fig, ax = plt.subplots(figsize=(10, 7))
        row = table5[table5["ROI"] == roi_name].iloc[0]
        draw_jn_panel(ax, jn, f"{row['Label']} ({fig_label})", slopes)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  Saved {fig_label}: {out_path}")

    # --- Figure S5: 2x2 merge of non-significant Krause ROIs ---
    available_s5 = [r for r in S5_ROIS if r in jn_results]
    if len(available_s5) >= 2:
        n_panels = len(available_s5)
        ncols = 2
        nrows = (n_panels + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows))
        axes = axes.ravel() if n_panels > 1 else [axes]
        for i, roi_name in enumerate(available_s5):
            jn = jn_results[roi_name]
            a2_draws = d[f"{roi_name}_a2_draws"]
            gamma_draws = d[f"{roi_name}_gamma_sp_draws"]
            slopes = []
            for z_label, z_val in [("z=-2", -2.0), ("z=0", 0.0), ("z=+2", 2.0)]:
                cond = a2_draws + gamma_draws * z_val
                slopes.append({
                    "label": z_label, "x_val": z_val,
                    "mean": float(np.mean(cond)),
                    "ci_lo": float(np.percentile(cond, 2.5)),
                    "ci_hi": float(np.percentile(cond, 97.5)),
                })
            row = table5[table5["ROI"] == roi_name].iloc[0]
            draw_jn_panel(axes[i], jn, row["Label"], slopes)
        for j in range(len(available_s5), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Non-significant Krause ROIs (SP moderation)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT_FIG_S5, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  Saved Figure S5: {OUT_FIG_S5}")

    # Save text numbers
    text_df = pd.DataFrame(text_rows)
    text_df.to_csv(OUT_TEXT_CSV, index=False)
    if verbose:
        print(f"  Saved text numbers: {OUT_TEXT_CSV}")
        print("\n" + "=" * 70)
        print("STEP 7 COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 7 — SP moderation JN analysis."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_step7(verbose=not args.quiet)


if __name__ == "__main__":
    main()
