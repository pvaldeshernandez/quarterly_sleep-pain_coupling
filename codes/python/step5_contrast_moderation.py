"""
Step 4 — Contrast moderation analysis (Johnson-Neyman).
======================================================================

Input:  derivatives/step3_posterior_draws.npz
        results/step3_table4_coupling.csv    (for Table 5 subset)
Output:
  derivatives/
    step4_jn_results.csv              — full JN grid for both directions
  results/
    step4_table5_contrast_mod.csv     — Table 5: contrast moderation params
    step4_figure4_jn_contrast_ps.png  — Figure 4: PS direction JN
    step4_figure_s3_jn_contrast_sp.png — Figure S3: SP direction JN (null)
    step4_text_numbers.csv            — JN boundary, simple slopes, etc.

This step reads the posterior draws from the VARX(1) fit (Step 3)
and runs the Bayesian Johnson-Neyman analysis on the contrast
moderation terms (omega_sp, omega_ps). It determines the range of
within-person pain localization values (K^w) over which the
conditional coupling is credibly different from zero.

The model code for ``compute_jn_curve`` lives in
``codes/lib/coupling_model.py``.

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


# =====================================================================
# Paths
# =====================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DERIV_DIR = os.path.join(ROOT, "derivatives")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step5")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step5")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

IN_DRAWS_NPZ = os.path.join(DERIV_DIR, "step4", "step4_posterior_draws.npz")

# Derivatives
OUT_JN_CSV = os.path.join(STEP_DERIV_DIR, "step5_jn_localization_results.csv")

# Results
OUT_FIG4 = os.path.join(STEP_RESULTS_DIR, "step5_figure4_jn_localization_ps.png")
OUT_FIG_S3 = os.path.join(STEP_RESULTS_DIR, "step5_figure_s3_jn_localization_sp.png")
OUT_TEXT_CSV = os.path.join(STEP_RESULTS_DIR, "step5_text_numbers.csv")


# =====================================================================
# Simple-slope computation
# =====================================================================

def compute_simple_slopes(intercept_draws, slope_draws, x_positions):
    """Compute posterior simple slopes at specified moderator values.

    For each (label, x_val) in ``x_positions``, computes the
    conditional coupling ``intercept + slope * x_val`` draw-by-draw
    and returns the posterior mean and 95% CrI.
    """
    rows = []
    for label, x_val in x_positions:
        conditional = intercept_draws + slope_draws * x_val
        mean = float(np.mean(conditional))
        lo = float(np.percentile(conditional, 2.5))
        hi = float(np.percentile(conditional, 97.5))
        credible = "yes" if (lo > 0 or hi < 0) else "no"
        rows.append({
            "label": label, "x_val": x_val,
            "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "credible": credible,
        })
    return rows


# =====================================================================
# JN figure drawing
# =====================================================================

def draw_jn_figure(jn, direction_label, simple_slopes, contrast_sd,
                   obs_coupling, obs_contrast, out_path):
    """Draw a Johnson-Neyman figure for one coupling direction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.8, 8.05))

    x_grid = jn["x_grid"]
    mean_curve = jn["mean"]
    lo_curve = jn["ci_lo"]
    hi_curve = jn["ci_hi"]

    # Credible region shading
    credible_mask = (lo_curve > 0) | (hi_curve < 0)
    for i in range(len(x_grid) - 1):
        if credible_mask[i]:
            ax.axvspan(x_grid[i], x_grid[i + 1],
                       color="#C8E6C9", alpha=0.4, linewidth=0)

    # Coupling curve + CrI
    ax.plot(x_grid, mean_curve, color="#1565C0", linewidth=2.5,
            label="Posterior mean")
    ax.plot(x_grid, lo_curve, color="#1565C0", linewidth=1,
            linestyle="--", alpha=0.7)
    ax.plot(x_grid, hi_curve, color="#1565C0", linewidth=1,
            linestyle="--", alpha=0.7, label="95% CrI")
    ax.fill_between(x_grid, lo_curve, hi_curve,
                     color="#BBDEFB", alpha=0.3)

    # Zero line
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)

    # JN boundary
    if jn["boundary"] is not None:
        ax.axvline(jn["boundary"], color="#D32F2F", linewidth=1.5,
                   linestyle=":", alpha=0.8, label="JN boundary")

    # Person dots
    if obs_coupling is not None and obs_contrast is not None:
        ax.scatter(obs_contrast, obs_coupling, s=8, alpha=0.15,
                   color="#1565C0", zorder=1)

    # Simple-slope markers
    level_colors = ["#E53935", "#FB8C00", "#43A047"]
    for i, ss in enumerate(simple_slopes):
        color = level_colors[i % len(level_colors)]
        ax.errorbar(ss["x_val"], ss["mean"],
                    yerr=[[ss["mean"] - ss["ci_lo"]],
                          [ss["ci_hi"] - ss["mean"]]],
                    fmt="D", markersize=10, color=color,
                    elinewidth=2.5, capsize=6, capthick=2,
                    zorder=5, label=ss["label"])

    ax.set_xlabel("Within-person pain localization ($K^w$)", fontsize=14)
    ax.set_ylabel(f"Conditional {direction_label} coupling", fontsize=14)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.tick_params(labelsize=11)

    # Body-dominant / knee-dominant labels
    xlim = ax.get_xlim()
    ax.text(xlim[0] + 0.02 * (xlim[1] - xlim[0]), ax.get_ylim()[0],
            "Body-dominant", fontsize=10, color="#666666",
            ha="left", va="bottom")
    ax.text(xlim[1] - 0.02 * (xlim[1] - xlim[0]), ax.get_ylim()[0],
            "Knee-dominant", fontsize=10, color="#666666",
            ha="right", va="bottom")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Main pipeline
# =====================================================================

def run_step5(verbose: bool = True):
    """Run contrast moderation JN analysis."""
    from coupling_model import compute_jn_curve

    if verbose:
        print("=" * 70)
        print("STEP 4 — Contrast moderation (Johnson-Neyman)")
        print("=" * 70)
        print(f"  Input: {IN_DRAWS_NPZ}")

    os.makedirs(DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    # Load posterior draws from Step 3
    d = np.load(IN_DRAWS_NPZ)
    a2_draws = d["a2_draws"]      # lambda_sp
    a4_draws = d["a4_draws"]      # omega_sp
    b1_draws = d["b1_draws"]      # lambda_ps
    b4_draws = d["b4_draws"]      # omega_ps
    u_sp_mean = d["u_sp_mean"]
    u_ps_mean = d["u_ps_mean"]
    obs_pid_idx = d["obs_pid_idx"]
    obs_contrast = d["obs_contrast"]
    contrast_vals = d["contrast_vals"]

    c_sd = float(np.std(contrast_vals))
    if verbose:
        print(f"  Contrast SD (within-person): {c_sd:.4f}")

    # Note: the contrast moderation parameters (delta_p, omega_sp,
    # delta_s, omega_ps) are already in Table 4 from Step 3. No
    # separate Table 5 is needed — the manuscript's Table 5 was
    # redundant with Table 4 and has been merged into it.

    # ==================================================================
    # JN analysis — both directions
    # ==================================================================
    x_positions = [
        ("Body-dominant (-2 SD)", -2 * c_sd),
        ("Balanced (0)", 0.0),
        ("Knee-dominant (+2 SD)", 2 * c_sd),
    ]

    # Helper to extract the first JN boundary (or None)
    def _first_boundary(jn_result):
        bds = jn_result["jn_boundaries"]
        return float(bds[0]) if len(bds) > 0 else None

    # ---- Pain-to-Sleep (PS) direction: Figure 4 ----
    jn_ps = compute_jn_curve(b1_draws, b4_draws, contrast_vals,
                             clip_pct=(0, 100))
    slopes_ps = compute_simple_slopes(b1_draws, b4_draws, x_positions)
    ps_boundary = _first_boundary(jn_ps)

    if verbose:
        print(f"\n  Pain-to-Sleep (PS) JN:")
        if ps_boundary is not None:
            print(f"    Boundary: K = {ps_boundary:.4f} "
                  f"({ps_boundary/c_sd:.2f} SD)")
            pct = float((contrast_vals > ps_boundary).mean() * 100)
            print(f"    {pct:.1f}% of observations in credible region")
        else:
            print(f"    No boundary within observed range")
        for ss in slopes_ps:
            sig = "*" if ss["credible"] == "yes" else ""
            print(f"    {ss['label']}: coupling = {ss['mean']:.4f} "
                  f"[{ss['ci_lo']:.4f}, {ss['ci_hi']:.4f}]{sig}")

    # Observation-level fitted coupling for dots
    b1_mean = float(np.mean(b1_draws))
    obs_ps = b1_mean + u_ps_mean[obs_pid_idx] + float(np.mean(b4_draws)) * obs_contrast

    # Adapt jn_ps keys for draw_jn_figure
    jn_ps_fig = {
        "x_grid": jn_ps["x_grid"],
        "mean": jn_ps["post_mean"],
        "ci_lo": jn_ps["ci_lo"],
        "ci_hi": jn_ps["ci_hi"],
        "boundary": ps_boundary,
    }
    draw_jn_figure(
        jn_ps_fig, "Pain → Sleep", slopes_ps, c_sd,
        obs_ps, obs_contrast, OUT_FIG4,
    )
    if verbose:
        print(f"  Saved Figure 4: {OUT_FIG4}")

    # ---- Sleep-to-Pain (SP) direction: Figure S3 ----
    jn_sp = compute_jn_curve(a2_draws, a4_draws, contrast_vals,
                             clip_pct=(0, 100))
    slopes_sp = compute_simple_slopes(a2_draws, a4_draws, x_positions)
    sp_boundary = _first_boundary(jn_sp)

    if verbose:
        print(f"\n  Sleep-to-Pain (SP) JN:")
        if sp_boundary is not None:
            print(f"    Boundary: K = {sp_boundary:.4f}")
        else:
            print(f"    No boundary within observed range (null)")
        for ss in slopes_sp:
            sig = "*" if ss["credible"] == "yes" else ""
            print(f"    {ss['label']}: coupling = {ss['mean']:.4f} "
                  f"[{ss['ci_lo']:.4f}, {ss['ci_hi']:.4f}]{sig}")

    a2_mean = float(np.mean(a2_draws))
    obs_sp = a2_mean + u_sp_mean[obs_pid_idx] + float(np.mean(a4_draws)) * obs_contrast

    jn_sp_fig = {
        "x_grid": jn_sp["x_grid"],
        "mean": jn_sp["post_mean"],
        "ci_lo": jn_sp["ci_lo"],
        "ci_hi": jn_sp["ci_hi"],
        "boundary": sp_boundary,
    }
    draw_jn_figure(
        jn_sp_fig, "Sleep → Pain", slopes_sp, c_sd,
        obs_sp, obs_contrast, OUT_FIG_S3,
    )
    if verbose:
        print(f"  Saved Figure S3: {OUT_FIG_S3}")

    # ==================================================================
    # Save JN grid results (derivative)
    # ==================================================================
    jn_rows = []
    for direction, jn_result in [("PS", jn_ps), ("SP", jn_sp)]:
        for i, x in enumerate(jn_result["x_grid"]):
            jn_rows.append({
                "direction": direction,
                "x": float(x),
                "mean": float(jn_result["post_mean"][i]),
                "ci_lo": float(jn_result["ci_lo"][i]),
                "ci_hi": float(jn_result["ci_hi"][i]),
            })
    pd.DataFrame(jn_rows).to_csv(OUT_JN_CSV, index=False)
    if verbose:
        print(f"\n  Saved JN grid: {OUT_JN_CSV}")

    # ==================================================================
    # Text numbers
    # ==================================================================
    text_rows = []

    def _t(metric, value, note=""):
        text_rows.append({"metric": metric, "value": str(value), "note": note})

    _t("contrast_sd_within", f"{c_sd:.4f}")

    # PS JN
    if ps_boundary is not None:
        _t("jn_ps_boundary_K", f"{ps_boundary:.4f}")
        _t("jn_ps_boundary_sd", f"{ps_boundary/c_sd:.2f}")
        pct_cred = float((contrast_vals > ps_boundary).mean() * 100)
        _t("jn_ps_pct_credible", f"{pct_cred:.1f}")
    else:
        _t("jn_ps_boundary", "none")

    for ss in slopes_ps:
        label = ss["label"].replace(" ", "_").replace("(", "").replace(")", "")
        _t(f"slope_ps_{label}", f"{ss['mean']:.4f}")
        _t(f"slope_ps_{label}_ci", f"[{ss['ci_lo']:.4f}, {ss['ci_hi']:.4f}]")

    # SP JN
    if sp_boundary is not None:
        _t("jn_sp_boundary_K", f"{sp_boundary:.4f}")
    else:
        _t("jn_sp_boundary", "none")

    for ss in slopes_sp:
        label = ss["label"].replace(" ", "_").replace("(", "").replace(")", "")
        _t(f"slope_sp_{label}", f"{ss['mean']:.4f}")
        _t(f"slope_sp_{label}_ci", f"[{ss['ci_lo']:.4f}, {ss['ci_hi']:.4f}]")

    text_df = pd.DataFrame(text_rows)
    text_df.to_csv(OUT_TEXT_CSV, index=False)
    if verbose:
        print(f"  Saved text numbers: {OUT_TEXT_CSV}")
        print("\n" + "=" * 70)
        print("STEP 4 COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 4 — contrast moderation Johnson-Neyman analysis."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()
    run_step5(verbose=not args.quiet)


if __name__ == "__main__":
    main()
