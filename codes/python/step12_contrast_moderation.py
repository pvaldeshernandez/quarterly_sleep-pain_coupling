"""
Step 04 — Contrast moderation analysis (Johnson-Neyman).
======================================================================

Input:  derivatives/step08_coupling_model/step08_posterior_draws.npz
Output:
  derivatives/
    step12_jn_localization_results.csv — full JN grid for both directions
  results/
    step12_figure4_jn_localization_ps.png  — Figure 4: PS direction JN
    step12_figure_jn_localization_sp.png — Figure S3: SP direction JN (null)
    step12_text_numbers.csv                — JN boundary, simple slopes, etc.

Note: contrast moderation parameters (delta_p, omega_sp, delta_s, omega_ps)
are in Table 4 (results/step08/step08_table4_coupling.csv), not a separate table.

This step reads the posterior draws from the VARX(1) fit (Step 03)
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
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step12_contrast_moderation")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step12_contrast_moderation")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

IN_DRAWS_NPZ = os.path.join(DERIV_DIR, "step08_coupling_model", "step08_posterior_draws.npz")

# Derivatives
OUT_JN_CSV = os.path.join(STEP_DERIV_DIR, "step12_jn_localization_results.csv")

# Results
OUT_FIG4 = os.path.join(STEP_RESULTS_DIR, "step12_figure4_jn_localization_ps.png")
SUPP_DIR = os.path.join(RESULTS_DIR, "supplementary_materials")
os.makedirs(SUPP_DIR, exist_ok=True)
OUT_FIG_S3 = os.path.join(SUPP_DIR, "figure_jn_localization_sp.png")
OUT_TEXT_CSV = os.path.join(STEP_DERIV_DIR, "step12_text_numbers.csv")


# =====================================================================
# Simple-slope computation
# =====================================================================

def compute_simple_slopes(intercept_draws, slope_draws, x_positions):
    """Compute posterior simple slopes at specified moderator values."""
    rows = []
    for label, x_val in x_positions:
        conditional = intercept_draws + slope_draws * x_val
        mean = float(np.mean(conditional))
        lo = float(np.percentile(conditional, 2.5))
        hi = float(np.percentile(conditional, 97.5))
        p_neg = float((conditional < 0).mean())
        credible = "yes" if (lo > 0 or hi < 0) else "no"
        rows.append({
            "label": label, "x_val": x_val,
            "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "prob_neg": p_neg, "credible": credible,
        })
    return rows


# =====================================================================
# JN figure drawing  — canonical implementation matching original scripts
# =====================================================================

def draw_jn_panel(ax, jn, panel_label, direction_label, slopes_dict,
                  level_labels, level_x_vals,
                  xlabel=None, body_knee_labels=False,
                  legend_loc="best", info_loc="upper left",
                  person_dots=None):
    """Draw a JN panel with full annotations (person dots, rug, labels, info box).

    Matches the original plot_manuscript_contrast_moderation.py implementation.
    """
    import matplotlib.patheffects
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    x_grid = jn["x_grid"]
    post_mean = jn["post_mean"]
    ci_lo = jn["ci_lo"]
    ci_hi = jn["ci_hi"]
    sig = jn["sig"]
    obs_vals = jn["obs_vals"]

    # Person dots (behind everything else)
    if person_dots is not None:
        ax.scatter(person_dots["x"], person_dots["y"],
                   s=30, color="#42A5F5", alpha=0.7, edgecolors="#0D47A1",
                   linewidths=0.8, zorder=2,
                   path_effects=[matplotlib.patheffects.withSimplePatchShadow(
                       offset=(0.5, -0.5), shadow_rgbFace="#1565C0", alpha=0.3)])

    # Shaded CrI band (green = credible, grey = non-credible)
    for i in range(len(x_grid) - 1):
        color = "#81C784" if sig[i] else "#BDBDBD"
        alpha = 0.35 if sig[i] else 0.25
        ax.fill_between(x_grid[i:i + 2], ci_lo[i:i + 2], ci_hi[i:i + 2],
                        color=color, alpha=alpha, linewidth=0)

    ax.plot(x_grid, post_mean, color="#1565C0", linewidth=2.5)
    ax.plot(x_grid, ci_lo, color="#1565C0", linewidth=1, linestyle="--", alpha=0.7)
    ax.plot(x_grid, ci_hi, color="#1565C0", linewidth=1, linestyle="--", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)

    # Rug plot
    ax.scatter(obs_vals, np.full_like(obs_vals, ax.get_ylim()[0]),
               marker="|", color="#424242", alpha=0.12, s=25, zorder=1)

    # Simple slope error bars with annotated labels
    levels = list(slopes_dict.keys())
    y_lo, y_hi = ax.get_ylim()
    y_range = y_hi - y_lo
    x_lo, x_hi = x_grid[0], x_grid[-1]
    x_range = x_hi - x_lo

    label_specs = []
    for k in range(len(levels)):
        if k == 0:
            label_specs.append((x_lo + x_range * 0.02, y_hi - y_range * 0.03, "left", "top"))
        elif k == 1:
            label_specs.append((level_x_vals[k] + x_range * 0.06, y_hi - y_range * 0.03, "left", "top"))
        else:
            label_specs.append((x_hi - x_range * 0.02, y_hi - y_range * 0.03, "right", "top"))

    for k, level in enumerate(levels):
        d = slopes_dict[level]
        x_pos = level_x_vals[k]
        beta = d.get("beta", d.get("mean"))
        cl = d["ci_lo"]
        ch = d["ci_hi"]
        p_neg = d.get("prob_neg", d.get("p_neg", 0.5))
        sig_mark = "*" if p_neg > 0.975 or p_neg < 0.025 else ""
        label_text = f"{level_labels[k]}\n{beta:.3f} [{cl:.3f}, {ch:.3f}]{sig_mark}"

        ax.errorbar(x_pos, beta,
                    yerr=[[beta - cl], [ch - beta]],
                    fmt="o", color="black", markersize=10,
                    capsize=7, capthick=2.5, elinewidth=2.5,
                    markeredgecolor="white", markeredgewidth=1.5,
                    zorder=6)

        spec = label_specs[k]
        txt_x, txt_y, ha, va = spec
        ax.annotate(label_text,
                    xy=(x_pos, ch if va == "bottom" else beta),
                    xytext=(txt_x, txt_y),
                    fontsize=10, fontfamily="monospace", color="black",
                    fontweight="bold", ha=ha, va=va,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="black", alpha=0.90),
                    arrowprops=dict(arrowstyle="->", color="black",
                                   lw=1.5, connectionstyle="arc3,rad=0.2"),
                    zorder=7)

    ax.set_ylabel(f"Coupling \u03bb ({direction_label})", fontsize=18)
    ax.tick_params(axis="y", labelsize=16)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=18)
    ax.tick_params(axis="x", labelsize=16)

    if body_knee_labels:
        ax.text(0.01, -0.10, "\u2190 Body-dominant", transform=ax.transAxes,
                fontsize=16, ha="left", va="top", color="black", fontstyle="italic")
        ax.text(0.99, -0.10, "Knee-dominant \u2192", transform=ax.transAxes,
                fontsize=16, ha="right", va="top", color="black", fontstyle="italic")

    # Title with equation
    prefix = f"{panel_label}.  " if panel_label else ""
    ax.set_title(
        f"{prefix}{direction_label} coupling: "
        f"\u03bb(K\u1d42) = {jn['intercept_mean']:.3f} + ({jn['slope_mean']:.3f})\u00b7K\u1d42",
        fontsize=20, fontweight="bold", loc="left", pad=10)

    # JN boundaries
    for x0 in jn["jn_boundaries"]:
        ax.axvline(x0, color="black", linewidth=2, linestyle=":", alpha=0.8, zorder=4)
        y_top = ax.get_ylim()[1]
        y_bot = ax.get_ylim()[0]
        ax.text(x0, y_top - (y_top - y_bot) * 0.04, f"{x0:.2f}",
                fontsize=14, color="black", fontweight="bold",
                va="top", ha="center")

    # Info box: % in credible region
    jn_text_parts = []
    if np.any(jn["sig_negative"]):
        sig_lo = x_grid[jn["sig_negative"]][0]
        sig_hi = x_grid[jn["sig_negative"]][-1]
        pct_sig = float(((obs_vals >= sig_lo) & (obs_vals <= sig_hi)).mean() * 100)
        jn_text_parts.append(f"{pct_sig:.1f}% of observations in credible region")
    elif np.any(jn["sig_positive"]):
        sig_lo = x_grid[jn["sig_positive"]][0]
        sig_hi = x_grid[jn["sig_positive"]][-1]
        pct_sig = float(((obs_vals >= sig_lo) & (obs_vals <= sig_hi)).mean() * 100)
        jn_text_parts.append(f"{pct_sig:.1f}% of observations in credible region")
    if not jn["jn_boundaries"]:
        if np.all(sig):
            jn_text_parts.append("Coupling credible across entire range")
        elif not np.any(sig):
            jn_text_parts.append("No JN boundary: coupling non-credible across range")

    if jn_text_parts:
        locs = {"upper left": (0.02, 0.97, "left", "top"),
                "upper right": (0.98, 0.97, "right", "top"),
                "lower right": (0.98, 0.10, "right", "bottom")}
        ix, iy, iha, iva = locs.get(info_loc, (0.02, 0.97, "left", "top"))
        ax.text(ix, iy, "\n".join(jn_text_parts), transform=ax.transAxes,
                fontsize=14, va=iva, ha=iha, fontfamily="monospace")

    # Legend
    legend_elements = [
        Line2D([0], [0], color="#1565C0", linewidth=2.5, label="Posterior mean"),
        Line2D([0], [0], color="#1565C0", linewidth=1, linestyle="--", label="95% CrI"),
        Patch(facecolor="#81C784", alpha=0.35, label="Credible (CrI excludes 0)"),
        Patch(facecolor="#BDBDBD", alpha=0.25, label="Non-credible"),
        Line2D([0], [0], color="black", linewidth=2, linestyle=":", label="JN boundary"),
        Line2D([0], [0], marker="o", color="black", markersize=8,
               markeredgecolor="white", markeredgewidth=1, linestyle="",
               label="Simple slope \u00b1 95% CrI"),
    ]
    if person_dots is not None:
        legend_elements.insert(0,
            Line2D([0], [0], marker="o", color="#42A5F5", markersize=7,
                   markeredgecolor="#0D47A1", markeredgewidth=0.8, linestyle="",
                   label="Fitted coupling"))
    ax.legend(handles=legend_elements, loc=legend_loc, fontsize=13,
              framealpha=0.9, edgecolor="#CCCCCC",
              bbox_to_anchor=(1.0, 0.08) if "right" in legend_loc else (0.0, 0.08),
              borderaxespad=0.3)


# =====================================================================
# Main pipeline
# =====================================================================

def run_step12(verbose: bool = True, refit: bool = False):
    """Run contrast moderation JN analysis."""
    from coupling_model import compute_jn_curve

    if not refit and verbose:
        print("  NOTE: Running in replot mode (loading saved posterior draws).")
        print("        Use --refit to re-run the MCMC computation from scratch.")

    if verbose:
        print("=" * 70)
        print("STEP 04 — Contrast moderation (Johnson-Neyman)")
        print("=" * 70)
        print(f"  Input: {IN_DRAWS_NPZ}")

    os.makedirs(DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    # Load posterior draws from Step 03
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

    # Convert slopes_ps list to dict keyed by label for draw_jn_panel
    slopes_ps_dict = {ss["label"]: ss for ss in slopes_ps}
    level_labels_ps = [ss["label"] for ss in slopes_ps]
    level_x_vals_ps = [ss["x_val"] for ss in slopes_ps]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.8, 8.05))
    draw_jn_panel(ax, jn_ps, "", "Pain \u2192 Sleep",
                  slopes_ps_dict, level_labels_ps, level_x_vals_ps,
                  xlabel="K\u1d42",
                  body_knee_labels=True,
                  legend_loc="lower left", info_loc="lower right",
                  person_dots={"x": obs_contrast, "y": obs_ps})
    os.makedirs(os.path.dirname(OUT_FIG4), exist_ok=True)
    fig.savefig(OUT_FIG4, dpi=300, bbox_inches="tight")
    plt.close(fig)
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
            sig_str = "*" if ss["credible"] == "yes" else ""
            print(f"    {ss['label']}: coupling = {ss['mean']:.4f} "
                  f"[{ss['ci_lo']:.4f}, {ss['ci_hi']:.4f}]{sig_str}")

    a2_mean = float(np.mean(a2_draws))
    obs_sp = a2_mean + u_sp_mean[obs_pid_idx] + float(np.mean(a4_draws)) * obs_contrast

    slopes_sp_dict = {ss["label"]: ss for ss in slopes_sp}
    level_labels_sp = [ss["label"] for ss in slopes_sp]
    level_x_vals_sp = [ss["x_val"] for ss in slopes_sp]

    fig, ax = plt.subplots(figsize=(12.8, 8.05))
    draw_jn_panel(ax, jn_sp, "", "Sleep \u2192 Pain",
                  slopes_sp_dict, level_labels_sp, level_x_vals_sp,
                  xlabel="K\u1d42",
                  body_knee_labels=True,
                  legend_loc="lower right", info_loc="upper left",
                  person_dots={"x": obs_contrast, "y": obs_sp})
    ax.set_xlim(jn_sp["x_grid"][0], jn_sp["x_grid"][-1])
    os.makedirs(os.path.dirname(OUT_FIG_S3), exist_ok=True)
    fig.savefig(OUT_FIG_S3, dpi=300, bbox_inches="tight")
    plt.close(fig)
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
        print("STEP 04 COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 04 — contrast moderation Johnson-Neyman analysis."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--refit", action="store_true",
        help="Re-run computation from scratch instead of loading saved derivatives",
    )
    args = parser.parse_args()
    run_step12(verbose=not args.quiet, refit=args.refit)


if __name__ == "__main__":
    main()
