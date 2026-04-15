"""
Step 09 — Johnson-Neyman analysis for SP moderation ROIs.
======================================================================

Input:  derivatives/step08/step08_sp_posterior_draws.npz
        results/step08/step08_table5_sp_moderation.csv
Output:
  derivatives/
    step09_jn_sp_results.csv              — full JN grids per ROI
  results/
    step09_figure5_jn_nacc.png            — Figure 5: Left NAcc JN
    step09_figure6_jn_acc.png             — Figure 6: ACC JN (Right + Left, 2 panels)
    step09_figure_s5_krause_jn.png        — Figure S5: 4 non-sig Krause JN
    step09_text_numbers.csv               — JN boundaries, % sample, slopes

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
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step09_sp_jn")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step09_sp_jn")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

IN_DRAWS_NPZ = os.path.join(DERIV_DIR, "step08_sp_moderation", "step08_sp_posterior_draws.npz")
IN_TABLE5_CSV = os.path.join(RESULTS_DIR, "step08_sp_moderation", "step08_table5_sp_moderation.csv")

OUT_JN_CSV = os.path.join(STEP_DERIV_DIR, "step09_jn_sp_results.csv")
OUT_FIG5 = os.path.join(STEP_RESULTS_DIR, "step09_figure5_jn_nacc.png")
OUT_FIG6 = os.path.join(STEP_RESULTS_DIR, "step09_figure6_jn_acc.png")
SUPP_DIR = os.path.join(RESULTS_DIR, "supplementary_materials")
os.makedirs(SUPP_DIR, exist_ok=True)
OUT_FIG_S5 = os.path.join(SUPP_DIR, "figure_s5_krause_jn.png")
OUT_TEXT_CSV = os.path.join(STEP_RESULTS_DIR, "step09_text_numbers.csv")

# ROI for Figure 5 (single panel)
FIG5_ROI = "Left_NAcc"
# ROIs for Figure 6 (two panels stacked: Right ACC on top, Left ACC below)
FIG6_ROIS = ["Right_dACC_MCC", "Left_dACC_MCC"]
# Non-credible Krause ROIs for the S5 2x2 merge
S5_ROIS = ["Right_S1", "Right_Middle_Insula", "Left_Thalamus", "Left_Anterior_Insula"]


# =====================================================================
# Canonical JN panel — matches original plot_manuscript_nacc_moderation.py
# =====================================================================

def draw_jn_panel(ax, jn, direction_label, slopes_dict,
                  level_labels, level_x_vals,
                  xlabel=None, legend_loc="lower right", info_loc="upper left",
                  person_dots=None, left_label_inside=False):
    """Draw a full JN panel with person dots, rug, annotated simple slopes,
    info box, and legend. Matches the original manuscript figure scripts.
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
    raw_mean = jn.get("raw_mean")
    raw_sd = jn.get("raw_sd")

    # Person dots (behind everything else)
    if person_dots is not None:
        ax.scatter(person_dots["x_raw"], person_dots["y"],
                   s=60, color="#42A5F5", alpha=0.7, edgecolors="#0D47A1",
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
               marker="|", color="#424242", alpha=0.4, s=60, zorder=1)

    # Simple slope error bars with annotated labels
    levels = list(slopes_dict.keys())
    y_lo, y_hi = ax.get_ylim()
    y_range = y_hi - y_lo
    x_lo, x_hi = x_grid[0], x_grid[-1]
    x_range = x_hi - x_lo

    label_specs = []
    for k in range(len(levels)):
        beta_k = slopes_dict[levels[k]]["beta"]
        if k == 0:
            label_specs.append((level_x_vals[k] + x_range * 0.04,
                                 beta_k + y_range * 0.25, "left"))
        elif k == 1:
            label_specs.append((level_x_vals[k] - x_range * 0.04,
                                 beta_k + y_range * 0.22, "right"))
        else:
            label_specs.append((level_x_vals[k] - x_range * 0.04,
                                 beta_k + y_range * 0.12, "right"))

    for k, level in enumerate(levels):
        d_s = slopes_dict[level]
        x_pos = level_x_vals[k]
        beta = d_s["beta"]
        cl = d_s["ci_lo"]
        ch = d_s["ci_hi"]
        is_sig = d_s.get("sig", False)
        sig_mark = "*" if is_sig else ""
        label_text = f"{level_labels[k]}\n{beta:.3f} [{cl:.3f}, {ch:.3f}]{sig_mark}"

        ax.errorbar(x_pos, beta,
                    yerr=[[beta - cl], [ch - beta]],
                    fmt="o", color="black", markersize=10,
                    capsize=7, capthick=2.5, elinewidth=2.5,
                    markeredgecolor="white", markeredgewidth=1.5,
                    zorder=6)

        if k == 0 and left_label_inside:
            ax.annotate(label_text,
                        xy=(x_pos, cl), xycoords="data",
                        xytext=(0.02, 0.03), textcoords="axes fraction",
                        fontsize=10, fontfamily="monospace", color="black",
                        fontweight="bold", ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="black", alpha=0.90),
                        arrowprops=dict(arrowstyle="->", color="black",
                                        lw=1.5, connectionstyle="arc3,rad=0.2"),
                        zorder=7)
        else:
            txt_x, txt_y, ha = label_specs[k]
            ax.annotate(label_text,
                        xy=(x_pos, ch if txt_y > beta else cl),
                        xytext=(txt_x, txt_y),
                        fontsize=10, fontfamily="monospace", color="black",
                        fontweight="bold", ha=ha, va="bottom",
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

    # Title with equation in raw units
    if raw_mean is not None and raw_sd is not None:
        gamma_raw = jn["slope_mean"] / raw_sd
        intercept_raw = jn["intercept_mean"] - jn["slope_mean"] * raw_mean / raw_sd
        ax.set_title(
            f"{direction_label} coupling: "
            f"\u03bb(X) = {intercept_raw:.3f} + ({gamma_raw:.3f})\u00b7X",
            fontsize=20, fontweight="bold", loc="left", pad=10)
    else:
        ax.set_title(
            f"{direction_label} coupling",
            fontsize=20, fontweight="bold", loc="left", pad=10)

    # JN boundaries
    y_top = ax.get_ylim()[1]
    y_bot = ax.get_ylim()[0]
    for x0 in jn["jn_boundaries"]:
        ax.axvline(x0, color="black", linewidth=2, linestyle=":", alpha=0.8, zorder=4)
        ax.text(x0, y_top - (y_top - y_bot) * 0.04, f"{x0:.3f}",
                fontsize=14, color="black", fontweight="bold",
                va="top", ha="center")

    # Info box
    jn_text_parts = []
    if jn["jn_boundaries"]:
        x_grid_z = jn["x_grid_z"]
        obs_z = jn["X_vals_z"]
        sig_at_obs = np.interp(obs_z, x_grid_z, sig.astype(float))
        pct = float((sig_at_obs > 0.5).mean() * 100)
        jn_text_parts.append(f"{pct:.0f}% of sample in credible region")
    else:
        if np.all(sig):
            jn_text_parts.append("Coupling credible across entire range")
        elif not np.any(sig):
            jn_text_parts.append("No JN boundary: coupling non-credible across range")

    if jn_text_parts:
        has_bounds = bool(jn["jn_boundaries"])
        if has_bounds:
            ix, iy, iha, iva = 0.98, 0.97, "right", "top"
        elif info_loc == "upper left":
            ix, iy, iha, iva = 0.02, 0.97, "left", "top"
        else:
            ix, iy, iha, iva = 0.98, 0.97, "right", "top"
        ax.text(ix, iy, "\n".join(jn_text_parts), transform=ax.transAxes,
                fontsize=14, va=iva, ha=iha, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.85) if has_bounds else None)

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
            Line2D([0], [0], marker="o", color="#42A5F5", markersize=8,
                   markeredgecolor="#0D47A1", markeredgewidth=0.8, linestyle="",
                   label="Fitted coupling"))
    ax.legend(handles=legend_elements, loc=legend_loc, fontsize=13,
              framealpha=0.9, edgecolor="#CCCCCC",
              bbox_to_anchor=(1.0, 0.08) if "right" in legend_loc else (0.12, 0.08),
              borderaxespad=0.3)


def run_step09(verbose=True, refit=False):
    from coupling_model import compute_jn_curve

    if verbose:
        print("=" * 70)
        print("STEP 09 — SP moderation Johnson-Neyman analysis")
        print("=" * 70)

    os.makedirs(DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    if not refit and os.path.exists(IN_DRAWS_NPZ) and os.path.exists(IN_TABLE5_CSV):
        if verbose:
            print("  WARNING: Running in replot mode -- loading saved derivatives.")
            print("  If you have changed upstream data or code, re-run with --refit.")

    d = np.load(IN_DRAWS_NPZ)
    table5 = pd.read_csv(IN_TABLE5_CSV)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jn_rows = []
    text_rows = []
    jn_results = {}
    slopes_all = {}

    def _t(metric, value, note=""):
        text_rows.append({"metric": metric, "value": str(value), "note": note})

    for _, row in table5.iterrows():
        roi_name = row["ROI"]

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

        # Simple slopes at Q1-1.5*IQR, median, Q3+1.5*IQR (raw units)
        raw_vals = X_vals * raw_sd + raw_mean
        q1_raw = float(np.percentile(raw_vals, 25))
        q3_raw = float(np.percentile(raw_vals, 75))
        iqr_raw = q3_raw - q1_raw
        median_raw = float(np.median(raw_vals))
        low_fence_raw = q1_raw - 1.5 * iqr_raw
        high_fence_raw = q3_raw + 1.5 * iqr_raw

        low_z = (low_fence_raw - raw_mean) / raw_sd
        med_z = (median_raw - raw_mean) / raw_sd
        high_z = (high_fence_raw - raw_mean) / raw_sd

        slopes = {}
        for label, x_z, x_raw in [
            ("low", low_z, low_fence_raw),
            ("median", med_z, median_raw),
            ("high", high_z, high_fence_raw),
        ]:
            cond = a2_draws + gamma_sp_draws * x_z
            slopes[label] = {
                "beta": float(np.mean(cond)),
                "ci_lo": float(np.percentile(cond, 2.5)),
                "ci_hi": float(np.percentile(cond, 97.5)),
                "sig": (float(np.percentile(cond, 97.5)) < 0) or
                       (float(np.percentile(cond, 2.5)) > 0),
                "x_val": x_raw,
                "x_val_z": x_z,
            }
        slopes_all[roi_name] = slopes

        bds = jn["jn_boundaries"]
        boundary = float(bds[0]) if len(bds) > 0 else None

        if verbose:
            print(f"\n  {row['Label']}:")
            if boundary is not None:
                pct_below = float((X_vals < (boundary - raw_mean) / raw_sd).mean() * 100)
                print(f"    JN boundary: raw={boundary:.3f}, {pct_below:.1f}% below")
            else:
                print(f"    JN boundary: none")
            for lbl, ss in slopes.items():
                sig_str = "*" if ss["sig"] else ""
                print(f"    {lbl}: coupling={ss['beta']:.4f} "
                      f"[{ss['ci_lo']:.4f}, {ss['ci_hi']:.4f}]{sig_str}")

        # Text numbers
        if boundary is not None:
            _t(f"jn_sp_{roi_name}_boundary_raw", f"{boundary:.4f}")
            x_grid_z = jn["x_grid_z"]
            obs_z = jn["X_vals_z"]
            sig_at_obs = np.interp(obs_z, x_grid_z, jn["sig"].astype(float))
            pct = float((sig_at_obs > 0.5).mean() * 100)
            _t(f"jn_sp_{roi_name}_pct_credible", f"{pct:.1f}")
        else:
            _t(f"jn_sp_{roi_name}_boundary", "none")
        for lbl, ss in slopes.items():
            _t(f"slope_sp_{roi_name}_{lbl}", f"{ss['beta']:.4f}")
            _t(f"slope_sp_{roi_name}_{lbl}_ci", f"[{ss['ci_lo']:.4f}, {ss['ci_hi']:.4f}]")

        # JN grid
        for i, x in enumerate(jn["x_grid"]):
            jn_rows.append({
                "ROI": roi_name, "x": float(x),
                "mean": float(jn["post_mean"][i]),
                "ci_lo": float(jn["ci_lo"][i]),
                "ci_hi": float(jn["ci_hi"][i]),
            })

    pd.DataFrame(jn_rows).to_csv(OUT_JN_CSV, index=False)
    if verbose:
        print(f"\n  Saved JN grid: {OUT_JN_CSV}")

    def _person_dots(roi_name):
        a2_draws = d[f"{roi_name}_a2_draws"]
        gamma_draws = d[f"{roi_name}_gamma_sp_draws"]
        X_vals = d[f"{roi_name}_X_vals"]
        u_sp_mean = d[f"{roi_name}_u_sp_mean"] if f"{roi_name}_u_sp_mean" in d else None
        raw_mean = float(d[f"{roi_name}_raw_mean"][0])
        raw_sd = float(d[f"{roi_name}_raw_sd"][0])
        a2_mean = float(np.mean(a2_draws))
        gamma_mean = float(np.mean(gamma_draws))
        person_x_raw = X_vals * raw_sd + raw_mean
        if u_sp_mean is not None:
            person_y = a2_mean + gamma_mean * X_vals + u_sp_mean
        else:
            person_y = a2_mean + gamma_mean * X_vals
        return {"x_raw": person_x_raw, "y": person_y}

    # --- Figure 5: Left NAcc (single panel) ---
    if FIG5_ROI in jn_results:
        jn = jn_results[FIG5_ROI]
        slopes = slopes_all[FIG5_ROI]
        row = table5[table5["ROI"] == FIG5_ROI].iloc[0]
        raw_mean = float(d[f"{FIG5_ROI}_raw_mean"][0])
        raw_sd = float(d[f"{FIG5_ROI}_raw_sd"][0])

        level_labels = [
            f"Q1\u22121.5\u00b7IQR\n({slopes['low']['x_val']:.3f})",
            f"Median\n({slopes['median']['x_val']:.3f})",
            f"Q3+1.5\u00b7IQR\n({slopes['high']['x_val']:.3f})",
        ]
        level_x_vals = [slopes["low"]["x_val"], slopes["median"]["x_val"],
                        slopes["high"]["x_val"]]

        dots = _person_dots(FIG5_ROI)

        fig, ax = plt.subplots(figsize=(12.8, 8.05))
        draw_jn_panel(ax, jn, "Sleep \u2192 Pain", slopes,
                      level_labels, level_x_vals,
                      xlabel=f"{row['Label']} BOLD activation (mean contrast)",
                      legend_loc="lower right", info_loc="upper left",
                      person_dots=dots)
        ax.set_xlim(jn["x_grid"][0], jn["x_grid"][-1])
        fig.savefig(OUT_FIG5, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  Saved Figure 5: {OUT_FIG5}")

    # --- Figure 6: Right + Left dACC/MCC (2 panels stacked vertically) ---
    available_f6 = [r for r in FIG6_ROIS if r in jn_results]
    if available_f6:
        fig, axes = plt.subplots(len(available_f6), 1,
                                 figsize=(12.8, 8.05 * len(available_f6)))
        if len(available_f6) == 1:
            axes = [axes]
        for ax, roi_name in zip(axes, available_f6):
            jn = jn_results[roi_name]
            slopes = slopes_all[roi_name]
            row_r = table5[table5["ROI"] == roi_name].iloc[0]
            level_labels = [
                f"Q1\u22121.5\u00b7IQR\n({slopes['low']['x_val']:.3f})",
                f"Median\n({slopes['median']['x_val']:.3f})",
                f"Q3+1.5\u00b7IQR\n({slopes['high']['x_val']:.3f})",
            ]
            level_x_vals = [slopes["low"]["x_val"], slopes["median"]["x_val"],
                            slopes["high"]["x_val"]]
            dots = _person_dots(roi_name)
            draw_jn_panel(ax, jn, "Sleep \u2192 Pain", slopes,
                          level_labels, level_x_vals,
                          xlabel=f"{row_r['Label']} BOLD activation (mean contrast)",
                          legend_loc="lower right", info_loc="upper left",
                          person_dots=dots)
            ax.set_xlim(jn["x_grid"][0], jn["x_grid"][-1])
        fig.tight_layout()
        fig.savefig(OUT_FIG6, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  Saved Figure 6: {OUT_FIG6}")

    # --- Figure S5: 2x2 merge of non-credible Krause ROIs ---
    available_s5 = [r for r in S5_ROIS if r in jn_results]
    if len(available_s5) >= 2:
        n_panels = len(available_s5)
        ncols = 2
        nrows = (n_panels + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(12.8 * ncols, 8.05 * nrows))
        axes = axes.ravel() if n_panels > 1 else [axes]
        for i, roi_name in enumerate(available_s5):
            jn = jn_results[roi_name]
            slopes = slopes_all[roi_name]
            row_r = table5[table5["ROI"] == roi_name].iloc[0]
            level_labels = [
                f"Q1\u22121.5\u00b7IQR\n({slopes['low']['x_val']:.3f})",
                f"Median\n({slopes['median']['x_val']:.3f})",
                f"Q3+1.5\u00b7IQR\n({slopes['high']['x_val']:.3f})",
            ]
            level_x_vals = [slopes["low"]["x_val"], slopes["median"]["x_val"],
                            slopes["high"]["x_val"]]
            dots = _person_dots(roi_name)
            draw_jn_panel(axes[i], jn, "Sleep \u2192 Pain", slopes,
                          level_labels, level_x_vals,
                          xlabel=f"{row_r['Label']} BOLD activation (mean contrast)",
                          legend_loc="lower right", info_loc="upper left",
                          person_dots=dots)
            axes[i].set_xlim(jn["x_grid"][0], jn["x_grid"][-1])
        for j in range(n_panels, len(axes)):
            axes[j].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT_FIG_S5, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"  Saved Figure S5: {OUT_FIG_S5}")

    pd.DataFrame(text_rows).to_csv(OUT_TEXT_CSV, index=False)
    if verbose:
        print(f"  Saved text numbers: {OUT_TEXT_CSV}")

    generate_text_paragraphs(verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 09 COMPLETE")
        print("=" * 70)


def generate_text_paragraphs(verbose: bool = True) -> None:
    """Generate step09_text.md with JN-specific numbers (boundaries,
    % sample in credible region, simple slopes) for NAcc and ACC ROIs.
    """
    OUT_TEXT_MD = os.path.join(STEP_RESULTS_DIR, "step09_text.md")

    if not os.path.exists(OUT_TEXT_CSV):
        if verbose:
            print("  SKIP: step09_text_numbers.csv not found — run step09 first")
        return

    if verbose:
        print("  Generating step09_text.md ...")

    tn = pd.read_csv(OUT_TEXT_CSV)
    v = dict(zip(tn["metric"], tn["value"]))

    # Helper: format a simple slope line
    def _slope_line(roi_label, roi_key, level, level_label):
        beta = v.get(f"slope_sp_{roi_key}_{level}", "N/A")
        ci = v.get(f"slope_sp_{roi_key}_{level}_ci", "N/A")
        # Check if credible (CrI excludes zero)
        try:
            lo_str, hi_str = ci.strip("[]").split(", ")
            lo, hi = float(lo_str), float(hi_str)
            sig = "*" if (lo > 0 or hi < 0) else ""
        except (ValueError, AttributeError):
            sig = ""
        return f"- {level_label}: $\\hat{{\\lambda}}_{{sp}}={beta}$, 95% CrI {ci}{sig}"

    sections = []

    # --- Left NAcc ---
    nacc_boundary = v.get("jn_sp_Left_NAcc_boundary_raw", None)
    nacc_pct = v.get("jn_sp_Left_NAcc_pct_credible", None)

    nacc_lines = ["### Left NAcc", ""]
    if nacc_boundary is not None and nacc_boundary != "none":
        try:
            pct_int = f"{float(nacc_pct):.0f}"
        except (ValueError, TypeError):
            pct_int = nacc_pct
        nacc_lines.append(
            f"JN boundary at raw activation = {nacc_boundary}, "
            f"with {pct_int}% of the sample in the credible region."
        )
    else:
        nacc_lines.append("No JN boundary identified.")
    nacc_lines.append("")
    nacc_lines.append("Simple slopes:")
    nacc_lines.append("")
    nacc_lines.append(_slope_line("Left NAcc", "Left_NAcc", "low", "Low (Q1 - 1.5 x IQR)"))
    nacc_lines.append(_slope_line("Left NAcc", "Left_NAcc", "median", "Median"))
    nacc_lines.append(_slope_line("Left NAcc", "Left_NAcc", "high", "High (Q3 + 1.5 x IQR)"))
    sections.append("\n".join(nacc_lines))

    # --- Right dACC/MCC ---
    for acc_key, acc_label in [("Right_dACC_MCC", "Right dACC/MCC"),
                                ("Left_dACC_MCC", "Left dACC/MCC")]:
        boundary = v.get(f"jn_sp_{acc_key}_boundary_raw", None)
        pct = v.get(f"jn_sp_{acc_key}_pct_credible", None)
        acc_lines = [f"### {acc_label}", ""]
        if boundary is not None and boundary != "none":
            try:
                pct_int = f"{float(pct):.0f}"
            except (ValueError, TypeError):
                pct_int = pct
            acc_lines.append(
                f"JN boundary at raw activation = {boundary}, "
                f"with {pct_int}% of the sample in the credible region."
            )
        else:
            acc_lines.append("No JN boundary identified.")
        acc_lines.append("")
        acc_lines.append("Simple slopes:")
        acc_lines.append("")
        acc_lines.append(_slope_line(acc_label, acc_key, "low", "Low (Q1 - 1.5 x IQR)"))
        acc_lines.append(_slope_line(acc_label, acc_key, "median", "Median"))
        acc_lines.append(_slope_line(acc_label, acc_key, "high", "High (Q3 + 1.5 x IQR)"))
        sections.append("\n".join(acc_lines))

    # --- Figure captions ---

    # Figure 5 caption is purely descriptive (no computed numbers);
    # lives only in docs/manuscript_pain.md. Not code-generated.

    # Figure 6: ACC JN — load gamma values from Table 5 (no p-values per CrI-only convention)
    table5 = pd.read_csv(IN_TABLE5_CSV)
    t5 = dict(zip(table5["ROI"], table5.itertuples(index=False)))

    def _fmt_gamma(row):
        return f"{row.gamma_sp:+.3f}"

    def _fmt_ci(row):
        return f"[{row.gamma_sp_ci_lo:+.3f}, {row.gamma_sp_ci_hi:+.3f}]"

    r_acc = t5.get("Right_dACC_MCC")
    l_acc = t5.get("Left_dACC_MCC")
    if r_acc is not None and l_acc is not None:
        fig6_caption = (
            "**Figure 6.** Johnson-Neyman analyses of bilateral dACC/MCC "
            "BOLD moderation of sleep-to-pain coupling. Top panel: right "
            "dACC/MCC (MNI 6, 12, 38; 6 mm sphere; unmasked contrasts; "
            f"$\\hat{{\\gamma}}_{{sp}}={_fmt_gamma(r_acc)}$, "
            f"95% CrI {_fmt_ci(r_acc)}). Bottom panel: left dACC/MCC "
            f"(MNI -6, 12, 38; 6 mm sphere; unmasked contrasts; "
            f"$\\hat{{\\gamma}}_{{sp}}={_fmt_gamma(l_acc)}$, "
            f"95% CrI {_fmt_ci(l_acc)}). In each panel, the blue line "
            "shows the posterior mean conditional coupling slope as a "
            "function of ACC activation (z-scored), dashed lines show "
            "the 95% credible interval, and green shading indicates the "
            "region where the CrI excludes zero. The dotted vertical "
            "line marks the JN boundary. Vertical markers show simple "
            "slopes at $z=-2$, $z=0$, and $z=+2$ with 95% CrI error "
            "bars. N = 174."
        )
    else:
        fig6_caption = "**Figure 6.** (ACC moderation — data unavailable)"

    # Figure S5 caption only cites static sample sizes (N=173/174);
    # no computed results. Not code-generated.

    text = f"""\
## Step 09 — Johnson-Neyman analysis: SP moderation

JN boundaries and simple slopes for the credible and near-credible
sleep-to-pain moderation ROIs. An asterisk (*) after the CrI indicates
the credible interval excludes zero.

{chr(10).join(sections)}

### Figure 6 caption

{fig6_caption}
"""

    with open(OUT_TEXT_MD, "w") as f:
        f.write(text)

    if verbose:
        print(f"    Saved: {OUT_TEXT_MD}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 09 — SP moderation JN analysis."
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--refit", action="store_true",
                        help="Re-run computation from scratch instead of loading saved derivatives")
    args = parser.parse_args()
    run_step09(verbose=not args.quiet, refit=args.refit)


if __name__ == "__main__":
    main()
