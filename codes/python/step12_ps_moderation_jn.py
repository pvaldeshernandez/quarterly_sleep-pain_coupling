"""
Step 12 — Johnson-Neyman analysis for PS arousal moderation ROIs.
======================================================================

Input:  derivatives/step11/step11_ps_fmri_posterior_draws.npz
        derivatives/step11/step11_ps_vbm_posterior_draws.npz
        results/step11/step11_table_s1_fmri_arousal.csv
        results/step11/step11_table_s1_vbm_arousal.csv
Output:
  derivatives/step12/
    step12_jn_ps_fmri_results.csv
    step12_jn_ps_vbm_results.csv
  results/step12/
    step12_figure_s7_fmri_arousal_jn.png   — Figure S7
    step12_figure_s8_vbm_arousal_jn.png    — Figure S8
    step12_text_numbers.csv

Author: Pedro Valdes-Hernandez (with Claude Sonnet 4.6)
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

    # Shaded CrI band (green = credible, grey = non-credible), per-segment
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


def run_jn_for_modality(modality_name, draws_path, table_path, verbose=True):
    """Run JN for all ROIs in one modality. Returns JN grid rows, text rows,
    jn_results dict, slopes_all dict, and the table DataFrame."""
    from coupling_model import compute_jn_curve

    d = np.load(draws_path)
    table = pd.read_csv(table_path)

    jn_rows = []
    text_rows = []
    jn_results = {}
    slopes_all = {}

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
            cond = b1_draws + gamma_ps_draws * x_z
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
            print(f"\n  {row['Label']} ({modality_name}):")
            if boundary is not None:
                pct_below = float((X_vals < (boundary - raw_mean) / raw_sd).mean() * 100)
                print(f"    JN boundary: raw={boundary:.4f}, {pct_below:.1f}% below")
            else:
                print(f"    JN boundary: none")
            for lbl, ss in slopes.items():
                sig_str = "*" if ss["sig"] else ""
                print(f"    {lbl}: coupling={ss['beta']:.4f} "
                      f"[{ss['ci_lo']:.4f}, {ss['ci_hi']:.4f}]{sig_str}")

        # Text numbers
        if boundary is not None:
            text_rows.append({
                "metric": f"jn_ps_{modality_name}_{roi_name}_boundary_raw",
                "value": f"{boundary:.4f}"})
            x_grid_z = jn["x_grid_z"]
            obs_z = jn["X_vals_z"]
            sig_at_obs = np.interp(obs_z, x_grid_z, jn["sig"].astype(float))
            pct = float((sig_at_obs > 0.5).mean() * 100)
            text_rows.append({
                "metric": f"jn_ps_{modality_name}_{roi_name}_pct_credible",
                "value": f"{pct:.1f}"})
        else:
            text_rows.append({
                "metric": f"jn_ps_{modality_name}_{roi_name}_boundary",
                "value": "none"})
        for lbl, ss in slopes.items():
            text_rows.append({
                "metric": f"slope_ps_{modality_name}_{roi_name}_{lbl}",
                "value": f"{ss['beta']:.4f}"})
            text_rows.append({
                "metric": f"slope_ps_{modality_name}_{roi_name}_{lbl}_ci",
                "value": f"[{ss['ci_lo']:.4f}, {ss['ci_hi']:.4f}]"})

        # JN grid
        for i, x in enumerate(jn["x_grid"]):
            jn_rows.append({
                "ROI": roi_name, "modality": modality_name,
                "x": float(x),
                "mean": float(jn["post_mean"][i]),
                "ci_lo": float(jn["ci_lo"][i]),
                "ci_hi": float(jn["ci_hi"][i]),
            })

    return jn_rows, text_rows, jn_results, slopes_all, table


def _person_dots_ps(d, roi_name):
    """Compute person-level fitted PS coupling values for scatter overlay."""
    b1_draws = d[f"{roi_name}_b1_draws"]
    gamma_draws = d[f"{roi_name}_gamma_ps_draws"]
    X_vals = d[f"{roi_name}_X_vals"]
    raw_mean = float(d[f"{roi_name}_raw_mean"][0])
    raw_sd = float(d[f"{roi_name}_raw_sd"][0])
    b1_mean = float(np.mean(b1_draws))
    gamma_mean = float(np.mean(gamma_draws))
    person_x_raw = X_vals * raw_sd + raw_mean
    u_ps_key = f"{roi_name}_u_ps_mean"
    if u_ps_key in d:
        person_y = b1_mean + gamma_mean * X_vals + d[u_ps_key]
    else:
        person_y = b1_mean + gamma_mean * X_vals
    return {"x_raw": person_x_raw, "y": person_y}


def _make_merged_figure(jn_results, slopes_all, d, table, modality_label,
                        fig_path, fig_title, verbose=True):
    """Generate merged multi-panel JN figure for one modality."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rois = list(jn_results.keys())
    if not rois:
        return

    ncols = 2
    nrows = (len(rois) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(12.8 * ncols, 8.05 * nrows))
    if len(rois) == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    for i, roi_name in enumerate(rois):
        row = table[table["ROI"] == roi_name].iloc[0]
        jn = jn_results[roi_name]
        slopes = slopes_all[roi_name]

        level_labels = [
            f"Q1\u22121.5\u00b7IQR\n({slopes['low']['x_val']:.3f})",
            f"Median\n({slopes['median']['x_val']:.3f})",
            f"Q3+1.5\u00b7IQR\n({slopes['high']['x_val']:.3f})",
        ]
        level_x_vals = [slopes["low"]["x_val"], slopes["median"]["x_val"],
                        slopes["high"]["x_val"]]

        dots = _person_dots_ps(d, roi_name)

        # For panels other than the first, place left label inside the plot
        left_inside = (i > 0)

        draw_jn_panel(axes[i], jn, "Pain \u2192 Sleep", slopes,
                      level_labels, level_x_vals,
                      xlabel=f"{row['Label']} ({modality_label})",
                      legend_loc="lower right", info_loc="upper left",
                      person_dots=dots, left_label_inside=left_inside)
        axes[i].set_xlim(jn["x_grid"][0], jn["x_grid"][-1])

    for j in range(len(rois), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"  Saved: {fig_path}")


def run_step12(verbose=True):
    if verbose:
        print("=" * 70)
        print("STEP 12 — PS arousal moderation Johnson-Neyman analysis")
        print("=" * 70)

    os.makedirs(DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    all_text = []

    # ---- fMRI BOLD ----
    if verbose:
        print("\n  fMRI BOLD JN:")
    fmri_jn_rows, fmri_text, fmri_jn_results, fmri_slopes, fmri_table = \
        run_jn_for_modality("fMRI", IN_FMRI_DRAWS, IN_FMRI_TABLE, verbose)
    pd.DataFrame(fmri_jn_rows).to_csv(OUT_FMRI_JN, index=False)
    all_text.extend(fmri_text)

    fmri_d = np.load(IN_FMRI_DRAWS)
    _make_merged_figure(
        fmri_jn_results, fmri_slopes, fmri_d, fmri_table,
        modality_label="fMRI BOLD",
        fig_path=OUT_FIG_S7,
        fig_title="Pain-to-Sleep arousal moderation (fMRI BOLD)",
        verbose=verbose,
    )

    # ---- VBM GM volume ----
    if verbose:
        print("\n  VBM GM volume JN:")
    vbm_jn_rows, vbm_text, vbm_jn_results, vbm_slopes, vbm_table = \
        run_jn_for_modality("VBM", IN_VBM_DRAWS, IN_VBM_TABLE, verbose)
    pd.DataFrame(vbm_jn_rows).to_csv(OUT_VBM_JN, index=False)
    all_text.extend(vbm_text)

    vbm_d = np.load(IN_VBM_DRAWS)
    _make_merged_figure(
        vbm_jn_results, vbm_slopes, vbm_d, vbm_table,
        modality_label="GM volume",
        fig_path=OUT_FIG_S8,
        fig_title="Pain-to-Sleep arousal moderation (VBM GM volume)",
        verbose=verbose,
    )

    # Save text numbers
    pd.DataFrame(all_text).to_csv(OUT_TEXT_CSV, index=False)
    if verbose:
        print(f"  Saved text numbers: {OUT_TEXT_CSV}")
        print("\n" + "=" * 70)
        print("STEP 12 COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 12 — PS arousal moderation JN analysis."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_step12(verbose=not args.quiet)


if __name__ == "__main__":
    main()
