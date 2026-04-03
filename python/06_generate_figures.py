#!/usr/bin/env python3
"""
06_generate_figures.py — Generate All Manuscript Figures
========================================================

Loads pre-computed results from disk (CSV, NPZ) and generates every
figure used in the manuscript and supplementary materials. This script
never calls model-fitting functions; all data comes from saved outputs.

Main Figures
------------
  Figure 1:  Data availability grid (participants x quarters)
  Figure 2:  Pain-to-sleep coupling (boxstrip + forest)
  Figure 3:  Sleep-to-pain coupling (boxstrip + forest)
  Figure 4:  Contrast moderation JN (pain-to-sleep)
  Figure 5:  Left NAcc moderation JN (sleep-to-pain)
  Figure 6:  ACC moderation JN (sleep-to-pain)

Supplementary Figures
---------------------
  Figure S1:  Endorsement + grouped barplot (factor validation)
  Figure S2:  Convergent validity scatter plots
  Figure S3:  Contrast JN for sleep-to-pain (null)
  Figure S4:  Stimulation ROI maps (skipped if atlas images unavailable)
  Figure S5:  Krause ROI JN panels (2x2 merged)
  Figure S6:  Arousal ROI maps (skipped if atlas images unavailable)
  Figure S7:  fMRI arousal JN panels
  Figure S8:  VBM arousal JN panels

Usage
-----
  python python/06_generate_figures.py                # Real data
  python python/06_generate_figures.py --synthetic     # Synthetic data

Author: Pedro Valdes-Hernandez
"""

import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
LIB_DIR = os.path.join(SCRIPT_DIR, "lib")
sys.path.insert(0, LIB_DIR)

from plotting import setup_style, COLORS, save_figure
from coupling_model import compute_jn_curve


# ===================================================================
# Helpers
# ===================================================================

def _resolve_paths(synthetic):
    """Return (data_dir, results_dir, figures_dir) for real or synthetic mode."""
    data_dir = os.path.join(REPO_ROOT, "data")
    if synthetic:
        results_dir = os.path.join(REPO_ROOT, "results", "synthetic")
        figures_dir = os.path.join(REPO_ROOT, "figures", "synthetic")
    else:
        results_dir = os.path.join(REPO_ROOT, "results")
        figures_dir = os.path.join(REPO_ROOT, "figures")
    return data_dir, results_dir, figures_dir


def _file_exists(path, label):
    """Check if a file exists and print status."""
    if os.path.exists(path):
        return True
    print(f"  SKIP {label}: {os.path.basename(path)} not found")
    return False


def _parse_population_params(summary_path):
    """Parse population-level coupling params from coupling_summary.txt."""
    with open(summary_path) as f:
        text = f.read()

    params = {}
    for direction, label in [("sp", "Sleep->Pain (a2)"),
                              ("ps", "Pain->Sleep (b1)")]:
        block = re.search(
            re.escape(label) + r".*?Population: mean=([-\d.]+), "
            r"95% CI=\[([-\d.]+), ([-\d.]+)\].*?"
            r"P\(beta < 0\) = ([\d.]+)",
            text, re.DOTALL,
        )
        if block:
            params[direction] = {
                "beta": float(block.group(1)),
                "ci_lo": float(block.group(2)),
                "ci_hi": float(block.group(3)),
                "prob_neg": float(block.group(4)),
            }
        else:
            params[direction] = {
                "beta": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "prob_neg": np.nan,
            }
    return params


def _compute_simple_slopes(intercept_draws, slope_draws, x_positions):
    """Compute simple slopes from posterior draws at given x positions."""
    slopes = {}
    for label, x_val in x_positions:
        draws = intercept_draws + slope_draws * x_val
        beta = draws.mean()
        cl = np.percentile(draws, 2.5)
        ch = np.percentile(draws, 97.5)
        p_neg = (draws < 0).mean()
        is_sig = (ch < 0) or (cl > 0)
        slopes[label] = {
            "beta": beta, "ci_lo": cl, "ci_hi": ch,
            "prob_neg": p_neg, "sig": is_sig,
        }
    return slopes


# ===================================================================
# Figure 1 — Data Availability Grid
# ===================================================================

def _find_segments(quarters_with_data):
    """Find maximal runs of consecutive quarters."""
    if len(quarters_with_data) == 0:
        return []
    segments = []
    current = [quarters_with_data[0]]
    for q in quarters_with_data[1:]:
        if q == current[-1] + 1:
            current.append(q)
        else:
            segments.append(current)
            current = [q]
    segments.append(current)
    return segments


def generate_figure1(data_dir, results_dir, figures_dir, synthetic):
    """Figure 1: Data availability grid (participants x quarters)."""
    print("  Figure 1: Data availability grid")

    if synthetic:
        csv_path = os.path.join(data_dir, "synthetic", "processed_data.csv")
    else:
        csv_path = os.path.join(data_dir, "processed_data_contrast.csv")

    if not _file_exists(csv_path, "Figure 1"):
        return

    proc = pd.read_csv(csv_path)

    # For synthetic data we don't have the raw quarterly_data_long with
    # individual items, so we treat all processed points as "observed"
    if synthetic:
        raw_path = os.path.join(data_dir, "synthetic", "quarterly_data_long.csv")
        if os.path.exists(raw_path):
            raw = pd.read_csv(raw_path)
            raw["has_both_raw"] = (
                raw["pain_severity"].notna() & raw["sleep_quality"].notna()
            )
        else:
            raw = None
    else:
        raw_path = os.path.join(data_dir, "quarterly_data_long.csv")
        if os.path.exists(raw_path):
            raw = pd.read_csv(raw_path)
            raw = raw[raw["quarter"] >= 1].copy()
            pain_items = [
                "q2_knee_pain", "q3_knee_pain", "q4_knee_pain", "q5_knee_pain",
                "q7_body_pain", "q8_body_pain", "q9_body_pain", "q10_body_pain",
            ]
            if pain_items[0] in raw.columns:
                raw["n_pain_items"] = raw[pain_items].notna().sum(axis=1)
                raw["has_both_raw"] = (
                    (raw["n_pain_items"] >= 2) & raw["q13_sleep"].notna()
                )
            else:
                raw["has_both_raw"] = (
                    raw["pain_severity"].notna() & raw["sleep_quality"].notna()
                )
        else:
            raw = None

    # Determine subject ID column
    id_col = "ID" if "ID" in proc.columns else "subject_id"
    q_col = "quarter"

    proc["has_both_proc"] = (
        proc["pain_within"].notna() & proc["sleep_within"].notna()
    )
    if "segment_id" in proc.columns:
        proc["retained_proc"] = proc["segment_id"].notna()
    else:
        # For synthetic data without segment_id, treat all as retained
        proc["retained_proc"] = proc["has_both_proc"]

    # Build merged grid
    if raw is not None:
        raw_id_col = "ID" if "ID" in raw.columns else "subject_id"
        raw_flag = raw[[raw_id_col, q_col, "has_both_raw"]].copy()
        raw_flag = raw_flag.rename(columns={raw_id_col: id_col})
        merged = proc[[id_col, q_col, "has_both_proc", "retained_proc"]].merge(
            raw_flag, on=[id_col, q_col], how="outer",
        )
        merged["has_both_raw"] = merged["has_both_raw"].fillna(False).astype(bool)
    else:
        merged = proc[[id_col, q_col, "has_both_proc", "retained_proc"]].copy()
        merged["has_both_raw"] = merged["has_both_proc"]

    merged["has_both_proc"] = merged["has_both_proc"].fillna(False).astype(bool)
    merged["retained_proc"] = merged["retained_proc"].fillna(False).astype(bool)
    merged["observed"] = merged["has_both_raw"]
    merged["interpolated"] = merged["has_both_proc"] & ~merged["has_both_raw"]
    merged["has_data"] = merged["has_both_raw"] | merged["has_both_proc"]
    merged["retained"] = merged["retained_proc"] & merged["has_data"]
    merged = merged.sort_values([id_col, q_col])

    # Compute segment info per person
    person_segments = {}
    stats = []
    for pid, grp in merged.groupby(id_col):
        retained_quarters = sorted(grp.loc[grp["retained"], q_col].values)
        retained_segs = _find_segments(retained_quarters)
        person_segments[pid] = retained_segs
        stats.append({
            "ID": pid,
            "n_retained": len(retained_quarters),
        })

    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values("n_retained").reset_index(drop=True)
    id_order = stats_df["ID"].tolist()

    n_subjects = len(id_order)
    n_excluded = (stats_df["n_retained"] == 0).sum()
    n_ret_interp = merged[merged["retained"] & merged["interpolated"]].shape[0]

    quarters = sorted(merged[q_col].dropna().unique().astype(int))
    if not quarters:
        quarters = list(range(1, 12))

    # Plot
    fig_height = max(6, n_subjects * 0.055 + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    for yi, pid in enumerate(id_order):
        pdata = merged[merged[id_col] == pid]
        retained_segs = person_segments[pid]

        for seg in retained_segs:
            ax.plot([seg[0], seg[-1]], [yi, yi],
                    color="#90CAF9", linewidth=2.5, alpha=0.5, zorder=1,
                    solid_capstyle="round")

        for _, row in pdata.iterrows():
            q = row[q_col]
            if not row["has_data"]:
                continue
            if row["retained"]:
                color = "#1565C0" if row["observed"] else "#D32F2F"
                alpha = 0.85
                zorder = 3
            else:
                color = "#BDBDBD"
                alpha = 0.5
                zorder = 2
            ax.plot(q, yi, "o", color=color, markersize=5, alpha=alpha,
                    zorder=zorder)

    if n_excluded > 0:
        ax.axhline(y=n_excluded - 0.5, color="black", linewidth=0.8,
                   linestyle="--", alpha=0.5)
        ax.text(0.7, n_excluded - 0.8, f"{n_excluded} excluded",
                fontsize=14, va="top", ha="left", color="#666666")

    ax.set_xlim(quarters[0] - 0.5, quarters[-1] + 0.5)
    ax.set_ylim(-1, n_subjects)
    ax.set_xticks(quarters)
    ax.set_xticklabels([f"Q{q}" for q in quarters], fontsize=16)
    ax.set_xlabel("Quarter", fontsize=20)
    ax.set_ylabel(f"Participant (N = {n_subjects})", fontsize=18)
    ax.set_yticks([])
    ax.set_title("", fontsize=1)

    legend_handles = [
        Line2D([0], [0], marker="o", color="#1565C0", markersize=10,
               linestyle="", label="Observed, retained"),
        Line2D([0], [0], marker="o", color="#BDBDBD", markersize=10,
               linestyle="", alpha=0.5, label="Discarded (segment < 3)"),
        Line2D([0], [0], color="#90CAF9", linewidth=4, alpha=0.5,
               label="Retained segment"),
    ]
    if n_ret_interp > 0:
        legend_handles.insert(1,
            Line2D([0], [0], marker="o", color="#D32F2F", markersize=10,
                   linestyle="", label="Interpolated, retained"),
        )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=15,
              framealpha=0.9, edgecolor="#CCCCCC")

    plt.tight_layout()
    save_figure(fig, "figure1", figures_dir)


# ===================================================================
# Figures 2 & 3 — Coupling Boxstrip + Forest
# ===================================================================

def _make_coupling_figure(df, pop_params, direction, title_label, fname,
                          figures_dir):
    """Create a two-panel (A: boxstrip, B: forest) coupling figure."""
    if direction == "sp":
        col_mean = "beta_sp_mean"
        col_ci_lo = "beta_sp_ci_lo"
        col_ci_hi = "beta_sp_ci_hi"
    else:
        col_mean = "beta_ps_mean"
        col_ci_lo = "beta_ps_ci_lo"
        col_ci_hi = "beta_ps_ci_hi"

    pp = pop_params[direction]
    vals = df[col_mean].dropna().values
    n_total = len(vals)

    s = df.dropna(subset=[col_mean]).sort_values(col_mean).reset_index(drop=True)
    n = len(s)

    if n > 0:
        forest_lo = s[col_ci_lo].min()
        forest_hi = s[col_ci_hi].max()
        box_lo = vals.min()
        box_hi = vals.max()
        x_lo = min(forest_lo, box_lo) - 0.05
        x_hi = max(forest_hi, box_hi) + 0.05
    else:
        x_lo, x_hi = -1, 1

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.12)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Panel A: Boxstrip
    if len(vals) > 0:
        ax_a.boxplot(vals, vert=False, widths=0.5,
                     patch_artist=True,
                     boxprops=dict(facecolor=COLORS["box_face"],
                                   edgecolor=COLORS["box_edge"],
                                   linewidth=1.5),
                     medianprops=dict(color=COLORS["median_line"],
                                      linewidth=2.5),
                     whiskerprops=dict(color=COLORS["box_edge"],
                                       linewidth=1.2),
                     capprops=dict(color=COLORS["box_edge"], linewidth=1.2),
                     flierprops=dict(marker="", markersize=0))

    rng = np.random.default_rng(42)
    if len(vals) > 0:
        jitter = rng.uniform(0.7, 1.3, len(vals))
        colors = [COLORS["negative"] if x < 0 else COLORS["positive"]
                  for x in vals]
        edge_colors = [COLORS["negative_dark"] if x < 0
                       else COLORS["positive_dark"] for x in vals]
        sc = ax_a.scatter(vals, jitter, c=colors, alpha=0.7, s=80, zorder=3,
                          marker="o", edgecolors=edge_colors, linewidths=0.8)
        sc.set_path_effects([patheffects.withSimplePatchShadow(
            offset=(0.5, -0.5), shadow_rgbFace="#555555", alpha=0.25)])

    ax_a.axvline(0, color="black", linewidth=1, linestyle="-",
                 zorder=2, alpha=0.5)

    if len(vals) > 0:
        mean_val = np.mean(vals)
        ax_a.plot(mean_val, 1.0, marker="D", color=COLORS["mean_diamond"],
                  markersize=12, zorder=5, markeredgecolor="white",
                  markeredgewidth=1.2)

    ax_a.set_title(f"A.  {title_label}", fontsize=20, fontweight="bold",
                   loc="left", pad=10)
    ax_a.set_xlabel("\u03bb (posterior mean)", fontsize=18)
    ax_a.set_yticks([])
    ax_a.tick_params(axis="x", labelsize=16)

    if np.isfinite(pp["beta"]):
        pop_text = (f"Population \u03bb = {pp['beta']:.3f}\n"
                    f"[{pp['ci_lo']:.3f}, {pp['ci_hi']:.3f}]\n"
                    f"P(\u03bb < 0) = {pp['prob_neg']:.3f}")
        ax_a.text(0.02, 0.98, pop_text, transform=ax_a.transAxes,
                  fontsize=14, va="top", ha="left", color="black",
                  fontweight="bold")

    if len(vals) > 0:
        med = np.median(vals)
        q25, q75 = np.percentile(vals, [25, 75])
        stats_text = (f"N = {n_total}\n"
                      f"mean = {mean_val:.3f}\n"
                      f"median = {med:.3f}\n"
                      f"IQR = [{q25:.3f}, {q75:.3f}]")
        ax_a.text(0.98, 0.98, stats_text, transform=ax_a.transAxes,
                  fontsize=13, va="top", ha="right", color="#333333",
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            edgecolor="#CCCCCC", alpha=0.9))

    legend_handles = [
        Line2D([0], [0], color=COLORS["median_line"], linewidth=2.5,
               label="Median"),
        Line2D([0], [0], marker="D", color=COLORS["mean_diamond"],
               markersize=10, markeredgecolor="white", markeredgewidth=0.8,
               linestyle="", label="Mean"),
        Line2D([0], [0], marker="o", color=COLORS["negative"], markersize=8,
               markeredgecolor=COLORS["negative_dark"], markeredgewidth=0.8,
               linestyle="", label="Individual (\u03bb < 0)"),
        Line2D([0], [0], marker="o", color=COLORS["positive"], markersize=8,
               markeredgecolor=COLORS["positive_dark"], markeredgewidth=0.8,
               linestyle="", label="Individual (\u03bb > 0)"),
    ]
    ax_a.legend(handles=legend_handles, loc="lower right", fontsize=13,
                framealpha=0.9, edgecolor="#CCCCCC")

    # Panel B: Forest
    for k, row in s.iterrows():
        color = COLORS["forest_point"] if row[col_mean] < 0 \
            else COLORS["positive_dark"]
        if np.isfinite(row[col_ci_lo]) and np.isfinite(row[col_ci_hi]):
            excludes_zero = (row[col_ci_lo] > 0) or (row[col_ci_hi] < 0)
            lw = 1.8 if excludes_zero else 0.7
            al = 0.7 if excludes_zero else 0.4
            ax_b.plot([row[col_ci_lo], row[col_ci_hi]], [k, k],
                      color=color, linewidth=lw, alpha=al, zorder=2)
        ax_b.plot(row[col_mean], k, marker="|", color=color,
                  markersize=3.5, alpha=0.8, zorder=3, markeredgewidth=0.9)

    ax_b.axvline(0, color="black", linewidth=0.8, linestyle="-",
                 alpha=0.5, zorder=1)

    if len(vals) > 0:
        m = np.mean(vals)
        ax_b.axvline(m, color=COLORS["mean_diamond"], linewidth=2,
                     linestyle="--", alpha=0.7, zorder=4)

    ax_b.set_title(f"B.  {title_label}", fontsize=20, fontweight="bold",
                   loc="left", pad=10)
    ax_b.set_xlabel("\u03bb (posterior mean \u00b1 95% CrI)", fontsize=18)
    ax_b.set_yticks([])
    ax_b.set_ylim(-1, n + 1)
    ax_b.tick_params(axis="x", labelsize=16)

    # Align panel heights
    fig.canvas.draw()
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    top = max(pos_a.y1, pos_b.y1)
    bottom = min(pos_a.y0, pos_b.y0)
    ax_a.set_position([pos_a.x0, bottom, pos_a.width, top - bottom])
    ax_b.set_position([pos_b.x0, bottom, pos_b.width, top - bottom])

    save_figure(fig, fname.replace(".png", ""), figures_dir)


def generate_figures2_3(data_dir, results_dir, figures_dir, synthetic):
    """Figures 2 and 3: Person-specific coupling (boxstrip + forest)."""
    print("  Figure 2: Pain-to-sleep coupling")
    print("  Figure 3: Sleep-to-pain coupling")

    csv_path = os.path.join(results_dir, "coupling_results.csv")
    summary_path = os.path.join(results_dir, "coupling_summary.txt")

    if not _file_exists(csv_path, "Figures 2-3"):
        return
    if not _file_exists(summary_path, "Figures 2-3"):
        return

    df = pd.read_csv(csv_path)
    pop_params = _parse_population_params(summary_path)

    _make_coupling_figure(df, pop_params, "ps",
                          "Pain \u2192 Sleep coupling",
                          "figure2.png", figures_dir)

    _make_coupling_figure(df, pop_params, "sp",
                          "Sleep \u2192 Pain coupling",
                          "figure3.png", figures_dir)


# ===================================================================
# JN Panel Drawing (shared by Figures 4, 5, 6, S3, S5, S7, S8)
# ===================================================================

def _draw_jn_panel(ax, jn, direction_label, slopes_dict,
                   level_labels, level_x_vals,
                   xlabel=None, body_knee_labels=False,
                   legend_loc="best", info_loc="upper left",
                   person_dots=None, panel_label=""):
    """Draw a Johnson-Neyman panel with simple slope annotations."""
    x_grid = jn["x_grid"]
    post_mean = jn["post_mean"]
    ci_lo = jn["ci_lo"]
    ci_hi = jn["ci_hi"]
    sig = jn["sig"]
    obs_vals = jn["obs_vals"]

    # Person dots (behind everything)
    if person_dots is not None:
        x_key = "x_raw" if "x_raw" in person_dots else "x"
        ax.scatter(person_dots[x_key], person_dots["y"],
                   s=40, color=COLORS["negative"], alpha=0.7,
                   edgecolors=COLORS["negative_dark"], linewidths=0.8,
                   zorder=2, label="Fitted coupling",
                   path_effects=[patheffects.withSimplePatchShadow(
                       offset=(0.5, -0.5), shadow_rgbFace="#1565C0",
                       alpha=0.3)])

    # Shaded CrI band
    for i in range(len(x_grid) - 1):
        color = COLORS["jn_ci_credible"] if sig[i] else COLORS["jn_ci_null"]
        alpha = 0.35 if sig[i] else 0.25
        ax.fill_between(x_grid[i:i + 2], ci_lo[i:i + 2], ci_hi[i:i + 2],
                        color=color, alpha=alpha, linewidth=0)

    ax.plot(x_grid, post_mean, color=COLORS["jn_line"], linewidth=2.5)
    ax.plot(x_grid, ci_lo, color=COLORS["jn_line"], linewidth=1,
            linestyle="--", alpha=0.7)
    ax.plot(x_grid, ci_hi, color=COLORS["jn_line"], linewidth=1,
            linestyle="--", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)

    # Rug plot
    ax.scatter(obs_vals, np.full_like(obs_vals, ax.get_ylim()[0]),
               marker="|", color=COLORS["rug"], alpha=0.3, s=40, zorder=1)

    # Simple slope error bars
    if slopes_dict is not None:
        levels = list(slopes_dict.keys())
        y_lo, y_hi = ax.get_ylim()
        y_range = y_hi - y_lo
        x_lo_plot, x_hi_plot = x_grid[0], x_grid[-1]
        x_range = x_hi_plot - x_lo_plot

        for k, level in enumerate(levels):
            d = slopes_dict[level]
            x_pos = level_x_vals[k]
            beta = d["beta"]
            cl = d["ci_lo"]
            ch = d["ci_hi"]
            is_sig = d.get("sig", False)

            ax.errorbar(x_pos, beta,
                        yerr=[[beta - cl], [ch - beta]],
                        fmt="o", color="black", markersize=10,
                        capsize=7, capthick=2.5, elinewidth=2.5,
                        markeredgecolor="white", markeredgewidth=1.5,
                        zorder=6)

            sig_mark = "*" if is_sig else ""
            label_text = (f"{level_labels[k]}\n"
                          f"{beta:.3f} [{cl:.3f}, {ch:.3f}]{sig_mark}")

            # Position labels: left, center, right tiers
            if k == 0:
                txt_x = x_pos + x_range * 0.04
                txt_y = beta + y_range * 0.25
                ha = "left"
            elif k == 1:
                txt_x = x_pos - x_range * 0.04
                txt_y = beta + y_range * 0.22
                ha = "right"
            else:
                txt_x = x_pos - x_range * 0.04
                txt_y = beta + y_range * 0.12
                ha = "right"

            ax.annotate(label_text,
                        xy=(x_pos, ch if txt_y > beta else cl),
                        xytext=(txt_x, txt_y),
                        fontsize=10, fontfamily="monospace", color="black",
                        fontweight="bold", ha=ha, va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="black", alpha=0.90),
                        arrowprops=dict(arrowstyle="->", color="black",
                                        lw=1.5,
                                        connectionstyle="arc3,rad=0.2"),
                        zorder=7)

    ax.set_ylabel(f"Coupling \u03bb ({direction_label})", fontsize=18)
    ax.tick_params(axis="y", labelsize=16)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=18)
    ax.tick_params(axis="x", labelsize=16)

    if body_knee_labels:
        ax.text(0.01, -0.10, "\u2190 Body-dominant",
                transform=ax.transAxes, fontsize=16, ha="left", va="top",
                color="black", fontstyle="italic")
        ax.text(0.99, -0.10, "Knee-dominant \u2192",
                transform=ax.transAxes, fontsize=16, ha="right", va="top",
                color="black", fontstyle="italic")

    # Title with equation
    prefix = f"{panel_label}.  " if panel_label else ""
    raw_mean = jn.get("raw_mean")
    raw_sd = jn.get("raw_sd")
    if raw_mean is not None and raw_sd is not None and raw_sd > 0:
        gamma_raw = jn["slope_mean"] / raw_sd
        intercept_raw = (jn["intercept_mean"]
                         - jn["slope_mean"] * raw_mean / raw_sd)
        ax.set_title(
            f"{prefix}{direction_label} coupling: "
            f"\u03bb(X) = {intercept_raw:.3f} + ({gamma_raw:.3f})\u00b7X",
            fontsize=20, fontweight="bold", loc="left", pad=10)
    else:
        ax.set_title(
            f"{prefix}{direction_label} coupling: "
            f"\u03bb(K) = {jn['intercept_mean']:.3f} "
            f"+ ({jn['slope_mean']:.3f})\u00b7K",
            fontsize=20, fontweight="bold", loc="left", pad=10)

    # JN boundaries
    for x0 in jn["jn_boundaries"]:
        ax.axvline(x0, color="black", linewidth=2, linestyle=":",
                   alpha=0.8, zorder=4)
        y_top = ax.get_ylim()[1]
        y_bot = ax.get_ylim()[0]
        ax.text(x0, y_top - (y_top - y_bot) * 0.04, f"{x0:.3f}",
                fontsize=14, color="black", fontweight="bold",
                va="top", ha="center")

    # JN info text
    jn_text_parts = []
    if jn["jn_boundaries"]:
        if "jn_boundaries_z" in jn and "X_vals_z" in jn:
            x0_z = jn["jn_boundaries_z"][0]
            if np.any(jn["sig_negative"]):
                pct = (jn["X_vals_z"] < x0_z).mean() * 100
            elif np.any(jn["sig_positive"]):
                pct = (jn["X_vals_z"] > x0_z).mean() * 100
            else:
                pct = 0
            jn_text_parts.append(f"{pct:.0f}% of sample in credible region")
    else:
        if np.all(sig):
            jn_text_parts.append("Coupling credible across entire range")
        elif not np.any(sig):
            jn_text_parts.append(
                "No JN boundary: coupling non-credible across range")

    if jn_text_parts:
        if info_loc == "upper left":
            ix, iy, iha, iva = 0.02, 0.97, "left", "top"
        elif info_loc == "upper right":
            ix, iy, iha, iva = 0.98, 0.97, "right", "top"
        elif info_loc == "lower right":
            ix, iy, iha, iva = 0.98, 0.10, "right", "bottom"
        else:
            ix, iy, iha, iva = 0.02, 0.97, "left", "top"
        ax.text(ix, iy, "\n".join(jn_text_parts), transform=ax.transAxes,
                fontsize=14, va=iva, ha=iha, fontfamily="monospace")

    # Legend
    legend_elements = [
        Line2D([0], [0], color=COLORS["jn_line"], linewidth=2.5,
               label="Posterior mean"),
        Line2D([0], [0], color=COLORS["jn_line"], linewidth=1, linestyle="--",
               label="95% CrI"),
        Patch(facecolor=COLORS["jn_ci_credible"], alpha=0.35,
              label="Credible (CrI excludes 0)"),
        Patch(facecolor=COLORS["jn_ci_null"], alpha=0.25,
              label="Non-credible"),
        Line2D([0], [0], color="black", linewidth=2, linestyle=":",
               label="JN boundary"),
        Line2D([0], [0], marker="o", color="black", markersize=8,
               markeredgecolor="white", markeredgewidth=1, linestyle="",
               label="Simple slope \u00b1 95% CrI"),
    ]
    if person_dots is not None:
        legend_elements.insert(0,
            Line2D([0], [0], marker="o", color=COLORS["negative"],
                   markersize=8, markeredgecolor=COLORS["negative_dark"],
                   markeredgewidth=0.8, linestyle="",
                   label="Fitted coupling"))
    ax.legend(handles=legend_elements, loc=legend_loc, fontsize=13,
              framealpha=0.9, edgecolor="#CCCCCC",
              bbox_to_anchor=(1.0, 0.08) if "right" in legend_loc
              else (0.0, 0.08),
              borderaxespad=0.3)


# ===================================================================
# Figure 4 — Contrast Moderation JN
# ===================================================================

def generate_figure4(data_dir, results_dir, figures_dir, synthetic):
    """Figure 4: Contrast moderation of pain-to-sleep coupling (JN)."""
    print("  Figure 4: Contrast moderation JN (pain-to-sleep)")

    npz_path = os.path.join(results_dir, "contrast_posterior_draws.npz")
    if not _file_exists(npz_path, "Figure 4"):
        return

    d = np.load(npz_path)
    b1_draws = d["b1_draws"]
    b4_draws = d["b4_draws"]
    u_ps_mean = d["u_ps_mean"]
    obs_pid_idx = d["obs_pid_idx"]
    obs_contrast = d["obs_contrast"]
    contrast_vals = d["contrast_vals"]
    c_sd = np.std(contrast_vals)

    # JN curve
    jn_ps = compute_jn_curve(b1_draws, b4_draws, contrast_vals,
                             clip_pct=(0, 100))

    # Simple slopes at -2 SD, 0, +2 SD
    x_positions = [("-2 SD", -2 * c_sd), ("0", 0.0), ("+2 SD", 2 * c_sd)]
    slopes_ps = _compute_simple_slopes(b1_draws, b4_draws, x_positions)

    # Observation-level adjusted coupling dots
    b1_mean = b1_draws.mean()
    b4_mean = b4_draws.mean()
    obs_ps = b1_mean + u_ps_mean[obs_pid_idx] + b4_mean * obs_contrast
    dots_ps = {"x": obs_contrast, "y": obs_ps}

    level_labels = [
        "Body-dominant\n(\u22122 SD)", "Balanced\n(0)",
        "Knee-dominant\n(+2 SD)",
    ]
    level_x_vals = [-2 * c_sd, 0.0, 2 * c_sd]

    fig, ax = plt.subplots(figsize=(12.8, 8.05))
    _draw_jn_panel(ax, jn_ps, "Pain \u2192 Sleep",
                   slopes_ps, level_labels, level_x_vals,
                   xlabel="K\u1d42", body_knee_labels=True,
                   legend_loc="lower left", info_loc="lower right",
                   person_dots=dots_ps)

    save_figure(fig, "figure4", figures_dir)


# ===================================================================
# Figure 5 — NAcc Moderation JN
# ===================================================================

def generate_figure5(data_dir, results_dir, figures_dir, synthetic):
    """Figure 5: Left NAcc moderation of sleep-to-pain coupling (JN)."""
    print("  Figure 5: Left NAcc moderation JN (sleep-to-pain)")

    npz_path = os.path.join(results_dir, "nacc_posterior_draws.npz")
    if not _file_exists(npz_path, "Figure 5"):
        return

    d = np.load(npz_path)
    a2_draws = d["a2_draws"]
    gamma_sp_draws = d["gamma_sp_draws"]
    u_sp_mean = d["u_sp_mean"]
    person_x_z = d["person_x_z"]
    X_vals = d["X_vals"]
    nacc_mean = float(d["nacc_mean"])
    nacc_sd = float(d["nacc_sd"])
    raw_nacc_vals = d["raw_nacc_vals"]

    # Compute median and 1.5*IQR fences
    q1_raw = np.percentile(raw_nacc_vals, 25)
    q3_raw = np.percentile(raw_nacc_vals, 75)
    iqr_raw = q3_raw - q1_raw
    median_raw = np.median(raw_nacc_vals)
    low_fence_raw = q1_raw - 1.5 * iqr_raw
    high_fence_raw = q3_raw + 1.5 * iqr_raw

    low_fence_z = (low_fence_raw - nacc_mean) / nacc_sd
    median_z = (median_raw - nacc_mean) / nacc_sd
    high_fence_z = (high_fence_raw - nacc_mean) / nacc_sd

    slopes_sp = {}
    for label, x_z in [("low", low_fence_z), ("median", median_z),
                        ("high", high_fence_z)]:
        sp_draws = a2_draws + gamma_sp_draws * x_z
        slopes_sp[label] = {
            "beta": sp_draws.mean(),
            "ci_lo": np.percentile(sp_draws, 2.5),
            "ci_hi": np.percentile(sp_draws, 97.5),
            "sig": (np.percentile(sp_draws, 97.5) < 0) or
                   (np.percentile(sp_draws, 2.5) > 0),
        }

    a2_mean = a2_draws.mean()
    gamma_sp_mean = gamma_sp_draws.mean()
    person_x_raw = person_x_z * nacc_sd + nacc_mean
    person_sp = a2_mean + gamma_sp_mean * person_x_z + u_sp_mean
    dots_sp = {"x_raw": person_x_raw, "y": person_sp}

    jn_sp = compute_jn_curve(a2_draws, gamma_sp_draws, X_vals,
                             raw_mean=nacc_mean, raw_sd=nacc_sd)

    level_x_vals = [low_fence_raw, median_raw, high_fence_raw]
    level_labels = [
        f"Q1\u22121.5\u00b7IQR\n({low_fence_raw:.3f})",
        f"Median\n({median_raw:.3f})",
        f"Q3+1.5\u00b7IQR\n({high_fence_raw:.3f})",
    ]

    fig, ax = plt.subplots(figsize=(12.8, 8.05))
    _draw_jn_panel(ax, jn_sp, "Sleep \u2192 Pain",
                   slopes_sp, level_labels, level_x_vals,
                   xlabel="Left NAcc BOLD activation (mean contrast)",
                   legend_loc="lower right", info_loc="upper left",
                   person_dots=dots_sp)
    ax.set_xlim(jn_sp["x_grid"][0], jn_sp["x_grid"][-1])

    save_figure(fig, "figure5", figures_dir)


# ===================================================================
# Figure 6 — ACC Moderation JN
# ===================================================================

def generate_figure6(data_dir, results_dir, figures_dir, synthetic):
    """Figure 6: ACC moderation of sleep-to-pain coupling (JN)."""
    print("  Figure 6: ACC moderation JN (sleep-to-pain)")

    npz_path = os.path.join(results_dir, "acc_posterior_draws.npz")
    if not _file_exists(npz_path, "Figure 6"):
        return

    d = np.load(npz_path)
    a2_draws = d["a2_draws"]
    gamma_sp_draws = d["gamma_sp_draws"]
    u_sp_mean = d["u_sp_mean"]
    person_x_z = d["person_x_z"]
    X_vals = d["X_vals"]
    acc_mean = float(d["acc_mean"])
    acc_sd = float(d["acc_sd"])
    raw_acc_vals = d["raw_acc_vals"]

    q1_raw = np.percentile(raw_acc_vals, 25)
    q3_raw = np.percentile(raw_acc_vals, 75)
    iqr_raw = q3_raw - q1_raw
    median_raw = np.median(raw_acc_vals)
    low_fence_raw = q1_raw - 1.5 * iqr_raw
    high_fence_raw = q3_raw + 1.5 * iqr_raw

    low_fence_z = (low_fence_raw - acc_mean) / acc_sd
    median_z = (median_raw - acc_mean) / acc_sd
    high_fence_z = (high_fence_raw - acc_mean) / acc_sd

    slopes_sp = {}
    for label, x_z in [("low", low_fence_z), ("median", median_z),
                        ("high", high_fence_z)]:
        sp_draws = a2_draws + gamma_sp_draws * x_z
        slopes_sp[label] = {
            "beta": sp_draws.mean(),
            "ci_lo": np.percentile(sp_draws, 2.5),
            "ci_hi": np.percentile(sp_draws, 97.5),
            "sig": (np.percentile(sp_draws, 97.5) < 0) or
                   (np.percentile(sp_draws, 2.5) > 0),
        }

    a2_mean = a2_draws.mean()
    gamma_sp_mean = gamma_sp_draws.mean()
    person_x_raw = person_x_z * acc_sd + acc_mean
    person_sp = a2_mean + gamma_sp_mean * person_x_z + u_sp_mean
    dots_sp = {"x_raw": person_x_raw, "y": person_sp}

    jn_sp = compute_jn_curve(a2_draws, gamma_sp_draws, X_vals,
                             raw_mean=acc_mean, raw_sd=acc_sd)

    level_x_vals = [low_fence_raw, median_raw, high_fence_raw]
    level_labels = [
        f"Q1\u22121.5\u00b7IQR\n({low_fence_raw:.3f})",
        f"Median\n({median_raw:.3f})",
        f"Q3+1.5\u00b7IQR\n({high_fence_raw:.3f})",
    ]

    fig, ax = plt.subplots(figsize=(12.8, 8.05))
    _draw_jn_panel(ax, jn_sp, "Sleep \u2192 Pain",
                   slopes_sp, level_labels, level_x_vals,
                   xlabel="ACC BOLD activation (mean contrast)",
                   legend_loc="lower right", info_loc="upper left",
                   person_dots=dots_sp)
    ax.set_xlim(jn_sp["x_grid"][0], jn_sp["x_grid"][-1])

    save_figure(fig, "figure6", figures_dir)


# ===================================================================
# Figure S1 — Endorsement + Grouped Barplot (Factor Validation)
# ===================================================================

def generate_figure_s1(data_dir, results_dir, figures_dir, synthetic):
    """Figure S1: Endorsement and grouped barplot for factor validation."""
    print("  Figure S1: Endorsement + grouped barplot")

    endorsement_path = os.path.join(results_dir, "endorsement_data.csv")
    if not _file_exists(endorsement_path, "Figure S1"):
        return

    edf = pd.read_csv(endorsement_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Point-biserial correlations (or loadings)
    ax = axes[0]
    if "loading" in edf.columns:
        items = edf["item"].values
        loadings = edf["loading"].values
        colors = [COLORS["negative"] if v < 0 else COLORS["positive"]
                  for v in loadings]
        ax.barh(range(len(items)), loadings, color=colors, edgecolor="white")
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels(items, fontsize=10)
        ax.set_xlabel("Factor loading", fontsize=14)
        ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("A.  Factor loadings", fontsize=16, fontweight="bold",
                 loc="left")

    # Panel B: Grouped barplot (if columns present)
    ax = axes[1]
    if "group" in edf.columns and "mean" in edf.columns:
        groups = edf["group"].unique()
        x = np.arange(len(groups))
        ax.bar(x, edf.groupby("group")["mean"].first().values,
               color=COLORS["forest_fill"], edgecolor=COLORS["forest_point"])
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=10)
    ax.set_title("B.  Grouped endorsement", fontsize=16, fontweight="bold",
                 loc="left")

    plt.tight_layout()
    save_figure(fig, "figure_s1", figures_dir)


# ===================================================================
# Figure S2 — Convergent Validity Scatter Plots
# ===================================================================

def generate_figure_s2(data_dir, results_dir, figures_dir, synthetic):
    """Figure S2: Convergent validity scatter plots."""
    print("  Figure S2: Convergent validity scatter plots")

    if synthetic:
        proc_path = os.path.join(data_dir, "synthetic", "processed_data.csv")
        wide_path = os.path.join(data_dir, "synthetic",
                                 "participants_wideformat.csv")
    else:
        proc_path = os.path.join(data_dir, "processed_data_contrast.csv")
        wide_path = os.path.join(data_dir, "participants_wideformat.xlsx")

    if not _file_exists(proc_path, "Figure S2"):
        return
    if not _file_exists(wide_path, "Figure S2"):
        return

    from scipy import stats as sp_stats

    df_long = pd.read_csv(proc_path)
    id_col = "ID" if "ID" in df_long.columns else "subject_id"

    if wide_path.endswith(".xlsx"):
        df_wide = pd.read_excel(wide_path)
    else:
        df_wide = pd.read_csv(wide_path)

    # Compute person-mean contrast
    if "contrast_person_mean" in df_long.columns:
        ki = df_long.groupby(id_col)["contrast_person_mean"].first() \
                     .reset_index()
        ki.columns = [id_col, "K_i"]
    elif "contrast_factor" in df_long.columns:
        ki = df_long.groupby(id_col)["contrast_factor"].mean().reset_index()
        ki.columns = [id_col, "K_i"]
    else:
        print("    No contrast column found, skipping Figure S2")
        return

    wide_id_col = "ID" if "ID" in df_wide.columns else "subject_id"
    df_wide = df_wide.rename(columns={wide_id_col: id_col})
    df = ki.merge(df_wide, on=id_col, how="inner")

    # Try standard clinical columns; fall back to available numeric columns
    pearson_panels = []
    candidate_cols = [
        ("phq_knee_pain_days__s1", "PHQ knee pain days per week"),
        ("phq_percent_pain__s1", "PHQ % waking day in knee pain"),
        ("womac_pain__s1", "WOMAC Pain"),
        ("total_womac__s1", "WOMAC Total"),
        ("womac_phys_function__s1", "WOMAC Physical Function"),
        ("womac_stiffness__s1", "WOMAC Stiffness"),
        ("qst_knee_pain_rating__s1", "Knee pain rating"),
    ]
    for col, label in candidate_cols:
        if col in df.columns:
            pearson_panels.append((col, label))

    # For synthetic data, use available numeric columns
    if not pearson_panels:
        numeric_cols = [c for c in df.columns
                        if c not in [id_col, "K_i", "age", "sex", "race"]
                        and pd.api.types.is_numeric_dtype(df[c])]
        for col in numeric_cols[:7]:
            pearson_panels.append((col, col))

    if not pearson_panels:
        print("    No clinical columns found for scatter plots, skipping")
        return

    n_panels = len(pearson_panels)
    n_cols = 4
    n_rows = max(1, (n_panels + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    scatter_kw = dict(alpha=0.35, s=18, color="steelblue", edgecolor="none")
    line_kw = dict(color="firebrick", linewidth=2)
    bbox_kw = dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.85,
                   edgecolor="gray")

    for i, (col, label) in enumerate(pearson_panels):
        ax = axes[i]
        tmp = df[["K_i", col]].dropna()
        if len(tmp) < 5:
            ax.set_visible(False)
            continue
        x, y = tmp["K_i"].values, tmp[col].values
        n = len(x)
        r, p = sp_stats.pearsonr(x, y)

        ax.scatter(x, y, **scatter_kw)
        slope, intercept = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 200)
        ax.plot(xline, intercept + slope * xline, **line_kw)

        pstr = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.05, 0.95, f"r = {r:.3f}\n{pstr}\nN = {n}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=bbox_kw)
        ax.set_xlabel("Person-mean contrast (K_i)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.tick_params(labelsize=9)

    for j in range(len(pearson_panels), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Convergent validity: person-mean contrast vs baseline "
                 "clinical measures",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "figure_s2", figures_dir)


# ===================================================================
# Figure S3 — Contrast JN for Sleep-to-Pain (Null)
# ===================================================================

def generate_figure_s3(data_dir, results_dir, figures_dir, synthetic):
    """Figure S3: Contrast moderation of sleep-to-pain coupling (null)."""
    print("  Figure S3: Contrast JN (sleep-to-pain, null)")

    npz_path = os.path.join(results_dir, "contrast_posterior_draws.npz")
    if not _file_exists(npz_path, "Figure S3"):
        return

    d = np.load(npz_path)
    a2_draws = d["a2_draws"]
    a4_draws = d["a4_draws"]
    contrast_vals = d["contrast_vals"]
    c_sd = np.std(contrast_vals)
    u_sp_mean = d["u_sp_mean"]
    obs_pid_idx = d["obs_pid_idx"]
    obs_contrast = d["obs_contrast"]

    jn_sp = compute_jn_curve(a2_draws, a4_draws, contrast_vals,
                             clip_pct=(0, 100))

    x_positions = [("-2 SD", -2 * c_sd), ("0", 0.0), ("+2 SD", 2 * c_sd)]
    slopes_sp = _compute_simple_slopes(a2_draws, a4_draws, x_positions)

    a2_mean = a2_draws.mean()
    a4_mean = a4_draws.mean()
    obs_sp = a2_mean + u_sp_mean[obs_pid_idx] + a4_mean * obs_contrast
    dots_sp = {"x": obs_contrast, "y": obs_sp}

    level_labels = [
        "Body-dominant\n(\u22122 SD)", "Balanced\n(0)",
        "Knee-dominant\n(+2 SD)",
    ]
    level_x_vals = [-2 * c_sd, 0.0, 2 * c_sd]

    fig, ax = plt.subplots(figsize=(12.8, 8.05))
    _draw_jn_panel(ax, jn_sp, "Sleep \u2192 Pain",
                   slopes_sp, level_labels, level_x_vals,
                   xlabel="K\u1d42", body_knee_labels=True,
                   legend_loc="lower left", info_loc="lower right",
                   person_dots=dots_sp)

    save_figure(fig, "figure_s3", figures_dir)


# ===================================================================
# Figure S4 — Stimulation ROI Maps (skip if no atlas images)
# ===================================================================

def generate_figure_s4(data_dir, results_dir, figures_dir, synthetic):
    """Figure S4: Stimulation ROI maps (skipped if atlas images unavailable)."""
    print("  Figure S4: Stimulation ROI maps")

    roi_map_path = os.path.join(results_dir, "stim_roi_maps.png")
    if not os.path.exists(roi_map_path):
        print("    SKIP: No pre-rendered ROI map image found")
        return

    # If a pre-rendered image exists, just copy it
    import shutil
    out_path = os.path.join(figures_dir, "figure_s4.png")
    os.makedirs(figures_dir, exist_ok=True)
    shutil.copy2(roi_map_path, out_path)
    print(f"  Saved: {out_path}")


# ===================================================================
# Figure S5 — Krause ROI JN Panels (2x2 Merged)
# ===================================================================

def generate_figure_s5(data_dir, results_dir, figures_dir, synthetic):
    """Figure S5: Krause ROI JN panels (2x2 merged, non-NAcc ROIs)."""
    print("  Figure S5: Krause ROI JN panels (2x2)")

    npz_path = os.path.join(results_dir, "krause_roi_posterior_draws.npz")
    if not _file_exists(npz_path, "Figure S5"):
        return

    d = np.load(npz_path, allow_pickle=True)

    # Expected ROIs: Right_S1, Right_Mid_Insula, Left_Thalamus, Left_Ant_Insula
    roi_keys = [k.replace("_a2_draws", "")
                for k in d.files if k.endswith("_a2_draws")]
    roi_keys = roi_keys[:4]  # Take up to 4 for 2x2 layout

    if not roi_keys:
        print("    No Krause ROI draws found in NPZ, skipping")
        return

    n_rois = len(roi_keys)
    n_cols = 2
    n_rows = max(1, (n_rois + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i, roi in enumerate(roi_keys):
        ax = axes[i]
        a2_draws = d[f"{roi}_a2_draws"]
        gamma_draws = d[f"{roi}_gamma_sp_draws"]
        X_vals = d[f"{roi}_X_vals"]

        raw_mean = float(d[f"{roi}_raw_mean"]) if f"{roi}_raw_mean" in d \
            else None
        raw_sd = float(d[f"{roi}_raw_sd"]) if f"{roi}_raw_sd" in d else None

        jn = compute_jn_curve(a2_draws, gamma_draws, X_vals,
                              raw_mean=raw_mean, raw_sd=raw_sd)

        _draw_jn_panel(ax, jn, "Sleep \u2192 Pain",
                       slopes_dict=None, level_labels=None,
                       level_x_vals=None,
                       xlabel=roi.replace("_", " "),
                       legend_loc="lower right", info_loc="upper left",
                       panel_label=chr(65 + i))

    for j in range(n_rois, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    save_figure(fig, "figure_s5", figures_dir)


# ===================================================================
# Figure S6 — Arousal ROI Maps (skip if no atlas images)
# ===================================================================

def generate_figure_s6(data_dir, results_dir, figures_dir, synthetic):
    """Figure S6: Arousal ROI maps (skipped if atlas images unavailable)."""
    print("  Figure S6: Arousal ROI maps")

    roi_map_path = os.path.join(results_dir, "arousal_roi_maps.png")
    if not os.path.exists(roi_map_path):
        print("    SKIP: No pre-rendered arousal ROI map image found")
        return

    import shutil
    out_path = os.path.join(figures_dir, "figure_s6.png")
    os.makedirs(figures_dir, exist_ok=True)
    shutil.copy2(roi_map_path, out_path)
    print(f"  Saved: {out_path}")


# ===================================================================
# Figure S7 — fMRI Arousal JN Panels
# ===================================================================

def generate_figure_s7(data_dir, results_dir, figures_dir, synthetic):
    """Figure S7: fMRI arousal ROI JN panels (pain-to-sleep direction)."""
    print("  Figure S7: fMRI arousal JN panels")

    npz_path = os.path.join(results_dir, "fmri_arousal_posterior_draws.npz")
    if not _file_exists(npz_path, "Figure S7"):
        return

    d = np.load(npz_path, allow_pickle=True)

    roi_keys = [k.replace("_b1_draws", "")
                for k in d.files if k.endswith("_b1_draws")]

    if not roi_keys:
        print("    No fMRI arousal ROI draws found in NPZ, skipping")
        return

    n_rois = len(roi_keys)
    n_cols = 2
    n_rows = max(1, (n_rois + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i, roi in enumerate(roi_keys):
        ax = axes[i]
        b1_draws = d[f"{roi}_b1_draws"]
        gamma_draws = d[f"{roi}_gamma_ps_draws"]
        X_vals = d[f"{roi}_X_vals"]

        raw_mean = float(d[f"{roi}_raw_mean"]) if f"{roi}_raw_mean" in d \
            else None
        raw_sd = float(d[f"{roi}_raw_sd"]) if f"{roi}_raw_sd" in d else None

        jn = compute_jn_curve(b1_draws, gamma_draws, X_vals,
                              raw_mean=raw_mean, raw_sd=raw_sd)

        _draw_jn_panel(ax, jn, "Pain \u2192 Sleep",
                       slopes_dict=None, level_labels=None,
                       level_x_vals=None,
                       xlabel=roi.replace("_", " "),
                       legend_loc="lower right", info_loc="upper left",
                       panel_label=chr(65 + i))

    for j in range(n_rois, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    save_figure(fig, "figure_s7", figures_dir)


# ===================================================================
# Figure S8 — VBM Arousal JN Panels
# ===================================================================

def generate_figure_s8(data_dir, results_dir, figures_dir, synthetic):
    """Figure S8: VBM arousal ROI JN panels (pain-to-sleep direction)."""
    print("  Figure S8: VBM arousal JN panels")

    npz_path = os.path.join(results_dir, "vbm_arousal_posterior_draws.npz")
    if not _file_exists(npz_path, "Figure S8"):
        return

    d = np.load(npz_path, allow_pickle=True)

    roi_keys = [k.replace("_b1_draws", "")
                for k in d.files if k.endswith("_b1_draws")]

    if not roi_keys:
        print("    No VBM arousal ROI draws found in NPZ, skipping")
        return

    n_rois = len(roi_keys)
    n_cols = 2
    n_rows = max(1, (n_rois + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i, roi in enumerate(roi_keys):
        ax = axes[i]
        b1_draws = d[f"{roi}_b1_draws"]
        gamma_draws = d[f"{roi}_gamma_ps_draws"]
        X_vals = d[f"{roi}_X_vals"]

        raw_mean = float(d[f"{roi}_raw_mean"]) if f"{roi}_raw_mean" in d \
            else None
        raw_sd = float(d[f"{roi}_raw_sd"]) if f"{roi}_raw_sd" in d else None

        jn = compute_jn_curve(b1_draws, gamma_draws, X_vals,
                              raw_mean=raw_mean, raw_sd=raw_sd)

        _draw_jn_panel(ax, jn, "Pain \u2192 Sleep",
                       slopes_dict=None, level_labels=None,
                       level_x_vals=None,
                       xlabel=roi.replace("_", " "),
                       legend_loc="lower right", info_loc="upper left",
                       panel_label=chr(65 + i))

    for j in range(n_rois, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    save_figure(fig, "figure_s8", figures_dir)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate all manuscript figures from saved results.",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Read from results/synthetic/ and save to figures/synthetic/.",
    )
    args = parser.parse_args()

    setup_style()

    data_dir, results_dir, figures_dir = _resolve_paths(args.synthetic)
    os.makedirs(figures_dir, exist_ok=True)

    mode_label = "SYNTHETIC" if args.synthetic else "REAL"
    print(f"\n{'=' * 60}")
    print(f"  06_generate_figures.py  [{mode_label} DATA]")
    print(f"  Results: {results_dir}")
    print(f"  Output:  {figures_dir}")
    print(f"{'=' * 60}\n")

    # Main figures
    generate_figure1(data_dir, results_dir, figures_dir, args.synthetic)
    generate_figures2_3(data_dir, results_dir, figures_dir, args.synthetic)
    generate_figure4(data_dir, results_dir, figures_dir, args.synthetic)
    generate_figure5(data_dir, results_dir, figures_dir, args.synthetic)
    generate_figure6(data_dir, results_dir, figures_dir, args.synthetic)

    # Supplementary figures
    generate_figure_s1(data_dir, results_dir, figures_dir, args.synthetic)
    generate_figure_s2(data_dir, results_dir, figures_dir, args.synthetic)
    generate_figure_s3(data_dir, results_dir, figures_dir, args.synthetic)
    generate_figure_s4(data_dir, results_dir, figures_dir, args.synthetic)
    generate_figure_s5(data_dir, results_dir, figures_dir, args.synthetic)
    generate_figure_s6(data_dir, results_dir, figures_dir, args.synthetic)
    generate_figure_s7(data_dir, results_dir, figures_dir, args.synthetic)
    generate_figure_s8(data_dir, results_dir, figures_dir, args.synthetic)

    print(f"\n{'=' * 60}")
    print(f"  All figures generated.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
