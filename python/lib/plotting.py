#!/usr/bin/env python3
"""
Shared Plotting Utilities for Sleep-Pain Coupling Manuscript Figures
====================================================================

This module provides reusable plotting functions for all manuscript figures.
Each function creates one standardized plot component (boxstrip, forest,
Johnson-Neyman curve) with consistent styling.

Figure-to-Function Mapping
--------------------------
  Figure 2 (PS coupling):  boxstrip_plot + forest_plot
  Figure 3 (SP coupling):  boxstrip_plot + forest_plot
  Figure 4 (Contrast JN):  jn_plot
  Figure 5 (NAcc JN):      jn_plot
  Figure 6 (ACC JN):       jn_plot

Style
-----
All figures use Times New Roman 12pt as the base font, consistent with
journal requirements. Colors follow a blue/red scheme for negative/positive
coupling, with green shading for credible (significant) JN regions and
gray for non-credible regions.

Author: Pedro Valdes-Hernandez
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ===================================================================
# Global Style Setup
# ===================================================================

def setup_style():
    """Configure matplotlib rcParams for publication-quality figures.

    Sets Times New Roman as the default font (12pt), enables minor ticks,
    and configures line widths and marker sizes for print clarity at
    journal column widths.

    Call this once at the start of any plotting script.
    """
    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        # Lines and markers
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        # Axes
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Ticks
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


# ===================================================================
# Color Palette
# ===================================================================

# Consistent color scheme across all figures
COLORS = {
    "negative": "#42A5F5",         # blue -- negative coupling
    "positive": "#EF5350",         # red -- positive coupling
    "negative_dark": "#0D47A1",    # dark blue edge
    "positive_dark": "#B71C1C",    # dark red edge
    "box_face": "#E3F2FD",         # light blue box fill
    "box_edge": "#1565C0",         # blue box border
    "median_line": "#D32F2F",      # red median line
    "mean_diamond": "#757575",     # gray diamond for mean
    "jn_line": "#1565C0",          # blue JN posterior mean line
    "jn_ci_credible": "#81C784",   # green shading (CrI excludes 0)
    "jn_ci_null": "#BDBDBD",       # gray shading (CrI includes 0)
    "jn_boundary": "#F44336",      # red dashed JN boundary
    "rug": "#424242",              # dark gray rug marks
    "forest_point": "#1565C0",     # blue forest plot points
    "forest_line": "#1565C0",      # blue CrI lines
    "forest_fill": "#BBDEFB",      # light blue forest fill
    "zero_line": "black",          # reference line at zero
}


# ===================================================================
# Box-Strip Plot (Figures 2A, 3A)
# ===================================================================

def boxstrip_plot(values, population_mean, ax, population_ci=None,
                  title=None, xlabel=None):
    """Combined box-and-strip plot for person-level coupling estimates.

    Displays the distribution of N person-specific posterior mean coupling
    coefficients as a horizontal boxplot overlaid with jittered strip
    points. Points are colored blue (negative) or red (positive).

    This corresponds to Panels A of Figures 2 and 3 in the manuscript.

    Parameters
    ----------
    values : ndarray, shape (n_persons,)
        Person-specific posterior mean coupling coefficients
        (lambda_sp_i or lambda_ps_i).
    population_mean : float
        Population-level posterior mean (lambda_sp or lambda_ps),
        plotted as a gray diamond.
    ax : matplotlib.axes.Axes
        Axes to draw on.
    population_ci : tuple (lo, hi) or None, default None
        If provided, the 95% CrI for the population mean.
        Displayed as a text annotation.
    title : str or None
        Panel title (e.g., "A.  Pain-to-Sleep Coupling").
    xlabel : str or None
        X-axis label (e.g., "lambda (posterior mean)").
    """
    if len(values) == 0:
        return

    # --- Boxplot ---
    bp = ax.boxplot(
        values, vert=False, widths=0.5,
        patch_artist=True,
        boxprops=dict(
            facecolor=COLORS["box_face"],
            edgecolor=COLORS["box_edge"],
            linewidth=1.5,
        ),
        medianprops=dict(color=COLORS["median_line"], linewidth=2.5),
        whiskerprops=dict(color=COLORS["box_edge"], linewidth=1.2),
        capprops=dict(color=COLORS["box_edge"], linewidth=1.2),
        flierprops=dict(marker="", markersize=0),  # suppress outlier marks
    )

    # --- Jittered strip points ---
    rng = np.random.default_rng(42)
    jitter_y = rng.uniform(0.7, 1.3, len(values))

    # Color each point by sign
    point_colors = [
        COLORS["negative"] if x < 0 else COLORS["positive"]
        for x in values
    ]
    edge_colors = [
        COLORS["negative_dark"] if x < 0 else COLORS["positive_dark"]
        for x in values
    ]

    sc = ax.scatter(
        values, jitter_y,
        c=point_colors, alpha=0.7, s=80, zorder=3,
        marker="o", edgecolors=edge_colors, linewidths=0.8,
    )
    # Subtle drop shadow for depth
    sc.set_path_effects([
        patheffects.withSimplePatchShadow(
            offset=(0.5, -0.5), shadow_rgbFace="#555555", alpha=0.25
        )
    ])

    # --- Reference line at zero ---
    ax.axvline(0, color=COLORS["zero_line"], linewidth=1, linestyle="-",
               zorder=2, alpha=0.5)

    # --- Population mean diamond ---
    ax.plot(
        population_mean, 1.0,
        marker="D", color=COLORS["mean_diamond"], markersize=12,
        zorder=5, markeredgecolor="white", markeredgewidth=1.2,
    )

    # --- Annotations ---
    if title:
        ax.set_title(title, fontsize=20, fontweight="bold",
                      loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=18)

    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=16)

    # Population parameter text box
    if population_ci is not None:
        pop_text = (
            f"Population \u03bb = {population_mean:.3f}\n"
            f"[{population_ci[0]:.3f}, {population_ci[1]:.3f}]"
        )
        ax.text(
            0.02, 0.98, pop_text, transform=ax.transAxes,
            fontsize=14, va="top", ha="left", color="black",
            fontweight="bold",
        )


# ===================================================================
# Forest Plot (Figures 2B, 3B)
# ===================================================================

def forest_plot(means, cris, labels, ax, title=None,
                highlight_credible=True):
    """Forest plot with credible intervals for person-level estimates.

    Displays each person's coupling coefficient as a point with a
    horizontal 95% CrI bar, sorted from most negative to most positive.

    This corresponds to Panels B of Figures 2 and 3 in the manuscript.

    Parameters
    ----------
    means : ndarray, shape (n_persons,)
        Posterior means for each person's coupling.
    cris : ndarray, shape (n_persons, 2)
        Lower and upper bounds of 95% CrIs. Column 0 = lower, column 1 = upper.
    labels : list of str or None
        Optional person labels (usually None for N>50).
    ax : matplotlib.axes.Axes
        Axes to draw on.
    title : str or None
        Panel title (e.g., "B.  Person-Specific Estimates").
    highlight_credible : bool, default True
        If True, CrIs that exclude zero are drawn in a darker color.
    """
    n = len(means)
    if n == 0:
        return

    # Sort by posterior mean
    sort_idx = np.argsort(means)
    sorted_means = means[sort_idx]
    sorted_cris = cris[sort_idx]

    y_positions = np.arange(n)

    for i in range(n):
        lo = sorted_cris[i, 0]
        hi = sorted_cris[i, 1]
        mean_val = sorted_means[i]

        # Determine if this CrI excludes zero
        credible = (lo > 0) or (hi < 0)

        if highlight_credible and credible:
            color = COLORS["forest_point"]
            alpha = 0.9
            lw = 1.2
        else:
            color = COLORS["forest_line"]
            alpha = 0.4
            lw = 0.6

        # Horizontal CrI bar
        ax.plot(
            [lo, hi], [y_positions[i], y_positions[i]],
            color=color, alpha=alpha, linewidth=lw, solid_capstyle="round",
        )
        # Point estimate
        ax.plot(
            mean_val, y_positions[i],
            "o", color=color, alpha=min(alpha + 0.1, 1.0), markersize=3,
        )

    # Reference line at zero
    ax.axvline(0, color=COLORS["zero_line"], linewidth=1, linestyle="-",
               alpha=0.5, zorder=0)

    # Formatting
    if title:
        ax.set_title(title, fontsize=20, fontweight="bold",
                      loc="left", pad=10)
    ax.set_yticks([])
    ax.set_xlabel("\u03bb (95% CrI)", fontsize=18)
    ax.tick_params(axis="x", labelsize=16)

    # Count credible intervals that exclude zero
    n_neg = int(np.sum(sorted_cris[:, 1] < 0))
    n_pos = int(np.sum(sorted_cris[:, 0] > 0))
    n_null = n - n_neg - n_pos
    count_text = (
        f"N = {n}\n"
        f"Credibly < 0: {n_neg}\n"
        f"Credibly > 0: {n_pos}\n"
        f"Includes 0: {n_null}"
    )
    ax.text(
        0.98, 0.02, count_text, transform=ax.transAxes,
        fontsize=12, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.8),
    )


# ===================================================================
# Johnson-Neyman Plot (Figures 4, 5, 6)
# ===================================================================

def jn_plot(moderator_grid, coupling_mean, coupling_ci, boundary_values,
            observations, ax, title=None, xlabel=None, ylabel=None,
            person_dots=None, simple_slopes=None):
    """Johnson-Neyman significance curve with shaded credible regions.

    Displays the conditional coupling coefficient as a continuous function
    of a moderator variable, with the 95% credible interval shaded green
    where it excludes zero (credible effect) and gray where it includes
    zero (null). JN boundaries are marked with dashed red lines.

    This corresponds to Figures 4, 5, and 6 in the manuscript.

    Parameters
    ----------
    moderator_grid : ndarray, shape (n_grid,)
        X-axis values (moderator values, 500 grid points).
    coupling_mean : ndarray, shape (n_grid,)
        Posterior mean of the conditional coupling at each grid point.
    coupling_ci : ndarray, shape (n_grid, 2)
        Lower (col 0) and upper (col 1) 95% CrI bounds.
    boundary_values : list of float
        JN boundary values where significance status changes.
    observations : ndarray
        Observed moderator values for all subjects (for rug plot).
    ax : matplotlib.axes.Axes
        Axes to draw on.
    title : str or None
        Panel/figure title.
    xlabel : str or None
        X-axis label (moderator name).
    ylabel : str or None
        Y-axis label (default: "Conditional coupling").
    person_dots : dict or None
        If provided, scatter person-specific coupling estimates.
        Must have keys 'x' (moderator values) and 'y' (coupling values).
    simple_slopes : dict or None
        If provided, overlay vertical error bars at specific moderator
        values. Format: {label: {"x": float, "beta": float, "ci_lo": float,
        "ci_hi": float, "color": str}}.
    """
    ci_lo = coupling_ci[:, 0]
    ci_hi = coupling_ci[:, 1]

    # Determine significance at each grid point
    sig_negative = ci_hi < 0
    sig_positive = ci_lo > 0
    sig = sig_negative | sig_positive

    # --- Person-specific coupling dots (behind everything) ---
    if person_dots is not None:
        ax.scatter(
            person_dots["x"], person_dots["y"],
            s=40, color=COLORS["negative"], alpha=0.7,
            edgecolors=COLORS["negative_dark"], linewidths=0.8,
            zorder=2, label="Fitted coupling",
            path_effects=[patheffects.withSimplePatchShadow(
                offset=(0.5, -0.5), shadow_rgbFace="#1565C0", alpha=0.3,
            )],
        )

    # --- Shaded credible interval band ---
    # Color-code by significance: green where CrI excludes zero, gray otherwise
    for i in range(len(moderator_grid) - 1):
        color = COLORS["jn_ci_credible"] if sig[i] else COLORS["jn_ci_null"]
        alpha = 0.35 if sig[i] else 0.25
        ax.fill_between(
            moderator_grid[i:i + 2],
            ci_lo[i:i + 2],
            ci_hi[i:i + 2],
            color=color, alpha=alpha, linewidth=0,
        )

    # --- Posterior mean line and CrI boundaries ---
    ax.plot(moderator_grid, coupling_mean,
            color=COLORS["jn_line"], linewidth=2.5)
    ax.plot(moderator_grid, ci_lo,
            color=COLORS["jn_line"], linewidth=1, linestyle="--", alpha=0.7)
    ax.plot(moderator_grid, ci_hi,
            color=COLORS["jn_line"], linewidth=1, linestyle="--", alpha=0.7)

    # --- Zero reference line ---
    ax.axhline(0, color=COLORS["zero_line"], linewidth=0.8,
               linestyle="-", alpha=0.5)

    # --- JN boundary lines ---
    for bval in boundary_values:
        ax.axvline(
            bval, color=COLORS["jn_boundary"], linewidth=1.5,
            linestyle="--", alpha=0.8, zorder=4,
        )

    # --- Rug plot of observed moderator values ---
    y_lo, y_hi = ax.get_ylim()
    ax.scatter(
        observations,
        np.full_like(observations, y_lo),
        marker="|", color=COLORS["rug"], alpha=0.3, s=40, zorder=1,
    )

    # --- Simple slope error bars (optional) ---
    if simple_slopes is not None:
        for label, ss in simple_slopes.items():
            ax.errorbar(
                ss["x"], ss["beta"],
                yerr=[[ss["beta"] - ss["ci_lo"]], [ss["ci_hi"] - ss["beta"]]],
                fmt="o", color=ss.get("color", COLORS["forest_point"]),
                markersize=8, capsize=5, capthick=1.5, linewidth=1.5,
                zorder=5,
            )
            ax.annotate(
                label, (ss["x"], ss["ci_hi"]),
                textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=10, fontweight="bold",
                color=ss.get("color", COLORS["forest_point"]),
            )

    # --- Labels ---
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold",
                      loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    else:
        ax.set_ylabel("Conditional coupling (\u03bb)", fontsize=14)

    ax.tick_params(axis="both", labelsize=12)

    # --- Legend ---
    legend_elements = [
        Patch(facecolor=COLORS["jn_ci_credible"], alpha=0.35,
              label="95% CrI excludes 0"),
        Patch(facecolor=COLORS["jn_ci_null"], alpha=0.25,
              label="95% CrI includes 0"),
        Line2D([0], [0], color=COLORS["jn_boundary"], linewidth=1.5,
               linestyle="--", label="JN boundary"),
    ]
    if person_dots is not None:
        legend_elements.insert(0, Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=COLORS["negative"], markersize=8,
            label="Person-specific coupling",
        ))
    ax.legend(handles=legend_elements, loc="best", fontsize=10,
              framealpha=0.9)


# ===================================================================
# Figure Saving
# ===================================================================

def save_figure(fig, name, figures_dir):
    """Save a figure at 300 DPI as PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    name : str
        Filename without extension (e.g., "figure2").
    figures_dir : str
        Path to the ``figures/`` directory.
    """
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, f"{name}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {out_path}")
