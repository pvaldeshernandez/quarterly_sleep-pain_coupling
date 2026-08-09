"""The stability heatmap: one cell per (measure, occasion), carrying r, p and n.

Written around the CONCEPT — "a grid of correlations with their p and n, sorted by
strength, with the non-significant cells marked" — not around one figure. It takes three
aligned frames and a list of column labels, so the same function draws Figure S3's
17 x 12 grid or any other grid of the same shape.

Factored out of `revision/a10d_render_heatmap.py`, which was the only thing that could
draw Figure S3 and was never migrated when the sandbox became the pipeline: step 06
computed the grid but nothing rendered it, so the figure in the document had no source.

Functions
---------
render(r, p, n, columns, row_labels, out_path, ...)
    Draw and save the heatmap. Returns the path written.
"""
from __future__ import annotations

import numpy as np

__all__ = ["render", "ALPHA", "VMIN", "VMAX"]

#: cells with p above this are hatched — "not significant" must be visible at a glance,
#: not inferred by reading the small number in the middle of the cell
ALPHA = 0.05

#: colour scale, centred on zero so the sign of r is readable from hue alone
VMIN, VMAX = -0.6, 0.6

# Layout. These are tuned to the published figure; changing them changes Figure S3.
FIG_W, FIG_H = 17.5, 12.0
AX_LEFT, AX_RIGHT = 0.190, 0.905      # grid spans from the row labels to the colorbar
AX_BOTTOM, AX_TOP = 0.055, 0.985
CBAR_RECT = [0.918, 0.20, 0.014, 0.60]
FS_R, FS_P, FS_N = 14, 12, 9          # r bold, then p, then n
FS_XTICK, FS_YTICK = 16, 13
Y_R, Y_P, Y_N = 0.74, 0.50, 0.26      # vertical placement within a unit-height cell
HATCH_LW = 0.7
HATCH_COLOR = (0, 0, 0, 0.45)         # light enough to read the numbers through


def fmt_p(p):
    """p as printed in a cell: scientific below .001, three decimals above."""
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def render(r, p, n, columns, row_labels, out_path,
           alpha=ALPHA, vmin=VMIN, vmax=VMAX, cbar_label="Pearson r", dpi=300):
    """Draw the grid and save it.

    Parameters
    ----------
    r, p, n : DataFrame
        Same index and same columns, one row per measure. `r` is the coefficient,
        `p` its p-value, `n` the pairs it was computed on. Row ORDER is the drawing
        order, top to bottom — sort before calling; this function never reorders,
        because a figure whose rows silently reordered would be a different figure.
    columns : sequence of str
        Which columns to draw, in order. Named explicitly rather than taken from the
        frame so a caller can draw a subset without slicing three frames consistently.
    row_labels : sequence of str
        Printed labels, in the SAME order as `r.index`.
    out_path : str
        Where to write the PNG. Parent directory must exist.

    Returns
    -------
    str : `out_path`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    for name, f in (("p", p), ("n", n)):
        if list(f.index) != list(r.index):
            raise ValueError(f"`{name}` rows are not aligned with `r` rows")
    missing = [c for c in columns if c not in r.columns]
    if missing:
        raise KeyError(f"columns absent from the grid: {missing}")
    if len(row_labels) != len(r.index):
        raise ValueError(f"{len(row_labels)} labels for {len(r.index)} rows")

    n_rows, n_cols = len(r.index), len(columns)
    plt.rcParams["hatch.linewidth"] = HATCH_LW

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([AX_LEFT, AX_BOTTOM, AX_RIGHT - AX_LEFT, AX_TOP - AX_BOTTOM])
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu_r

    for i, var in enumerate(r.index):
        y = n_rows - 1 - i                      # row 0 at the top
        for j, col in enumerate(columns):
            rv, pv, nv = r.at[var, col], p.at[var, col], int(n.at[var, col])
            ax.add_patch(Rectangle((j, y), 1, 1, facecolor=cmap(norm(rv)),
                                   edgecolor="black", linewidth=0.5, zorder=1))
            if pv > alpha:
                ax.add_patch(Rectangle((j, y), 1, 1, facecolor="none", hatch="///",
                                       edgecolor=HATCH_COLOR, linewidth=0.0, zorder=2))
            ax.text(j + .5, y + Y_R, f"{rv:+.3f}", ha="center", va="center",
                    fontsize=FS_R, fontweight="bold", zorder=3)
            ax.text(j + .5, y + Y_P, fmt_p(pv), ha="center", va="center",
                    fontsize=FS_P, zorder=3)
            ax.text(j + .5, y + Y_N, f"{nv}", ha="center", va="center",
                    fontsize=FS_N, zorder=3)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + .5)
    ax.set_xticklabels(list(columns), fontsize=FS_XTICK, fontweight="bold")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(np.arange(n_rows) + .5)
    ax.set_yticklabels(list(row_labels)[::-1], fontsize=FS_YTICK)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(length=0)

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                        cax=fig.add_axes(CBAR_RECT))
    cbar.set_label(cbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
