"""
Step 7b — Plot Sleep-to-Pain ROI maps (Figure S4).
======================================================================

Input:  MNI152 template (via nilearn)
Output:
  results/step7b_sp_roi_maps/
    step7b_figure_s4_stim_rois.png   — Figure S4

Creates a 2-column grid of orthogonal brain slices (sagittal/coronal/axial)
for each SP moderation ROI: 6 Krause ROIs + 2 Sardi ACC ROIs (bilateral).

Author: Pedro Valdes-Hernandez (with Claude Sonnet 4.6)
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR = os.path.join(ROOT, "results")
SUPP_DIR = os.path.join(RESULTS_DIR, "supplementary_materials")
os.makedirs(SUPP_DIR, exist_ok=True)

OUT_FIG_S4 = os.path.join(SUPP_DIR, "figure_s4_stim_rois.png")

# ROIs: 6 Krause + 2 Sardi (Right + Left dACC/MCC)
ROIS = {
    "Contra_S1": {
        "label": "Contralateral Somatosensory Cortex (S1)",
        "mni": (36, -45, 59),
        "mni_mirror": (-36, -45, 59),
        "radius_mm": 8,
    },
    "Contra_Middle_Insula": {
        "label": "Contralateral Middle Insula",
        "mni": (32, 4, 11),
        "mni_mirror": (-32, 4, 11),
        "radius_mm": 8,
    },
    "Left_Thalamus": {
        "label": "Left Thalamus",
        "mni": (-10, -6, 10),
        "radius_mm": 4,
    },
    "Left_Anterior_Insula": {
        "label": "Left Anterior Insula",
        "mni": (-27, 25, 0),
        "radius_mm": 8,
    },
    "Left_NAcc": {
        "label": "Left Nucleus Accumbens (NAcc)",
        "mni": (-9, 2, -7),
        "radius_mm": 6,
    },
    "Right_NAcc": {
        "label": "Right Nucleus Accumbens (NAcc)",
        "mni": (9, 2, -7),
        "radius_mm": 6,
    },
    "Left_dACC_MCC": {
        "label": "Left dACC/MCC (Xu et al. 2020)",
        "mni": (-6, 12, 38),
        "radius_mm": 6,
    },
    "Right_dACC_MCC": {
        "label": "Right dACC/MCC (Xu et al. 2020)",
        "mni": (6, 12, 38),
        "radius_mm": 6,
    },
}

ROI_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00",
    "#984ea3", "#984ea3", "#a65628", "#a65628",
]


def make_solid_cmap(hex_color):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("solid", [hex_color, hex_color], N=2)


def make_sphere_img(template, mni_center, radius_mm):
    from nilearn.image import new_img_like
    affine = template.affine
    shape = template.shape[:3]
    data = np.zeros(shape, dtype=np.float32)
    i, j, k = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    ijk = np.column_stack([i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)])
    mni_coords = (affine @ ijk.T).T[:, :3]
    center = np.array(mni_center, dtype=float)
    dist = np.sqrt(np.sum((mni_coords - center) ** 2, axis=1))
    data[dist.reshape(shape) <= radius_mm] = 1.0
    return new_img_like(template, data, affine=affine)


def render_roi_to_image(roi_img, template, cut_coords, title, cmap):
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_roi
    from PIL import Image
    display = plot_roi(
        roi_img, bg_img=template, display_mode="ortho",
        cut_coords=cut_coords, title=title, cmap=cmap,
        alpha=0.7, dim=-0.5, annotate=True, draw_cross=True, colorbar=False,
    )
    fig = display.frame_axes.figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def make_composite(panels, ncols=2):
    from PIL import Image
    nrows = math.ceil(len(panels) / ncols)
    target_w = max(p.width for p in panels)
    scaled = []
    for p in panels:
        new_h = int(p.height * target_w / p.width)
        scaled.append(p.resize((target_w, new_h), Image.LANCZOS))
    max_h = max(p.height for p in scaled)
    padded = []
    for p in scaled:
        if p.height < max_h:
            canvas = Image.new("RGB", (target_w, max_h), (0, 0, 0))
            canvas.paste(p, (0, (max_h - p.height) // 2))
            padded.append(canvas)
        else:
            padded.append(p)
    gap = 4
    composite = Image.new("RGB",
                          (ncols * target_w + (ncols - 1) * gap,
                           nrows * max_h + (nrows - 1) * gap),
                          (255, 255, 255))
    for idx, p in enumerate(padded):
        composite.paste(p, (idx % ncols * (target_w + gap),
                            idx // ncols * (max_h + gap)))
    return composite


def run_step7b(verbose=True):
    from nilearn.datasets import load_mni152_template
    from nilearn.image import new_img_like

    if verbose:
        print("=" * 70)
        print("STEP 7b — Plot Sleep-to-Pain ROI maps (Figure S4)")
        print("=" * 70)

    template = load_mni152_template(resolution=1)
    if verbose:
        print("  Loaded MNI152 template (1mm)")

    panels = []
    for idx, (roi_key, cfg) in enumerate(ROIS.items()):
        if verbose:
            print(f"  Building: {cfg['label']}")
        roi_img = make_sphere_img(template, cfg["mni"], cfg["radius_mm"])
        if "mni_mirror" in cfg:
            mirror = make_sphere_img(template, cfg["mni_mirror"], cfg["radius_mm"])
            combined = np.clip(roi_img.get_fdata() + mirror.get_fdata(), 0, 1)
            roi_img = new_img_like(template, combined, affine=template.affine)
            x = abs(cfg["mni"][0])
            title = (f"{cfg['label']}  "
                     f"(MNI: \u00b1{x}, {cfg['mni'][1]}, {cfg['mni'][2]}, "
                     f"r = {cfg['radius_mm']} mm)")
        else:
            title = (f"{cfg['label']}  "
                     f"(MNI: {cfg['mni']}, r = {cfg['radius_mm']} mm)")

        cmap = make_solid_cmap(ROI_COLORS[idx])
        panel = render_roi_to_image(roi_img, template, cfg["mni"], title, cmap)
        panels.append(panel)

    composite = make_composite(panels, ncols=2)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)
    composite.save(OUT_FIG_S4, dpi=(200, 200))
    if verbose:
        print(f"  Saved: {OUT_FIG_S4}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 7b — plot SP ROI maps (Figure S4)."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_step7b(verbose=not args.quiet)


if __name__ == "__main__":
    main()
