"""
Step 10b — Plot Pain-to-Sleep arousal relay ROI maps (Figure S6).
======================================================================

Input:  data/atlases/   (atlas NIfTI files)
        MNI152 template (via nilearn)
Output:
  results/step10b_ps_roi_maps/
    step10b_figure_s6_arousal_rois.png   — Figure S6

Creates a 2-column grid of orthogonal brain slices (sagittal/coronal/axial)
for each PS arousal relay ROI: PBN, SI-BF/Ch4, CeA, BNST, LH.

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
ATLAS_DIR = os.path.join(ROOT, "data", "atlases")
RESULTS_DIR = os.path.join(ROOT, "results")
SUPP_DIR = os.path.join(RESULTS_DIR, "supplementary_materials")
os.makedirs(SUPP_DIR, exist_ok=True)

OUT_FIG_S6 = os.path.join(SUPP_DIR, "figure_s6_arousal_rois.png")

ROIS = {
    "PBN": {
        "label": "Lateral Parabrachial Nucleus (PBN)",
        "atlas_file": os.path.join(ATLAS_DIR, "atlas_b2_brainstem.nii.gz"),
        "atlas_type": "label",
        "labels": [19, 20],
        "title": "PBN  (Brainstem Navigator [39])",
    },
    "SI-BF/Ch4": {
        "label": "Substantia Innominata / Basal Forebrain (SI-BF/Ch4)",
        "atlas_file": os.path.join(ATLAS_DIR, "zaborszky_bf",
                                   "Ch4_basal_forebrain_prob_MNI152.nii.gz"),
        "atlas_type": "prob",
        "title": "SI-BF/Ch4  (Zaborszky et al. [40])",
    },
    "CeA": {
        "label": "Central Nucleus of the Amygdala (CeA)",
        "atlas_file": os.path.join(ATLAS_DIR,
                                   "CIT168_CeA_prob_bilat_MNI152_1mm.nii.gz"),
        "atlas_type": "prob",
        "title": "CeA  (CIT168 [41])",
    },
    "BNST": {
        "label": "Bed Nucleus of the Stria Terminalis (BNST)",
        "atlas_file": os.path.join(ATLAS_DIR, "Blackford_BNST_3T.nii.gz"),
        "atlas_type": "prob",
        "title": "BNST  (Theiss et al. [42])",
    },
    "LH": {
        "label": "Lateral Hypothalamus (LH)",
        "atlas_file": os.path.join(ATLAS_DIR, "hypothalamus_neudorfer2020",
                                   "atlas_labels_0.5mm.nii.gz"),
        "atlas_type": "label",
        "labels": [25, 26],
        "title": "LH  (Neudorfer et al. [43])",
    },
}

ROI_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]


def make_solid_cmap(hex_color):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("solid", [hex_color, hex_color], N=2)


def load_atlas_roi(roi_cfg, template):
    import nibabel as nib
    from nilearn.image import resample_to_img, new_img_like
    atlas_img = nib.load(roi_cfg["atlas_file"])
    atlas_data = atlas_img.get_fdata()
    if roi_cfg["atlas_type"] == "label":
        mask = np.zeros_like(atlas_data, dtype=np.float32)
        for lab in roi_cfg["labels"]:
            mask[atlas_data == lab] = 1.0
        mask_img = nib.Nifti1Image(mask, atlas_img.affine)
    else:
        mask_img = nib.Nifti1Image((atlas_data > 0).astype(np.float32),
                                   atlas_img.affine)
    resampled = resample_to_img(mask_img, template, interpolation="linear")
    data = (resampled.get_fdata() > 0.1).astype(np.float32)
    return new_img_like(template, data)


def get_center_of_mass(roi_img):
    data = roi_img.get_fdata()
    coords = np.array(np.where(data > 0)).T
    if len(coords) == 0:
        return (0, 0, 0)
    com_ijk = coords.mean(axis=0)
    com_mni = roi_img.affine[:3, :3] @ com_ijk + roi_img.affine[:3, 3]
    return tuple(com_mni.astype(int))


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


def run_step10b(verbose=True):
    from nilearn.datasets import load_mni152_template

    if verbose:
        print("=" * 70)
        print("STEP 10b — Plot Pain-to-Sleep arousal relay ROI maps (Figure S6)")
        print("=" * 70)

    template = load_mni152_template(resolution=1)
    if verbose:
        print("  Loaded MNI152 template (1mm)")

    panels = []
    for idx, (roi_key, cfg) in enumerate(ROIS.items()):
        if verbose:
            print(f"  Loading: {cfg['label']}")
        roi_img = load_atlas_roi(cfg, template)
        nvox = int((roi_img.get_fdata() > 0).sum())
        com = get_center_of_mass(roi_img)
        if verbose:
            print(f"    {nvox} voxels, center of mass: {com}")
        cmap = make_solid_cmap(ROI_COLORS[idx])
        panel = render_roi_to_image(roi_img, template, com, cfg["title"], cmap)
        panels.append(panel)

    composite = make_composite(panels, ncols=2)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)
    composite.save(OUT_FIG_S6, dpi=(200, 200))
    if verbose:
        print(f"  Saved: {OUT_FIG_S6}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 10b — plot PS arousal relay ROI maps (Figure S6)."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_step10b(verbose=not args.quiet)


if __name__ == "__main__":
    main()
