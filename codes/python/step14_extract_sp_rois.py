"""
Step 07 — Extract Sleep-to-Pain fMRI ROI values + Figure S4.
======================================================================

Input:  derivatives/step13_fmri_contrasts/masked/   (NAcc ROIs)
        derivatives/step13_fmri_contrasts/unmasked/ (all other ROIs)
        MNI152 template (via nilearn, for Figure S4)
Output:
  derivatives/step14_sp_roi_values/
    step14_sp_roi_values.csv    — per-subject z-scored ROI values
  results/<this step>/
    figure_stim_rois.png    — Figure S4: ROI brain maps

Extracts mean fMRI BOLD contrast (stimulation > baseline) within
8 spherical ROIs for the Sleep-to-Pain moderation analysis:
  - 6 Krause et al. (2019) ROIs: S1, Middle Insula, Thalamus,
    Anterior Insula, Left NAcc, Right NAcc
  - 2 Sardi-motivated ACC ROIs: Left + Right dACC/MCC (Xu et al. 2020)

NAcc ROIs use GM-masked contrasts; all others use unmasked
re-estimated contrasts. Values are z-scored across subjects.

Also generates Figure S4: orthogonal brain slices for all 8 ROIs.

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
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DERIV_DIR = os.path.join(ROOT, "derivatives")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step14_sp_roi_values")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
#: A step writes into its OWN results folder. tools/collect_deliverables.py
#: copies what the documents need into results/manuscript/ and
#: results/supplementary_materials/ under their document-facing names.
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step14_sp_roi_extraction")
SUPP_DIR = STEP_RESULTS_DIR
os.makedirs(SUPP_DIR, exist_ok=True)

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

OUT_ROI_CSV = os.path.join(STEP_DERIV_DIR, "step14_sp_roi_values.csv")
OUT_FIG_S4 = os.path.join(SUPP_DIR, "figure_stim_rois.png")

# Contrast image directories from Step 06
FMRI_MASKED_DIR   = os.path.join(DERIV_DIR, "step13_fmri_contrasts", "masked")
FMRI_UNMASKED_DIR = os.path.join(DERIV_DIR, "step13_fmri_contrasts", "unmasked")

# Stimulation-side inputs (for contralateralized ROIs: S1, Middle Insula)
DATA_DIR = os.path.join(ROOT, "data")
WIDE_XLSX = os.path.join(DATA_DIR, "original", "participants_wideformat.xlsx")
LONG_CSV  = os.path.join(DATA_DIR, "step00_extracted_long.csv")

# ROI definitions for extraction
SP_ROIS = {
    "Contra_S1": {
        "label": "Contralateral Somatosensory Cortex (S1)",
        "framework": "Krause",
        "mni": (36, -45, 59), "radius_mm": 8,
        "mni_mirror": (-36, -45, 59),
        "contralateralize": True,
        "expected_sign_sp": "-",
        "mask": "unmasked",
    },
    "Contra_Middle_Insula": {
        "label": "Contralateral Middle Insula",
        "framework": "Krause",
        "mni": (32, 4, 11), "radius_mm": 8,
        "mni_mirror": (-32, 4, 11),
        "contralateralize": True,
        "expected_sign_sp": "+",
        "mask": "unmasked",
    },
    "Left_Thalamus": {
        "label": "Left Thalamus",
        "framework": "Krause",
        "mni": (-10, -6, 10), "radius_mm": 4,
        "expected_sign_sp": "+",
        "mask": "unmasked",
    },
    "Left_Anterior_Insula": {
        "label": "Left Anterior Insula",
        "framework": "Krause",
        "mni": (-27, 25, 0), "radius_mm": 8,
        "expected_sign_sp": "+",
        "mask": "unmasked",
    },
    "Left_NAcc": {
        "label": "Left Nucleus Accumbens",
        "framework": "Krause",
        "mni": (-9, 2, -7), "radius_mm": 6,
        "expected_sign_sp": "+",
        "mask": "gm_masked",
    },
    "Right_NAcc": {
        "label": "Right Nucleus Accumbens",
        "framework": "Krause",
        "mni": (9, 2, -7), "radius_mm": 6,
        "expected_sign_sp": "+",
        "mask": "gm_masked",
    },
    "Left_dACC_MCC": {
        "label": "Left dACC/MCC",
        "framework": "Sardi",
        "mni": (-6, 12, 38), "radius_mm": 6,
        "expected_sign_sp": "+",
        "mask": "unmasked",
    },
    "Right_dACC_MCC": {
        "label": "Right dACC/MCC",
        "framework": "Sardi",
        "mni": (6, 12, 38), "radius_mm": 6,
        "expected_sign_sp": "+",
        "mask": "unmasked",
    },
}

# ROI definitions for Figure S4 (display order, with contralateral labels)
FIG_S4_ROIS = {
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

FIG_S4_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00",
    "#984ea3", "#984ea3", "#a65628", "#a65628",
]


# =====================================================================
# ROI extraction
# =====================================================================

def build_spherical_mask(mni_center, radius_mm, affine, shape):
    """Create a boolean 3D mask for a sphere at MNI coordinates."""
    i, j, k = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    ijk = np.column_stack([i.ravel(), j.ravel(), k.ravel(),
                           np.ones(np.prod(shape))])
    mni = (affine @ ijk.T).T[:, :3]
    center = np.array(mni_center, dtype=float)
    dist = np.sqrt(np.sum((mni - center) ** 2, axis=1))
    mask = (dist <= radius_mm).reshape(shape)
    return mask, int(mask.sum())


def load_stim_side_map(verbose=True):
    """Return {ID -> 'left' or 'right'} with the hemisphere CONTRALATERAL to
    the stimulated knee, restricted to the analytic cohort.

    - Analytic cohort = IDs present in step00_extracted_long.csv. This
      drops baseline-only subjects (6) who cannot contribute to any
      coupling analysis.
    - Stimulated side from img_test_site__s1; fall back to
      img_test_site__s2 for subjects missing s1.
    - Values in the source file: 1 = Right knee stimulated
      (contralateral = LEFT hemisphere), 2 = Left knee stimulated
      (contralateral = RIGHT hemisphere).
    """
    long_df = pd.read_csv(LONG_CSV)
    long_df["ID"] = long_df["ID"].astype(str)
    analytic_ids = set(long_df["ID"].unique())

    wide = pd.read_excel(WIDE_XLSX)
    wide["ID"] = wide["ID"].astype(str)
    wide_idx = wide.set_index("ID")

    contra = {}
    n_s1 = n_s2 = n_missing = 0
    for sid in analytic_ids:
        if sid not in wide_idx.index:
            n_missing += 1
            continue
        s1 = wide_idx.loc[sid, "img_test_site__s1"]
        s2 = wide_idx.loc[sid, "img_test_site__s2"]
        if pd.notna(s1):
            ts = int(s1); n_s1 += 1
        elif pd.notna(s2):
            ts = int(s2); n_s2 += 1
        else:
            n_missing += 1
            continue
        contra[sid] = "left" if ts == 1 else "right"
    if verbose:
        print(f"  Stimulation-side lookup: {len(contra)} subjects "
              f"(s1: {n_s1}, s2 fallback: {n_s2}, missing: {n_missing})")
    return contra


def extract_rois(verbose=True):
    """Extract fMRI BOLD in all SP ROIs and save to CSV.

    For ROIs with ``contralateralize=True``, the ROI sphere is placed on
    the hemisphere contralateral to the stimulated knee for each
    subject (x-coordinate mirroring, not image flipping). For all
    other ROIs the fixed atlas coordinate is used.
    """
    import nibabel as nib

    if verbose:
        print(f"  Masked contrasts:   {FMRI_MASKED_DIR}")
        print(f"  Unmasked contrasts: {FMRI_UNMASKED_DIR}")

    ref_dir = FMRI_UNMASKED_DIR if os.path.isdir(FMRI_UNMASKED_DIR) else FMRI_MASKED_DIR
    ref_ids = sorted(os.listdir(ref_dir))
    ref_path = os.path.join(ref_dir, ref_ids[0], "con_0001.nii")
    ref_img = nib.load(ref_path)
    affine = ref_img.affine
    shape = ref_img.shape[:3]

    # Build per-ROI mask(s). Contralateralized ROIs carry both
    # hemispheres so we can pick per subject.
    roi_masks = {}
    for roi_name, cfg in SP_ROIS.items():
        right_mask, n_right = build_spherical_mask(
            cfg["mni"], cfg["radius_mm"], affine, shape
        )
        if cfg.get("contralateralize", False):
            left_mask, n_left = build_spherical_mask(
                cfg["mni_mirror"], cfg["radius_mm"], affine, shape
            )
            roi_masks[roi_name] = {"right": right_mask, "left": left_mask}
            if verbose:
                print(f"    {cfg['label']}: right={cfg['mni']} "
                      f"({n_right} vox), left={cfg['mni_mirror']} "
                      f"({n_left} vox)")
        else:
            roi_masks[roi_name] = right_mask
            if verbose:
                print(f"    {cfg['label']}: MNI={cfg['mni']}, "
                      f"r={cfg['radius_mm']}mm, {n_right} voxels")

    # Stim-side lookup (only needed for contralateralized ROIs).
    side_map = load_stim_side_map(verbose=verbose)

    all_rows = []
    for roi_name, cfg in SP_ROIS.items():
        src_dir = FMRI_MASKED_DIR if cfg["mask"] == "gm_masked" else FMRI_UNMASKED_DIR
        is_contra = cfg.get("contralateralize", False)

        values = {}
        for subj_id in sorted(os.listdir(src_dir)):
            con_path = os.path.join(src_dir, subj_id, "con_0001.nii")
            if not os.path.isfile(con_path):
                continue
            data = nib.load(con_path).get_fdata()
            if is_contra:
                contra_side = side_map.get(subj_id)
                if contra_side is None:
                    # No side info -> cannot contralateralize; skip.
                    continue
                m = roi_masks[roi_name][contra_side]
            else:
                m = roi_masks[roi_name]
            values[subj_id] = float(np.nanmean(data[m]))

        clean = {k: v for k, v in values.items() if np.isfinite(v)}
        vals = np.array(list(clean.values()))
        mean_x, sd_x = vals.mean(), vals.std()

        for subj_id, raw_val in clean.items():
            all_rows.append({
                "ID": subj_id,
                "ROI": roi_name,
                "label": cfg["label"],
                "framework": cfg["framework"],
                "expected_sign_sp": cfg["expected_sign_sp"],
                "mask_type": cfg["mask"],
                "raw_value": raw_val,
                "z_value": (raw_val - mean_x) / sd_x,
                "raw_mean": mean_x,
                "raw_sd": sd_x,
            })

        if verbose:
            print(f"    {cfg['label']}: N={len(clean)}, "
                  f"mean={mean_x:.4f}, SD={sd_x:.4f}")

    roi_df = pd.DataFrame(all_rows)
    roi_df.to_csv(OUT_ROI_CSV, index=False)
    if verbose:
        n_rois = roi_df["ROI"].nunique()
        n_subj = roi_df["ID"].nunique()
        print(f"\n  Saved: {OUT_ROI_CSV}")
        print(f"    {n_rois} ROIs, {n_subj} subjects")


# =====================================================================
# Figure S4 — ROI brain maps
# =====================================================================

def _make_solid_cmap(hex_color):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("solid", [hex_color, hex_color], N=2)


def _make_sphere_img(template, mni_center, radius_mm):
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


def _render_roi_to_image(roi_img, template, cut_coords, title, cmap):
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


def _make_composite(panels, ncols=2):
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


def generate_figure_s4(verbose=True):
    """Generate Figure S4: orthogonal brain slices for all SP ROIs."""
    from nilearn.datasets import load_mni152_template
    from nilearn.image import new_img_like

    if verbose:
        print("\n  Generating Figure S4 (SP ROI brain maps)...")

    template = load_mni152_template(resolution=1)
    if verbose:
        print("    Loaded MNI152 template (1mm)")

    panels = []
    for idx, (roi_key, cfg) in enumerate(FIG_S4_ROIS.items()):
        if verbose:
            print(f"    Building: {cfg['label']}")
        roi_img = _make_sphere_img(template, cfg["mni"], cfg["radius_mm"])
        if "mni_mirror" in cfg:
            mirror = _make_sphere_img(template, cfg["mni_mirror"], cfg["radius_mm"])
            combined = np.clip(roi_img.get_fdata() + mirror.get_fdata(), 0, 1)
            roi_img = new_img_like(template, combined, affine=template.affine)
            x = abs(cfg["mni"][0])
            title = (f"{cfg['label']}  "
                     f"(MNI: \u00b1{x}, {cfg['mni'][1]}, {cfg['mni'][2]}, "
                     f"r = {cfg['radius_mm']} mm)")
        else:
            title = (f"{cfg['label']}  "
                     f"(MNI: {cfg['mni']}, r = {cfg['radius_mm']} mm)")

        cmap = _make_solid_cmap(FIG_S4_COLORS[idx])
        panel = _render_roi_to_image(roi_img, template, cfg["mni"], title, cmap)
        panels.append(panel)

    composite = _make_composite(panels, ncols=2)
    composite.save(OUT_FIG_S4, dpi=(200, 200))
    if verbose:
        print(f"    Saved: {OUT_FIG_S4}")


# =====================================================================
# Main
# =====================================================================


def run_step14(verbose=True, refit=False):
    if verbose:
        print("=" * 70)
        print("STEP 07 — Extract Sleep-to-Pain fMRI ROI values + Figure S4")
        print("=" * 70)

    if not refit and os.path.exists(OUT_ROI_CSV):
        if verbose:
            print("  WARNING: Running in replot mode — loading saved derivatives.")
            print("  If you have changed upstream data or code, re-run with --refit.")
            print(f"  Loading: {OUT_ROI_CSV}")
    else:
        extract_rois(verbose)

    generate_figure_s4(verbose)

    if verbose:
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 07 — extract SP fMRI ROI values + Figure S4."
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--refit", action="store_true",
                        help="Re-extract ROI values from fMRI contrasts")
    args = parser.parse_args()
    run_step14(verbose=not args.quiet, refit=args.refit)


if __name__ == "__main__":
    main()
