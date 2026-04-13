"""
Step 6 — Extract Sleep-to-Pain fMRI ROI values.
======================================================================

Input:  derivatives/step5_fmri_contrasts_masked/   (NAcc ROIs)
        derivatives/step5_fmri_contrasts_unmasked/ (all other ROIs)
Output:
  derivatives/
    step6_sp_roi_values.csv    — per-subject z-scored ROI values

Extracts mean fMRI BOLD contrast (stimulation > baseline) within
7 spherical ROIs for the Sleep-to-Pain moderation analysis:
  - 6 Krause et al. (2019) ROIs: S1, Middle Insula, Thalamus,
    Anterior Insula, Left NAcc, Right NAcc
  - 1 Sardi-motivated ACC ROI: Right dACC/MCC (Xu et al. 2020)

NAcc ROIs use GM-masked contrasts; all others use unmasked
re-estimated contrasts. Values are z-scored across subjects.

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
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step6")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

OUT_ROI_CSV = os.path.join(STEP_DERIV_DIR, "step6_sp_roi_values.csv")

# Contrast image directories from Step 5
FMRI_MASKED_DIR   = os.path.join(DERIV_DIR, "step5_fmri_contrasts_masked")
FMRI_UNMASKED_DIR = os.path.join(DERIV_DIR, "step5_fmri_contrasts_unmasked")

# ROI definitions
SP_ROIS = {
    "Right_S1": {
        "label": "Right Somatosensory Cortex (S1)",
        "framework": "Krause",
        "mni": (36, -45, 59), "radius_mm": 8,
        "expected_sign_sp": "-",
        "mask": "unmasked",
    },
    "Right_Middle_Insula": {
        "label": "Right Middle Insula",
        "framework": "Krause",
        "mni": (32, 4, 11), "radius_mm": 8,
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
    "Right_dACC_MCC": {
        "label": "Right dACC/MCC",
        "framework": "Sardi",
        "mni": (6, 12, 38), "radius_mm": 6,
        "expected_sign_sp": "+",
        "mask": "unmasked",
    },
    "Left_dACC_MCC": {
        "label": "Left dACC/MCC",
        "framework": "Sardi",
        "mni": (-6, 12, 38), "radius_mm": 6,
        "expected_sign_sp": "+",
        "mask": "unmasked",
    },
}


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


def run_step6(verbose: bool = True):
    import nibabel as nib

    if verbose:
        print("=" * 70)
        print("STEP 6 — Extract Sleep-to-Pain fMRI ROI values")
        print("=" * 70)
        print(f"  Masked contrasts:   {FMRI_MASKED_DIR}")
        print(f"  Unmasked contrasts: {FMRI_UNMASKED_DIR}")

    os.makedirs(DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)

    fmri_masked_dir   = FMRI_MASKED_DIR
    fmri_unmasked_dir = FMRI_UNMASKED_DIR

    # Reference image for affine + shape
    ref_dir = fmri_unmasked_dir if os.path.isdir(fmri_unmasked_dir) else fmri_masked_dir
    ref_ids = sorted(os.listdir(ref_dir))
    ref_path = os.path.join(ref_dir, ref_ids[0], "con_0001.nii")
    ref_img = nib.load(ref_path)
    affine = ref_img.affine
    shape = ref_img.shape[:3]

    # Build masks
    roi_masks = {}
    for roi_name, cfg in SP_ROIS.items():
        mask, n_vox = build_spherical_mask(
            cfg["mni"], cfg["radius_mm"], affine, shape
        )
        roi_masks[roi_name] = mask
        if verbose:
            print(f"    {cfg['label']}: MNI={cfg['mni']}, "
                  f"r={cfg['radius_mm']}mm, {n_vox} voxels")

    # Extract per-subject mean contrast
    all_rows = []
    for roi_name, cfg in SP_ROIS.items():
        src_dir = fmri_masked_dir if cfg["mask"] == "gm_masked" else fmri_unmasked_dir
        mask = roi_masks[roi_name]

        values = {}
        for subj_id in sorted(os.listdir(src_dir)):
            con_path = os.path.join(src_dir, subj_id, "con_0001.nii")
            if not os.path.isfile(con_path):
                continue
            data = nib.load(con_path).get_fdata()
            values[subj_id] = float(np.nanmean(data[mask]))

        # Z-score
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
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 6 — extract SP fMRI ROI values."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_step6(verbose=not args.quiet)


if __name__ == "__main__":
    main()
