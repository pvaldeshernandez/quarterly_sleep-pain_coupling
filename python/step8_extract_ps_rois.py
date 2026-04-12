"""
Step 8 — Extract Pain-to-Sleep arousal relay ROI values.
======================================================================

Input:  data/original neuroimaging (spm_nomask/, fmri_contrasts/, vbm/, atlases/)
Output:
  derivatives/
    step8_ps_fmri_roi_values.csv   — per-subject fMRI BOLD z-scored ROI values
    step8_ps_vbm_roi_values.csv    — per-subject VBM GM volume z-scored ROI values

Extracts probability-weighted mean fMRI BOLD and VBM GM volume
from 5 atlas-defined arousal relay ROIs (Lynch et al. 2025):
  PBN, SI-BF/Ch4, CeA, BNST, LH

fMRI: probability-weighted mean BOLD from stimulation > baseline
      contrasts. Most ROIs use unmasked contrasts; LH uses
      GM-masked contrasts.
VBM:  probability-weighted total GM volume from smwc1 images.

Author: Pedro Valdes-Hernandez (with Claude Opus 4.6)
"""
from __future__ import annotations

import argparse
import glob as glob_mod
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DERIV_DIR = os.path.join(ROOT, "derivatives")

DEFAULT_NII_DIR = os.path.join(ROOT, "data")

OUT_FMRI_CSV = os.path.join(DERIV_DIR, "step8_ps_fmri_roi_values.csv")
OUT_VBM_CSV = os.path.join(DERIV_DIR, "step8_ps_vbm_roi_values.csv")

AROUSAL_ROIS = {
    "PBN": {
        "label": "Lateral Parabrachial Nucleus",
        "atlas_type": "label", "labels": [19, 20],
        "expected_sign_ps": "-",
    },
    "SI_BF_Ch4": {
        "label": "Substantia Innominata / Basal Forebrain (Ch4)",
        "atlas_type": "prob",
        "expected_sign_ps": "-",
    },
    "CeA": {
        "label": "Central Nucleus of the Amygdala",
        "atlas_type": "prob",
        "expected_sign_ps": "-",
    },
    "BNST": {
        "label": "Bed Nucleus of the Stria Terminalis",
        "atlas_type": "prob",
        "expected_sign_ps": "-",
    },
    "LH": {
        "label": "Lateral Hypothalamus",
        "atlas_type": "label", "labels": [25, 26],
        "expected_sign_ps": "-",
        "fmri_mask": "gm_masked",
    },
}

ATLAS_FILES = {
    "PBN": "atlas_b2_brainstem.nii.gz",
    "SI_BF_Ch4": os.path.join("zaborszky_bf", "Ch4_basal_forebrain_prob_MNI152.nii.gz"),
    "CeA": "CIT168_CeA_prob_bilat_MNI152_1mm.nii.gz",
    "BNST": "Blackford_BNST_3T.nii.gz",
    "LH": os.path.join("hypothalamus_neudorfer2020", "atlas_labels_0.5mm.nii.gz"),
}


def extract_fmri_arousal(nii_dir, verbose=True):
    """Extract probability-weighted mean BOLD per subject per ROI."""
    import nibabel as nib
    from nilearn.image import resample_to_img

    atlas_dir = os.path.join(nii_dir, "atlases")
    fmri_masked_dir = os.path.join(nii_dir, "fmri_contrasts")
    fmri_unmasked_dir = os.path.join(nii_dir, "spm_nomask")
    default_fmri_dir = fmri_unmasked_dir if os.path.isdir(fmri_unmasked_dir) else fmri_masked_dir

    ref_ids = sorted(os.listdir(default_fmri_dir))
    ref_path = os.path.join(default_fmri_dir, ref_ids[0], "con_0001.nii")
    ref_img = nib.load(ref_path)
    brain_mask = np.isfinite(ref_img.get_fdata())

    # Build weight maps
    roi_weights = {}
    for roi_name, cfg in AROUSAL_ROIS.items():
        atlas_path = os.path.join(atlas_dir, ATLAS_FILES[roi_name])
        if not os.path.isfile(atlas_path):
            if verbose:
                print(f"    {roi_name}: atlas not found at {atlas_path}, skipping")
            continue

        atlas_img = nib.load(atlas_path)
        atlas_data = atlas_img.get_fdata()

        if cfg["atlas_type"] == "label":
            mask = np.zeros_like(atlas_data, dtype=np.float32)
            for lab in cfg["labels"]:
                mask[atlas_data == lab] = 1.0
            mask_img = nib.Nifti1Image(mask, atlas_img.affine)
            resampled = resample_to_img(mask_img, ref_img, interpolation="linear")
            weights = resampled.get_fdata().clip(0, 1)
        else:
            resampled = resample_to_img(atlas_img, ref_img, interpolation="linear")
            weights = resampled.get_fdata().clip(0, 1)

        n_brain = int(np.sum((weights * brain_mask) > 0))
        if verbose:
            print(f"    {cfg['label']}: {n_brain} brain voxels at fMRI resolution")
        if n_brain < 1:
            if verbose:
                print(f"      WARNING: 0 brain voxels, skipping")
            continue
        roi_weights[roi_name] = weights

    # Extract per-subject
    all_rows = []
    for subj_id in sorted(os.listdir(default_fmri_dir)):
        con_path = os.path.join(default_fmri_dir, subj_id, "con_0001.nii")
        if not os.path.isfile(con_path):
            continue
        data = nib.load(con_path).get_fdata()
        for roi_name, weights in roi_weights.items():
            valid = np.isfinite(data)
            w = weights * valid
            wsum = w.sum()
            if wsum > 0:
                all_rows.append({
                    "ID": subj_id, "ROI": roi_name,
                    "label": AROUSAL_ROIS[roi_name]["label"],
                    "modality": "fMRI_BOLD",
                    "expected_sign_ps": AROUSAL_ROIS[roi_name]["expected_sign_ps"],
                    "raw_value": float(np.nansum(data * w) / wsum),
                })

    df = pd.DataFrame(all_rows)

    # Z-score per ROI
    for roi_name in df["ROI"].unique():
        mask = df["ROI"] == roi_name
        vals = df.loc[mask, "raw_value"]
        mean_x, sd_x = vals.mean(), vals.std()
        df.loc[mask, "z_value"] = (vals - mean_x) / sd_x
        df.loc[mask, "raw_mean"] = mean_x
        df.loc[mask, "raw_sd"] = sd_x
        if verbose:
            print(f"    {roi_name} fMRI: N={mask.sum()}, mean={mean_x:.4f}, SD={sd_x:.4f}")

    return df


def extract_vbm_arousal(nii_dir, verbose=True):
    """Extract probability-weighted GM volume per subject per ROI."""
    import nibabel as nib
    from nilearn.image import resample_to_img

    atlas_dir = os.path.join(nii_dir, "atlases")
    vbm_dir = os.path.join(nii_dir, "vbm")

    pattern = os.path.join(vbm_dir, "smwc1*_ses-01_T1w.nii")
    gm_files = sorted(glob_mod.glob(pattern))
    if not gm_files:
        if verbose:
            print(f"  No smwc1 files found in {vbm_dir}")
        return pd.DataFrame()

    ref_img = nib.load(gm_files[0])
    vox_vol = float(np.abs(np.linalg.det(ref_img.affine[:3, :3])))

    # Build weight maps at VBM resolution
    roi_weights = {}
    for roi_name, cfg in AROUSAL_ROIS.items():
        atlas_path = os.path.join(atlas_dir, ATLAS_FILES[roi_name])
        if not os.path.isfile(atlas_path):
            continue

        atlas_img = nib.load(atlas_path)
        atlas_data = atlas_img.get_fdata()

        if cfg["atlas_type"] == "label":
            mask = np.zeros_like(atlas_data, dtype=np.float32)
            for lab in cfg["labels"]:
                mask[atlas_data == lab] = 1.0
            mask_img = nib.Nifti1Image(mask, atlas_img.affine)
            resampled = resample_to_img(mask_img, ref_img, interpolation="linear")
            weights = resampled.get_fdata().clip(0, 1)
        else:
            resampled = resample_to_img(atlas_img, ref_img, interpolation="linear")
            weights = resampled.get_fdata().clip(0, 1)

        n_vox = int(np.sum(weights > 0))
        if verbose:
            print(f"    {cfg['label']}: {n_vox} voxels at VBM resolution")
        roi_weights[roi_name] = weights

    # Extract per-subject
    all_rows = []
    for gm_path in gm_files:
        fname = os.path.basename(gm_path)
        # Parse subject ID from smwc1<subj_id>_ses-01_T1w.nii
        # VBM filenames use 'x' instead of '-' in subject IDs
        # (BIDS-safe convention), e.g. "2014x2" in VBM corresponds
        # to "2014-2" in the fMRI dirs and the quarterly data table.
        subj_id = fname.replace("smwc1", "").replace("_ses-01_T1w.nii", "")
        subj_id = subj_id.replace("x", "-")
        data = nib.load(gm_path).get_fdata()

        for roi_name, weights in roi_weights.items():
            vol = float(np.nansum(data * weights) * vox_vol)
            all_rows.append({
                "ID": subj_id, "ROI": roi_name,
                "label": AROUSAL_ROIS[roi_name]["label"],
                "modality": "VBM_GM_volume",
                "expected_sign_ps": AROUSAL_ROIS[roi_name]["expected_sign_ps"],
                "raw_value": vol,
            })

    df = pd.DataFrame(all_rows)

    # Z-score per ROI
    for roi_name in df["ROI"].unique():
        mask = df["ROI"] == roi_name
        vals = df.loc[mask, "raw_value"]
        mean_x, sd_x = vals.mean(), vals.std()
        df.loc[mask, "z_value"] = (vals - mean_x) / sd_x
        df.loc[mask, "raw_mean"] = mean_x
        df.loc[mask, "raw_sd"] = sd_x
        if verbose:
            print(f"    {roi_name} VBM: N={mask.sum()}, mean={mean_x:.2f}, SD={sd_x:.2f}")

    return df


def run_step8(nii_dir=None, verbose=True):
    if nii_dir is None:
        nii_dir = DEFAULT_NII_DIR

    if verbose:
        print("=" * 70)
        print("STEP 8 — Extract Pain-to-Sleep arousal relay ROI values")
        print("=" * 70)
        print(f"  NIfTI dir: {nii_dir}")

    os.makedirs(DERIV_DIR, exist_ok=True)

    # fMRI BOLD
    if verbose:
        print("\n  fMRI BOLD extraction:")
    fmri_df = extract_fmri_arousal(nii_dir, verbose)
    fmri_df.to_csv(OUT_FMRI_CSV, index=False)
    if verbose:
        print(f"  Saved: {OUT_FMRI_CSV} ({fmri_df['ROI'].nunique()} ROIs, "
              f"{fmri_df['ID'].nunique()} subjects)")

    # VBM GM volume
    if verbose:
        print("\n  VBM GM volume extraction:")
    vbm_df = extract_vbm_arousal(nii_dir, verbose)
    vbm_df.to_csv(OUT_VBM_CSV, index=False)
    if verbose:
        print(f"  Saved: {OUT_VBM_CSV} ({vbm_df['ROI'].nunique()} ROIs, "
              f"{vbm_df['ID'].nunique()} subjects)")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 8 — extract PS arousal relay ROI values."
    )
    parser.add_argument("--nii-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_step8(nii_dir=args.nii_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
