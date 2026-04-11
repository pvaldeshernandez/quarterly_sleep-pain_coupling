#!/usr/bin/env python3
"""
Moderator Variable Loaders for Neuroimaging ROI Analyses
========================================================

This module provides functions to load brain-based moderator variables
for cross-level moderation analysis of sleep-pain coupling. Each function
returns a dictionary mapping subject IDs to z-scored moderator values,
ready for use with ``coupling_model.fit_bayesian_varx1(X_person=...)``.

The paper tests four families of neuroimaging moderators:

  1. **Krause fMRI ROIs** (sleep-to-pain direction):
     Six regions from Krause et al. (2019) J Neurosci that mediate
     the relationship between sleep deprivation and pain sensitivity.
     Spherical ROIs at published MNI coordinates, applied to first-level
     fMRI contrast images (painful stimulation > baseline).

  2. **ACC ROI** (sleep-to-pain direction):
     Right dorsal ACC/MCC from Xu et al. (2020) meta-analysis. Motivated
     by Sardi et al. (2018) showing ACC and NAcc are parallel D2-gated
     nodes whose activation prevents sleep-restriction hyperalgesia.

  3. **Lynch fMRI atlas arousal ROIs** (pain-to-sleep direction):
     Five subcortical regions in the PBelCGRP-to-forebrain arousal pathway
     from Lynch et al. (2025) Advanced Science, using published probabilistic
     atlases applied to fMRI contrast images.

  4. **Lynch VBM atlas arousal ROIs** (pain-to-sleep direction):
     Same five atlas-defined regions applied to modulated gray matter
     images (smwc1 from VBM/CAT12), testing whether structural volume
     in arousal-relay nuclei moderates pain-to-sleep coupling.

Masking Convention
------------------
  - NAcc and LH: GM-masked contrast images (original SPM first-level)
  - All other fMRI ROIs: Unmasked contrast images (re-estimated without
    individual GM masks to avoid NaN in subcortical voxels)
  - VBM ROIs: Smoothed modulated gray matter images (smwc1)

Synthetic Data Mode
-------------------
When ``synthetic=True``, each function loads from pre-generated CSV files
in ``data/synthetic/`` instead of neuroimaging files. This allows running
the full analysis pipeline without access to the original brain images.

Author: Pedro Valdes-Hernandez
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ===================================================================
# Path resolution
# ===================================================================
#
# Raw neuroimaging directories (fmri_contrasts, spm_nomask, vbm,
# atlases) can live outside the ``data/`` directory that the user
# passes via ``--data-dir``. This is common in sandbox runs where
# only the processed CSV is redirected into a fresh folder while the
# large image directories stay in the repo's default data/. The
# helper below looks for a subdirectory first in the user-provided
# data_dir, then falls back to the default ``{repo}/data/``.

_DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "data",
    )
)


def _resolve_raw_data_path(data_dir, subdir):
    """Return ``{data_dir}/{subdir}`` if it exists, else the default."""
    primary = os.path.join(data_dir, subdir)
    if os.path.isdir(primary):
        return primary
    fallback = os.path.join(_DEFAULT_DATA_DIR, subdir)
    return fallback


# ===================================================================
# ROI Definitions
# ===================================================================

# --- Krause et al. (2019) J Neurosci 39:2291-2300 ---
# Regions mediating sleep deprivation -> pain sensitivity.
# MNI coordinates and radii from Table 1 of the original paper.
#
# Expected sign of gamma_sp:
#   S1:              negative
#   Middle insula:   positive
#   Thalamus:        positive
#   Anterior insula: positive
#   Left NAcc:       positive
#   Right NAcc:      positive
# All six estimated gamma_sp values matching this pattern gives the
# manuscript's p = (1/2)^6 = 0.016 sign test.
KRAUSE_ROIS = {
    "Right_S1": {
        "label": "Right Somatosensory Cortex (S1)",
        "mni": (36, -45, 59),
        "radius_mm": 8,
        "expected_sign_sp": "-",
        "mask": "unmasked",
    },
    "Right_Middle_Insula": {
        "label": "Right Middle Insula",
        "mni": (32, 4, 11),
        "radius_mm": 8,
        "expected_sign_sp": "+",
        "mask": "unmasked",
    },
    "Left_Thalamus": {
        "label": "Left Thalamus",
        "mni": (-10, -6, 10),
        "radius_mm": 4,
        "expected_sign_sp": "+",
        "mask": "unmasked",
    },
    "Left_Anterior_Insula": {
        "label": "Left Anterior Insula",
        "mni": (-27, 25, 0),
        "radius_mm": 8,
        "expected_sign_sp": "+",
        "mask": "unmasked",
    },
    "Left_NAcc": {
        "label": "Left Nucleus Accumbens",
        "mni": (-9, 2, -7),
        "radius_mm": 6,
        "expected_sign_sp": "+",
        "mask": "gm_masked",
        "note": "Key finding: gamma_sp=+0.040, p=0.027",
    },
    "Right_NAcc": {
        "label": "Right Nucleus Accumbens",
        "mni": (9, 2, -7),
        "radius_mm": 6,
        "expected_sign_sp": "+",
        "mask": "gm_masked",
        "note": "Same direction as left, but null (p=0.194)",
    },
}

# --- Xu et al. (2020) Neurosci Biobehav Rev 112:300-323 ---
# Right dorsal ACC/MCC from pain meta-analysis. Tested as SP moderator
# based on Sardi et al. (2018) showing ACC and NAcc are parallel D2-gated
# nodes in the sleep-pain pathway.
ACC_ROI = {
    "label": "Right dACC/MCC",
    "mni": (6, 12, 38),
    "radius_mm": 6,
    "expected_sign_sp": "+",
    "mask": "unmasked",
    "source": "Xu et al. 2020 meta-analysis",
    "rationale": "Sardi et al. 2018: ACC parallels NAcc as D2-gated node",
}

# --- Lynch et al. (2025) Advanced Science ---
# PBelCGRP-to-forebrain arousal pathway. These subcortical nuclei relay
# nociceptive signals that disrupt sleep via ascending arousal.
# Expected sign of gamma_ps: negative (greater volume/activation in
# arousal-promoting regions should amplify pain-to-sleep disruption,
# i.e., more negative coupling).

# Probabilistic atlas definitions for each ROI:
ATLAS_AROUSAL_ROIS = {
    "PBN": {
        "label": "Lateral Parabrachial Nucleus",
        "atlas_source": "Singh et al. 2020, EagleVAC brainstem atlas",
        "atlas_type": "label",
        "labels": [19, 20],  # LPB left (19), LPB right (20)
        "expected_sign_ps": "-",
        "note": "Gateway from spinal nociceptive input to forebrain arousal",
    },
    "SI_BF_Ch4": {
        "label": "Substantia Innominata / Basal Forebrain (Ch4)",
        "atlas_source": "Zaborszky et al. 2008",
        "atlas_type": "prob",
        "expected_sign_ps": "-",
        "note": "Cholinergic arousal center receiving PBN projections",
    },
    "CeA": {
        "label": "Central Nucleus of the Amygdala",
        "atlas_source": "CIT168 (Pauli et al. 2018)",
        "atlas_type": "prob",
        "expected_sign_ps": "-",
        "note": "Integrates nociceptive and affective signals",
    },
    "BNST": {
        "label": "Bed Nucleus of the Stria Terminalis",
        "atlas_source": "Theiss et al. 2017, 3T probabilistic atlas",
        "atlas_type": "prob",
        "expected_sign_ps": "-",
        "note": "Extended amygdala; sustained threat/arousal processing",
    },
    "LH": {
        "label": "Lateral Hypothalamus",
        "atlas_source": "Neudorfer et al. 2020 hypothalamic atlas",
        "atlas_type": "label",
        "labels": [25, 26],  # lateral hypothalamus right (25), left (26)
        "expected_sign_ps": "-",
        "mask": "gm_masked",
        "note": "Orexin/hypocretin neurons driving wakefulness",
    },
}


# ===================================================================
# Helper Functions
# ===================================================================

def _build_spherical_mask(mni_center, radius_mm, affine, shape):
    """Create a boolean 3D mask for a sphere at given MNI coordinates.

    Parameters
    ----------
    mni_center : tuple (x, y, z)
        Center of the sphere in MNI space (mm).
    radius_mm : float
        Radius of the sphere in mm.
    affine : ndarray (4, 4)
        Voxel-to-world affine transformation matrix.
    shape : tuple (nx, ny, nz)
        3D image dimensions.

    Returns
    -------
    mask : ndarray, shape (*shape), dtype bool
        True for voxels within the sphere.
    n_voxels : int
        Number of voxels in the mask.
    """
    # Build a coordinate grid in world (MNI) space
    i_coords, j_coords, k_coords = np.mgrid[
        0:shape[0], 0:shape[1], 0:shape[2]
    ]
    ijk = np.column_stack([
        i_coords.ravel(),
        j_coords.ravel(),
        k_coords.ravel(),
        np.ones(np.prod(shape)),
    ])
    mni_coords = (affine @ ijk.T).T[:, :3]

    # Euclidean distance from sphere center
    center = np.array(mni_center, dtype=float)
    distances = np.sqrt(np.sum((mni_coords - center) ** 2, axis=1))
    mask_flat = distances <= radius_mm
    mask_3d = mask_flat.reshape(shape)

    return mask_3d, int(mask_flat.sum())


def _zscore_dict(values_dict, min_n=20):
    """Z-score a {subject_ID: value} dictionary.

    Parameters
    ----------
    values_dict : dict
        Raw values keyed by subject ID.
    min_n : int
        Minimum number of finite values required; returns None if fewer.

    Returns
    -------
    z_dict : dict or None
        Z-scored values, or None if insufficient data.
    raw_mean : float
        Mean of raw values (for back-transformation).
    raw_sd : float
        SD of raw values (for back-transformation).
    """
    # Drop non-finite values
    clean = {k: v for k, v in values_dict.items() if np.isfinite(v)}
    if len(clean) < min_n:
        return None, np.nan, np.nan

    vals = np.array(list(clean.values()))
    mean_x = vals.mean()
    sd_x = vals.std()
    if sd_x == 0:
        return None, mean_x, 0.0

    z_dict = {k: (v - mean_x) / sd_x for k, v in clean.items()}
    return z_dict, mean_x, sd_x


# ===================================================================
# Krause fMRI ROI Loaders
# ===================================================================

def load_fmri_krause_rois(data_dir, synthetic=False):
    """Load fMRI BOLD activation in Krause et al. (2019) pain ROIs.

    Extracts mean contrast values from first-level SPM contrasts
    (painful knee stimulation > baseline) within spherical ROIs at
    the published MNI coordinates.

    For the paper, these moderate the sleep-to-pain coupling direction
    (gamma_sp). All six ROIs showed sign concordance with predictions
    from Krause et al. (sign test p = 1/64 = 0.016).

    Parameters
    ----------
    data_dir : str
        Path to the ``data/`` directory.
    synthetic : bool, default False
        If True, load from ``data/synthetic/roi_values.csv`` instead
        of neuroimaging files.

    Returns
    -------
    moderators : dict
        {roi_name: {subject_ID: z_scored_value, ...}, ...}
        for all six Krause ROIs.
    labels : dict
        {roi_name: human-readable label string}
    raw_stats : dict
        {roi_name: {"mean": float, "sd": float}} for back-transformation.
    """
    if synthetic:
        return _load_synthetic_roi_moderators(
            data_dir, roi_names=list(KRAUSE_ROIS.keys()),
            label_dict={k: v["label"] for k, v in KRAUSE_ROIS.items()},
        )

    # Real data: requires nibabel for NIfTI I/O
    import nibabel as nib

    moderators = {}
    labels = {}
    raw_stats = {}

    # Determine paths based on masking convention
    # NAcc uses GM-masked images; others use unmasked re-estimated images
    fmri_masked_dir = _resolve_raw_data_path(data_dir, "fmri_contrasts")
    fmri_unmasked_dir = _resolve_raw_data_path(data_dir, "spm_nomask")

    # Load a reference image to get the affine and shape
    # (any subject's contrast image will do)
    ref_dir = fmri_masked_dir if os.path.isdir(fmri_masked_dir) else fmri_unmasked_dir
    ref_ids = sorted(os.listdir(ref_dir))
    ref_path = os.path.join(ref_dir, ref_ids[0], "con_0001.nii")
    ref_img = nib.load(ref_path)
    affine = ref_img.affine
    shape = ref_img.shape[:3]

    # Build spherical masks for each ROI
    roi_masks = {}
    for roi_name, roi_cfg in KRAUSE_ROIS.items():
        mask, n_vox = _build_spherical_mask(
            roi_cfg["mni"], roi_cfg["radius_mm"], affine, shape
        )
        roi_masks[roi_name] = mask
        print(
            f"    {roi_cfg['label']}: MNI={roi_cfg['mni']}, "
            f"r={roi_cfg['radius_mm']}mm, {n_vox} voxels"
        )

    # Extract mean contrast per subject per ROI
    for roi_name, roi_cfg in KRAUSE_ROIS.items():
        # Choose masked or unmasked directory based on ROI convention
        if roi_cfg.get("mask") == "gm_masked":
            src_dir = fmri_masked_dir
        else:
            src_dir = fmri_unmasked_dir

        mask = roi_masks[roi_name]
        values = {}
        for subj_id in sorted(os.listdir(src_dir)):
            con_path = os.path.join(src_dir, subj_id, "con_0001.nii")
            if not os.path.isfile(con_path):
                continue
            img = nib.load(con_path)
            data = img.get_fdata()
            values[subj_id] = float(np.nanmean(data[mask]))

        z_dict, raw_mean, raw_sd = _zscore_dict(values)
        if z_dict is not None:
            moderators[roi_name] = z_dict
            labels[roi_name] = roi_cfg["label"]
            raw_stats[roi_name] = {"mean": raw_mean, "sd": raw_sd}

    print(f"    {len(moderators)} ROI moderators loaded")
    return moderators, labels, raw_stats


# ===================================================================
# ACC ROI Loader
# ===================================================================

def load_acc_roi(data_dir, synthetic=False):
    """Load fMRI BOLD activation in the right dACC/MCC ROI.

    Spherical ROI at MNI (6, 12, 38), radius 6mm, from Xu et al. (2020)
    pain meta-analysis. Uses unmasked contrast images.

    Motivated by Sardi et al. (2018): ACC and NAcc are parallel D2-gated
    nodes whose activation prevents sleep-restriction hyperalgesia. Since
    NAcc moderates sleep-to-pain coupling, ACC should too.

    Parameters
    ----------
    data_dir : str
        Path to the ``data/`` directory.
    synthetic : bool, default False
        If True, load from ``data/synthetic/roi_values.csv``.

    Returns
    -------
    moderator : dict
        {subject_ID: z_scored_ACC_value, ...}
    label : str
        Human-readable ROI label.
    raw_stats : dict
        {"mean": float, "sd": float} for back-transformation.
    """
    if synthetic:
        mods, labs, stats = _load_synthetic_roi_moderators(
            data_dir, roi_names=["ACC"],
            label_dict={"ACC": ACC_ROI["label"]},
        )
        return mods.get("ACC", {}), labs.get("ACC", ""), stats.get("ACC", {})

    # Real data
    import nibabel as nib

    fmri_dir = _resolve_raw_data_path(data_dir, "spm_nomask")  # unmasked
    ref_ids = sorted(os.listdir(fmri_dir))
    ref_path = os.path.join(fmri_dir, ref_ids[0], "con_0001.nii")
    ref_img = nib.load(ref_path)
    affine = ref_img.affine
    shape = ref_img.shape[:3]

    # Build the spherical mask
    mask, n_vox = _build_spherical_mask(
        ACC_ROI["mni"], ACC_ROI["radius_mm"], affine, shape
    )
    print(
        f"    {ACC_ROI['label']}: MNI={ACC_ROI['mni']}, "
        f"r={ACC_ROI['radius_mm']}mm, {n_vox} voxels"
    )

    # Extract mean contrast per subject
    values = {}
    for subj_id in sorted(os.listdir(fmri_dir)):
        con_path = os.path.join(fmri_dir, subj_id, "con_0001.nii")
        if not os.path.isfile(con_path):
            continue
        img = nib.load(con_path)
        data = img.get_fdata()
        values[subj_id] = float(np.nanmean(data[mask]))

    z_dict, raw_mean, raw_sd = _zscore_dict(values)
    return (
        z_dict if z_dict is not None else {},
        ACC_ROI["label"],
        {"mean": raw_mean, "sd": raw_sd},
    )


# ===================================================================
# Lynch Atlas Arousal ROI Loaders (fMRI)
# ===================================================================

def load_fmri_atlas_arousal(data_dir, synthetic=False):
    """Load fMRI BOLD in arousal pathway ROIs using probabilistic atlases.

    Uses published probabilistic or label atlases resampled to fMRI
    resolution. Extracts probability-weighted mean BOLD contrast
    (stimulation > baseline) per ROI.

    For the paper, these moderate the pain-to-sleep coupling direction
    (gamma_ps). Expected sign: negative (greater arousal-relay activation
    should amplify pain-induced sleep disruption).

    ROI atlases:
      PBN:     Singh et al. (2020) EagleVAC brainstem atlas
      SI-BF:   Zaborszky et al. (2008) Ch4 basal forebrain
      CeA:     CIT168 (Pauli et al. 2018) crowd-sourced probabilistic
      BNST:    Theiss et al. (2017) 3T probabilistic
      LH:      Neudorfer et al. (2020) hypothalamic atlas

    Parameters
    ----------
    data_dir : str
        Path to the ``data/`` directory. Expects atlas files under
        ``data/atlases/`` and fMRI contrast images under
        ``data/fmri_contrasts/`` or ``data/spm_nomask/``.
    synthetic : bool, default False
        If True, load from ``data/synthetic/roi_values.csv``.

    Returns
    -------
    moderators : dict
        {roi_name: {subject_ID: z_scored_value, ...}, ...}
    labels : dict
        {roi_name: human-readable label string}
    raw_stats : dict
        {roi_name: {"mean": float, "sd": float}}
    """
    if synthetic:
        return _load_synthetic_roi_moderators(
            data_dir, roi_names=list(ATLAS_AROUSAL_ROIS.keys()),
            label_dict={
                k: v["label"] + " (atlas BOLD)"
                for k, v in ATLAS_AROUSAL_ROIS.items()
            },
        )

    # Real data: requires nibabel and nilearn
    import nibabel as nib
    from nilearn.image import resample_to_img

    atlas_dir = _resolve_raw_data_path(data_dir, "atlases")

    # Determine fMRI source directory based on masking convention
    # Most ROIs use unmasked; LH uses GM-masked
    fmri_masked_dir = _resolve_raw_data_path(data_dir, "fmri_contrasts")
    fmri_unmasked_dir = _resolve_raw_data_path(data_dir, "spm_nomask")
    default_fmri_dir = (
        fmri_unmasked_dir if os.path.isdir(fmri_unmasked_dir)
        else fmri_masked_dir
    )

    # Load a reference image for target space
    ref_ids = sorted(os.listdir(default_fmri_dir))
    ref_path = os.path.join(default_fmri_dir, ref_ids[0], "con_0001.nii")
    ref_img = nib.load(ref_path)
    ref_data = ref_img.get_fdata()
    brain_mask = np.isfinite(ref_data)

    # Atlas file paths -- these must exist under data/atlases/
    atlas_files = {
        "PBN": os.path.join(atlas_dir, "atlas_b2_brainstem.nii.gz"),
        "SI_BF_Ch4": os.path.join(
            atlas_dir, "zaborszky_bf",
            "Ch4_basal_forebrain_prob_MNI152.nii.gz",
        ),
        "CeA": os.path.join(
            atlas_dir, "CIT168_CeA_prob_bilat_MNI152_1mm.nii.gz"
        ),
        "BNST": os.path.join(atlas_dir, "Blackford_BNST_3T.nii.gz"),
        "LH": os.path.join(
            atlas_dir, "hypothalamus_neudorfer2020",
            "atlas_labels_0.5mm.nii.gz",
        ),
    }

    # Resample each atlas to fMRI resolution and build weight maps
    roi_weights = {}
    print("  fMRI Atlas Arousal ROIs:")
    for roi_name, roi_cfg in ATLAS_AROUSAL_ROIS.items():
        atlas_path = atlas_files.get(roi_name)
        if atlas_path is None or not os.path.isfile(atlas_path):
            print(f"    {roi_name}: atlas not found, skipping")
            continue

        atlas_img = nib.load(atlas_path)
        atlas_data = atlas_img.get_fdata()

        if roi_cfg["atlas_type"] == "label":
            # Convert label indices to a binary mask, then resample
            mask = np.zeros_like(atlas_data, dtype=np.float32)
            for lab in roi_cfg["labels"]:
                mask[atlas_data == lab] = 1.0
            mask_img = nib.Nifti1Image(mask, atlas_img.affine)
            resampled = resample_to_img(
                mask_img, ref_img, interpolation="linear"
            )
            weights = resampled.get_fdata().clip(0, 1)
        else:
            # Probabilistic atlas -- resample directly
            resampled = resample_to_img(
                atlas_img, ref_img, interpolation="linear"
            )
            weights = resampled.get_fdata().clip(0, 1)

        # Check brain coverage
        weights_in_brain = weights * brain_mask
        n_brain = int(np.sum(weights_in_brain > 0))
        print(
            f"    {roi_cfg['label']}: {n_brain} voxels in brain mask"
        )

        if n_brain < 1:
            print(f"      WARNING: 0 brain voxels, skipping")
            continue

        roi_weights[roi_name] = weights

    # Extract probability-weighted mean BOLD per subject per ROI
    moderators = {}
    labels = {}
    raw_stats = {}

    # Collect values for all subjects
    roi_values = {roi_name: {} for roi_name in roi_weights}
    for subj_id in sorted(os.listdir(default_fmri_dir)):
        con_path = os.path.join(default_fmri_dir, subj_id, "con_0001.nii")
        if not os.path.isfile(con_path):
            continue
        img = nib.load(con_path)
        data = img.get_fdata()
        for roi_name, weights in roi_weights.items():
            valid = np.isfinite(data)
            w = weights * valid
            wsum = w.sum()
            if wsum > 0:
                roi_values[roi_name][subj_id] = float(
                    np.nansum(data * w) / wsum
                )

    # Z-score each ROI
    for roi_name in roi_weights:
        roi_cfg = ATLAS_AROUSAL_ROIS[roi_name]
        z_dict, raw_mean, raw_sd = _zscore_dict(roi_values[roi_name])
        if z_dict is not None:
            moderators[roi_name] = z_dict
            labels[roi_name] = roi_cfg["label"] + " (atlas BOLD)"
            raw_stats[roi_name] = {"mean": raw_mean, "sd": raw_sd}

    print(f"    {len(moderators)} ROI moderators loaded")
    return moderators, labels, raw_stats


# ===================================================================
# Lynch Atlas Arousal ROI Loaders (VBM)
# ===================================================================

def load_vbm_atlas_arousal(data_dir, synthetic=False):
    """Load gray matter volume in arousal pathway ROIs using atlases.

    Uses the same probabilistic/label atlases as the fMRI version, but
    applied to smoothed modulated gray matter images (smwc1) from the
    VBM/CAT12 pipeline. Because VBM images have higher spatial resolution
    (~1.5mm) than fMRI (~3mm), all five ROIs are viable including PBN.

    For probabilistic atlases: total GM volume = sum(smwc1 * prob * vox_vol)
    For label atlases: total GM volume = sum(smwc1 * vox_vol) within labels.
    Both give the integral of modulated GM in the ROI = total tissue volume.

    Parameters
    ----------
    data_dir : str
        Path to the ``data/`` directory. Expects atlas files under
        ``data/atlases/`` and VBM images under ``data/vbm/``.
    synthetic : bool, default False
        If True, load from ``data/synthetic/vbm_volumes.csv``.

    Returns
    -------
    moderators : dict
        {roi_name: {subject_ID: z_scored_value, ...}, ...}
    labels : dict
        {roi_name: human-readable label string}
    raw_stats : dict
        {roi_name: {"mean": float, "sd": float}} in mm^3.
    """
    if synthetic:
        return _load_synthetic_vbm_moderators(data_dir)

    # Real data: requires nibabel, nilearn, and glob
    import nibabel as nib
    import glob as glob_mod
    from nilearn.image import resample_to_img

    atlas_dir = _resolve_raw_data_path(data_dir, "atlases")
    vbm_dir = _resolve_raw_data_path(data_dir, "vbm")

    # Find all smoothed modulated GM images
    pattern = os.path.join(vbm_dir, "smwc1*_ses-01_T1w.nii")
    gm_files = sorted(glob_mod.glob(pattern))
    if not gm_files:
        print(f"  No smwc1 files found in {vbm_dir}")
        return {}, {}, {}

    # Load reference for target space
    ref_img = nib.load(gm_files[0])
    ref_affine = ref_img.affine
    vox_vol = float(np.abs(np.linalg.det(ref_affine[:3, :3])))  # mm^3

    # Atlas file paths
    atlas_files = {
        "PBN": os.path.join(atlas_dir, "atlas_b2_brainstem.nii.gz"),
        "SI_BF_Ch4": os.path.join(
            atlas_dir, "zaborszky_bf",
            "Ch4_basal_forebrain_prob_MNI152.nii.gz",
        ),
        "CeA": os.path.join(
            atlas_dir, "CIT168_CeA_prob_bilat_MNI152_1mm.nii.gz"
        ),
        "BNST": os.path.join(atlas_dir, "Blackford_BNST_3T.nii.gz"),
        "LH": os.path.join(
            atlas_dir, "hypothalamus_neudorfer2020",
            "atlas_labels_0.5mm.nii.gz",
        ),
    }

    # Resample each atlas to VBM resolution
    roi_weights = {}
    print("  VBM Atlas Arousal ROIs:")
    for roi_name, roi_cfg in ATLAS_AROUSAL_ROIS.items():
        atlas_path = atlas_files.get(roi_name)
        if atlas_path is None or not os.path.isfile(atlas_path):
            print(f"    {roi_name}: atlas not found, skipping")
            continue

        atlas_img = nib.load(atlas_path)
        atlas_data = atlas_img.get_fdata()

        if roi_cfg["atlas_type"] == "label":
            mask = np.zeros_like(atlas_data, dtype=np.float32)
            for lab in roi_cfg["labels"]:
                mask[atlas_data == lab] = 1.0
            mask_img = nib.Nifti1Image(mask, atlas_img.affine)
            resampled = resample_to_img(
                mask_img, ref_img, interpolation="linear"
            )
            weights = resampled.get_fdata().clip(0, 1)
        else:
            resampled = resample_to_img(
                atlas_img, ref_img, interpolation="linear"
            )
            weights = resampled.get_fdata().clip(0, 1)

        n_nonzero = int(np.sum(weights > 0))
        print(f"    {roi_cfg['label']}: {n_nonzero} voxels > 0")

        if n_nonzero < 1:
            continue
        roi_weights[roi_name] = weights

    # Extract probability-weighted GM integral per subject per ROI
    roi_values = {roi_name: {} for roi_name in roi_weights}
    for gm_path in gm_files:
        fname = os.path.basename(gm_path)
        # Parse subject ID from filename:
        #   smwc11001_ses-01_T1w.nii -> 1001
        #   smwc1836x500_ses-01_T1w.nii -> 836-500
        subj_id = fname.replace("smwc1", "").replace("_ses-01_T1w.nii", "")
        subj_id = subj_id.replace("x", "-")

        img = nib.load(gm_path)
        data = img.get_fdata()
        for roi_name, weights in roi_weights.items():
            gm_integral = float(np.nansum(data * weights) * vox_vol)
            roi_values[roi_name][subj_id] = gm_integral

    # Z-score each ROI
    moderators = {}
    labels = {}
    raw_stats = {}
    for roi_name in roi_weights:
        roi_cfg = ATLAS_AROUSAL_ROIS[roi_name]
        z_dict, raw_mean, raw_sd = _zscore_dict(roi_values[roi_name])
        if z_dict is not None:
            moderators[roi_name] = z_dict
            labels[roi_name] = roi_cfg["label"] + " (atlas GM volume)"
            raw_stats[roi_name] = {"mean": raw_mean, "sd": raw_sd}

    print(f"    {len(moderators)} ROI moderators loaded")
    return moderators, labels, raw_stats


# ===================================================================
# Synthetic Data Loaders
# ===================================================================

def _load_synthetic_roi_moderators(data_dir, roi_names, label_dict):
    """Load ROI moderator values from synthetic CSV.

    The synthetic file ``data/synthetic/roi_values.csv`` has columns:
    ID, roi_name, value (already z-scored).

    Parameters
    ----------
    data_dir : str
        Path to the ``data/`` directory.
    roi_names : list of str
        Which ROIs to extract.
    label_dict : dict
        {roi_name: human-readable label}.

    Returns
    -------
    moderators, labels, raw_stats : dicts
    """
    csv_path = os.path.join(data_dir, "synthetic", "roi_values.csv")
    df = pd.read_csv(csv_path)

    moderators = {}
    labels = {}
    raw_stats = {}

    for roi_name in roi_names:
        roi_df = df[df["roi_name"] == roi_name]
        if len(roi_df) == 0:
            continue

        # Values in synthetic CSV are already z-scored
        values = dict(zip(roi_df["ID"].astype(str), roi_df["value"]))
        moderators[roi_name] = values
        labels[roi_name] = label_dict.get(roi_name, roi_name)
        # Synthetic data: raw stats are 0/1 by convention (z-scored)
        raw_stats[roi_name] = {"mean": 0.0, "sd": 1.0}

    return moderators, labels, raw_stats


def _load_synthetic_vbm_moderators(data_dir):
    """Load VBM arousal ROI volumes from synthetic CSV.

    The synthetic file ``data/synthetic/vbm_volumes.csv`` has columns:
    ID, roi_name, value (already z-scored).

    Parameters
    ----------
    data_dir : str
        Path to the ``data/`` directory.

    Returns
    -------
    moderators, labels, raw_stats : dicts
    """
    csv_path = os.path.join(data_dir, "synthetic", "vbm_volumes.csv")
    df = pd.read_csv(csv_path)

    moderators = {}
    labels = {}
    raw_stats = {}

    for roi_name in ATLAS_AROUSAL_ROIS:
        roi_cfg = ATLAS_AROUSAL_ROIS[roi_name]
        roi_df = df[df["roi_name"] == roi_name]
        if len(roi_df) == 0:
            continue

        values = dict(zip(roi_df["ID"].astype(str), roi_df["value"]))
        moderators[roi_name] = values
        labels[roi_name] = roi_cfg["label"] + " (atlas GM volume)"
        raw_stats[roi_name] = {"mean": 0.0, "sd": 1.0}

    return moderators, labels, raw_stats
