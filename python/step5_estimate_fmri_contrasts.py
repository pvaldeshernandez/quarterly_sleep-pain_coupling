"""
Step 5 — Re-estimate fMRI first-level contrasts without GM mask.
======================================================================

Input:  data/original/fmri_glm/<subject_id>/SPM.mat + 4D functional data
Output:
  derivatives/
    step5_fmri_contrasts/<subject_id>/con_0001.nii  — unmasked contrasts
    step5_fmri_voxel_coverage.csv                    — per-subject per-ROI coverage

For each subject:
  1. Load SPM.mat (design matrix X, high-pass filter K, whitening W,
     global scaling factors gSF, contrast vector c)
  2. Load the 4D functional time series referenced in SPM.mat
  3. Identify valid voxels: any voxel with nonzero signal across time
     (this is the whole-brain mask — no individual GM mask)
  4. Apply global scaling: Y_scaled = Y * gSF
  5. High-pass filter: KY = Y - X0 @ pinv(X0) @ Y  (DCT basis)
  6. Whiten: WKY = W @ KY
  7. Compute betas: beta = pKX @ WKY  (using SPM's precomputed pseudoinverse)
  8. Apply contrast: con = c' @ beta
  9. Save con_0001.nii with NaN outside brain, finite inside

This mirrors what SPM12 does internally but removes the individual
GM mask, giving whole-brain coverage for subcortical and brainstem
ROIs. The original script was ``scripts/reestimate_nomask.py``.

After re-estimation, this step also computes per-subject per-ROI
voxel coverage for all spherical ROIs used in Steps 7 and 10
(Krause + ACC + arousal atlas ROIs), reporting how many voxels
within each sphere have finite data for each subject.

Source data path (to be copied into data/original/fmri_glm/):
  /orange/cruzalmeida/pvaldeshernandez/Data/UPLOAD2/
  CONN2SPM_dartel_BIDS_indirect/
  GLM-CONN2SPM-conn_stim_ses-01_dartel_BIDS_indirect-simplest/

Author: Pedro Valdes-Hernandez (with Claude Opus 4.6)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo root
DERIV_DIR = os.path.join(ROOT, "derivatives")
DATA_DIR = os.path.join(ROOT, "data")

# Default source: data/original/fmri_glm/<subject_id>/
DEFAULT_GLM_DIR = os.path.join(DATA_DIR, "original", "fmri_glm")

OUT_CON_DIR = os.path.join(DERIV_DIR, "step5_fmri_contrasts")
OUT_COVERAGE_CSV = os.path.join(DERIV_DIR, "step5_fmri_voxel_coverage.csv")


def process_subject(subj_id, glm_dir, out_dir):
    """Re-estimate con_0001.nii without mask for one subject.

    Returns the number of valid (non-zero) voxels, or 'skip' if
    the output already exists.
    """
    import nibabel as nib
    import scipy.io as sio
    from scipy import sparse

    spm_path = os.path.join(glm_dir, subj_id, "SPM.mat")
    out_subj = os.path.join(out_dir, subj_id)
    out_con = os.path.join(out_subj, "con_0001.nii")

    if os.path.isfile(out_con):
        return "skip"

    os.makedirs(out_subj, exist_ok=True)

    # Load SPM.mat
    mat = sio.loadmat(spm_path, squeeze_me=False, struct_as_record=False)
    spm = mat["SPM"][0, 0]
    xX = spm.xX[0, 0]
    xGX = spm.xGX[0, 0]

    # pKX: precomputed pseudoinverse = pinv(W @ K @ X)
    pKX = xX.pKX
    if sparse.issparse(pKX):
        pKX = pKX.toarray()

    # Global scaling factors
    gSF = xGX.gSF.flatten()

    # High-pass filter: K is DCT basis
    K = xX.K[0, 0]
    X0 = K.X0
    X0_pinv = np.linalg.pinv(X0)

    # Whitening matrix
    W = xX.W
    if sparse.issparse(W):
        W = W.toarray()

    # Contrast vector
    c = spm.xCon[0, 0].c.flatten()

    # Load functional data
    fname = str(spm.xY[0, 0].VY[0, 0].fname).strip("[] ").strip("'")
    img = nib.load(fname)
    Y_4d = img.get_fdata()
    nx, ny, nz, nt = Y_4d.shape
    Y = Y_4d.reshape(-1, nt).T  # (nt, n_voxels)

    # Valid voxels: any nonzero across time (whole-brain mask)
    valid_mask = np.any(Y != 0, axis=0)
    Y_valid = Y[:, valid_mask]

    # Global scaling
    Y_scaled = Y_valid * gSF[:, np.newaxis]

    # High-pass filter
    KY = Y_scaled - X0 @ (X0_pinv @ Y_scaled)

    # Whitening
    WKY = W @ KY

    # Betas
    beta = pKX @ WKY

    # Contrast
    con_valid = c @ beta

    # Reconstruct volume
    con_vol = np.full(nx * ny * nz, np.nan, dtype=np.float32)
    con_vol[valid_mask] = con_valid.astype(np.float32)
    con_vol = con_vol.reshape(nx, ny, nz)

    # Save
    out_img = nib.Nifti1Image(con_vol, img.affine, img.header)
    out_img.header.set_data_dtype(np.float32)
    nib.save(out_img, out_con)

    return int(valid_mask.sum())


def run_step5(glm_dir=None, verbose=True):
    if glm_dir is None:
        glm_dir = DEFAULT_GLM_DIR

    if verbose:
        print("=" * 70)
        print("STEP 5 — Re-estimate fMRI contrasts without GM mask")
        print("=" * 70)
        print(f"  GLM source: {glm_dir}")
        print(f"  Output: {OUT_CON_DIR}")

    if not os.path.isdir(glm_dir):
        print(f"\n  ERROR: GLM directory not found: {glm_dir}")
        print("  This step requires the original SPM first-level GLM data.")
        print("  Copy it to data/original/fmri_glm/ or pass --glm-dir.")
        return

    os.makedirs(OUT_CON_DIR, exist_ok=True)

    subjects = sorted([
        s for s in os.listdir(glm_dir)
        if os.path.isfile(os.path.join(glm_dir, s, "SPM.mat"))
    ])

    if verbose:
        print(f"  {len(subjects)} subjects with SPM.mat")

    t0 = time.time()
    results = []
    for i, subj in enumerate(subjects):
        try:
            result = process_subject(subj, glm_dir, OUT_CON_DIR)
            if result == "skip":
                if verbose:
                    print(f"  [{i+1:3d}/{len(subjects)}] {subj}: already done")
            else:
                results.append({"ID": subj, "n_valid_voxels": result})
                if verbose:
                    print(f"  [{i+1:3d}/{len(subjects)}] {subj}: {result} voxels")
        except Exception as e:
            if verbose:
                print(f"  [{i+1:3d}/{len(subjects)}] {subj}: ERROR - {e}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Done in {elapsed:.1f}s "
              f"({elapsed/max(len(subjects),1):.1f}s/subject)")

    # TODO: per-subject per-ROI voxel coverage audit
    # (to be implemented when ROI definitions are finalized)

    if verbose:
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 5 — re-estimate fMRI contrasts without GM mask."
    )
    parser.add_argument("--glm-dir", default=None,
                        help="Path to the GLM directory with SPM.mat per subject")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_step5(glm_dir=args.glm_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
