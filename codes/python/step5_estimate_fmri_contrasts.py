"""
Step 5 — Re-estimate first-level fMRI contrasts (masked and unmasked).
======================================================================

Inputs:
  data/original/spm_mats/<subj>/SPM.mat
      Design matrix, high-pass filter, whitening, contrast vector,
      and per-subject GM mask filename.
  data/original/spm_mats/<subj>/sUPLOAD2_T1w__gm-2mm-binarized(0.25).nii
      Individual binary GM mask (used for the masked variant).
  data/original/fmri_4d/sub-<subj>/ses-01/func/
      ssub-<subj>_ses-01_task-stim_space-MNI152Dartel_desc-preproc_bold.nii
      Smoothed, normalised 4D functional time series.

Outputs (per subject, per variant):
  derivatives/step5_fmri_contrasts_masked/<subj>/
      con_0001.nii   — contrast image (stimulation > baseline), GM-masked
      beta_*.nii     — beta images, GM-masked
  derivatives/step5_fmri_contrasts_unmasked/<subj>/
      con_0001.nii   — contrast image, whole-brain (no mask)
      beta_*.nii     — beta images, whole-brain

Method (replicates SPM12 OLS):
  1. Load design matrix X, precomputed pseudoinverse pKX = pinv(W K X),
     global scaling factors gSF, DCT high-pass basis X0, and whitening W
     from SPM.mat.
  2. Apply global scaling:   Y_scaled = Y * gSF
  3. High-pass filter:       KY = Y_scaled - X0 @ pinv(X0) @ Y_scaled
  4. Whitening:              WKY = W @ KY
  5. Betas:                  B = pKX @ WKY
  6. Contrast:               con = c' @ B  (c = [1, 0, ..., 0])
  For the masked variant, voxels outside the individual GM mask are set
  to NaN. For the unmasked variant, all finite voxels are retained.

Author: Pedro Valdes-Hernandez (with Claude Sonnet 4.6)
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import nibabel as nib
import scipy.io as sio
import scipy.sparse

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

GLM_DIR   = os.path.join(ROOT, "data", "original", "spm_mats")
FMRI4D_DIR = os.path.join(ROOT, "data", "original", "fmri_4d")
DERIV_DIR = os.path.join(ROOT, "derivatives")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step5")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)

OUT_MASKED   = os.path.join(DERIV_DIR, "step5_fmri_contrasts_masked")
OUT_UNMASKED = os.path.join(DERIV_DIR, "step5_fmri_contrasts_unmasked")

GM_MASK_FNAME = "sUPLOAD2_T1w__gm-2mm-binarized(0.25).nii"


def _load_spm(spm_path: str) -> dict:
    """Load SPM.mat and return a flat dict of the key arrays."""
    mat = sio.loadmat(spm_path, squeeze_me=False, struct_as_record=False)
    spm = mat["SPM"][0, 0]
    xX  = spm.xX[0, 0]
    xGX = spm.xGX[0, 0]

    # Pseudoinverse of whitened, filtered design matrix
    pKX = xX.pKX
    if scipy.sparse.issparse(pKX):
        pKX = pKX.toarray()

    # Global scaling factors (one per scan)
    gSF = xGX.gSF.flatten()

    # High-pass filter DCT basis
    K    = xX.K[0, 0]
    X0   = K.X0                          # (n_scans, n_dct)
    X0_pinv = np.linalg.pinv(X0)

    # Whitening matrix
    W = xX.W
    if scipy.sparse.issparse(W):
        W = W.toarray()

    # Contrast vector (first contrast = stimulation > baseline)
    c = spm.xCon[0, 0].c.flatten()      # (n_betas,)

    # Path to 4D data as stored in SPM.mat
    raw_fname = str(spm.xY[0, 0].VY[0, 0].fname).strip("[] '")

    # Number of betas
    n_betas = pKX.shape[0]

    return dict(pKX=pKX, gSF=gSF, X0=X0, X0_pinv=X0_pinv, W=W, c=c,
                raw_fname=raw_fname, n_betas=n_betas)


def _find_4d(subj_id: str) -> str | None:
    """Find the 4D NIfTI for a subject in the BIDS fmri_4d tree.

    BIDS directories use 'x' instead of '-' in subject IDs.
    """
    bids_id = "sub-" + subj_id.replace("-", "x")
    candidate = os.path.join(
        FMRI4D_DIR, bids_id, "ses-01", "func",
        f"s{bids_id}_ses-01_task-stim_space-MNI152Dartel_desc-preproc_bold.nii"
    )
    if os.path.isfile(candidate):
        return candidate
    # Also try without the 'x' replacement (some IDs have no dashes)
    bids_id2 = "sub-" + subj_id
    candidate2 = os.path.join(
        FMRI4D_DIR, bids_id2, "ses-01", "func",
        f"s{bids_id2}_ses-01_task-stim_space-MNI152Dartel_desc-preproc_bold.nii"
    )
    if os.path.isfile(candidate2):
        return candidate2
    return None


def _save_vol(data: np.ndarray, affine, header, path: str):
    img = nib.Nifti1Image(data.astype(np.float32), affine, header)
    img.header.set_data_dtype(np.float32)
    nib.save(img, path)


def process_subject(subj_id: str, overwrite: bool = False,
                    verbose: bool = True) -> str:
    """Re-estimate contrasts (masked + unmasked) for one subject."""
    spm_path = os.path.join(GLM_DIR, subj_id, "SPM.mat")
    if not os.path.isfile(spm_path):
        return "no SPM.mat"

    out_masked   = os.path.join(OUT_MASKED,   subj_id)
    out_unmasked = os.path.join(OUT_UNMASKED, subj_id)
    masked_con   = os.path.join(out_masked,   "con_0001.nii")
    unmasked_con = os.path.join(out_unmasked, "con_0001.nii")

    if not overwrite and os.path.isfile(masked_con) and os.path.isfile(unmasked_con):
        return "skip"

    # Load SPM model
    p = _load_spm(spm_path)

    # Find 4D file
    fmri_path = _find_4d(subj_id)
    if fmri_path is None:
        return "no 4D file"

    # Load 4D data
    img = nib.load(fmri_path)
    Y_4d = img.get_fdata(dtype=np.float32)
    nx, ny, nz, nt = Y_4d.shape
    Y = Y_4d.reshape(-1, nt).T          # (n_scans, n_voxels)
    del Y_4d

    # Valid voxels: any non-zero across time
    valid = np.any(Y != 0, axis=0)      # (n_voxels,)
    Yv = Y[:, valid]                    # (n_scans, n_valid)
    del Y

    # GLM steps
    Yv = Yv * p["gSF"][:, np.newaxis]                         # global scale
    Yv = Yv - p["X0"] @ (p["X0_pinv"] @ Yv)                  # high-pass
    Yv = p["W"] @ Yv                                          # whiten
    B  = p["pKX"] @ Yv                                        # betas (n_betas, n_valid)
    con_valid = p["c"] @ B                                     # contrast (n_valid,)

    affine, header = img.affine, img.header
    n_vox = nx * ny * nz

    # ---- UNMASKED ----
    os.makedirs(out_unmasked, exist_ok=True)
    if overwrite or not os.path.isfile(unmasked_con):
        con_vol = np.full(n_vox, np.nan, dtype=np.float32)
        con_vol[valid] = con_valid
        _save_vol(con_vol.reshape(nx, ny, nz), affine, header, unmasked_con)

    for k in range(p["n_betas"]):
        beta_path = os.path.join(out_unmasked, f"beta_{k+1:04d}.nii")
        if overwrite or not os.path.isfile(beta_path):
            beta_vol = np.full(n_vox, np.nan, dtype=np.float32)
            beta_vol[valid] = B[k]
            _save_vol(beta_vol.reshape(nx, ny, nz), affine, header, beta_path)

    # ---- MASKED ----
    os.makedirs(out_masked, exist_ok=True)
    gm_mask_path = os.path.join(GLM_DIR, subj_id, GM_MASK_FNAME)
    if os.path.isfile(gm_mask_path):
        gm_img  = nib.load(gm_mask_path)
        gm_data = (gm_img.get_fdata() > 0).flatten()
        # Resample if shapes differ
        if gm_data.shape[0] != n_vox:
            from nilearn.image import resample_to_img
            gm_res  = resample_to_img(gm_img, img, interpolation="nearest")
            gm_data = (gm_res.get_fdata() > 0).flatten()
        in_mask = valid & gm_data
    else:
        # Fallback: use the SPM analysis mask (mask.nii) if present
        mask_path = os.path.join(GLM_DIR, subj_id, "mask.nii")
        if os.path.isfile(mask_path):
            mk = nib.load(mask_path).get_fdata().flatten() > 0
            in_mask = valid & mk
        else:
            in_mask = valid

    if overwrite or not os.path.isfile(masked_con):
        con_vol = np.full(n_vox, np.nan, dtype=np.float32)
        con_vol[valid] = con_valid
        con_vol[~in_mask] = np.nan
        _save_vol(con_vol.reshape(nx, ny, nz), affine, header, masked_con)

    for k in range(p["n_betas"]):
        beta_path = os.path.join(out_masked, f"beta_{k+1:04d}.nii")
        if overwrite or not os.path.isfile(beta_path):
            beta_vol = np.full(n_vox, np.nan, dtype=np.float32)
            beta_vol[valid] = B[k]
            beta_vol[~in_mask] = np.nan
            _save_vol(beta_vol.reshape(nx, ny, nz), affine, header, beta_path)

    return f"{int(in_mask.sum())} masked / {int(valid.sum())} unmasked voxels"


def run_step5(overwrite: bool = False, verbose: bool = True):
    os.makedirs(OUT_MASKED,   exist_ok=True)
    os.makedirs(OUT_UNMASKED, exist_ok=True)

    subjects = sorted(s for s in os.listdir(GLM_DIR)
                      if os.path.isfile(os.path.join(GLM_DIR, s, "SPM.mat")))

    if verbose:
        print("=" * 70)
        print("STEP 5 — Re-estimate fMRI contrasts (masked + unmasked)")
        print("=" * 70)
        print(f"  GLM dir:      {GLM_DIR}")
        print(f"  4D fMRI dir:  {FMRI4D_DIR}")
        print(f"  Out masked:   {OUT_MASKED}")
        print(f"  Out unmasked: {OUT_UNMASKED}")
        print(f"  Subjects:     {len(subjects)}")
        print()

    t0 = time.time()
    n_done = n_skip = n_err = 0

    for i, subj in enumerate(subjects):
        result = process_subject(subj, overwrite=overwrite, verbose=verbose)
        tag = f"[{i+1:3d}/{len(subjects)}] {subj}"
        if result == "skip":
            n_skip += 1
            if verbose:
                print(f"  {tag}: already done")
        elif result.startswith("no "):
            n_err += 1
            if verbose:
                print(f"  {tag}: WARNING — {result}")
        else:
            n_done += 1
            if verbose:
                print(f"  {tag}: {result}")

    elapsed = time.time() - t0
    if verbose:
        print()
        print(f"  Done: {n_done} estimated, {n_skip} skipped, {n_err} errors")
        print(f"  Elapsed: {elapsed:.1f}s  ({elapsed/max(n_done,1):.1f}s/subject)")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 5 — Re-estimate fMRI contrasts (masked + unmasked)."
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-estimate even if output files already exist.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_step5(overwrite=args.overwrite, verbose=not args.quiet)


if __name__ == "__main__":
    main()
