"""
Step 6 — Prepare VBM modulated GM images for arousal ROI analysis.
======================================================================

Input:  data/original/vbm/smwc1<subject_id>_ses-01_T1w.nii
Output:
  derivatives/
    step6_vbm_subjects.csv   — list of subjects with VBM data + ID mapping

This step copies/symlinks the modulated GM images into a
standardized location and builds the subject-ID mapping (handling
the BIDS x-for-dash convention in filenames). It does NOT compute
ROI volumes — that happens in Step 10 (extract PS ROIs) where the
atlas probability maps are applied.

Source data path (to be copied into data/original/vbm/):
  /orange/cruzalmeida/jperazagoicolea/UPLOAD2/
  dset/derivatives/LocalizedBrainAge/ses-01/gm_data/
  smwc1*_ses-01_T1w.nii

Author: Pedro Valdes-Hernandez (with Claude Opus 4.6)
"""
from __future__ import annotations

import argparse
import glob as glob_mod
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo root
DATA_DIR = os.path.join(ROOT, "data")
DERIV_DIR = os.path.join(ROOT, "derivatives")

DEFAULT_VBM_DIR = os.path.join(DATA_DIR, "original", "vbm")
OUT_SUBJECTS_CSV = os.path.join(DERIV_DIR, "step6_vbm_subjects.csv")


def run_step6(vbm_dir=None, verbose=True):
    if vbm_dir is None:
        vbm_dir = DEFAULT_VBM_DIR

    if verbose:
        print("=" * 70)
        print("STEP 6 — Prepare VBM modulated GM images")
        print("=" * 70)
        print(f"  VBM source: {vbm_dir}")

    if not os.path.isdir(vbm_dir):
        print(f"\n  ERROR: VBM directory not found: {vbm_dir}")
        print("  Copy smwc1*_ses-01_T1w.nii files to data/original/vbm/")
        return

    pattern = os.path.join(vbm_dir, "smwc1*_ses-01_T1w.nii")
    gm_files = sorted(glob_mod.glob(pattern))

    if verbose:
        print(f"  Found {len(gm_files)} smwc1 files")

    rows = []
    for gm_path in gm_files:
        fname = os.path.basename(gm_path)
        raw_id = fname.replace("smwc1", "").replace("_ses-01_T1w.nii", "")
        # VBM filenames use 'x' instead of '-' (BIDS convention)
        clean_id = raw_id.replace("x", "-")
        rows.append({
            "ID": clean_id,
            "raw_filename_id": raw_id,
            "filename": fname,
            "path": gm_path,
        })

    df = pd.DataFrame(rows)
    os.makedirs(DERIV_DIR, exist_ok=True)
    df.to_csv(OUT_SUBJECTS_CSV, index=False)

    if verbose:
        n_unique = df["ID"].nunique()
        n_with_x = sum("x" in r["raw_filename_id"] for r in rows)
        print(f"  {n_unique} unique subjects ({n_with_x} with x-for-dash mapping)")
        print(f"  Saved: {OUT_SUBJECTS_CSV}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 6 — prepare VBM modulated GM images."
    )
    parser.add_argument("--vbm-dir", default=None,
                        help="Path to directory containing smwc1*_ses-01_T1w.nii")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_step6(vbm_dir=args.vbm_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
