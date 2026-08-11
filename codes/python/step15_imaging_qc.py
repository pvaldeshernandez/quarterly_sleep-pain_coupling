#!/usr/bin/env python3
"""
Step 15 — Imaging quality control of the fMRI subsample (Section S9).
=====================================================================

Head motion (framewise displacement read from each participant's SPM design matrix),
scanner site, and the maximum pain rating evoked during the fMRI run -- plus how each of
those tracks the eight sleep-to-pain ROI values.

The sixteen nuisance-adjusted refits that used to live here moved to STEP 18 on 11 Aug
2026. They read Table 5 for their unadjusted column, so keeping them here dragged this QC
after the moderation fits, while the paper reports it under "Final MRI samples", before
them. `prepare_inputs()` is the setup both steps share; step 18 imports it rather than
repeating it, so the two sample definitions cannot drift apart.

SAMPLE. Every DESCRIPTIVE is computed on the fitted fMRI subsample — the intersection
of the ROI table with the analytic sample, N = 174 — not on the 188 rows of the ROI
table. That is the sample the surrounding text describes and the sample all 16 fits use.
The RESIDUALIZATION and z-scoring stay on the full ROI-table sample (188 rows; 182 for
the two contralateralized ROIs), because step 13 z-scored the unadjusted moderator over
exactly those rows and the adjusted columns of Table S8 must remain comparable to the
unadjusted column of Table 5. The two samples are deliberately different and both are
recorded (`n_fmri_sample`, `n_resid_sample_<ROI>`).

The per-ROI scanner-site test is written for BOTH scopes as data (`sample` column of
step15_site_differences.csv) but published to numbers.json for the fMRI subsample only,
so a document number cannot match the wrong sample silently. See the module note below.

Input:
  data/original/spm_mats/<ID>/SPM.mat                      (READ ONLY)
  data/original/participants_wideformat.xlsx               (READ ONLY)
  derivatives/step07_varx_data/step07_processed_long.csv   (step 04)
  derivatives/step14_sp_roi_values/step14_sp_roi_values.csv (step 13)
  results/step16_sp_moderation/step16_table5_sp_moderation.csv (step 14)
  derivatives/step15_imaging_qc/diagnostics_step15_*.json  (self, written by run_fit)

Output:
  derivatives/step15_imaging_qc/
    step15_motion_qc.csv                  — per-participant framewise displacement
    step15_nuisance_gamma_draws.npz       — gamma_sp draws + adjusted moderators, 16 fits
    step15_diagnostics.csv                — the 16 run_fit diagnostics records
  results/step15_imaging_qc/
    step15_motion_correlations.csv        — mean FD vs evoked pain and vs each ROI
    step15_evokedpain_correlations.csv    — evoked pain vs each ROI
    step15_site_differences.csv           — per-ROI Welch test by scanner site
    step15_tableS8_nuisance_sensitivity.csv — Table S8 DATA (no note, no caption)
    numbers.json                          — every quantity, via registry.write_numbers

This step writes no prose: no *_text.md, no table notes, no captions (Decision 3).

Usage:
    python step15_imaging_qc.py            # rebuild CSVs and numbers from saved fits
    python step15_imaging_qc.py --refit    # rescan SPM.mat and run the 16 fits

Author: Pedro Valdes-Hernandez (with Claude Opus 5)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

DERIV_DIR = os.path.join(ROOT, "derivatives")
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step15_imaging_qc")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step15_imaging_qc")

# --- inputs ---------------------------------------------------------------
IN_SPM_DIR = os.path.join(ROOT, "data", "original", "spm_mats")
IN_WIDE_XLSX = os.path.join(ROOT, "data", "original", "participants_wideformat.xlsx")
IN_PROCESSED_CSV = os.path.join(DERIV_DIR, "step07_varx_data",
                                "step07_processed_long.csv")
IN_ROI_CSV = os.path.join(DERIV_DIR, "step14_sp_roi_values",
                          "step14_sp_roi_values.csv")
IN_TABLE5_CSV = os.path.join(RESULTS_DIR, "step16_sp_moderation",
                             "step16_table5_sp_moderation.csv")

# --- outputs --------------------------------------------------------------
OUT_MOTION_CSV = os.path.join(STEP_DERIV_DIR, "step15_motion_qc.csv")
OUT_DRAWS_NPZ = os.path.join(STEP_DERIV_DIR, "step15_nuisance_gamma_draws.npz")
OUT_DIAG_CSV = os.path.join(STEP_DERIV_DIR, "step15_diagnostics.csv")
OUT_FD_CORR_CSV = os.path.join(STEP_RESULTS_DIR, "step15_motion_correlations.csv")
OUT_PAIN_CORR_CSV = os.path.join(STEP_RESULTS_DIR, "step15_evokedpain_correlations.csv")
OUT_SITE_CSV = os.path.join(STEP_RESULTS_DIR, "step15_site_differences.csv")
OUT_TABLE_S8_CSV = os.path.join(STEP_RESULTS_DIR,
                                "step15_tableS8_nuisance_sensitivity.csv")

# --- library --------------------------------------------------------------
from coupling_model import (  # noqa: E402
    extract_results, fit_bayesian_varx1, load_varx_frame, read_diagnostics, two_tail_p,
)
from nuisance import (  # noqa: E402
    align, cohens_d, fd_from_design, person_covariate, read_spm_design, residualize,
    roi_moderator,
)
from registry import write_numbers  # noqa: E402

# --- constants ------------------------------------------------------------
SITE_COL = "site__s1"          # 1 = UF, 2 = UAB
PAIN_COL = "img_fmri_pain_score__s1"   # maximum pain rating during the fMRI run, 0-100
SITE_UF, SITE_UAB = 1.0, 2.0

#: the eight sleep-to-pain ROIs, in the order they are reported
ALL_SP_ROIS = [
    "Contra_S1", "Contra_Middle_Insula", "Left_Thalamus", "Left_Anterior_Insula",
    "Left_NAcc", "Right_NAcc", "Left_dACC_MCC", "Right_dACC_MCC",
]

#: the four ROIs that appear in Table S8. Only these are refit — the other four site
#: fits a06b ran are dead compute, they enter no table and no sentence.
TABLE_S8_ROIS = ["Left_NAcc", "Right_NAcc", "Left_dACC_MCC", "Right_dACC_MCC"]

#: adjustment schemes; the covariate maps themselves are assembled at run time
SCHEMES = ["site", "pain", "motion", "all3"]

#: ROIs named individually in the text for each correlation, so the "all the rest"
#: bound is computed over a set that is defined here rather than by whichever ROI
#: happened to clear p < .05 on this run.
FD_NAMED_ROIS = ["Left_Thalamus", "Contra_Middle_Insula"]
PAIN_NAMED_ROIS = ["Left_dACC_MCC", "Right_dACC_MCC"]

FD_THRESHOLDS_MM = [0.2, 0.3, 0.5]   # mean-FD exclusion thresholds, descriptive only
FD_VOLUME_THRESHOLD_MM = 0.5         # per-volume threshold for pct_fd_gt_05
MIN_N_CORR = 30                      # below this a correlation is not reported


# ======================================================================
# Small orchestration helpers (the reusable concepts live in lib/nuisance.py)
# ======================================================================

def corr_on(ids, map_a, map_b):
    """Pearson correlation of two person-level maps over one ID set.

    `align` (lib/nuisance) does the ID -> value lookup, so this shares its NaN
    handling with `residualize` instead of re-deriving it.
    """
    A = align(ids, map_a, map_b)
    ok = np.isfinite(A).all(axis=1)
    n = int(ok.sum())
    if n < MIN_N_CORR:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(A[ok, 0], A[ok, 1])
    return float(r), float(p), n


def site_contrast(ids, roi_map, site_map):
    """Welch t-test and Cohen's d of one ROI's values by scanner site."""
    A = align(ids, roi_map, site_map)
    v, s = A[:, 0], A[:, 1]
    a = v[np.isfinite(v) & (s == SITE_UF)]
    b = v[np.isfinite(v) & (s == SITE_UAB)]
    if len(a) < 2 or len(b) < 2:
        return {"n_UF": len(a), "n_UAB": len(b), "t": np.nan,
                "p": np.nan, "d": np.nan}
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"n_UF": int(len(a)), "n_UAB": int(len(b)), "t": float(t),
            "p": float(p), "d": float(cohens_d(a, b))}


def scan_motion(spm_dir, verbose=True):
    """Framewise displacement for every SPM.mat under `spm_dir`.

    Returns (DataFrame, failures). A file that cannot be READ is counted and listed —
    never swallowed, which is what a04 did — while a design matrix whose layout or
    units contradict the FD assumption raises out of `fd_from_design`, because that is
    a wrong number rather than a missing one.
    """
    mats = sorted(glob.glob(os.path.join(spm_dir, "*", "SPM.mat")))
    rows, failures = [], []
    for path in mats:
        sid = str(os.path.basename(os.path.dirname(path)))
        try:
            X = read_spm_design(path)
        except Exception as exc:                       # noqa: BLE001
            failures.append({"ID": sid, "error": f"{type(exc).__name__}: {exc}"})
            continue
        fd = fd_from_design(X)                         # raises on a bad layout/units
        rows.append({
            "ID": sid,
            "n_vols": int(X.shape[0]),
            "n_cols": int(X.shape[1]),
            "fd_mean": float(fd.mean()),
            "fd_median": float(np.median(fd)),
            "fd_max": float(fd.max()),
            "n_fd_gt_05": int((fd > FD_VOLUME_THRESHOLD_MM).sum()),
            "pct_fd_gt_05": float(100.0 * (fd > FD_VOLUME_THRESHOLD_MM).mean()),
        })
    qc = pd.DataFrame(rows)
    if verbose:
        print(f"  SPM.mat found: {len(mats)}, read: {len(qc)}, "
              f"failed: {len(failures)}")
        for f in failures:
            print(f"    !! {f['ID']}: {f['error']}")
    return qc, failures


def summarize(values, prefix, out):
    """mean / SD / median / min / max of one column, under `prefix`."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    out[f"{prefix}_mean"] = float(v.mean())
    out[f"{prefix}_sd"] = float(v.std(ddof=1))
    out[f"{prefix}_median"] = float(np.median(v))
    out[f"{prefix}_min"] = float(v.min())
    out[f"{prefix}_max"] = float(v.max())


def absmax_other(per_roi, named):
    """Largest |r| among the ROIs the text does NOT name individually."""
    others = {k: v for k, v in per_roi.items() if k not in named and np.isfinite(v)}
    if not others:
        return np.nan, ""
    roi = max(others, key=lambda k: abs(others[k]))
    return float(abs(others[roi])), roi


# ======================================================================
# Step
# ======================================================================


def prepare_inputs(verbose=True, refit=False):
    """ROI values, samples and nuisance covariates — shared with step 18.

    Returns a dict. `fmri_ids` (N=174) is the fitted subsample every DESCRIPTIVE
    is computed on; `roi_maps` is keyed over the full ROI table (188 rows, 182 for
    the two contralateralized ROIs), which is the base step 14 z-scored the
    unadjusted moderator over. Table S8 stays comparable to Table 5 only if the
    residualization uses that same base, so both live here in one definition.
    """
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("STEP 15 — Imaging QC and nuisance-adjusted SP moderation")
        print("=" * 70)

    nums = {}

    # ------------------------------------------------------------------
    # Samples: the ROI table, the analytic frame, and their intersection
    # ------------------------------------------------------------------
    roi_df = pd.read_csv(IN_ROI_CSV)
    roi_df["ID"] = roi_df["ID"].astype(str)
    present = set(roi_df["ROI"].unique())
    unexpected = sorted(present - set(ALL_SP_ROIS))
    if unexpected and verbose:
        print(f"  WARNING: ROI table has ROIs this step does not report: {unexpected}")
    rois = [r for r in ALL_SP_ROIS if r in present]
    missing_targets = [r for r in TABLE_S8_ROIS if r not in present]
    if missing_targets:
        raise ValueError(f"Table S8 ROIs absent from {IN_ROI_CSV}: {missing_targets}")

    df_full, model_df, unique_ids, id_map = load_varx_frame(IN_PROCESSED_CSV,
                                                            verbose=False)
    analytic_ids = {str(i) for i in unique_ids}
    roi_ids = set(roi_df["ID"])
    fmri_ids = sorted(roi_ids & analytic_ids)

    nums["n_roi_table"] = len(roi_ids)
    nums["n_analytic"] = len(analytic_ids)
    nums["n_fmri_sample"] = len(fmri_ids)
    if verbose:
        print(f"  ROI table {len(roi_ids)} x analytic {len(analytic_ids)} "
              f"-> fMRI subsample N = {len(fmri_ids)}")

    # {ID: z_value} per ROI — one concept, one function (lib/nuisance.roi_moderator)
    roi_maps = {r: roi_moderator(roi_df, r) for r in rois}
    for r in rois:
        nums[f"n_resid_sample_{r}"] = len(roi_maps[r])

    # ------------------------------------------------------------------
    # Person-level nuisance covariates
    # ------------------------------------------------------------------
    site_map = person_covariate(IN_WIDE_XLSX, SITE_COL)
    pain_map = person_covariate(IN_WIDE_XLSX, PAIN_COL)

    # How many SPM.mat files EXIST is counted the same way on both paths — it is a
    # cheap read-only glob, and deriving it from the saved table instead would make
    # `n_spm_dirs` mean "designs successfully read" on the load path and "designs
    # present" on the rescan path. numbers.json would then change between a --refit
    # run and the next default run with nothing having changed on disk, and the
    # read-failure count would silently reset to zero.
    n_spm_dirs = len(glob.glob(os.path.join(IN_SPM_DIR, "*", "SPM.mat")))

    need_motion = refit or not os.path.exists(OUT_MOTION_CSV)
    if need_motion:
        if verbose and not refit:
            print("  step15_motion_qc.csv not found — rescanning SPM.mat.")
        motion, failures = scan_motion(IN_SPM_DIR, verbose=verbose)
        motion.to_csv(OUT_MOTION_CSV, index=False)
        n_failures = len(failures)
    else:
        if verbose:
            print("  Loading saved motion QC (re-run with --refit to rescan SPM.mat).")
        motion = pd.read_csv(OUT_MOTION_CSV)
        motion["ID"] = motion["ID"].astype(str)
        n_failures = n_spm_dirs - len(motion)

    motion["ID"] = motion["ID"].astype(str)
    fd_map = dict(zip(motion["ID"], motion["fd_mean"].astype(float)))
    motion["in_fmri_sample"] = motion["ID"].isin(fmri_ids)
    motion.to_csv(OUT_MOTION_CSV, index=False)

    # A participant silently missing from the motion table would shrink the
    # motion-adjusted fits without changing anything visible. Refuse instead.
    no_fd = [i for i in fmri_ids if not np.isfinite(fd_map.get(i, np.nan))]
    if no_fd:
        raise ValueError(
            f"{len(no_fd)} fMRI-subsample participant(s) have no framewise "
            f"displacement (no readable SPM.mat): {no_fd[:10]}"
        )

    nums["n_spm_dirs"] = int(n_spm_dirs)
    nums["n_spm_read_failures"] = int(n_failures)
    nums["n_motion_qc"] = int(len(motion))

    return dict(nums=nums, rois=rois, roi_df=roi_df, roi_maps=roi_maps,
                roi_ids=roi_ids,
                fmri_ids=fmri_ids, analytic_ids=analytic_ids,
                site_map=site_map, pain_map=pain_map, fd_map=fd_map,
                motion=motion, df_full=df_full, model_df=model_df,
                unique_ids=unique_ids, id_map=id_map)


def run_step15(verbose=True, refit=False):
    """Imaging QC descriptives for the fMRI subsample."""
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)
    if verbose:
        print("=" * 70)
        print("STEP 15 — Imaging quality control")
        print("=" * 70)

    ctx = prepare_inputs(verbose=verbose, refit=refit)
    nums = ctx["nums"]
    rois, roi_maps = ctx["rois"], ctx["roi_maps"]
    fmri_ids, motion = ctx["fmri_ids"], ctx["motion"]
    roi_ids = ctx["roi_ids"]
    site_map, pain_map, fd_map = ctx["site_map"], ctx["pain_map"], ctx["fd_map"]

    # ------------------------------------------------------------------
    # QC descriptives — all on the 174-participant fMRI subsample
    # ------------------------------------------------------------------
    sub = motion[motion["ID"].isin(fmri_ids)]
    summarize(sub["fd_mean"], "fd_mean", nums)
    summarize(sub["fd_median"], "fd_median", nums)
    nums["fd_max_mean"] = float(sub["fd_max"].mean())
    nums["fd_max_max"] = float(sub["fd_max"].max())
    for thr in FD_THRESHOLDS_MM:
        tag = f"0{int(round(thr * 10))}"          # 0.2 -> "02", 0.5 -> "05"
        n_over = int((sub["fd_mean"] > thr).sum())
        nums[f"n_fd_mean_gt_{tag}"] = n_over
        nums[f"pct_fd_mean_gt_{tag}"] = float(100.0 * n_over / len(sub))
    nums["pct_volumes_fd_gt_05_mean"] = float(sub["pct_fd_gt_05"].mean())
    nums["n_vols_min"] = int(sub["n_vols"].min())
    nums["n_vols_max"] = int(sub["n_vols"].max())

    # evoked pain distribution on the same sample
    pain_vals = align(fmri_ids, pain_map)[:, 0]
    nums["n_evokedpain_available"] = int(np.isfinite(pain_vals).sum())
    summarize(pain_vals, "evokedpain", nums)

    # scanner site split — the source of "111 were scanned at UF and 63 at UAB"
    site_vals = align(fmri_ids, site_map)[:, 0]
    nums["n_site_uf"] = int((site_vals == SITE_UF).sum())
    nums["n_site_uab"] = int((site_vals == SITE_UAB).sum())
    nums["n_site_missing"] = int((~np.isfinite(site_vals)).sum())

    # ------------------------------------------------------------------
    # Correlations of the nuisances with each ROI value
    # ------------------------------------------------------------------
    fd_rows, r_fd = [], {}
    r, p, n = corr_on(fmri_ids, fd_map, pain_map)
    nums["r_fd_evokedpain"], nums["p_fd_evokedpain"], nums["n_fd_evokedpain"] = r, p, n
    fd_rows.append({"sample": "fmri_subsample", "x": "fd_mean",
                    "y": "evoked_pain", "r": r, "p": p, "n": n})
    for roi in rois:
        r, p, n = corr_on(fmri_ids, fd_map, roi_maps[roi])
        r_fd[roi] = r
        nums[f"r_fd_roi_{roi}"], nums[f"p_fd_roi_{roi}"] = r, p
        nums[f"n_fd_roi_{roi}"] = n
        fd_rows.append({"sample": "fmri_subsample", "x": "fd_mean",
                        "y": roi, "r": r, "p": p, "n": n})
    val, who = absmax_other(r_fd, FD_NAMED_ROIS)
    nums["r_fd_roi_absmax_other"], nums["r_fd_roi_absmax_other_roi"] = val, who
    pd.DataFrame(fd_rows).to_csv(OUT_FD_CORR_CSV, index=False)

    pain_rows, r_pain = [], {}
    for roi in rois:
        r, p, n = corr_on(fmri_ids, pain_map, roi_maps[roi])
        r_pain[roi] = r
        nums[f"r_evokedpain_roi_{roi}"], nums[f"p_evokedpain_roi_{roi}"] = r, p
        nums[f"n_evokedpain_roi_{roi}"] = n
        pain_rows.append({"sample": "fmri_subsample", "x": "evoked_pain",
                          "y": roi, "r": r, "p": p, "r2": r * r, "n": n})
    for roi in PAIN_NAMED_ROIS:
        if roi in r_pain and np.isfinite(r_pain[roi]):
            nums[f"r2_evokedpain_roi_{roi}"] = float(r_pain[roi] ** 2)
    val, who = absmax_other(r_pain, PAIN_NAMED_ROIS)
    nums["r_evokedpain_roi_absmax_other"] = val
    nums["r_evokedpain_roi_absmax_other_roi"] = who
    pd.DataFrame(pain_rows).to_csv(OUT_PAIN_CORR_CSV, index=False)

    # ------------------------------------------------------------------
    # Scanner-site differences in the ROI values
    #
    # Written for two scopes as DATA. Only the fMRI-subsample scope reaches
    # numbers.json: the sentence in the Results describes the 174, and publishing
    # the 188 numbers too would let a token match the sample the text does not use.
    # Which scope the sentence should claim ("no ROI" over eight ROIs, or over the
    # four of Table S8) is Pedro's call, not the code's — both counts are emitted.
    # ------------------------------------------------------------------
    site_rows, site_p = [], {}
    for sample_label, ids in (("fmri_subsample", fmri_ids),
                              ("roi_table", sorted(roi_ids))):
        for roi in rois:
            rec = site_contrast(ids, roi_maps[roi], site_map)
            site_rows.append({"sample": sample_label, "ROI": roi, **rec})
            if sample_label == "fmri_subsample":
                site_p[roi] = rec["p"]
                nums[f"site_diff_d_{roi}"] = rec["d"]
                nums[f"site_diff_t_{roi}"] = rec["t"]
                nums[f"site_diff_p_{roi}"] = rec["p"]
                nums[f"site_diff_n_uf_{roi}"] = rec["n_UF"]
                nums[f"site_diff_n_uab_{roi}"] = rec["n_UAB"]
    pd.DataFrame(site_rows).to_csv(OUT_SITE_CSV, index=False)

    finite_p = {k: v for k, v in site_p.items() if np.isfinite(v)}
    if finite_p:
        worst = min(finite_p, key=finite_p.get)
        nums["site_diff_p_min"] = float(finite_p[worst])
        nums["site_diff_p_min_roi"] = worst
    nums["site_diff_n_credible_8roi"] = int(sum(v < 0.05 for v in finite_p.values()))
    nums["site_diff_n_credible_tableS8"] = int(
        sum(site_p.get(r, np.nan) < 0.05 for r in TABLE_S8_ROIS
            if np.isfinite(site_p.get(r, np.nan)))
    )
    if verbose:
        print(f"  Site split: UF {nums['n_site_uf']} / UAB {nums['n_site_uab']}; "
              f"ROIs differing at p<.05 — {nums['site_diff_n_credible_8roi']}/8 "
              f"(all eight), {nums['site_diff_n_credible_tableS8']}/4 (Table S8)")

    # n_resid_sample_* describes the base the RESIDUALIZATION uses, which is step 18's
    # concern, not this step's. It is computed here because prepare_inputs builds the ROI
    # maps, but step 18 publishes it -- one quantity, one owner, one registry.
    nums = {k: v for k, v in nums.items() if not k.startswith("n_resid_sample_")}
    path = write_numbers(STEP_RESULTS_DIR, nums, prefix="step15")
    if verbose:
        print(f"\n  Wrote {len(nums)} numbers: {path}")
        print("=" * 70)
        print("STEP 15 COMPLETE")
        print("=" * 70)
    return nums


def main():
    ap = argparse.ArgumentParser(description="Step 15 — Imaging quality control.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--refit", action="store_true",
                    help="rescan every SPM.mat instead of loading the saved motion table")
    a = ap.parse_args()
    run_step15(verbose=not a.quiet, refit=a.refit)


if __name__ == "__main__":
    main()
