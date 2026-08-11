#!/usr/bin/env python3
"""
Step 18 — Nuisance-adjusted sleep-to-pain fMRI moderation (Section S11, Table S8).
==================================================================================

Each of the four Table S8 ROIs is refit with its moderator residualized on scanner site,
on evoked pain, on mean framewise displacement, and on all three jointly.
4 ROIs x 4 schemes = 16 fits.

SPLIT OUT OF step 15 on 11 Aug 2026. This half reads Table 5 for its unadjusted column,
so it must run AFTER the moderation fits; the QC half must run BEFORE them, where the
paper reports it under "Final MRI samples". One step could not be in both places, and
leaving it whole meant step 15 silently read the PREVIOUS run's Table 5.

SAMPLES are imported from step 15's `prepare_inputs`, never redefined here. The
residualization and z-scoring use the FULL ROI table (188 rows; 182 for the two
contralateralized ROIs), because step 14 z-scored the unadjusted moderator over exactly
those rows. gamma is "per one SD of the moderator", so standardizing the adjusted gamma
over a different base would leave Table S8's columns incomparable to Table 5's -- with no
error and no visible symptom.

Input:  results/step16_sp_moderation/step16_table5_sp_moderation.csv
Output: results/step18_nuisance_adjusted/step18_tableS8_nuisance_sensitivity.csv
        derivatives/step18_nuisance_adjusted/step18_nuisance_gamma_draws.npz
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
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step18_nuisance_adjusted")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step18_nuisance_adjusted")

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
OUT_DRAWS_NPZ = os.path.join(STEP_DERIV_DIR, "step18_nuisance_gamma_draws.npz")
OUT_DIAG_CSV = os.path.join(STEP_DERIV_DIR, "step18_diagnostics.csv")
OUT_FD_CORR_CSV = os.path.join(STEP_RESULTS_DIR, "step15_motion_correlations.csv")
OUT_PAIN_CORR_CSV = os.path.join(STEP_RESULTS_DIR, "step15_evokedpain_correlations.csv")
OUT_SITE_CSV = os.path.join(STEP_RESULTS_DIR, "step15_site_differences.csv")
OUT_TABLE_S8_CSV = os.path.join(STEP_RESULTS_DIR,
                                "step18_tableS8_nuisance_sensitivity.csv")

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


from step15_imaging_qc import prepare_inputs  # noqa: E402


def run_step18(verbose=True, refit=False):
    """The 16 nuisance-adjusted moderation fits."""
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)
    if verbose:
        print("=" * 70)
        print("STEP 18 — Nuisance-adjusted SP moderation")
        print("=" * 70)

    ctx = prepare_inputs(verbose=verbose, refit=False)
    # step15 owns and publishes the sample counts; republishing them here would put the
    # same named quantity in two registries, which is exactly how one quantity ends up
    # reported with two values. Only the per-ROI residualization sizes, which are this
    # step's own, are carried through.
    nums = {k: v for k, v in ctx["nums"].items() if k.startswith("n_resid_sample_")}
    roi_maps = ctx["roi_maps"]
    site_map, pain_map, fd_map = ctx["site_map"], ctx["pain_map"], ctx["fd_map"]

    # ------------------------------------------------------------------
    # The 16 nuisance-adjusted fits
    # ------------------------------------------------------------------
    covmaps = {"site": [site_map], "pain": [pain_map], "motion": [fd_map],
               "all3": [site_map, pain_map, fd_map]}

    have_fits = os.path.exists(OUT_DRAWS_NPZ) and os.path.exists(OUT_TABLE_S8_CSV)
    if not refit and not have_fits:
        if verbose:
            print("  Saved fits not found — forcing refit of the 16 models.")
        refit_fits = True
    else:
        refit_fits = refit

    if not refit_fits:
        # ------ LOAD MODE ------
        if verbose:
            print("  WARNING: Running in load mode -- using saved posterior draws.")
            print("  If you have changed upstream data or code, re-run with --refit.")
        table_s8 = pd.read_csv(OUT_TABLE_S8_CSV)
        draws = dict(np.load(OUT_DRAWS_NPZ, allow_pickle=False))
    else:
        # ------ FULL MCMC: 4 ROIs x 4 schemes ------
        pub = pd.read_csv(IN_TABLE5_CSV).set_index("ROI")
        draws, table_rows = {}, []
        for roi in TABLE_S8_ROIS:
            mod = roi_maps[roi]
            ids = list(mod)
            z = np.array([mod[i] for i in ids], dtype=float)
            row = {"ROI": roi, "n_resid_sample": len(ids)}
            for col, key in (("gamma_sp_unadjusted", "gamma_sp"),
                             ("ci_lo_unadjusted", "gamma_sp_ci_lo"),
                             ("ci_hi_unadjusted", "gamma_sp_ci_hi"),
                             ("p_unadjusted", "gamma_sp_p")):
                row[col] = float(pub.at[roi, key]) if key in pub.columns else np.nan

            for scheme in SCHEMES:
                resid, ok = residualize(ids, z, covmaps[scheme])
                X_person = {i: float(v) for i, v, k in zip(ids, resid, ok) if k}
                fit_id = f"step18_{scheme}_{roi}"
                if verbose:
                    print(f"\n  Fitting {fit_id} "
                          f"(moderator residualized on {scheme}, "
                          f"n = {len(X_person)} with covariates)...")

                idata, sub_df, valid_ids = fit_bayesian_varx1(
                    model_df, unique_ids, id_map,
                    X_person=X_person,
                    include_agesex=True,
                    moderator_direction="sp",
                    progressbar=False,
                    fit_id=fit_id, out_dir=STEP_DERIV_DIR,
                )
                res = extract_results(idata, moderator_name=roi)
                g_mean = res.get("gamma_sp_mean", np.nan)
                g_lo = res.get("gamma_sp_ci_lo", np.nan)
                g_hi = res.get("gamma_sp_ci_hi", np.nan)
                g_p = two_tail_p(res.get("gamma_sp_prob_neg", np.nan))

                row[f"gamma_sp_{scheme}"] = g_mean
                row[f"ci_lo_{scheme}"] = g_lo
                row[f"ci_hi_{scheme}"] = g_hi
                row[f"p_{scheme}"] = g_p
                row[f"n_persons_{scheme}"] = int(len(valid_ids))
                row[f"n_obs_{scheme}"] = int(len(sub_df))
                row[f"n_resid_ok_{scheme}"] = int(ok.sum())

                stem = f"{roi}_{scheme}"
                draws[f"{stem}_gamma_sp_draws"] = \
                    idata.posterior["gamma_sp"].values.flatten()
                # The adjusted moderator itself, in fit order: without it the
                # residualization cannot be checked afterwards from saved output.
                draws[f"{stem}_X_vals"] = np.array(
                    [X_person[s] for s in valid_ids], dtype=float)
                draws[f"{stem}_n_persons"] = np.array([len(valid_ids)])

                if verbose:
                    print(f"    gamma_sp = {g_mean:+.4f} [{g_lo:+.4f}, {g_hi:+.4f}], "
                          f"p = {g_p:.4f}, N = {len(valid_ids)}")
                del idata

            table_rows.append(row)

        table_s8 = pd.DataFrame(table_rows)
        table_s8.to_csv(OUT_TABLE_S8_CSV, index=False)
        np.savez(OUT_DRAWS_NPZ, **draws)
        if verbose:
            print(f"\n  Saved Table S8 data: {OUT_TABLE_S8_CSV}")
            print(f"  Saved draws: {OUT_DRAWS_NPZ}")

    # The NPZ is the provenance of the table: one gamma_sp draw array per fit.
    expected_draws = [f"{roi}_{scheme}_gamma_sp_draws"
                      for roi in TABLE_S8_ROIS for scheme in SCHEMES]
    missing_draws = [k for k in expected_draws if k not in draws]
    if missing_draws:
        raise ValueError(f"{OUT_DRAWS_NPZ} is missing draws for: {missing_draws}")
    nums["n_draws_per_fit"] = int(len(draws[expected_draws[0]]))

    # numbers from the table, identically on both paths
    n_persons_seen = []
    for _, row in table_s8.iterrows():
        roi = row["ROI"]
        for scheme in SCHEMES:
            nums[f"gamma_sp_{roi}_{scheme}"] = float(row[f"gamma_sp_{scheme}"])
            nums[f"gamma_sp_p_{roi}_{scheme}"] = float(row[f"p_{scheme}"])
            nums[f"gamma_sp_ci_lo_{roi}_{scheme}"] = float(row[f"ci_lo_{scheme}"])
            nums[f"gamma_sp_ci_hi_{roi}_{scheme}"] = float(row[f"ci_hi_{scheme}"])
            nums[f"n_resid_ok_{roi}_{scheme}"] = int(row[f"n_resid_ok_{scheme}"])
            n_persons_seen.append(int(row[f"n_persons_{scheme}"]))
    nums["n_fits"] = len(n_persons_seen)
    nums["n_persons_fit_min"] = int(min(n_persons_seen))
    nums["n_persons_fit_max"] = int(max(n_persons_seen))
    nums["n_persons_fit"] = int(max(set(n_persons_seen), key=n_persons_seen.count))
    if nums["n_persons_fit_min"] != nums["n_persons_fit_max"] and verbose:
        print(f"  WARNING: the 16 fits do not share one N "
              f"({nums['n_persons_fit_min']}-{nums['n_persons_fit_max']}); "
              f"'N = 174 throughout' would be false.")

    # ------------------------------------------------------------------
    # Diagnostics of the 16 fits (written by run_fit; aggregated here)
    # ------------------------------------------------------------------
    diag = read_diagnostics(STEP_DERIV_DIR, fit_id_prefix="step18_")
    if (diag is None or len(diag) == 0) and os.path.exists(OUT_DIAG_CSV):
        diag = pd.read_csv(OUT_DIAG_CSV)
    if diag is not None and len(diag):
        diag.to_csv(OUT_DIAG_CSV, index=False)
        # "all-parameter" flavor: run_fit's rhat_max/ess_* are computed over every
        # parameter except the non-centered offsets. Step 22 must read the same
        # columns, or the paper's two diagnostics sentences will disagree.
        nums["n_fit_records"] = int(len(diag))
        nums["rhat_max_all_16"] = float(diag["rhat_max"].max())
        nums["rhat_max_all_16_fit"] = str(diag.loc[diag["rhat_max"].idxmax(), "fit_id"])
        nums["ess_bulk_min_all_16"] = float(diag["ess_bulk_min"].min())
        nums["ess_bulk_min_all_16_fit"] = str(
            diag.loc[diag["ess_bulk_min"].idxmin(), "fit_id"])
        nums["ess_tail_min_all_16"] = float(diag["ess_tail_min"].min())
        nums["ess_tail_min_all_16_fit"] = str(
            diag.loc[diag["ess_tail_min"].idxmin(), "fit_id"])
        nums["n_divergences_16"] = int(diag["divergences"].sum())
        if "bfmi_min" in diag.columns:
            nums["bfmi_min_16"] = float(diag["bfmi_min"].min())
        if verbose:
            print(f"  Diagnostics over {len(diag)} fit(s): "
                  f"R-hat <= {nums['rhat_max_all_16']:.4f}, "
                  f"bulk ESS >= {nums['ess_bulk_min_all_16']:.0f}, "
                  f"tail ESS >= {nums['ess_tail_min_all_16']:.0f}, "
                  f"{nums['n_divergences_16']} divergence(s)")
    elif verbose:
        print("  WARNING: no diagnostics records for step 15 — the Table S8 note's "
              "R-hat/ESS/divergence numbers cannot be published. Re-run with --refit.")

    path = write_numbers(STEP_RESULTS_DIR, nums, prefix="step18")
    if verbose:
        print(f"\n  Wrote {len(nums)} numbers: {path}")
        print("=" * 70)
        print("STEP 18 COMPLETE")
        print("=" * 70)
    return nums




def main():
    ap = argparse.ArgumentParser(description="Step 18 — Nuisance-adjusted moderation.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--refit", action="store_true")
    a = ap.parse_args()
    run_step18(verbose=not a.quiet, refit=a.refit)


if __name__ == "__main__":
    main()
