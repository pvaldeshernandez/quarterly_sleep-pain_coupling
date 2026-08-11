"""
Step 11 — Fit Pain-to-Sleep moderation models (arousal relay ROIs).
======================================================================

Input:  derivatives/step07_varx_data/step07_processed_long.csv
        derivatives/step20_ps_roi_values/step20_ps_fmri_roi_values.csv
        derivatives/step20_ps_roi_values/step20_ps_vbm_roi_values.csv
Output:
  derivatives/
    step21_ps_fmri_posterior_draws.npz
    step21_ps_vbm_posterior_draws.npz
  results/
    step21_table_s1_fmri_arousal.csv    — Table S1 fMRI panel
    step21_table_s1_vbm_arousal.csv     — Table S1 VBM panel
    step21_vbm_sign_concordance.csv     — VBM 5/5 sign test
    step21_text_numbers.csv

Fits fit_bayesian_varx1 with X_person = z-scored ROI value for
each arousal ROI, separately for fMRI BOLD and VBM GM volume.
Runs the VBM sign concordance test (5/5 negative expected).

Author: Pedro Valdes-Hernandez (with Claude Opus 4.6)
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from math import comb

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DERIV_DIR = os.path.join(ROOT, "derivatives")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step21_ps_moderation")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step21_ps_moderation")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

IN_PROCESSED_CSV = os.path.join(DERIV_DIR, "step07_varx_data", "step07_processed_long.csv")
IN_FMRI_CSV = os.path.join(DERIV_DIR, "step20_ps_roi_values", "step20_ps_fmri_roi_values.csv")
IN_VBM_CSV = os.path.join(DERIV_DIR, "step20_ps_roi_values", "step20_ps_vbm_roi_values.csv")

OUT_FMRI_DRAWS = os.path.join(STEP_DERIV_DIR, "step21_ps_fmri_posterior_draws.npz")
OUT_VBM_DRAWS = os.path.join(STEP_DERIV_DIR, "step21_ps_vbm_posterior_draws.npz")
#: A step writes into its OWN results folder. tools/collect_deliverables.py
#: copies what the documents need into results/manuscript/ and
#: results/supplementary_materials/ under their document-facing names.
SUPP_DIR = STEP_RESULTS_DIR
os.makedirs(SUPP_DIR, exist_ok=True)
OUT_FMRI_TABLE = os.path.join(SUPP_DIR, "table_s1_fmri_arousal.csv")
OUT_VBM_TABLE = os.path.join(SUPP_DIR, "table_s1_vbm_arousal.csv")
OUT_VBM_SIGN = os.path.join(STEP_DERIV_DIR, "step21_vbm_sign_concordance.csv")
OUT_TEXT_CSV = os.path.join(STEP_DERIV_DIR, "step21_text_numbers.csv")


def load_step05_data(csv_path):
    """Thin wrapper around lib.coupling_model.load_varx_frame."""
    from coupling_model import load_varx_frame
    return load_varx_frame(csv_path, verbose=False)


def fit_modality(modality_name, roi_csv_path, model_df, unique_ids, id_map,
                 verbose=True):
    """Fit moderation models for all ROIs in one modality (fMRI or VBM)."""
    from coupling_model import fit_bayesian_varx1, extract_results

    roi_df = pd.read_csv(roi_csv_path)
    all_rois = roi_df["ROI"].unique()
    if verbose:
        print(f"\n  {modality_name}: {len(all_rois)} ROIs to fit")

    table_rows = []
    draws_dict = {}

    for roi_name in all_rois:
        roi_data = roi_df[roi_df["ROI"] == roi_name]
        label = roi_data["label"].iloc[0]
        expected = roi_data["expected_sign_ps"].iloc[0]
        raw_mean = roi_data["raw_mean"].iloc[0]
        raw_sd = roi_data["raw_sd"].iloc[0]

        X_person = dict(zip(
            roi_data["ID"].astype(str), roi_data["z_value"].values
        ))

        if verbose:
            print(f"\n    Fitting: {label}...")

        idata, sub_df, valid_ids = fit_bayesian_varx1(
            model_df, unique_ids, id_map,
            X_person=X_person, include_agesex=True,
            moderator_direction="ps",
            progressbar=True,
            fit_id=f"step21_ps_{modality_name}_{roi_name}", out_dir=STEP_DERIV_DIR,
        )

        n_valid = len(valid_ids)
        n_obs = len(sub_df)
        results = extract_results(idata, moderator_name=roi_name)

        from coupling_model import two_tail_p
        # With moderator_direction="ps", gamma_sp is absent from the posterior.
        sp_mean = results.get("gamma_sp_mean", np.nan)
        sp_p = two_tail_p(results.get("gamma_sp_prob_neg", np.nan))

        ps_mean = results.get("gamma_ps_mean", np.nan)
        ps_lo = results.get("gamma_ps_ci_lo", np.nan)
        ps_hi = results.get("gamma_ps_ci_hi", np.nan)
        ps_p = two_tail_p(results.get("gamma_ps_prob_neg", np.nan))

        rhat_max = results.get("rhat_max", np.nan)

        if verbose:
            print(f"      N={n_valid}, obs={n_obs}")
            print(f"      gamma_ps = {ps_mean:+.4f} [{ps_lo:+.4f}, {ps_hi:+.4f}], p={ps_p:.3f}")
            print(f"      R-hat max: {rhat_max:.3f}")

        table_rows.append({
            "ROI": roi_name, "Label": label, "Modality": modality_name,
            "N": n_valid, "N_obs": n_obs,
            "gamma_sp": sp_mean, "gamma_sp_p": sp_p,
            "gamma_ps": ps_mean, "gamma_ps_ci_lo": ps_lo,
            "gamma_ps_ci_hi": ps_hi, "gamma_ps_p": ps_p,
            "expected_sign_ps": expected,
            "rhat_max": rhat_max,
        })

        # Save draws for JN
        b1_draws = idata.posterior["b1"].values.flatten()
        gamma_ps_draws = idata.posterior["gamma_ps"].values.flatten()
        X_vals = np.array([X_person[sid] for sid in valid_ids if sid in X_person])
        u_ps_mean = idata.posterior["u_ps"].values.reshape(-1, len(valid_ids)).mean(axis=0)
        draws_dict[f"{roi_name}_b1_draws"] = b1_draws
        draws_dict[f"{roi_name}_gamma_ps_draws"] = gamma_ps_draws
        draws_dict[f"{roi_name}_X_vals"] = X_vals
        draws_dict[f"{roi_name}_raw_mean"] = np.array([raw_mean])
        draws_dict[f"{roi_name}_raw_sd"] = np.array([raw_sd])
        draws_dict[f"{roi_name}_u_ps_mean"] = u_ps_mean

        del idata

    return pd.DataFrame(table_rows), draws_dict


def run_step21(verbose=True, refit=False):
    if verbose:
        print("=" * 70)
        print("STEP 11 — Fit Pain-to-Sleep moderation (arousal relay ROIs)")
        print("=" * 70)

    os.makedirs(DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    # Check whether saved derivatives exist
    saved_exist = (os.path.exists(OUT_FMRI_DRAWS) and os.path.exists(OUT_VBM_DRAWS)
                   and os.path.exists(OUT_FMRI_TABLE) and os.path.exists(OUT_VBM_TABLE))
    if not refit and not saved_exist:
        if verbose:
            print("  Saved derivatives not found — forcing refit.")
        refit = True

    if not refit and saved_exist:
        # ------ REPLOT MODE: load saved derivatives ------
        if verbose:
            print("  WARNING: Running in replot mode -- loading saved derivatives.")
            print("  If you have changed upstream data or code, re-run with --refit.")
        fmri_table = pd.read_csv(OUT_FMRI_TABLE)
        fmri_draws = dict(np.load(OUT_FMRI_DRAWS))
        vbm_table = pd.read_csv(OUT_VBM_TABLE)
        vbm_draws = dict(np.load(OUT_VBM_DRAWS))
    else:
        # ------ FULL MCMC FIT ------
        _, model_df, unique_ids, id_map = load_step05_data(IN_PROCESSED_CSV)

        # ---- fMRI BOLD ----
        fmri_table, fmri_draws = fit_modality(
            "fMRI_BOLD", IN_FMRI_CSV, model_df, unique_ids, id_map, verbose
        )
        fmri_table.to_csv(OUT_FMRI_TABLE, index=False)
        np.savez(OUT_FMRI_DRAWS, **fmri_draws)
        if verbose:
            print(f"\n  Saved fMRI table: {OUT_FMRI_TABLE}")
            print(f"  Saved fMRI draws: {OUT_FMRI_DRAWS}")

        # ---- VBM GM volume ----
        vbm_table, vbm_draws = fit_modality(
            "VBM_GM_volume", IN_VBM_CSV, model_df, unique_ids, id_map, verbose
        )
        vbm_table.to_csv(OUT_VBM_TABLE, index=False)
        np.savez(OUT_VBM_DRAWS, **vbm_draws)
        if verbose:
            print(f"\n  Saved VBM table: {OUT_VBM_TABLE}")
            print(f"  Saved VBM draws: {OUT_VBM_DRAWS}")

    text_rows = []
    def _t(metric, value, note=""):
        text_rows.append({"metric": metric, "value": str(value), "note": note})

    # ---- VBM sign concordance (5/5 all negative expected) ----
    n_concordant = 0
    n_tested = 0
    sign_rows = []
    for _, row in vbm_table.iterrows():
        gamma = row["gamma_ps"]
        expected = row["expected_sign_ps"]
        observed = "-" if gamma < 0 else "+"
        match = observed == expected
        if match:
            n_concordant += 1
        n_tested += 1
        sign_rows.append({
            "ROI": row["ROI"], "gamma_ps": gamma,
            "expected": expected, "observed": observed, "match": match,
        })

    from coupling_model import sign_concordance_p
    sign_p = sign_concordance_p(n_concordant, n_tested)

    sign_df = pd.DataFrame(sign_rows)
    sign_df.to_csv(OUT_VBM_SIGN, index=False)
    if verbose:
        print(f"\n  VBM sign concordance: {n_concordant}/{n_tested}, p = {sign_p:.4f}")

    # ---- Text numbers ----
    for _, row in fmri_table.iterrows():
        _t(f"gamma_ps_fmri_{row['ROI']}", f"{row['gamma_ps']:+.4f}")
        _t(f"gamma_ps_fmri_{row['ROI']}_p", f"{row['gamma_ps_p']:.3f}")
        _t(f"N_fmri_{row['ROI']}", str(row["N"]))
    for _, row in vbm_table.iterrows():
        _t(f"gamma_ps_vbm_{row['ROI']}", f"{row['gamma_ps']:+.4f}")
        _t(f"gamma_ps_vbm_{row['ROI']}_p", f"{row['gamma_ps_p']:.3f}")
        _t(f"N_vbm_{row['ROI']}", str(row["N"]))
    _t("vbm_sign_concordance", f"{n_concordant}/{n_tested}")
    _t("vbm_sign_concordance_p", f"{sign_p:.4f}")

    pd.DataFrame(text_rows).to_csv(OUT_TEXT_CSV, index=False)
    if verbose:
        print(f"  Saved text numbers: {OUT_TEXT_CSV}")


    if verbose:
        print("\n" + "=" * 70)
        print("STEP 11 COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 11 — fit PS moderation (arousal relay ROIs)."
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--refit", action="store_true",
                        help="Re-run computation from scratch instead of loading saved derivatives")
    args = parser.parse_args()
    run_step21(verbose=not args.quiet, refit=args.refit)


if __name__ == "__main__":
    main()
