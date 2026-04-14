"""
Step 10 — Fit Pain-to-Sleep moderation models (arousal relay ROIs).
======================================================================

Input:  derivatives/step2_processed_long.csv
        derivatives/step9_ps_fmri_roi_values.csv
        derivatives/step9_ps_vbm_roi_values.csv
Output:
  derivatives/
    step10_ps_fmri_posterior_draws.npz
    step10_ps_vbm_posterior_draws.npz
  results/
    step9_table_s1_fmri_arousal.csv    — Table S1 fMRI panel
    step9_table_s1_vbm_arousal.csv     — Table S1 VBM panel
    step9_vbm_sign_concordance.csv     — VBM 5/5 sign test
    step9_text_numbers.csv

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
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step11_ps_moderation")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step11_ps_moderation")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

IN_PROCESSED_CSV = os.path.join(DERIV_DIR, "step3_varx_data", "step3_processed_long.csv")
IN_FMRI_CSV = os.path.join(DERIV_DIR, "step10_ps_roi_values", "step10_ps_fmri_roi_values.csv")
IN_VBM_CSV = os.path.join(DERIV_DIR, "step10_ps_roi_values", "step10_ps_vbm_roi_values.csv")

OUT_FMRI_DRAWS = os.path.join(STEP_DERIV_DIR, "step11_ps_fmri_posterior_draws.npz")
OUT_VBM_DRAWS = os.path.join(STEP_DERIV_DIR, "step11_ps_vbm_posterior_draws.npz")
SUPP_DIR = os.path.join(RESULTS_DIR, "supplementary_materials")
os.makedirs(SUPP_DIR, exist_ok=True)
OUT_FMRI_TABLE = os.path.join(SUPP_DIR, "table_s1_fmri_arousal.csv")
OUT_VBM_TABLE = os.path.join(SUPP_DIR, "table_s1_vbm_arousal.csv")
OUT_VBM_SIGN = os.path.join(SUPP_DIR, "vbm_sign_concordance.csv")
OUT_TEXT_CSV = os.path.join(STEP_RESULTS_DIR, "step11_text_numbers.csv")


def load_step2_data(csv_path):
    df = pd.read_csv(csv_path)
    df["Age_z"] = (df["Age"] - df["Age"].mean()) / df["Age"].std()
    df["Sex_coded"] = (df["Sex"] == 2).astype(float)
    df["Sex_c"] = df["Sex_coded"] - df["Sex_coded"].mean()
    unique_ids = sorted(df["ID"].unique())
    id_map = {sid: i for i, sid in enumerate(unique_ids)}
    df["pid_idx"] = df["ID"].map(id_map)
    model_df = df.dropna(subset=["pain_within_lag1", "sleep_within_lag1"]).copy()
    return df, model_df, unique_ids, id_map


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
            progressbar=True,
        )

        n_valid = len(valid_ids)
        n_obs = len(sub_df)
        results = extract_results(idata, moderator_name=roi_name)

        sp_mean = results.get("gamma_sp_mean", np.nan)
        sp_p_neg = results.get("gamma_sp_prob_neg", np.nan)
        sp_p = 2 * min(sp_p_neg, 1 - sp_p_neg) if np.isfinite(sp_p_neg) else np.nan

        ps_mean = results.get("gamma_ps_mean", np.nan)
        ps_lo = results.get("gamma_ps_ci_lo", np.nan)
        ps_hi = results.get("gamma_ps_ci_hi", np.nan)
        ps_p_neg = results.get("gamma_ps_prob_neg", np.nan)
        ps_p = 2 * min(ps_p_neg, 1 - ps_p_neg) if np.isfinite(ps_p_neg) else np.nan

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


def run_step11(verbose=True):
    if verbose:
        print("=" * 70)
        print("STEP 9 — Fit Pain-to-Sleep moderation (arousal relay ROIs)")
        print("=" * 70)

    os.makedirs(DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    _, model_df, unique_ids, id_map = load_step2_data(IN_PROCESSED_CSV)

    text_rows = []
    def _t(metric, value, note=""):
        text_rows.append({"metric": metric, "value": str(value), "note": note})

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

    sign_p = sum(comb(n_tested, k) * 0.5 ** n_tested
                 for k in range(n_concordant, n_tested + 1))

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
        print("\n" + "=" * 70)
        print("STEP 9 COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 10 — fit PS moderation (arousal relay ROIs)."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_step11(verbose=not args.quiet)


if __name__ == "__main__":
    main()
