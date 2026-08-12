"""
Step 08 — Fit Sleep-to-Pain moderation models (7 ROIs).
======================================================================

Input:  derivatives/step07_varx_data/step07_processed_long.csv
        derivatives/step14_sp_roi_values/step14_sp_roi_values.csv
Output:
  derivatives/
    step16_sp_posterior_draws.npz     — per-ROI posterior draws
  results/
    step16_table5_sp_moderation.csv   — Table 5: fMRI SP moderators
    step16_sign_concordance.csv       — sign test results
    step16_text_numbers.csv           — gamma estimates, p-values, etc.

Fits fit_bayesian_varx1 with X_person = z-scored ROI value for
each of the 7 SP ROIs (6 Krause + 1 ACC). Extracts gamma_sp and
gamma_ps posterior summaries. Runs the 6-ROI Krause sign
concordance test.

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
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step16_sp_moderation")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step16_sp_moderation")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

IN_PROCESSED_CSV = os.path.join(DERIV_DIR, "step07_varx_data", "step07_processed_long.csv")
IN_ROI_CSV = os.path.join(DERIV_DIR, "step14_sp_roi_values", "step14_sp_roi_values.csv")

OUT_DRAWS_NPZ = os.path.join(STEP_DERIV_DIR, "step16_sp_posterior_draws.npz")
OUT_TABLE5_CSV = os.path.join(STEP_RESULTS_DIR, "step16_table5_sp_moderation.csv")
OUT_SIGN_CSV = os.path.join(STEP_RESULTS_DIR, "step16_sign_concordance.csv")
OUT_TEXT_CSV = os.path.join(STEP_RESULTS_DIR, "step16_text_numbers.csv")

# ROIs included in the Krause sign-concordance test (ACC excluded)
KRAUSE_ROIS = [
    "Contra_S1", "Contra_Middle_Insula", "Left_Thalamus",
    "Left_Anterior_Insula", "Left_NAcc", "Right_NAcc",
]

# Expected signs for the sign test
EXPECTED_SIGNS = {
    "Contra_S1": "-",
    "Contra_Middle_Insula": "+",
    "Left_Thalamus": "+",
    "Left_Anterior_Insula": "+",
    "Left_NAcc": "+",
    "Right_NAcc": "+",
}


def load_step05_data(csv_path):
    """Thin wrapper around lib.coupling_model.load_varx_frame."""
    from coupling_model import load_varx_frame
    return load_varx_frame(csv_path, verbose=False)


def run_step16(verbose=True, refit=False):
    from coupling_model import fit_bayesian_varx1, extract_results

    if verbose:
        print("=" * 70)
        print("STEP 08 — Fit Sleep-to-Pain moderation (7 ROIs)")
        print("=" * 70)

    os.makedirs(DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    roi_df = pd.read_csv(IN_ROI_CSV)

    # Check whether saved derivatives exist
    saved_exist = os.path.exists(OUT_DRAWS_NPZ) and os.path.exists(OUT_TABLE5_CSV)
    if not refit and not saved_exist:
        if verbose:
            print("  Saved derivatives not found — forcing refit.")
        refit = True

    if not refit and saved_exist:
        # ------ REPLOT MODE: load saved derivatives ------
        if verbose:
            print("  WARNING: Running in replot mode -- loading saved derivatives.")
            print("  If you have changed upstream data or code, re-run with --refit.")
        table5 = pd.read_csv(OUT_TABLE5_CSV)
        draws_dict = dict(np.load(OUT_DRAWS_NPZ))
    else:
        # ------ FULL MCMC FIT ------
        df_full, model_df, unique_ids, id_map = load_step05_data(IN_PROCESSED_CSV)

        all_rois = roi_df["ROI"].unique()
        if verbose:
            print(f"  {len(all_rois)} ROIs to fit")

        table_rows = []
        draws_dict = {}

        for roi_name in all_rois:
            roi_data = roi_df[roi_df["ROI"] == roi_name]
            label = roi_data["label"].iloc[0]
            framework = roi_data["framework"].iloc[0]
            mask_type = roi_data["mask_type"].iloc[0]
            expected = roi_data["expected_sign_sp"].iloc[0]
            raw_mean = roi_data["raw_mean"].iloc[0]
            raw_sd = roi_data["raw_sd"].iloc[0]

            # Build X_person dict {ID: z_value}
            X_person = dict(zip(
                roi_data["ID"].astype(str),
                roi_data["z_value"].values,
            ))

            if verbose:
                print(f"\n  Fitting: {label} ({framework}, {mask_type})...")

            idata, sub_df, valid_ids = fit_bayesian_varx1(
                model_df, unique_ids, id_map,
                X_person=X_person,
                include_agesex=True,
                moderator_direction="sp",
                progressbar=True,
                fit_id=f"step16_sp_{roi_name}", out_dir=STEP_DERIV_DIR,
            )

            n_valid = len(valid_ids)
            n_obs = len(sub_df)
            results = extract_results(idata, moderator_name=roi_name)

            from coupling_model import two_tail_p
            sp_mean = results.get("gamma_sp_mean", np.nan)
            sp_lo = results.get("gamma_sp_ci_lo", np.nan)
            sp_hi = results.get("gamma_sp_ci_hi", np.nan)
            sp_p = two_tail_p(results.get("gamma_sp_prob_neg", np.nan))

            # With moderator_direction="sp" the PS slope has no gamma_ps
            # term, so its posterior is absent. Record NaN.
            ps_mean = results.get("gamma_ps_mean", np.nan)
            ps_lo = results.get("gamma_ps_ci_lo", np.nan)
            ps_hi = results.get("gamma_ps_ci_hi", np.nan)
            ps_p = two_tail_p(results.get("gamma_ps_prob_neg", np.nan))

            rhat_max = results.get("rhat_max", np.nan)

            if verbose:
                print(f"    N={n_valid}, obs={n_obs}")
                print(f"    gamma_sp = {sp_mean:+.4f} "
                      f"[{sp_lo:+.4f}, {sp_hi:+.4f}], p={sp_p:.4f}")
                print(f"    R-hat max: {rhat_max:.3f}")

            table_rows.append({
                "ROI": roi_name, "Label": label, "Framework": framework,
                "Source": mask_type, "N": n_valid, "N_obs": n_obs,
                "gamma_sp": sp_mean, "gamma_sp_ci_lo": sp_lo,
                "gamma_sp_ci_hi": sp_hi, "gamma_sp_p": sp_p,
                "gamma_ps": ps_mean, "gamma_ps_ci_lo": ps_lo,
                "gamma_ps_ci_hi": ps_hi, "gamma_ps_p": ps_p,
                "rhat_max": rhat_max,
            })

            # Save posterior draws for JN step
            a2_draws = idata.posterior["a2"].values.flatten()
            gamma_sp_draws = idata.posterior["gamma_sp"].values.flatten()
            u_sp_mean = idata.posterior["u_sp"].values.reshape(-1, len(valid_ids)).mean(axis=0)
            X_vals = np.array([X_person[sid] for sid in valid_ids if sid in X_person])
            draws_dict[f"{roi_name}_a2_draws"] = a2_draws
            draws_dict[f"{roi_name}_gamma_sp_draws"] = gamma_sp_draws
            draws_dict[f"{roi_name}_X_vals"] = X_vals
            draws_dict[f"{roi_name}_raw_mean"] = np.array([raw_mean])
            draws_dict[f"{roi_name}_raw_sd"] = np.array([raw_sd])
            draws_dict[f"{roi_name}_u_sp_mean"] = u_sp_mean

            del idata

        # Save Table 5
        table5 = pd.DataFrame(table_rows)
        table5.to_csv(OUT_TABLE5_CSV, index=False)
        if verbose:
            print(f"\n  Saved Table 5: {OUT_TABLE5_CSV}")

        # Save posterior draws
        np.savez(OUT_DRAWS_NPZ, **draws_dict)
        if verbose:
            print(f"  Saved draws: {OUT_DRAWS_NPZ}")

    text_rows = []

    # Sign concordance test (Krause 6 ROIs only)
    n_concordant = 0
    n_tested = 0
    sign_rows = []
    for roi_name in KRAUSE_ROIS:
        row = table5[table5["ROI"] == roi_name]
        if len(row) == 0:
            continue
        gamma = row["gamma_sp"].iloc[0]
        expected = EXPECTED_SIGNS[roi_name]
        observed = "+" if gamma > 0 else "-"
        match = observed == expected
        if match:
            n_concordant += 1
        n_tested += 1
        sign_rows.append({
            "ROI": roi_name, "gamma_sp": gamma,
            "expected": expected, "observed": observed,
            "match": match,
        })

    from coupling_model import sign_concordance_p
    sign_p = sign_concordance_p(n_concordant, n_tested)

    sign_df = pd.DataFrame(sign_rows)
    sign_df.to_csv(OUT_SIGN_CSV, index=False)
    if verbose:
        print(f"\n  Sign concordance: {n_concordant}/{n_tested}, p = {sign_p:.4f}")
        print(f"  Saved: {OUT_SIGN_CSV}")

    # Text numbers
    def _t(metric, value, note=""):
        text_rows.append({"metric": metric, "value": str(value), "note": note})

    for _, row in table5.iterrows():
        roi = row["ROI"]
        _t(f"gamma_sp_{roi}", f"{row['gamma_sp']:+.4f}")
        _t(f"gamma_sp_{roi}_ci", f"[{row['gamma_sp_ci_lo']:+.4f}, {row['gamma_sp_ci_hi']:+.4f}]")
        _t(f"gamma_sp_{roi}_p", f"{row['gamma_sp_p']:.4f}")
        _t(f"gamma_ps_{roi}", f"{row['gamma_ps']:+.4f}")
        _t(f"gamma_ps_{roi}_p", f"{row['gamma_ps_p']:.4f}")
        _t(f"N_{roi}", str(row["N"]))

    _t("sign_concordance", f"{n_concordant}/{n_tested}")
    _t("sign_concordance_p", f"{sign_p:.4f}")

    # NAcc-ACC correlations (left NAcc vs each ACC hemisphere)
    nacc_data = roi_df[roi_df["ROI"] == "Left_NAcc"][["ID", "z_value"]].rename(
        columns={"z_value": "nacc"})
    for acc_roi, acc_label in [("Left_dACC_MCC", "left"), ("Right_dACC_MCC", "right")]:
        acc_data = roi_df[roi_df["ROI"] == acc_roi][["ID", "z_value"]].rename(
            columns={"z_value": "acc"})
        merged = nacc_data.merge(acc_data, on="ID")
        if len(merged) > 10:
            r_val = float(np.corrcoef(merged["nacc"], merged["acc"])[0, 1])
            _t(f"r_nacc_acc_{acc_label}", f"{r_val:.2f}")
            if verbose:
                print(f"  Left NAcc vs {acc_label} ACC BOLD correlation: r = {r_val:.2f}")

    text_df = pd.DataFrame(text_rows)
    text_df.to_csv(OUT_TEXT_CSV, index=False)
    if verbose:
        print(f"  Saved: {OUT_TEXT_CSV}")


    if verbose:
        print("\n" + "=" * 70)
        print("STEP 08 COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 08 — fit SP moderation models (7 ROIs)."
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--refit", action="store_true",
                        help="Re-run computation from scratch instead of loading saved derivatives")
    args = parser.parse_args()
    run_step16(verbose=not args.quiet, refit=args.refit)


if __name__ == "__main__":
    main()
