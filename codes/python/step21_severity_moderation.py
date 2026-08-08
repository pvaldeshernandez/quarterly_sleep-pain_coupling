"""
Step 13 — Person-mean severity moderation of coupling (Table S2).
======================================================================

Tests whether person-mean pain severity or sleep quality moderate
the coupling parameters. Three models:
  1. Mean pain severity alone
  2. Mean sleep quality alone
  3. Both jointly

Input:  derivatives/step04_varx_data/step04_processed_long.csv
Output:
  results/step21_severity_moderation/
    step21_table_s2_severity.csv   — Table S2
    step21_text_numbers.csv        — estimates for manuscript text

Author: Pedro Valdes-Hernandez (with Claude Sonnet 4.6)
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DERIV_DIR = os.path.join(ROOT, "derivatives")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step21_severity_moderation")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
SUPP_DIR = os.path.join(RESULTS_DIR, "supplementary_materials")
os.makedirs(SUPP_DIR, exist_ok=True)

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

IN_PROCESSED_CSV = os.path.join(DERIV_DIR, "step04_varx_data",
                                "step04_processed_long.csv")

OUT_TABLE_CSV = os.path.join(SUPP_DIR, "table_s2_severity.csv")
OUT_TEXT_CSV = os.path.join(STEP_DERIV_DIR, "text_numbers_severity.csv")


def load_data(csv_path):
    """Thin wrapper around lib.coupling_model.load_varx_frame."""
    from coupling_model import load_varx_frame
    return load_varx_frame(csv_path, verbose=False)


MODELS = [
    {
        "name": "pain_alone",
        "label": "Mean Pain Severity",
        "model_label": "Alone",
        "moderators": ["pain_person_mean"],
    },
    {
        "name": "sleep_alone",
        "label": "Mean Sleep Quality",
        "model_label": "Alone",
        "moderators": ["sleep_person_mean"],
    },
    {
        "name": "joint",
        "label": None,  # both reported
        "model_label": "Joint",
        "moderators": ["pain_person_mean", "sleep_person_mean"],
    },
]


def run_step21(verbose=True, refit=False):
    from coupling_model import fit_bayesian_varx1, extract_results

    if verbose:
        print("=" * 70)
        print("STEP 13 — Person-mean severity moderation (Table S2)")
        print("=" * 70)

    # Check whether saved derivatives exist
    saved_exist = os.path.exists(OUT_TABLE_CSV) and os.path.exists(OUT_TEXT_CSV)
    if not refit and not saved_exist:
        if verbose:
            print("  Saved derivatives not found — forcing refit.")
        refit = True

    if not refit and saved_exist:
        # ------ REPLOT MODE: load saved derivatives ------
        if verbose:
            print("  WARNING: Running in replot mode -- loading saved derivatives.")
            print("  If you have changed upstream data or code, re-run with --refit.")
            print(f"\n  Loaded Table S2: {OUT_TABLE_CSV}")
            print(f"  Loaded text numbers: {OUT_TEXT_CSV}")
        if verbose:
            print("=" * 70)
        return

    # ------ FULL MCMC FIT ------
    df_full, model_df, unique_ids, id_map = load_data(IN_PROCESSED_CSV)

    # Compute person-mean z-scores for each moderator
    person_means = model_df.groupby("ID").agg(
        pain_pm=("pain_person_mean", "first"),
        sleep_pm=("sleep_person_mean", "first"),
    ).reset_index()
    pain_mean, pain_sd = person_means["pain_pm"].mean(), person_means["pain_pm"].std()
    sleep_mean, sleep_sd = person_means["sleep_pm"].mean(), person_means["sleep_pm"].std()

    if verbose:
        print(f"  Pain severity:  mean={pain_mean:.3f}, SD={pain_sd:.3f}")
        print(f"  Sleep quality:  mean={sleep_mean:.3f}, SD={sleep_sd:.3f}")
        print(f"  N={len(person_means)}, obs={len(model_df)}")

    table_rows = []
    text_rows = []

    def _t(metric, value, note=""):
        text_rows.append({"metric": metric, "value": str(value), "note": note})

    _t("pain_severity_mean", f"{pain_mean:.3f}")
    _t("pain_severity_sd", f"{pain_sd:.3f}")
    _t("sleep_quality_mean", f"{sleep_mean:.3f}")
    _t("sleep_quality_sd", f"{sleep_sd:.3f}")

    for model_cfg in MODELS:
        mod_name = model_cfg["name"]
        mod_label = model_cfg["model_label"]
        moderators = model_cfg["moderators"]

        if verbose:
            print(f"\n  Model: {mod_name} ({', '.join(moderators)})")

        # For single-moderator models, use fit_bayesian_varx1 with X_person
        # For joint model, we need to fit twice (once per moderator) or
        # use a custom approach. Since coupling_model only supports one
        # X_person at a time, fit twice for joint with both z-scored.
        # Actually — for joint, we need both in the same model.
        # coupling_model.fit_bayesian_varx1 only supports one moderator.
        # For the joint model, we'll fit with pain controlling for sleep
        # by residualizing, or we fit a custom model.
        #
        # Simpler approach: since Table S2 shows the joint model has
        # essentially the same results, and coupling_model supports only
        # one moderator, we fit each moderator separately for "Alone"
        # models. For "Joint", we need a 2-moderator model.
        #
        # Let's check if coupling_model supports multiple moderators...
        # It doesn't — X_person is a single dict. So for the joint model
        # we need to extend it or use a workaround.
        #
        # Workaround: for the joint model, residualize each moderator
        # against the other and fit with the residual.

        if mod_name == "joint":
            # Fit a 2-moderator model by extending the PyMC model directly
            _fit_joint_model(model_df, unique_ids, id_map,
                             pain_mean, pain_sd, sleep_mean, sleep_sd,
                             table_rows, text_rows, verbose,
                             out_dir=STEP_DERIV_DIR)
            continue

        # Single moderator
        col = moderators[0]
        if col == "pain_person_mean":
            pm_mean, pm_sd = pain_mean, pain_sd
            label = "Mean Pain Severity"
        else:
            pm_mean, pm_sd = sleep_mean, sleep_sd
            label = "Mean Sleep Quality"

        # Build X_person: z-scored person means
        X_person = {}
        for sid in unique_ids:
            sub = model_df[model_df["ID"] == sid]
            if len(sub) == 0:
                continue
            raw = sub[col].iloc[0]
            X_person[str(sid)] = (raw - pm_mean) / pm_sd

        idata, sub_df, valid_ids = fit_bayesian_varx1(
            model_df, unique_ids, id_map,
            X_person=X_person,
            include_agesex=True,
            progressbar=True,
            fit_id=f"step21_severity_{mod_name}", out_dir=STEP_DERIV_DIR,
        )

        from coupling_model import two_tail_p
        results = extract_results(idata, moderator_name=col)
        sp_mean = results.get("gamma_sp_mean", np.nan)
        sp_lo = results.get("gamma_sp_ci_lo", np.nan)
        sp_hi = results.get("gamma_sp_ci_hi", np.nan)
        sp_p = two_tail_p(results.get("gamma_sp_prob_neg", np.nan))

        ps_mean = results.get("gamma_ps_mean", np.nan)
        ps_lo = results.get("gamma_ps_ci_lo", np.nan)
        ps_hi = results.get("gamma_ps_ci_hi", np.nan)
        ps_p = two_tail_p(results.get("gamma_ps_prob_neg", np.nan))

        rhat = results.get("rhat_max", np.nan)

        if verbose:
            print(f"    gamma_sp = {sp_mean:+.4f} [{sp_lo:+.4f}, {sp_hi:+.4f}], "
                  f"p={sp_p:.4f}")
            print(f"    gamma_ps = {ps_mean:+.4f} [{ps_lo:+.4f}, {ps_hi:+.4f}], "
                  f"p={ps_p:.4f}")
            print(f"    R-hat max: {rhat:.3f}")

        table_rows.append({
            "Moderator": label, "Model": mod_label, "Direction": "SP",
            "gamma": sp_mean, "CrI_lo": sp_lo, "CrI_hi": sp_hi, "p": sp_p,
        })
        if mod_label == "Alone":
            # Also record the PS direction for Alone models (both directions
            # share the single moderator; Joint model only records SP pair).
            table_rows.append({
                "Moderator": label, "Model": mod_label, "Direction": "PS",
                "gamma": ps_mean, "CrI_lo": ps_lo, "CrI_hi": ps_hi, "p": ps_p,
            })

        _t(f"{mod_name}_gamma_sp", f"{sp_mean:+.4f}")
        _t(f"{mod_name}_gamma_sp_ci", f"[{sp_lo:+.4f}, {sp_hi:+.4f}]")
        _t(f"{mod_name}_gamma_sp_p", f"{sp_p:.4f}")
        _t(f"{mod_name}_gamma_ps", f"{ps_mean:+.4f}")
        _t(f"{mod_name}_gamma_ps_ci", f"[{ps_lo:+.4f}, {ps_hi:+.4f}]")
        _t(f"{mod_name}_gamma_ps_p", f"{ps_p:.4f}")
        _t(f"{mod_name}_rhat_max", f"{rhat:.3f}")

        del idata

    # Save Table S2
    table_s2 = pd.DataFrame(table_rows)
    table_s2.to_csv(OUT_TABLE_CSV, index=False)
    if verbose:
        print(f"\n  Saved Table S2: {OUT_TABLE_CSV}")

    pd.DataFrame(text_rows).to_csv(OUT_TEXT_CSV, index=False)
    if verbose:
        print(f"  Saved text numbers: {OUT_TEXT_CSV}")


    if verbose:
        print("=" * 70)


def _fit_joint_model(model_df, unique_ids, id_map,
                     pain_mean, pain_sd, sleep_mean, sleep_sd,
                     table_rows, text_rows, verbose=True,
                     fit_id="step21_severity_joint", out_dir=None):
    """Fit a 2-moderator VARX(1) model with both pain and sleep severity."""
    import pymc as pm
    import arviz as az
    from coupling_model import add_bivariate_innovations_likelihood, run_fit

    # Build data arrays
    sub_df = model_df.dropna(subset=["pain_within_lag1", "sleep_within_lag1"]).copy()

    # Person-level moderators (z-scored)
    person_pain_z = {}
    person_sleep_z = {}
    for sid in unique_ids:
        rows = sub_df[sub_df["ID"] == sid]
        if len(rows) == 0:
            continue
        person_pain_z[str(sid)] = (rows["pain_person_mean"].iloc[0] - pain_mean) / pain_sd
        person_sleep_z[str(sid)] = (rows["sleep_person_mean"].iloc[0] - sleep_mean) / sleep_sd

    valid_ids = [sid for sid in unique_ids if str(sid) in person_pain_z]
    valid_set = set(valid_ids)
    sub_df = sub_df[sub_df["ID"].isin(valid_set)].copy()
    id_map_local = {sid: i for i, sid in enumerate(valid_ids)}
    sub_df["pid_idx"] = sub_df["ID"].map(id_map_local)

    N = len(valid_ids)
    pain_w = sub_df["pain_within"].values
    sleep_w = sub_df["sleep_within"].values
    pain_lag = sub_df["pain_within_lag1"].values
    sleep_lag = sub_df["sleep_within_lag1"].values
    contrast_lag = sub_df["contrast_within_lag1"].values
    sleep_x_contrast = sub_df["sleep_x_contrast_lag1"].values
    pain_x_contrast = sub_df["pain_x_contrast_lag1"].values
    age_z = sub_df["Age_z"].values
    sex_c = sub_df["Sex_c"].values
    pid = sub_df["pid_idx"].values.astype(int)

    X_pain = np.array([person_pain_z[str(sid)] for sid in valid_ids])
    X_sleep = np.array([person_sleep_z[str(sid)] for sid in valid_ids])
    x_pain_obs = X_pain[pid]
    x_sleep_obs = X_sleep[pid]

    if verbose:
        print(f"    Joint model: N={N}, obs={len(sub_df)}")

    with pm.Model() as model:
        # Fixed effects
        a0 = pm.Normal("a0", 0, 5)   # pain intercept
        a1 = pm.Normal("a1", 0, 5)   # pain AR
        a2 = pm.Normal("a2", 0, 5)   # SP coupling (sleep->pain)
        a3 = pm.Normal("a3", 0, 5)   # contrast->pain
        a4 = pm.Normal("a4", 0, 5)   # sleep*contrast->pain

        b0 = pm.Normal("b0", 0, 5)   # sleep intercept
        b1 = pm.Normal("b1", 0, 5)   # PS coupling (pain->sleep)
        b2 = pm.Normal("b2", 0, 5)   # sleep AR
        b3 = pm.Normal("b3", 0, 5)   # contrast->sleep
        b4 = pm.Normal("b4", 0, 5)   # pain*contrast->sleep

        # Age/Sex moderation
        g_sp_age = pm.Normal("g_sp_age", 0, 1)
        g_sp_sex = pm.Normal("g_sp_sex", 0, 1)
        g_ps_age = pm.Normal("g_ps_age", 0, 1)
        g_ps_sex = pm.Normal("g_ps_sex", 0, 1)

        # Severity moderation: both moderators enter the SP slope only,
        # matching the manuscript's accounting of "six moderation
        # parameters" (individual-pain SP + PS, individual-sleep SP + PS,
        # joint-model pain SP + sleep SP = 6). The PS slope is left as
        # baseline + random effects + Age/Sex only in the joint model.
        gamma_sp_pain = pm.Normal("gamma_sp_pain", 0, 1)
        gamma_sp_sleep = pm.Normal("gamma_sp_sleep", 0, 1)

        # Random effects
        tau_sp = pm.HalfCauchy("tau_sp", 1)
        tau_ps = pm.HalfCauchy("tau_ps", 1)
        u_sp = pm.Normal("u_sp", 0, tau_sp, shape=N)
        u_ps = pm.Normal("u_ps", 0, tau_ps, shape=N)

        # SP coupling (sleep -> pain)
        lambda_sp = (a2
                     + g_sp_age * age_z
                     + g_sp_sex * sex_c
                     + gamma_sp_pain * x_pain_obs
                     + gamma_sp_sleep * x_sleep_obs
                     + u_sp[pid])

        # PS coupling (pain -> sleep): severity moderators not attached
        # (see "Severity moderation" comment above).
        lambda_ps = (b1
                     + g_ps_age * age_z
                     + g_ps_sex * sex_c
                     + u_ps[pid])

        # Means
        mu_p = a0 + a1 * pain_lag + lambda_sp * sleep_lag + a3 * contrast_lag + a4 * sleep_x_contrast
        mu_s = b0 + lambda_ps * pain_lag + b2 * sleep_lag + b3 * contrast_lag + b4 * pain_x_contrast

        # Correlated innovations — shared helper from lib/coupling_model.py.
        # Declares sigma_pain, sigma_sleep, rho_raw, rho_innov and attaches
        # y_pain / y_sleep observation nodes via sequential conditioning.
        add_bivariate_innovations_likelihood(
            mu_pain=mu_p, mu_sleep=mu_s,
            y_pain=pain_w, y_sleep=sleep_w,
        )

    # Sampling goes through the library's single entry point. This block used to
    # call pm.sample directly with its own 2000/2000/0.95, so severity moderation
    # sampled at settings the Methods did not describe and produced no diagnostics.
    idata = run_fit(model, fit_id=fit_id, out_dir=out_dir, cores=4)

    # Extract results
    def _summarize(var_name):
        from coupling_model import two_tail_p
        draws = idata.posterior[var_name].values.flatten()
        mean = float(np.mean(draws))
        lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        pneg = float(np.mean(draws < 0))
        return mean, lo, hi, two_tail_p(pneg)

    rhat_vals = az.rhat(idata)
    rhat_max = float(max(rhat_vals[v].values.max() for v in rhat_vals
                         if hasattr(rhat_vals[v], "values")))

    for var, label, direction in [
        ("gamma_sp_pain", "Mean Pain Severity", "SP"),
        ("gamma_sp_sleep", "Mean Sleep Quality", "SP"),
    ]:
        mean, lo, hi, p = _summarize(var)
        if verbose:
            print(f"    {var} = {mean:+.4f} [{lo:+.4f}, {hi:+.4f}], p={p:.4f}")
        table_rows.append({
            "Moderator": label, "Model": "Joint", "Direction": direction,
            "gamma": mean, "CrI_lo": lo, "CrI_hi": hi, "p": p,
        })
        text_rows.append({"metric": f"joint_{var}", "value": f"{mean:+.4f}", "note": ""})
        text_rows.append({"metric": f"joint_{var}_ci", "value": f"[{lo:+.4f}, {hi:+.4f}]", "note": ""})
        text_rows.append({"metric": f"joint_{var}_p", "value": f"{p:.4f}", "note": ""})

    if verbose:
        print(f"    R-hat max: {rhat_max:.3f}")
    text_rows.append({"metric": "joint_rhat_max", "value": f"{rhat_max:.3f}", "note": ""})


def main():
    parser = argparse.ArgumentParser(
        description="Step 13 — Person-mean severity moderation (Table S2)."
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--refit", action="store_true",
                        help="Re-run computation from scratch instead of loading saved derivatives")
    args = parser.parse_args()
    run_step21(verbose=not args.quiet, refit=args.refit)


if __name__ == "__main__":
    main()
