#!/usr/bin/env python3
"""
02 — Fit the Bayesian VARX(1) Coupling Model and Run LOO-CV Comparison
=======================================================================

This script fits the bivariate vector autoregressive model with exogenous
inputs (VARX(1)) that estimates bidirectional temporal coupling between
sleep quality and pain severity across quarterly assessments.

Model Equations
---------------
The model operates on within-person deviations (Y^w), which by construction
remove all stable between-person confounders (Curran & Bauer, 2011). At
each time point t for person i:

  Pain equation (sleep-to-pain coupling, SP):

    P^w_it = mu_p                              (population intercept)
           + phi_p * P^w_{i,t-1}               (pain autoregression)
           + lambda_{sp,it} * S^w_{i,t-1}      (sleep-to-pain coupling)
           + delta_p * K^w_{i,t-1}             (direct contrast effect)
           + omega_sp * S^w_{i,t-1} * K^w_{i,t-1}  (contrast moderation)
           + epsilon_{p,it}

  Sleep equation (pain-to-sleep coupling, PS):

    S^w_it = mu_s                              (population intercept)
           + lambda_{ps,it} * P^w_{i,t-1}      (pain-to-sleep coupling)
           + phi_s * S^w_{i,t-1}               (sleep autoregression)
           + delta_s * K^w_{i,t-1}             (direct contrast effect)
           + omega_ps * P^w_{i,t-1} * K^w_{i,t-1}  (contrast moderation)
           + epsilon_{s,it}

where K^w is the within-person pain localization contrast (Factor 2 from the
PAF model: knee-dominant = positive, body-dominant = negative).

Coupling Slopes
---------------
The cross-lagged coupling slopes vary across persons and conditions:

  lambda_{sp,it} = lambda_sp                   (population mean coupling)
                 + gamma_sp_age * Age_z_i       (age moderation, z-scored)
                 + gamma_sp_sex * Sex_c_i       (sex moderation, centered)
                 + u_{sp,i}                     (person random effect)

  lambda_{ps,it} = lambda_ps
                 + gamma_ps_age * Age_z_i
                 + gamma_ps_sex * Sex_c_i
                 + u_{ps,i}

Age is z-scored (mean=0, SD=1). Sex is coded 0=male, 1=female, then
mean-centered so that the main coupling effects (lambda_sp, lambda_ps)
are evaluated at the sample-average sex composition, not at Sex=0 (all male).

Correlated Innovations (Cholesky Trick)
----------------------------------------
The innovation pair (epsilon_p, epsilon_s) is bivariate normal with
off-diagonal correlation rho. Rather than MvNormal (O(n^3) per evaluation
due to Cholesky decomposition of the covariance matrix), we use sequential
conditioning:

  epsilon_p ~ N(0, sigma_p^2)
  epsilon_s | epsilon_p ~ N(rho * sigma_s/sigma_p * epsilon_p,
                            sigma_s^2 * (1 - rho^2))

This is mathematically equivalent but O(n) per evaluation, yielding ~100x
speedup for our sample sizes.

MCMC Configuration
------------------
  - Sampler: NUTS (No-U-Turn Sampler, Hoffman & Gelman, 2014)
  - Chains: 4 independent chains (for R-hat diagnostics)
  - Draws: 2000 post-warmup draws per chain (8000 total)
  - Warmup: 2000 tuning iterations per chain
  - Target acceptance: 0.95 (high to reduce divergences in hierarchical model)
  - Random seed: 42 (for reproducibility)

LOO-CV Model Comparison
------------------------
Four nested models are compared via PSIS-LOO (Pareto-smoothed importance
sampling leave-one-out cross-validation; Vehtari et al., 2017):

  M1: Base VARX(1) — AR + cross-lag only (no contrast, no age/sex)
  M2: + pain localization contrast (direct + interaction terms)
  M3: + age and sex moderation of coupling slopes
  M4: Full model (contrast + age/sex + interaction)

PSIS-LOO approximates exact LOO-CV without refitting the model N times.
It uses the posterior predictive distribution and importance weights to
estimate the expected log pointwise predictive density (ELPD). Models
are ranked by ELPD (higher = better out-of-sample prediction).

Output Files
------------
  results/coupling_results.csv         — Full parameter estimates table
  results/coupling_summary.txt         — Human-readable summary
  results/contrast_posterior_draws.npz — Posterior draws for plotting
  results/loo_comparison.txt           — LOO-CV model comparison table

References
----------
Curran, P. J., & Bauer, D. J. (2011). The disaggregation of within-person
  and between-person effects in longitudinal models of change. Annual Review
  of Psychology, 62, 583-619.

Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model
  evaluation using leave-one-out cross-validation and WAIC. Statistics and
  Computing, 27, 1413-1432.

Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively
  setting path lengths in Hamiltonian Monte Carlo. Journal of Machine
  Learning Research, 15, 1593-1623.

Author: Pedro Valdes-Hernandez
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import arviz as az

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)

# Add the library directory to the Python path so we can import the shared
# model module (coupling_model.py) and other library code
LIB_DIR = os.path.join(SCRIPT_DIR, "lib")
sys.path.insert(0, LIB_DIR)

DATA_DIR = os.path.join(REPO_DIR, "data")
RESULTS_DIR = os.path.join(REPO_DIR, "results")


# ===================================================================
# Result Saving
# ===================================================================

def save_coupling_results(idata, model_df, unique_ids, results_dir, synthetic):
    """Extract and save all posterior summaries from the full model fit.

    Saves three files:
      1. coupling_results.csv — tabular parameter estimates
      2. coupling_summary.txt — human-readable text summary
      3. contrast_posterior_draws.npz — raw posterior arrays for plotting

    Parameters
    ----------
    idata : arviz.InferenceData
        Full posterior from the VARX(1) fit.
    model_df : DataFrame
        The data subset used for fitting.
    unique_ids : list
        Subject IDs in the model.
    results_dir : str
        Directory to save results into.
    synthetic : bool
        Whether this is synthetic data (affects file naming).
    """
    # Import the shared model module for result extraction
    from coupling_model import extract_results, extract_person_posteriors

    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Extract population-level parameter summaries
    # ------------------------------------------------------------------
    results = extract_results(idata)
    n_persons = len(unique_ids)
    n_obs = len(model_df)
    results["n_persons"] = n_persons
    results["n_obs"] = n_obs

    # Save tabular results
    results_csv = os.path.join(results_dir, "coupling_results.csv")
    pd.DataFrame([results]).to_csv(results_csv, index=False)
    print(f"  Saved: {results_csv}")

    # ------------------------------------------------------------------
    # Generate human-readable summary text
    # ------------------------------------------------------------------
    summary_path = os.path.join(results_dir, "coupling_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Bayesian VARX(1) Coupling Model Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Sample: {n_persons} subjects, {n_obs} observations\n")
        f.write(f"R-hat max: {results['rhat_max']:.3f}\n\n")

        f.write("Population Coupling Parameters\n")
        f.write("-" * 50 + "\n")

        # SP coupling: lambda_sp = a2
        f.write(
            f"Sleep->Pain (a2/lambda_sp): "
            f"{results['a2_mean']:.4f} "
            f"[{results['a2_ci_lo']:.4f}, {results['a2_ci_hi']:.4f}]  "
            f"P(< 0) = {results['a2_prob_neg']:.3f}\n"
        )

        # PS coupling: lambda_ps = b1
        f.write(
            f"Pain->Sleep (b1/lambda_ps): "
            f"{results['b1_mean']:.4f} "
            f"[{results['b1_ci_lo']:.4f}, {results['b1_ci_hi']:.4f}]  "
            f"P(< 0) = {results['b1_prob_neg']:.3f}\n"
        )

        # AR coefficients
        f.write(
            f"Pain AR (a1/phi_p):     "
            f"{results['a1_mean']:.4f} "
            f"[{results['a1_ci_lo']:.4f}, {results['a1_ci_hi']:.4f}]\n"
        )
        f.write(
            f"Sleep AR (b2/phi_s):    "
            f"{results['b2_mean']:.4f} "
            f"[{results['b2_ci_lo']:.4f}, {results['b2_ci_hi']:.4f}]\n"
        )

        # Contrast effects
        f.write(
            f"\nContrast -> Pain (a3/delta_p):  "
            f"{results['a3_mean']:.4f} "
            f"[{results['a3_ci_lo']:.4f}, {results['a3_ci_hi']:.4f}]\n"
        )
        f.write(
            f"Contrast -> Sleep (b3/delta_s): "
            f"{results['b3_mean']:.4f} "
            f"[{results['b3_ci_lo']:.4f}, {results['b3_ci_hi']:.4f}]\n"
        )

        # Interaction (contrast moderation of coupling)
        f.write(
            f"Sleep*Contrast -> Pain (a4/omega_sp): "
            f"{results['a4_mean']:.4f} "
            f"[{results['a4_ci_lo']:.4f}, {results['a4_ci_hi']:.4f}]\n"
        )
        f.write(
            f"Pain*Contrast -> Sleep (b4/omega_ps): "
            f"{results['b4_mean']:.4f} "
            f"[{results['b4_ci_lo']:.4f}, {results['b4_ci_hi']:.4f}]\n"
        )

        # Age/Sex moderation
        if "g_sp_age_mean" in results:
            f.write("\nAge/Sex Moderation of Coupling\n")
            f.write("-" * 50 + "\n")
            for var, label in [
                ("g_sp_age", "SP x Age"),
                ("g_sp_sex", "SP x Sex"),
                ("g_ps_age", "PS x Age"),
                ("g_ps_sex", "PS x Sex"),
            ]:
                f.write(
                    f"  {label}: {results[f'{var}_mean']:.4f} "
                    f"[{results[f'{var}_ci_lo']:.4f}, "
                    f"{results[f'{var}_ci_hi']:.4f}]\n"
                )

        # Innovation parameters
        f.write("\nInnovation Parameters\n")
        f.write("-" * 50 + "\n")
        f.write(f"  sigma_pain:  {results['sigma_pain_mean']:.4f}\n")
        f.write(f"  sigma_sleep: {results['sigma_sleep_mean']:.4f}\n")
        f.write(f"  rho:         {results['rho_innov_mean']:.4f}\n")

        # Random effect SDs
        f.write(f"\n  tau_sp: {results['tau_sp_mean']:.4f}\n")
        f.write(f"  tau_ps: {results['tau_ps_mean']:.4f}\n")

    print(f"  Saved: {summary_path}")

    # ------------------------------------------------------------------
    # Save posterior draws for figure scripts (03_contrast_moderation.py)
    # ------------------------------------------------------------------
    # These are the draws needed for JN analysis and contrast moderation
    # figures, saved separately so plotting scripts don't need to re-fit.
    posterior_path = os.path.join(results_dir, "contrast_posterior_draws.npz")

    # Flatten posterior draws across chains: (chains, draws) -> (n_total,)
    a2_draws = idata.posterior["a2"].values.flatten()  # lambda_sp
    a4_draws = idata.posterior["a4"].values.flatten()  # omega_sp
    b1_draws = idata.posterior["b1"].values.flatten()  # lambda_ps
    b4_draws = idata.posterior["b4"].values.flatten()  # omega_ps

    # Person-specific random effects (posterior means, for person-level dots)
    u_sp_mean = idata.posterior["u_sp"].values.mean(axis=(0, 1))  # (n_persons,)
    u_ps_mean = idata.posterior["u_ps"].values.mean(axis=(0, 1))  # (n_persons,)

    # Observation-level arrays for person-level plotting
    obs_pid_idx = model_df["pid_idx"].values.astype(int)
    obs_contrast = model_df["contrast_within_lag1"].values.astype(float)
    contrast_vals = model_df["contrast_within_lag1"].dropna().values.astype(float)

    np.savez(
        posterior_path,
        # Full posterior draws (n_chains * n_draws = 8000 each)
        a2_draws=a2_draws,     # lambda_sp (SP coupling intercept)
        a4_draws=a4_draws,     # omega_sp (contrast x sleep interaction)
        b1_draws=b1_draws,     # lambda_ps (PS coupling intercept)
        b4_draws=b4_draws,     # omega_ps (contrast x pain interaction)
        # Person-specific random effects (posterior means)
        u_sp_mean=u_sp_mean,
        u_ps_mean=u_ps_mean,
        # Observation-level arrays for scatter plots
        obs_pid_idx=obs_pid_idx,
        obs_contrast=obs_contrast,
        # All contrast values (for JN grid range)
        contrast_vals=contrast_vals,
    )
    print(f"  Saved: {posterior_path}")

    # ------------------------------------------------------------------
    # Print key results to console
    # ------------------------------------------------------------------
    print("\n  Key Population Parameters:")
    print(
        f"    Sleep->Pain (lambda_sp): {results['a2_mean']:.4f} "
        f"[{results['a2_ci_lo']:.4f}, {results['a2_ci_hi']:.4f}]"
    )
    print(
        f"    Pain->Sleep (lambda_ps): {results['b1_mean']:.4f} "
        f"[{results['b1_ci_lo']:.4f}, {results['b1_ci_hi']:.4f}]"
    )
    print(f"    rho (innovation correlation): {results['rho_innov_mean']:.4f}")
    print(f"    R-hat max: {results['rhat_max']:.3f}")


def save_loo_comparison(comparison, results_dir):
    """Save the LOO-CV model comparison table.

    Parameters
    ----------
    comparison : DataFrame
        ArviZ LOO comparison table (from az.compare).
    results_dir : str
        Directory to save results into.
    """
    loo_path = os.path.join(results_dir, "loo_comparison.txt")
    with open(loo_path, "w") as f:
        f.write("LOO-CV Model Comparison (PSIS-LOO, Vehtari et al. 2017)\n")
        f.write("=" * 70 + "\n\n")
        f.write("Models (nested hierarchy):\n")
        f.write("  M1_base:     AR + cross-lag only (no contrast, no age/sex)\n")
        f.write("  M2_contrast: + pain localization contrast (direct + interaction)\n")
        f.write("  M3_agesex:   + age and sex moderation of coupling slopes\n")
        f.write("  M4_full:     Full model (contrast + age/sex + interaction)\n\n")
        f.write("ELPD = expected log pointwise predictive density\n")
        f.write("Higher ELPD = better out-of-sample prediction\n\n")
        f.write(comparison.to_string() + "\n")
    print(f"  Saved: {loo_path}")


# ===================================================================
# Main Pipeline
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fit the Bayesian VARX(1) coupling model."
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data from data/synthetic/"
    )
    parser.add_argument(
        "--loo", action="store_true",
        help="Also run LOO-CV model comparison (fits 4 models, ~4x slower)"
    )
    parser.add_argument(
        "--no-agesex", action="store_true",
        help="Omit age and sex moderation from the full model"
    )
    args = parser.parse_args()

    # Determine output directory
    results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("BAYESIAN VARX(1) COUPLING MODEL")
    print("=" * 70)
    if args.synthetic:
        print("  Mode: SYNTHETIC data")
    else:
        print("  Mode: REAL data")

    # Import the shared model module
    from coupling_model import (
        load_data, fit_bayesian_varx1, compute_loo_comparison
    )

    # ------------------------------------------------------------------
    # Load the processed data (output of 01_prepare_data.py)
    # ------------------------------------------------------------------
    print("\n  Loading processed data...")
    df_full, model_df, unique_ids, id_map = load_data(
        DATA_DIR, synthetic=args.synthetic
    )

    # ------------------------------------------------------------------
    # Fit the full VARX(1) model
    # ------------------------------------------------------------------
    include_agesex = not args.no_agesex
    print(f"\n  Fitting full model (include_agesex={include_agesex})...")
    print(f"    {len(unique_ids)} subjects, {len(model_df)} observations")
    print(f"    MCMC: 4 chains x 2000 draws, 2000 tuning")

    idata, sub_df, valid_ids = fit_bayesian_varx1(
        model_df, unique_ids, id_map,
        include_agesex=include_agesex,
        progressbar=True,
    )

    # ------------------------------------------------------------------
    # Save results and posterior draws
    # ------------------------------------------------------------------
    print("\n  Extracting and saving results...")
    save_coupling_results(idata, sub_df, valid_ids, results_dir, args.synthetic)

    # ------------------------------------------------------------------
    # Optional: LOO-CV model comparison
    # ------------------------------------------------------------------
    if args.loo:
        print("\n" + "=" * 70)
        print("LOO-CV MODEL COMPARISON")
        print("=" * 70)
        print("  Fitting 4 nested models for LOO comparison...")
        print("  (This will take approximately 4x the single-model fitting time)")
        comparison = compute_loo_comparison(DATA_DIR, synthetic=args.synthetic)
        save_loo_comparison(comparison, results_dir)

    print("\n" + "=" * 70)
    print("MODEL FITTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
