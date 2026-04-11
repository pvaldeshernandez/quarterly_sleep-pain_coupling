#!/usr/bin/env python3
"""
Bayesian VARX(1) Model for Bidirectional Sleep-Pain Coupling
=============================================================

This module implements a bivariate vector autoregressive model with exogenous
inputs (VARX(1)) to estimate the bidirectional temporal coupling between
sleep quality and pain severity in a longitudinal panel design.

Model Specification
-------------------
The within-person deviations at each time point are modeled as:

  Pain equation (sleep-to-pain coupling):

    P^w_it = mu_p + phi_p * P^w_{i,t-1}
             + lambda_{sp,it} * S^w_{i,t-1}
             + delta_p * K^w_{i,t-1}
             + epsilon_{p,it}

  Sleep equation (pain-to-sleep coupling):

    S^w_it = mu_s
             + lambda_{ps,it} * P^w_{i,t-1}
             + phi_s * S^w_{i,t-1}
             + delta_s * K^w_{i,t-1}
             + epsilon_{s,it}

where:
  - P^w_it, S^w_it   = within-person centered pain and sleep at time t
  - K^w_{i,t-1}      = within-person centered pain localization contrast at t-1
  - mu_p, mu_s        = population intercepts
  - phi_p, phi_s      = autoregressive coefficients
  - delta_p, delta_s  = direct effects of pain localization
  - lambda_{sp,it}    = person- and time-varying sleep-to-pain coupling
  - lambda_{ps,it}    = person- and time-varying pain-to-sleep coupling
  - epsilon_{p,it}, epsilon_{s,it} = correlated innovations

Coupling slopes vary across persons and conditions:

  lambda_{sp,it} = lambda_sp
                   + gamma_sp_age * Age_z_i
                   + gamma_sp_sex * Sex_c_i
                   + omega_sp * K^w_{i,t-1}
                   + [gamma_sp * X_i]          (Aim 2 only)
                   + u_{sp,i}

  lambda_{ps,it} = lambda_ps
                   + gamma_ps_age * Age_z_i
                   + gamma_ps_sex * Sex_c_i
                   + omega_ps * K^w_{i,t-1}
                   + [gamma_ps * X_i]          (Aim 2 only)
                   + u_{ps,i}

The terms in [brackets] are present only when an external moderator X is
supplied (Aim 2 analyses). u_{sp,i} and u_{ps,i} are person-specific random
coupling deviations.

Correlated Innovations (Cholesky Trick)
----------------------------------------
Rather than specifying a full bivariate normal for the innovation vector,
we use a sequential conditioning approach (Cholesky decomposition):

  epsilon_p  ~  N(0, sigma_p^2)
  epsilon_s | epsilon_p  ~  N(rho * (sigma_s / sigma_p) * epsilon_p,
                              sigma_s^2 * (1 - rho^2))

This factorization is mathematically equivalent to drawing from the
bivariate normal N(0, Sigma) where Sigma has off-diagonal correlation rho,
but it avoids PyMC's MvNormal sampler, which scales as O(n^3) per evaluation
due to repeated Cholesky decompositions of the covariance matrix. The
sequential conditioning approach is O(n) per evaluation, yielding roughly
100x faster sampling for our sample sizes.

Prior Specification
-------------------
  Fixed effects (mu, phi, lambda, delta, omega): N(0, 5)
    -- Weakly informative; coupling effects are typically |lambda| < 0.5.

  Moderation coefficients (gamma): N(0, 1)
    -- Tighter prior reflecting that moderators should have smaller effects.

  Random effect SDs (tau_sp, tau_ps): HalfCauchy(1)
    -- Allows substantial heterogeneity while regularizing toward zero.

  Innovation SDs (sigma_p, sigma_s): HalfCauchy(2)
    -- Weakly informative for residual variance.

  Innovation correlation (rho):
    rho_raw ~ Beta(2, 2);  rho = 2 * rho_raw - 1
    -- Symmetric prior on [-1, 1] with mode at 0, downweighting extreme
       correlations. Equivalent to a regularized uniform on correlations.

Variable Naming Convention
--------------------------
Internal code uses compact variable names inherited from the analysis scripts.
The mapping to manuscript notation is:

  Code    Manuscript       Meaning
  ----    ----------       -------
  a0      mu_p             Pain intercept
  a1      phi_p            Pain autoregressive coefficient
  a2      lambda_sp        Population sleep-to-pain coupling
  a3      delta_p          Direct effect of contrast on pain
  a4      omega_sp         Contrast x sleep interaction (pain eq.)
  b0      mu_s             Sleep intercept
  b1      lambda_ps        Population pain-to-sleep coupling
  b2      phi_s            Sleep autoregressive coefficient
  b3      delta_s          Direct effect of contrast on sleep
  b4      omega_ps         Contrast x pain interaction (sleep eq.)
  u_sp    u_{sp,i}         Person-specific sleep-to-pain random effect
  u_ps    u_{ps,i}         Person-specific pain-to-sleep random effect
  gamma_sp, gamma_ps       External moderator coefficients (Aim 2)
  g_sp_age, g_sp_sex       Age/Sex moderation of SP coupling
  g_ps_age, g_ps_sex       Age/Sex moderation of PS coupling

References
----------
Curran, P. J., & Bauer, D. J. (2011). The disaggregation of within-person
  and between-person effects in longitudinal models of change. Annual Review
  of Psychology, 62, 583-619.

Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model
  evaluation using leave-one-out cross-validation and WAIC. Statistics and
  Computing, 27, 1413-1432.

Johnson, P. O., & Neyman, J. (1936). Tests of certain linear hypotheses and
  their application to some educational problems. Statistical Research
  Memoirs, 1, 57-93.

Bauer, D. J., & Curran, P. J. (2005). Probing interactions in fixed and
  multilevel regression: Inferential and graphical techniques. Multivariate
  Behavioral Research, 40, 373-400.

Author: Pedro Valdes-Hernandez
"""

import os
import warnings
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# MCMC settings -- shared across all model fits
# ---------------------------------------------------------------------------
N_SAMPLES = 2000    # posterior draws per chain
N_TUNE = 2000       # warmup/tuning iterations per chain
N_CHAINS = 4        # number of independent Markov chains
TARGET_ACCEPT = 0.95  # NUTS target acceptance probability (high for HMC)


# ===================================================================
# Data Loading
# ===================================================================

def load_data(data_dir, synthetic=False):
    """Load processed quarterly data with within-person decomposition.

    Reads the CSV produced by the data preparation pipeline (either real
    processed data or synthetic data for reproducibility demonstrations)
    and prepares all arrays needed for model fitting.

    Parameters
    ----------
    data_dir : str
        Path containing the processed CSV. The loader tries the standard
        layouts first and falls back to looking directly inside
        ``data_dir``:

          * Real data:
              ``{data_dir}/processed_data_contrast.csv``         (default)
              ``{data_dir}/processed_data.csv``                  (sandbox)

          * Synthetic data:
              ``{data_dir}/synthetic/processed_data.csv``        (default)
              ``{data_dir}/processed_data.csv``                  (sandbox)

        Passing a sandbox directory (where the processed CSV sits at the
        top level) works for both modes without having to create a
        ``synthetic/`` subfolder.
    synthetic : bool, default False
        If True, interpret the processed CSV as synthetic (already has
        pre-computed factor scores and no factor-analysis parameters
        file). Synthetic data preserves the statistical structure of the
        original dataset but contains no identifiable information.

    Returns
    -------
    df_full : DataFrame
        Full processed data (before dropping rows with missing lags).
    model_df : DataFrame
        Analysis-ready subset with all required lagged variables present.
        Includes columns: pid_idx (integer person index), Age_z (z-scored
        age), Sex_coded (0=male, 1=female), Sex_c (mean-centered sex).
    unique_ids : list
        Sorted unique subject identifiers in model_df.
    id_map : dict
        Mapping from subject ID to integer index {ID: int}.
    """
    # Build a search list of candidate CSV paths and take the first one
    # that exists. This supports both the legacy layout (data/, with a
    # synthetic/ subfolder) and the sandbox layout (a single directory
    # where the processed CSV sits at the top level).
    if synthetic:
        candidates = [
            os.path.join(data_dir, "synthetic", "processed_data.csv"),
            os.path.join(data_dir, "processed_data.csv"),
        ]
    else:
        candidates = [
            os.path.join(data_dir, "processed_data_contrast.csv"),
            os.path.join(data_dir, "processed_data.csv"),
        ]

    csv_path = next((p for p in candidates if os.path.exists(p)), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"No processed data found in {data_dir!r}. Tried: {candidates}"
        )

    df = pd.read_csv(csv_path)

    # Required columns for the VARX(1) model: current outcomes and all lags
    required = [
        "pain_within",              # P^w_it (current pain deviation)
        "sleep_within",             # S^w_it (current sleep deviation)
        "pain_within_lag1",         # P^w_{i,t-1}
        "sleep_within_lag1",        # S^w_{i,t-1}
        "contrast_within_lag1",     # K^w_{i,t-1}
        "sleep_x_contrast_lag1",    # S^w_{i,t-1} * K^w_{i,t-1}
        "pain_x_contrast_lag1",     # P^w_{i,t-1} * K^w_{i,t-1}
    ]
    model_df = df.dropna(subset=required).copy()

    # Create integer person index for PyMC shape parameters
    unique_ids = sorted(model_df["ID"].unique())
    id_map = {pid: i for i, pid in enumerate(unique_ids)}
    model_df["pid_idx"] = model_df["ID"].map(id_map)

    # Compute person-level summary statistics (used for diagnostics)
    pain_sd = df.groupby("ID")["pain_factor"].std().rename("sd_pain")
    sleep_sd = df.groupby("ID")["sleep_factor"].std().rename("sd_sleep")
    pain_mean = df.groupby("ID")["pain_factor"].mean().rename("mean_pain")
    sleep_mean = df.groupby("ID")["sleep_factor"].mean().rename("mean_sleep")
    obs_counts = model_df.groupby("ID").size().rename("n_obs")
    model_df = (
        model_df
        .merge(pain_sd, on="ID", how="left")
        .merge(sleep_sd, on="ID", how="left")
        .merge(pain_mean, on="ID", how="left")
        .merge(sleep_mean, on="ID", how="left")
        .merge(obs_counts, on="ID", how="left")
    )

    # Prepare demographic moderators: z-score Age, recode and center Sex
    #   Raw data: Sex=1 (male), Sex=2 (female)
    #   Recoded:  Sex_coded=0 (male), Sex_coded=1 (female)
    #   Centered: Sex_c = Sex_coded - mean(Sex_coded)
    #   This ensures main coupling effects (a2, b1) are evaluated at the
    #   sample-average sex composition, not at Sex=0 (all male).
    person_demo = (
        model_df.groupby("ID").first()[["Age", "Sex"]].reset_index()
    )
    age_mean = person_demo["Age"].mean()
    age_sd = person_demo["Age"].std()
    person_demo["Age_z"] = (person_demo["Age"] - age_mean) / age_sd

    person_demo["Sex_coded"] = (person_demo["Sex"] == 2).astype(float)
    sex_mean = person_demo["Sex_coded"].mean()
    person_demo["Sex_c"] = person_demo["Sex_coded"] - sex_mean

    model_df = model_df.merge(
        person_demo[["ID", "Age_z", "Sex_coded", "Sex_c"]], on="ID", how="left"
    )

    # Print summary for verification
    n = model_df["ID"].nunique()
    n_female = int((person_demo["Sex_coded"] == 1).sum())
    n_male = int((person_demo["Sex_coded"] == 0).sum())
    print(f"  Loaded: {n} subjects, {len(model_df)} observations")
    print(
        f"  Obs per person: median="
        f"{model_df.groupby('ID').size().median():.0f}, "
        f"range=[{model_df.groupby('ID').size().min()}, "
        f"{model_df.groupby('ID').size().max()}]"
    )
    print(f"  Age: mean={age_mean:.1f}, SD={age_sd:.1f}")
    print(f"  Sex: {n_female} female (coded=1), {n_male} male (coded=0)")
    print(
        f"  Sex centering: mean(Sex_coded)={sex_mean:.4f}, "
        f"Sex_c = Sex_coded - {sex_mean:.4f}"
    )

    return df, model_df, unique_ids, id_map


# ===================================================================
# Model Fitting
# ===================================================================

def fit_bayesian_varx1(model_df, unique_ids, id_map,
                       X_person=None, include_agesex=True,
                       include_sp=True, include_ps=True,
                       idata_kwargs=None,
                       cores=None,
                       progressbar=True):
    """Fit the Bayesian VARX(1) model with contrast and optional moderators.

    This is the core model-fitting function used by all analyses. It builds
    the PyMC model graph, runs NUTS sampling across multiple chains, and
    returns the full posterior in ArviZ format.

    Parameters
    ----------
    model_df : DataFrame
        Analysis-ready data from ``load_data()``. Must contain: pain_within,
        sleep_within, all lag columns, pid_idx, Age_z, Sex_c.
    unique_ids : list
        Sorted unique subject IDs (defines the person indexing).
    id_map : dict
        Mapping {subject_ID: integer_index}.
    X_person : dict or None, default None
        If provided, a dictionary {subject_ID: z-scored_moderator_value}
        adding gamma_sp and gamma_ps parameters for the external moderator.
        Subjects not in X_person are excluded from the analysis.
    include_agesex : bool, default True
        Whether to include Age (z-scored) and Sex (centered) as moderators
        of the coupling slopes. Set to False for simpler nested models.
    include_sp : bool, default True
        Whether to include the Sleep->Pain cross-lagged coupling path.
        When False, the population slope lambda_sp, the person-specific
        random effects u_sp, tau_sp, and all SP moderators are removed
        from the model. The contrast-by-sleep interaction (omega_sp) is
        also removed, since it requires the SP path. Setting False is
        used in the LOO-CV comparison to test whether removing the SP
        pathway degrades out-of-sample prediction.
    include_ps : bool, default True
        Same as ``include_sp`` but for the Pain->Sleep cross-lagged
        coupling path. Setting both to False gives the null model with
        only autoregressions, intercepts, contrast direct effects, and
        correlated innovations.
    idata_kwargs : dict, optional
        Extra keyword arguments forwarded to ``pm.sample`` via its
        ``idata_kwargs``. Use ``{"log_likelihood": True}`` to compute
        and store the pointwise log-likelihood required by
        ``az.loo`` / ``az.compare``.
    progressbar : bool, default True
        Whether to display MCMC progress bars during sampling.

    Returns
    -------
    idata : arviz.InferenceData
        Full posterior samples, including all fixed effects, random effects,
        and derived quantities. Access chains via
        ``idata.posterior['param_name']``.
    sub_df : DataFrame
        The subset of model_df actually used (may be smaller than input
        if X_person filters out subjects without moderator data).
    valid_ids : list
        Subject IDs included in this particular fit.
    """
    # ------------------------------------------------------------------
    # Filter to subjects with external moderator data (Aim 2 only)
    # ------------------------------------------------------------------
    if X_person is not None:
        # Keep only subjects present in both the coupling data and the
        # moderator dictionary, and with finite moderator values
        valid_ids = [
            pid for pid in unique_ids
            if pid in X_person and np.isfinite(X_person[pid])
        ]
        sub_df = model_df[model_df["ID"].isin(valid_ids)].copy()
        # Rebuild integer indices for the filtered subset
        valid_id_map = {pid: i for i, pid in enumerate(valid_ids)}
        sub_df["pid_idx"] = sub_df["ID"].map(valid_id_map)
    else:
        valid_ids = unique_ids
        sub_df = model_df.copy()

    n_persons = len(valid_ids)

    # ------------------------------------------------------------------
    # Extract data arrays for PyMC (float64 for numerical stability)
    # ------------------------------------------------------------------
    y_pain = sub_df["pain_within"].values.astype(float)
    y_sleep = sub_df["sleep_within"].values.astype(float)
    pain_lag = sub_df["pain_within_lag1"].values.astype(float)
    sleep_lag = sub_df["sleep_within_lag1"].values.astype(float)
    contrast_lag = sub_df["contrast_within_lag1"].values.astype(float)
    sleep_x_contrast = sub_df["sleep_x_contrast_lag1"].values.astype(float)
    pain_x_contrast = sub_df["pain_x_contrast_lag1"].values.astype(float)
    idx = sub_df["pid_idx"].values.astype(int)

    if include_agesex:
        age_z = sub_df["Age_z"].values.astype(float)
        sex_c = sub_df["Sex_c"].values.astype(float)

    # Build observation-level moderator array from person-level dict
    has_moderator = X_person is not None
    if has_moderator:
        X_array = np.array([X_person[pid] for pid in valid_ids])
        X_obs = X_array[idx]  # broadcast person-level to observation-level

    # ------------------------------------------------------------------
    # Build PyMC model
    # ------------------------------------------------------------------
    with pm.Model() as model:

        # --- Fixed effects (weakly informative priors) ---
        # Pain equation parameters (autoregression, intercept, contrast
        # direct effect, and — if the SP path is enabled — the SP
        # coupling slope and its contrast interaction).
        a0 = pm.Normal("a0", mu=0, sigma=5)   # mu_p: intercept
        a1 = pm.Normal("a1", mu=0, sigma=5)   # phi_p: pain AR
        a3 = pm.Normal("a3", mu=0, sigma=5)   # delta_p: contrast -> pain
        if include_sp:
            a2 = pm.Normal("a2", mu=0, sigma=5)  # lambda_sp: SP coupling
            a4 = pm.Normal("a4", mu=0, sigma=5)  # omega_sp: sleep*contrast

        # Sleep equation parameters (analogous).
        b0 = pm.Normal("b0", mu=0, sigma=5)   # mu_s: intercept
        b2 = pm.Normal("b2", mu=0, sigma=5)   # phi_s: sleep AR
        b3 = pm.Normal("b3", mu=0, sigma=5)   # delta_s: contrast -> sleep
        if include_ps:
            b1 = pm.Normal("b1", mu=0, sigma=5)  # lambda_ps: PS coupling
            b4 = pm.Normal("b4", mu=0, sigma=5)  # omega_ps: pain*contrast

        # --- Age & Sex moderation of coupling slopes (gamma ~ N(0,1)) ---
        # Only attach to whichever coupling direction is present.
        if include_agesex and include_sp:
            g_sp_age = pm.Normal("g_sp_age", mu=0, sigma=1)
            g_sp_sex = pm.Normal("g_sp_sex", mu=0, sigma=1)
        if include_agesex and include_ps:
            g_ps_age = pm.Normal("g_ps_age", mu=0, sigma=1)
            g_ps_sex = pm.Normal("g_ps_sex", mu=0, sigma=1)

        # --- External moderator (Aim 2 analyses only) ---
        # Also tied to whichever coupling direction is present.
        if has_moderator and include_sp:
            gamma_sp = pm.Normal("gamma_sp", mu=0, sigma=1)
        if has_moderator and include_ps:
            gamma_ps = pm.Normal("gamma_ps", mu=0, sigma=1)

        # --- Person-specific random coupling slopes ---
        # tau controls the between-person heterogeneity in coupling.
        # Random effects are only sampled for directions that are
        # present in the model.
        if include_sp:
            tau_sp = pm.HalfCauchy("tau_sp", beta=1)
            u_sp = pm.Normal("u_sp", mu=0, sigma=tau_sp, shape=n_persons)
        if include_ps:
            tau_ps = pm.HalfCauchy("tau_ps", beta=1)
            u_ps = pm.Normal("u_ps", mu=0, sigma=tau_ps, shape=n_persons)

        # --- Assemble person-varying coupling slopes ---
        # If the corresponding direction is disabled, a2_i / b1_i stay
        # at zero and the corresponding cross-lag term drops out of the
        # linear predictor below.
        if include_sp:
            a2_i = a2 + u_sp[idx]
            if include_agesex:
                a2_i = a2_i + g_sp_age * age_z + g_sp_sex * sex_c
            if has_moderator:
                a2_i = a2_i + gamma_sp * X_obs
        if include_ps:
            b1_i = b1 + u_ps[idx]
            if include_agesex:
                b1_i = b1_i + g_ps_age * age_z + g_ps_sex * sex_c
            if has_moderator:
                b1_i = b1_i + gamma_ps * X_obs

        # --- Linear predictors ---
        mu_pain = a0 + a1 * pain_lag + a3 * contrast_lag
        if include_sp:
            mu_pain = mu_pain + a2_i * sleep_lag + a4 * sleep_x_contrast

        mu_sleep = b0 + b2 * sleep_lag + b3 * contrast_lag
        if include_ps:
            mu_sleep = mu_sleep + b1_i * pain_lag + b4 * pain_x_contrast

        # --- Correlated innovations via Cholesky trick ---
        # See module docstring for why this is ~100x faster than MvNormal.
        sigma_pain = pm.HalfCauchy("sigma_pain", beta=2)
        sigma_sleep = pm.HalfCauchy("sigma_sleep", beta=2)

        # rho_raw ~ Beta(2,2) on [0,1], rescaled to rho on [-1,1]
        rho_raw = pm.Beta("rho_raw", alpha=2, beta=2)
        rho_innov = pm.Deterministic("rho_innov", 2 * rho_raw - 1)

        # Step 1: pain innovation (marginal)
        pm.Normal("y_pain", mu=mu_pain, sigma=sigma_pain, observed=y_pain)

        # Step 2: sleep innovation conditioned on pain residual
        eps_pain = y_pain - mu_pain
        mu_sleep_cond = (
            mu_sleep + rho_innov * (sigma_sleep / sigma_pain) * eps_pain
        )
        sigma_sleep_cond = sigma_sleep * pt.sqrt(1 - rho_innov**2)
        pm.Normal(
            "y_sleep",
            mu=mu_sleep_cond,
            sigma=sigma_sleep_cond,
            observed=y_sleep,
        )

        # --- MCMC sampling ---
        sample_kwargs = dict(
            draws=N_SAMPLES,
            tune=N_TUNE,
            chains=N_CHAINS,
            target_accept=TARGET_ACCEPT,
            return_inferencedata=True,
            progressbar=progressbar,
            random_seed=42,
            # compute_convergence_checks=False is required due to a known
            # bug in PyMC 5.27.1 + Python 3.13 where the R-hat computation
            # crashes with large random-effect arrays. Convergence is
            # checked manually via ArviZ summary after sampling.
            compute_convergence_checks=False,
        )
        if cores is not None:
            sample_kwargs["cores"] = cores
        if idata_kwargs is not None:
            sample_kwargs["idata_kwargs"] = idata_kwargs
        idata = pm.sample(**sample_kwargs)

    return idata, sub_df, valid_ids


# ===================================================================
# LOO-CV Model Comparison
# ===================================================================

def compute_loo_comparison(data_dir, synthetic=False):
    """Fit four nested coupling-direction models and compare via LOO-CV.

    The four models test whether each cross-lagged coupling direction
    improves out-of-sample prediction. All four share the same
    baseline structure (intercepts, autoregressions, contrast direct
    effects delta_p/delta_s, correlated innovations); they differ only
    in whether the Sleep->Pain and Pain->Sleep paths are included:

      full:    both directions present (lambda_sp, lambda_ps, u_sp, u_ps,
               and the corresponding age/sex and contrast interactions).
      no_PS:   Pain->Sleep direction removed (lambda_ps = 0, no u_ps,
               no omega_ps, no g_ps_age/sex). Equivalently, SP-only.
      no_SP:   Sleep->Pain direction removed (lambda_sp = 0, no u_sp,
               no omega_sp, no g_sp_age/sex). Equivalently, PS-only.
      null:    both cross-lagged paths removed. Only AR + contrast
               direct effects + correlated innovations remain.

    LOO-CV (Leave-One-Out Cross-Validation) is computed via
    Pareto-smoothed importance sampling (PSIS-LOO), which provides an
    efficient approximation to exact LOO without refitting (Vehtari
    et al., 2017). `az.compare` then reports the ELPD difference
    between each model and the best-ranked model, with a standard
    error for the difference.

    This nesting corresponds to what the manuscript reports: it
    directly tests whether adding each coupling direction to the
    baseline improves predictive accuracy, with Delta/SE > 2 as the
    threshold for "substantial improvement."

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the processed CSV.
    synthetic : bool, default False
        Whether to use synthetic data.

    Returns
    -------
    comparison : DataFrame
        ArviZ LOO comparison table with ELPD differences and standard
        errors, sorted by best model first.
    loo_dict : dict
        {model_name: arviz.ElPDData} for each fitted model. Contains
        the pointwise log likelihood and Pareto k-hat diagnostics.
    pairwise : dict
        {(model_a, model_b): {"delta_elpd": ..., "se": ..., "ratio": ...}}
        giving the specific pairwise comparisons reported in the
        manuscript (full vs no_PS, no_SP vs null, etc.). Ratios >2 in
        magnitude are considered substantive improvements.
    """
    # Load data once, then fit each of the four coupling-structure
    # variants. All fits use the same Cholesky-parameterised correlated
    # innovations, same weakly informative priors, same 4x2000 NUTS
    # settings, and the same random seed — the only thing that changes
    # is whether the SP and/or PS paths are included in the likelihood.
    #
    # idata_kwargs={"log_likelihood": True} is crucial: it tells PyMC
    # to store the observation-level log-likelihood evaluated on each
    # MCMC draw, which is the input to az.loo(). Without it,
    # `az.loo(idata)` raises because there is no log_likelihood group
    # in the InferenceData object.

    df_full, model_df, unique_ids, id_map = load_data(
        data_dir, synthetic=synthetic
    )

    idata_kwargs = {"log_likelihood": True}

    # To keep memory usage bounded we fit each model one at a time,
    # compute its LOO (which only needs the pointwise log-likelihood),
    # store the pointwise ELPD contributions, and then release the
    # InferenceData. Keeping all 4 idata objects with full
    # log_likelihood groups in memory simultaneously is enough to OOM
    # the analysis on a 16 GB node because each log-likelihood array is
    # 4 chains x 2000 draws x ~1800 obs x 2 observed variables.
    #
    # The model has two observed variables (y_pain, y_sleep) so by
    # default the inference data contains two separate log-likelihood
    # arrays. For LOO we want the *joint* per-observation log-
    # likelihood, which is log p(y_pain_i, y_sleep_i | theta). Because
    # the Cholesky trick factors this as
    #   p(y_pain_i) * p(y_sleep_i | y_pain_i)
    # and the second factor is what the conditional `y_sleep` likelihood
    # computes, the joint log-likelihood is simply the sum of the two
    # per-draw, per-observation log-likelihood arrays. We add a new
    # variable ``y_joint`` to the log_likelihood group and point
    # ``az.loo`` at it.
    import gc

    model_configs = [
        ("full",  True,  True,  "full model (both SP and PS paths)"),
        ("no_PS", True,  False, "no_PS model (SP only, Pain->Sleep removed)"),
        ("no_SP", False, True,  "no_SP model (PS only, Sleep->Pain removed)"),
        ("null",  False, False, "null model (no cross-lagged coupling)"),
    ]

    loo_dict = {}
    for name, inc_sp, inc_ps, desc in model_configs:
        print(f"\n  Fitting {desc}...")
        # cores=1 runs chains sequentially to avoid multiprocess fork
        # memory overhead (each worker would need its own copy of the
        # compiled model + log-likelihood buffers, which overflows the
        # 16 GB cgroup limit on standard HiPerGator jobs).
        idata, _, _ = fit_bayesian_varx1(
            model_df, unique_ids, id_map,
            include_agesex=True,
            include_sp=inc_sp, include_ps=inc_ps,
            idata_kwargs=idata_kwargs,
            cores=1,
            progressbar=True,
        )
        print(f"  Computing LOO for {name}...")
        ll = idata.log_likelihood
        # Sum y_pain and y_sleep log-likelihoods into a single "y_joint"
        # variable in-place without holding three copies at once. We
        # compute the joint array, drop the originals to free memory,
        # then add back a single y_joint variable.
        if "y_joint" not in ll.data_vars:
            joint_vals = ll["y_pain"].values + ll["y_sleep"].values
            dims = ll["y_pain"].dims
            coords = {d: ll[d] for d in dims if d in ll.coords}
            idata.log_likelihood = idata.log_likelihood.drop_vars(
                ["y_pain", "y_sleep"]
            )
            import xarray as xr
            idata.log_likelihood["y_joint"] = xr.DataArray(
                joint_vals, dims=dims, coords=coords
            )
            del joint_vals
            gc.collect()
        loo_dict[name] = az.loo(idata, pointwise=True, var_name="y_joint")
        # Drop the idata now — we have the pointwise LOO we need.
        del idata, ll
        gc.collect()

    comparison = az.compare(loo_dict)
    print("\n  LOO-CV Comparison (sorted by best):")
    print(comparison)

    # Pairwise comparisons the manuscript reports:
    #   full vs no_PS   — does adding PS to the SP-only model help?
    #   no_SP vs null   — does adding PS to the null model help?
    #   full vs no_SP   — does adding SP to the PS-only model help?
    #   no_PS vs null   — does adding SP to the null model help?
    #
    # For each pair, the pointwise ELPD difference is d_i = elpd_a_i
    # - elpd_b_i; the SE of the summed difference is
    # sqrt(N * var(d_i)), and Delta/SE > 2 is the conventional
    # "substantial improvement" threshold.
    pairs = [
        ("full", "no_PS"),
        ("no_SP", "null"),
        ("full", "no_SP"),
        ("no_PS", "null"),
    ]
    pairwise = {}
    for a, b in pairs:
        loo_a_pw = loo_dict[a].loo_i.values
        loo_b_pw = loo_dict[b].loo_i.values
        d = loo_a_pw - loo_b_pw
        n = len(d)
        delta = float(d.sum())
        se = float(np.sqrt(n * d.var(ddof=1)))
        ratio = delta / se if se > 0 else float("nan")
        pairwise[(a, b)] = {
            "delta_elpd": delta,
            "se": se,
            "ratio": ratio,
        }
        print(
            f"  {a:>6} vs {b:<6}: ΔELPD = {delta:+.2f}, SE = {se:.2f}, "
            f"Δ/SE = {ratio:+.2f}"
        )

    # Pareto k-hat diagnostics (max and fraction >0.7) for the full model
    khat = loo_dict["full"].pareto_k.values
    k_max = float(np.nanmax(khat))
    k_bad = int((khat > 0.7).sum())
    n_obs = len(khat)
    print(
        f"  Pareto k̂ (full model): max = {k_max:.2f}, "
        f"{k_bad} / {n_obs} observations > 0.7 "
        f"({100 * k_bad / n_obs:.1f}%)"
    )

    return comparison, loo_dict, pairwise


# ===================================================================
# Johnson-Neyman Analysis
# ===================================================================

def compute_jn_curve(intercept_draws, slope_draws, X_vals,
                     raw_mean=None, raw_sd=None, clip_pct=(1, 99)):
    """Compute a Bayesian Johnson-Neyman (JN) significance curve.

    The Johnson-Neyman technique identifies regions of a continuous
    moderator where a conditional effect is statistically significant
    (Johnson & Neyman, 1936). In the frequentist version, JN boundaries
    are found by solving a quadratic equation for the moderator value
    where the t-statistic equals the critical value.

    The Bayesian version implemented here is simpler and more flexible:
    instead of a quadratic formula, we evaluate the full posterior of the
    conditional effect at each point on a dense moderator grid (500 points).
    At each grid point, if the 95% credible interval excludes zero, the
    effect is declared credibly nonzero. The JN boundary is found by linear
    interpolation between adjacent grid points where significance status
    changes.

    This approach naturally handles non-normal posteriors and avoids the
    assumption of a known sampling distribution required by the frequentist
    JN formula.

    Parameters
    ----------
    intercept_draws : ndarray, shape (n_draws,)
        Posterior draws of the conditional effect intercept (e.g., lambda_sp
        or lambda_ps -- the coupling at moderator = 0).
    slope_draws : ndarray, shape (n_draws,)
        Posterior draws of the moderator slope (e.g., gamma_sp or omega_sp).
    X_vals : ndarray, shape (n_subjects,)
        Observed moderator values for all subjects (used to determine the
        grid range and for rug plots). Can be in z-scored or raw units.
    raw_mean : float or None, default None
        If the moderator was z-scored, provide the original mean to
        back-transform the x-grid to raw units for plotting.
    raw_sd : float or None, default None
        If the moderator was z-scored, provide the original SD.
    clip_pct : tuple (lo, hi), default (1, 99)
        Percentiles of X_vals used to set the grid range. Use (0, 100)
        for the full range (may be affected by outliers).

    Returns
    -------
    result : dict
        Keys:
          x_grid       -- plotting grid (raw units if raw_mean/sd given)
          x_grid_z     -- plotting grid (z-scored or original units)
          post_mean    -- posterior mean coupling at each grid point
          ci_lo, ci_hi -- 95% credible interval bounds
          sig          -- boolean array, True where CrI excludes zero
          sig_negative -- boolean array, True where CrI entirely below zero
          sig_positive -- boolean array, True where CrI entirely above zero
          jn_boundaries   -- boundary values (raw or original units)
          jn_boundaries_z -- boundary values (z-scored or original units)
          obs_vals     -- observed values (raw or original units)
          X_vals_z     -- observed values (z-scored or original units)
          intercept_mean, slope_mean -- posterior means
          raw_mean, raw_sd -- back-transformation parameters

    References
    ----------
    Johnson, P. O., & Neyman, J. (1936). Tests of certain linear hypotheses
      and their application to some educational problems. Statistical
      Research Memoirs, 1, 57-93.

    Bauer, D. J., & Curran, P. J. (2005). Probing interactions in fixed and
      multilevel regression: Inferential and graphical techniques.
      Multivariate Behavioral Research, 40, 373-400.
    """
    # Build a dense grid spanning the observed moderator range
    x_min = np.percentile(X_vals, clip_pct[0])
    x_max = np.percentile(X_vals, clip_pct[1])
    x_grid_z = np.linspace(x_min, x_max, 500)

    # Compute the conditional coupling at every grid point for every
    # posterior draw: coupling(x) = intercept + slope * x
    # Shape: (n_draws, 500)
    coupling_grid = (
        intercept_draws[:, None] + slope_draws[:, None] * x_grid_z[None, :]
    )

    # Posterior summary at each grid point
    post_mean = coupling_grid.mean(axis=0)
    ci_lo = np.percentile(coupling_grid, 2.5, axis=0)
    ci_hi = np.percentile(coupling_grid, 97.5, axis=0)

    # Determine significance: CrI excludes zero
    sig_negative = ci_hi < 0   # entirely below zero
    sig_positive = ci_lo > 0   # entirely above zero
    sig = sig_negative | sig_positive

    # Find JN boundaries by linear interpolation where significance changes
    jn_boundaries_z = []
    for i in range(1, len(x_grid_z)):
        if sig[i] != sig[i - 1]:
            # Significance status changed between grid points i-1 and i.
            # Interpolate to find the exact boundary.
            if sig_negative[i] != sig_negative[i - 1]:
                # Upper CI crossed zero
                x0 = x_grid_z[i - 1] + (x_grid_z[i] - x_grid_z[i - 1]) * (
                    -ci_hi[i - 1] / (ci_hi[i] - ci_hi[i - 1])
                )
                jn_boundaries_z.append(x0)
            elif sig_positive[i] != sig_positive[i - 1]:
                # Lower CI crossed zero
                x0 = x_grid_z[i - 1] + (x_grid_z[i] - x_grid_z[i - 1]) * (
                    -ci_lo[i - 1] / (ci_lo[i] - ci_lo[i - 1])
                )
                jn_boundaries_z.append(x0)

    # Back-transform to raw units if z-score parameters were provided
    if raw_mean is not None and raw_sd is not None:
        x_grid = x_grid_z * raw_sd + raw_mean
        obs_raw = X_vals * raw_sd + raw_mean
        jn_boundaries = [z * raw_sd + raw_mean for z in jn_boundaries_z]
    else:
        x_grid = x_grid_z
        obs_raw = X_vals
        jn_boundaries = jn_boundaries_z

    return {
        "x_grid": x_grid,
        "x_grid_z": x_grid_z,
        "post_mean": post_mean,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "sig": sig,
        "sig_negative": sig_negative,
        "sig_positive": sig_positive,
        "jn_boundaries": jn_boundaries,
        "jn_boundaries_z": jn_boundaries_z,
        "obs_vals": obs_raw,
        "X_vals_z": X_vals,
        "intercept_mean": intercept_draws.mean(),
        "slope_mean": slope_draws.mean(),
        "raw_mean": raw_mean,
        "raw_sd": raw_sd,
    }


# ===================================================================
# Result Extraction
# ===================================================================

def extract_results(trace, moderator_name=None):
    """Extract summary statistics from the posterior.

    Computes posterior means, 95% credible intervals, probability of
    direction (P(beta < 0)), and convergence diagnostics (R-hat) for
    all model parameters.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples from ``fit_bayesian_varx1()``.
    moderator_name : str or None, default None
        If provided, also extract gamma_sp and gamma_ps summaries for
        the named moderator. The name is included in the output for
        identification.

    Returns
    -------
    results : dict
        Dictionary containing:
          - Population parameters (a2/lambda_sp, b1/lambda_ps, etc.)
          - Their posterior means, SDs, 95% CrIs
          - P(beta < 0) for each coupling parameter
          - R-hat maximum across all scalar parameters
          - Random effect SDs (tau_sp, tau_ps)
          - Innovation parameters (sigma_pain, sigma_sleep, rho)
          - If moderator_name is set: gamma_sp and gamma_ps summaries
    """
    # Identify which scalar parameters are in the posterior
    scalar_vars = [
        "a0", "a1", "a2", "a3", "a4",
        "b0", "b1", "b2", "b3", "b4",
        "tau_sp", "tau_ps", "sigma_pain", "sigma_sleep", "rho_innov",
    ]

    has_agesex = "g_sp_age" in trace.posterior
    if has_agesex:
        scalar_vars.extend(["g_sp_age", "g_sp_sex", "g_ps_age", "g_ps_sex"])

    has_moderator = "gamma_sp" in trace.posterior
    if has_moderator:
        scalar_vars.extend(["gamma_sp", "gamma_ps"])

    # ArviZ summary gives mean, sd, hdi, r_hat, ess
    summary = az.summary(trace, var_names=scalar_vars)
    rhat_max = summary["r_hat"].max()

    # Build results dictionary
    results = {
        "rhat_max": rhat_max,
    }

    # Extract each parameter's posterior summary
    for var in scalar_vars:
        if var in summary.index:
            post = trace.posterior[var].values.flatten()
            results[f"{var}_mean"] = float(np.mean(post))
            results[f"{var}_sd"] = float(np.std(post))
            results[f"{var}_ci_lo"] = float(np.percentile(post, 2.5))
            results[f"{var}_ci_hi"] = float(np.percentile(post, 97.5))
            results[f"{var}_prob_neg"] = float((post < 0).mean())

    # Compute two-tailed posterior p-values for coupling parameters
    for var in ["a2", "b1"]:
        p_neg = results.get(f"{var}_prob_neg", 0.5)
        results[f"{var}_p_twotail"] = 2 * min(p_neg, 1 - p_neg)

    if has_moderator:
        for var in ["gamma_sp", "gamma_ps"]:
            p_neg = results.get(f"{var}_prob_neg", 0.5)
            results[f"{var}_p_twotail"] = 2 * min(p_neg, 1 - p_neg)

    if moderator_name is not None:
        results["moderator_name"] = moderator_name

    return results


def extract_person_posteriors(idata, unique_ids, direction):
    """Extract person-specific coupling posterior summaries.

    Computes the full posterior distribution of each person's coupling
    slope by combining the population mean with their random effect:
      person_coupling_i = population_mean + u_i

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior samples from ``fit_bayesian_varx1()``.
    unique_ids : list
        Subject IDs corresponding to person indices 0..N-1.
    direction : str
        ``'sp'`` for sleep-to-pain (a2 + u_sp) or
        ``'ps'`` for pain-to-sleep (b1 + u_ps).

    Returns
    -------
    person_df : DataFrame
        One row per person with columns: ID, posterior_mean, posterior_median,
        posterior_sd, ci_lo, ci_hi, prob_negative, prob_positive.
    coupling_samples : ndarray, shape (n_draws, n_persons)
        Full posterior draws for each person's coupling coefficient.
    """
    n_persons = len(unique_ids)

    if direction == "sp":
        beta_samples = idata.posterior["a2"].values   # (chains, draws)
        u_samples = idata.posterior["u_sp"].values    # (chains, draws, persons)
    else:
        beta_samples = idata.posterior["b1"].values
        u_samples = idata.posterior["u_ps"].values

    # Flatten chains: (chains * draws,) and (chains * draws, persons)
    beta_flat = beta_samples.reshape(-1)
    u_flat = u_samples.reshape(-1, n_persons)

    # Person-specific coupling = population mean + person random effect
    coupling_samples = beta_flat[:, None] + u_flat

    # Summarize each person's posterior
    rows = []
    for i, pid in enumerate(unique_ids):
        post = coupling_samples[:, i]
        rows.append({
            "ID": pid,
            "posterior_mean": float(np.mean(post)),
            "posterior_median": float(np.median(post)),
            "posterior_sd": float(np.std(post)),
            "ci_lo": float(np.percentile(post, 2.5)),
            "ci_hi": float(np.percentile(post, 97.5)),
            "prob_negative": float((post < 0).mean()),
            "prob_positive": float((post > 0).mean()),
        })

    return pd.DataFrame(rows), coupling_samples
