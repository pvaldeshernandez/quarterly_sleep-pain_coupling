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

import glob
import json
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
N_SAMPLES = 4000    # posterior draws per chain
N_TUNE = 4000       # warmup/tuning iterations per chain
N_CHAINS = 4        # number of independent Markov chains
TARGET_ACCEPT = 0.99  # NUTS target acceptance probability (high for HMC)
RANDOM_SEED = 42

#: The one place the sampler is configured. `run_fit` is the only function in the
#: repository that calls `pm.sample`; every step reaches it through this dict, so a
#: fit cannot silently run at settings the Methods does not describe. Raised from
#: 2000/2000/0.95 on Aug 8 2026 together with the non-centered parameterization.
SAMPLER = dict(draws=N_SAMPLES, tune=N_TUNE, chains=N_CHAINS,
               target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED)



# ===================================================================
# Sampling -- the only place in the repository that calls pm.sample
# ===================================================================

def _fmt(x):
    """JSON-safe scalar."""
    if x is None:
        return None
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def write_diagnostics(idata, fit_id, cfg, out_dir):
    """Record how well one fit sampled, beside the fit itself.

    Written for EVERY fit, without exception, because the paper reports diagnostics
    for every model it fits and the alternative -- remembering to compute them at
    each call site -- is what left the submitted manuscript quoting a placeholder.

    The sampler configuration actually used is stored alongside the numbers, so a fit
    that deviated is visible in its own output rather than nowhere.

    Returns the dict it wrote; writes nothing if `out_dir` is None.
    """
    summary = az.summary(idata, round_to="none")          # NOT round_to=None: in
    # ArviZ 0.23 the sentinel that disables rounding is the STRING "none"; passing
    # None silently rounds to 1 decimal and would corrupt every diagnostic here.

    offsets = [i for i in summary.index if str(i).endswith("_z") or "_z[" in str(i)]
    interp = summary.drop(index=offsets, errors="ignore")

    sample_stats = idata.sample_stats
    depth = sample_stats["tree_depth"].values if "tree_depth" in sample_stats else None
    diverging = sample_stats["diverging"].values if "diverging" in sample_stats else None
    accept = sample_stats["acceptance_rate"].values if "acceptance_rate" in sample_stats else None
    try:
        bfmi = np.asarray(az.bfmi(idata))
    except Exception:
        bfmi = np.array([np.nan])

    rec = {
        "fit_id": fit_id,
        "n_params": int(len(interp)),
        "rhat_max": _fmt(interp["r_hat"].max()),
        "rhat_max_param": str(interp["r_hat"].idxmax()),
        "ess_bulk_min": _fmt(interp["ess_bulk"].min()),
        "ess_bulk_min_param": str(interp["ess_bulk"].idxmin()),
        "ess_tail_min": _fmt(interp["ess_tail"].min()),
        "ess_tail_min_param": str(interp["ess_tail"].idxmin()),
        "divergences": int(diverging.sum()) if diverging is not None else None,
        "n_draws_total": int(idata.posterior.draw.size * idata.posterior.chain.size),
        "max_tree_depth": int(depth.max()) if depth is not None else None,
        "tree_depth_cap": 10,
        "pct_draws_at_cap": _fmt(float((depth >= 10).mean() * 100)) if depth is not None else None,
        "mean_accept": _fmt(float(accept.mean())) if accept is not None else None,
        "bfmi_min": _fmt(float(np.nanmin(bfmi))),
        "bfmi_per_chain": "; ".join(f"{b:.3f}" for b in np.atleast_1d(bfmi)),
    }
    rec.update({f"cfg_{k}": _fmt(v) for k, v in cfg.items()})

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"diagnostics_{fit_id}.json")
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=1)
        # The PER-PARAMETER table too, not just the extrema. Table S3 reports
        # convergence by parameter family, and that cannot be recovered afterwards:
        # the saved posterior npz holds flattened draws with no chain structure, so
        # R-hat and ESS are computable only here, while the InferenceData exists.
        interp.to_csv(os.path.join(out_dir, f"diagnostics_{fit_id}_by_param.csv"))
    return rec


def read_diagnostics(out_dir, fit_id_prefix=""):
    """Every diagnostics record `write_diagnostics` left in `out_dir`, as one frame.

    The counterpart of ``write_diagnostics``. Step 22 aggregates across the whole
    pipeline and several steps aggregate over their own fits; both read the JSON
    rather than recompute, because R-hat and ESS need the chain structure and the
    saved posteriors keep only flattened draws.

    Parameters
    ----------
    out_dir : str
        Directory holding ``diagnostics_<fit_id>.json`` files. A missing
        directory gives an empty frame, not an error: a step that has not been
        refit yet has no diagnostics, and that is a reportable state.
    fit_id_prefix : str, default ""
        Keep only fits whose id starts with this. Use it to separate several
        steps' fits when they share a directory.

    Returns
    -------
    DataFrame
        One row per fit, sorted by ``fit_id``; empty (no columns) when nothing
        matched.
    """
    if not os.path.isdir(out_dir):
        return pd.DataFrame()
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "diagnostics_*.json"))):
        try:
            with open(path) as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        if fit_id_prefix and not str(rec.get("fit_id", "")).startswith(fit_id_prefix):
            continue
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("fit_id").reset_index(drop=True)


def summarize_posterior(idata, labels=None, n_obs=None, n_persons=None,
                        var_names=None):
    """One tidy row per scalar parameter of one fit.

    The canonical replacement for the near-identical summary blocks that used to
    live in the sandbox (a07, a09, a09b). Every consumer gets the same column
    names, so summaries of different fits stack without renaming.

    Non-centered offsets (``*_z``) are dropped: they are an implementation
    detail of the parameterization, carry no interpretation, and would swamp a
    table with hundreds of rows.

    Parameters
    ----------
    idata : arviz.InferenceData
        A fit produced by ``fit_bayesian_varx1``.
    labels : dict, optional
        Extra constant columns, inserted BEFORE ``param`` so they read as keys
        (e.g. ``{"covariate_set": "clinical", "model": "C1"}``).
    n_obs, n_persons : int, optional
        The fit's sample size, carried as columns so a stacked summary can state
        what each row was estimated on.
    var_names : sequence of str, optional
        Restrict to these parameters. Default is every scalar in the posterior.

    Returns
    -------
    DataFrame
        Columns: [*labels], param, mean, sd, ci_lo_2.5, ci_hi_97.5, P_neg,
        rhat, ess_bulk, ess_tail, [n_obs], [n_persons].

        ``P_neg`` is P(theta < 0) computed from the draws themselves, not from a
        normal approximation to the summary. ``ci_lo_2.5``/``ci_hi_97.5`` are
        equal-tailed percentiles of the draws, matching how every table in the
        paper reports its 95% credible intervals -- ArviZ's default ``hdi`` is a
        DIFFERENT interval and would silently change published bounds.
    """
    post = idata.posterior
    summary = az.summary(idata, round_to="none")

    rows = []
    for name in post.data_vars:
        if str(name).endswith("_z"):
            continue
        if var_names is not None and name not in var_names:
            continue
        draws = post[name].values
        # Scalars only. A vector parameter (u_sp, u_ps) has one entry per person
        # and belongs in the person-level output, not in a parameter table.
        if draws.ndim != 2:
            continue
        flat = draws.reshape(-1)
        row = dict(labels) if labels else {}
        row["param"] = str(name)
        row["mean"] = float(np.mean(flat))
        row["sd"] = float(np.std(flat))
        row["ci_lo_2.5"] = float(np.percentile(flat, 2.5))
        row["ci_hi_97.5"] = float(np.percentile(flat, 97.5))
        row["P_neg"] = float((flat < 0).mean())
        if name in summary.index:
            row["rhat"] = float(summary.loc[name, "r_hat"])
            row["ess_bulk"] = float(summary.loc[name, "ess_bulk"])
            row["ess_tail"] = float(summary.loc[name, "ess_tail"])
        else:
            row["rhat"] = row["ess_bulk"] = row["ess_tail"] = np.nan
        if n_obs is not None:
            row["n_obs"] = int(n_obs)
        if n_persons is not None:
            row["n_persons"] = int(n_persons)
        rows.append(row)

    return pd.DataFrame(rows)


#: Historical name. `summarize_posterior` is canonical; this alias exists because
#: step 10 was written against `tidy_posterior_summary` while step 09 was written
#: against `summarize_posterior`. One function, two names -- never two functions.
tidy_posterior_summary = summarize_posterior


def run_fit(model, fit_id=None, out_dir=None, progressbar=True, **overrides):
    """Sample `model` with the project's settings and record how it went.

    Every fit in the pipeline goes through here. `pm.sample` is called nowhere else,
    which is what makes the Methods description of the sampler true by construction
    instead of by vigilance -- before this existed, step23 sampled at its own
    hardcoded 2000/2000/0.95 while the library said otherwise.

    `overrides` are for genuine exceptions and are recorded in the diagnostics, so a
    deviation leaves a trace.
    """
    cfg = {**SAMPLER, **overrides}
    kwargs = dict(cfg)
    kwargs.update(
        return_inferencedata=True,
        progressbar=progressbar,
        # Required by a known PyMC 5.27.1 + Python 3.13 bug where R-hat computation
        # crashes on large random-effect arrays. Convergence is computed below from
        # the ArviZ summary instead.
        compute_convergence_checks=False,
    )
    with model:
        idata = pm.sample(**kwargs)
    if fit_id is not None:
        write_diagnostics(idata, fit_id, cfg, out_dir)
    return idata


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
# Small shared helpers
# ===================================================================

def two_tail_p(prob_neg):
    """Two-tailed Bayesian posterior "p-value" from P(parameter < 0).

        p = 2 * min(P(<0), 1 - P(<0))

    Returns NaN for NaN input. Shared by all step-* scripts that
    summarize γ posteriors.
    """
    if prob_neg is None or not np.isfinite(prob_neg):
        return np.nan
    return 2.0 * min(prob_neg, 1.0 - prob_neg)


def sign_concordance_p(n_concordant, n_tested):
    """Upper-tail binomial p-value for observing ``n_concordant`` out
    of ``n_tested`` expected-sign matches under a fair-coin null:

        p = P(X >= n_concordant | X ~ Binomial(n_tested, 0.5))

    Used by step16 (Krause 6-ROI sign test) and step21 (Lynch 5-ROI
    sign tests, fMRI and VBM).
    """
    from math import comb
    if n_tested == 0:
        return np.nan
    return sum(
        comb(n_tested, k) * 0.5 ** n_tested
        for k in range(n_concordant, n_tested + 1)
    )


# ===================================================================
# Shared dataframe loader for step08 / step16 / step21 / step23
# ===================================================================

def load_varx_frame(csv_path, verbose=False):
    """Load the VARX-ready long dataframe and prepare it for any
    ``fit_bayesian_varx1`` / step23 call.

    Centralizes four previously-duplicated loader functions
    (``step08.load_data``, ``step16.load_step05_data``,
    ``step21.load_step05_data``, ``step23.load_data``). Performs:

      1. Read ``csv_path``.
      2. Compute Age_z and Sex_c at the PERSON level (one row per
         subject; D4 fix — not observation-weighted).
      3. Merge Age_z / Sex_coded / Sex_c back onto the long frame.
      4. Build ``pid_idx`` column from ``sorted(unique(ID))``.
      5. Drop rows with NaN in the lag columns to produce
         ``model_df`` suitable for fitting.

    Returns
    -------
    df : DataFrame
        Full long frame (all rows) with Age_z, Sex_coded, Sex_c,
        pid_idx columns added.
    model_df : DataFrame
        ``df`` minus rows with NaN in ``pain_within_lag1`` or
        ``sleep_within_lag1``.
    unique_ids : list
        Sorted subject IDs.
    id_map : dict
        Mapping {subject_ID: integer_index}.
    """
    df = pd.read_csv(csv_path)

    person_demo = (
        df.groupby("ID").first()[["Age", "Sex"]].reset_index()
    )
    age_mean = person_demo["Age"].mean()
    age_sd = person_demo["Age"].std()
    person_demo["Age_z"] = (person_demo["Age"] - age_mean) / age_sd
    person_demo["Sex_coded"] = (person_demo["Sex"] == 2).astype(float)
    sex_mean = person_demo["Sex_coded"].mean()
    person_demo["Sex_c"] = person_demo["Sex_coded"] - sex_mean

    df = df.drop(
        columns=[c for c in ("Age_z", "Sex_coded", "Sex_c")
                 if c in df.columns]
    )
    df = df.merge(
        person_demo[["ID", "Age_z", "Sex_coded", "Sex_c"]],
        on="ID", how="left",
    )

    unique_ids = sorted(df["ID"].unique())
    id_map = {sid: i for i, sid in enumerate(unique_ids)}
    df["pid_idx"] = df["ID"].map(id_map)

    model_df = df.dropna(
        subset=["pain_within_lag1", "sleep_within_lag1"]
    ).copy()

    if verbose:
        n_female = int((person_demo["Sex_coded"] == 1).sum())
        n_male = int((person_demo["Sex_coded"] == 0).sum())
        print(
            f"  Loaded: {len(unique_ids)} subjects, "
            f"{len(model_df)} observations"
        )
        print(f"  Age (person-level): mean={age_mean:.1f}, SD={age_sd:.1f}")
        print(f"  Sex: {n_female} female, {n_male} male")
        print(
            f"  Sex centering (person-level): "
            f"mean(Sex_coded)={sex_mean:.4f}"
        )

    return df, model_df, unique_ids, id_map


# ===================================================================
# Correlated-innovations likelihood (shared helper)
# ===================================================================

def add_bivariate_innovations_likelihood(
    mu_pain, mu_sleep, y_pain, y_sleep, name_suffix=""
):
    """Attach the bivariate-normal innovations likelihood to the current
    PyMC model via the sequential-conditioning Cholesky trick.

    Factoring the bivariate normal as ``p(pain) * p(sleep | pain)``
    avoids the O(n^3) per-evaluation cost of ``pm.MvNormal`` and
    samples ~100x faster while being mathematically identical (see the
    module docstring). Both the main VARX fit (``fit_bayesian_varx1``)
    and the severity-joint model (``step23``) attach their likelihood
    through this helper so the speed and priors stay in sync.

    Priors (declared here and returned as Deterministics/RVs):

      sigma_pain, sigma_sleep ~ HalfCauchy(2)
      rho_raw                 ~ Beta(2, 2)
      rho_innov = 2 * rho_raw - 1  on [-1, 1]

    Parameters
    ----------
    mu_pain, mu_sleep : tensors
        Observation-level predicted means for pain and sleep.
    y_pain, y_sleep : arrays
        Observation-level observed values.
    name_suffix : str, optional
        Appended to the two observation-node names (``y_pain`` and
        ``y_sleep``) so multiple likelihoods can coexist in one model
        if ever needed. Default empty.

    Returns
    -------
    dict : keys ``sigma_pain``, ``sigma_sleep``, ``rho_innov``
        The prior RVs, so the caller can reuse them in deterministics
        or posterior extraction.
    """
    sigma_pain = pm.HalfCauchy(f"sigma_pain{name_suffix}", beta=2)
    sigma_sleep = pm.HalfCauchy(f"sigma_sleep{name_suffix}", beta=2)
    rho_raw = pm.Beta(f"rho_raw{name_suffix}", alpha=2, beta=2)
    rho_innov = pm.Deterministic(
        f"rho_innov{name_suffix}", 2 * rho_raw - 1
    )

    # Step 1: pain innovation (marginal).
    pm.Normal(
        f"y_pain{name_suffix}",
        mu=mu_pain, sigma=sigma_pain, observed=y_pain,
    )

    # Step 2: sleep innovation conditioned on pain residual.
    eps_pain = y_pain - mu_pain
    mu_sleep_cond = (
        mu_sleep + rho_innov * (sigma_sleep / sigma_pain) * eps_pain
    )
    sigma_sleep_cond = sigma_sleep * pt.sqrt(1 - rho_innov ** 2)
    pm.Normal(
        f"y_sleep{name_suffix}",
        mu=mu_sleep_cond, sigma=sigma_sleep_cond, observed=y_sleep,
    )

    return {
        "sigma_pain": sigma_pain,
        "sigma_sleep": sigma_sleep,
        "rho_innov": rho_innov,
    }


# ===================================================================
# Model Fitting
# ===================================================================

def build_timevarying(model_df, df_full, raw_long, covars,
                      standardize=True, binary=False, interpolate=True):
    """Within-person-centered time-varying covariates and their within-segment lag-1.

    The merged canonical form of the two near-identical builders that used to
    live in the sandbox (a09's continuous mood/fatigue version and a09b's binary
    treatment version). They differed in exactly two places -- whether the
    centered deviations were z-scored, and whether interpolated values were
    rounded back to {0, 1} -- so those are the two parameters.

    The recipe mirrors how pain and sleep themselves are treated, deliberately:
    interpolate single interior gaps, subtract the person mean, optionally
    z-score the pooled within-person deviations, then lag WITHIN
    ``(ID, segment_id)`` so a lag never crosses a break in a person's run.

    Parameters
    ----------
    model_df : DataFrame
        The transitions being modelled; keyed on (ID, quarter).
    df_full : DataFrame
        The full long frame, which is where ``segment_id`` lives. The lag has to
        be taken on the full frame and merged back: taking it on ``model_df``
        would shift across the rows that lag-completeness already removed.
    raw_long : DataFrame
        The raw item-level frame holding the covariate columns, with ID and
        quarter. Passed in rather than read from a path, so the same function
        serves any source frame.
    covars : dict
        ``{source_column: output_name}``. The output frame gains
        ``<name>_within`` and ``<name>_within_lag1`` for each entry.
    standardize : bool, default True
        Z-score the within-person deviations (population SD, ``ddof=0``) so a
        theta is per within-person SD. Set False to keep the raw units, which is
        what a binary item wants.
    binary : bool, default False
        Round interpolated values back to whole numbers, so a 0/1 item cannot
        acquire a 0.5. Only affects interpolated cells.
    interpolate : bool, default True
        Fill single interior gaps (pandas linear, ``limit=1``,
        ``limit_area="inside"``) exactly as ``step01`` does for the factor
        scores. Consistency with the primary variables is the point: a covariate
        that dropped rows the outcome kept would change the sample for a reason
        that has nothing to do with the adjustment.

    Returns
    -------
    DataFrame
        ``model_df`` with the ``_within`` and ``_within_lag1`` columns merged on.
        Rows where a covariate was never observed keep NaN; selecting the
        complete subset is the caller's decision, because which block of
        covariates must be complete depends on the model being fit.
    """
    need = ["ID", "quarter"] + list(covars)
    missing = [c for c in need if c not in raw_long.columns]
    if missing:
        raise KeyError(f"raw frame is missing {missing}")
    if "segment_id" not in df_full.columns:
        raise KeyError(
            "df_full has no segment_id; the lag would silently cross a break "
            "in a person's run of quarters"
        )

    raw = raw_long[need].copy().sort_values(["ID", "quarter"])

    if interpolate:
        for src in covars:
            filled = raw.groupby("ID")[src].transform(
                lambda s: s.interpolate(method="linear", limit=1,
                                        limit_area="inside")
            )
            raw[src] = filled.round() if binary else filled

    for src, name in covars.items():
        w = raw[src] - raw.groupby("ID")[src].transform("mean")
        raw[f"{name}_within"] = (w - w.mean()) / w.std(ddof=0) if standardize else w

    within = ["ID", "quarter"] + [f"{n}_within" for n in covars.values()]
    out = model_df.merge(raw[within], on=["ID", "quarter"], how="left")

    src_full = (df_full[["ID", "quarter", "segment_id"]]
                .merge(raw[within], on=["ID", "quarter"], how="left")
                .sort_values(["ID", "segment_id", "quarter"]))
    for n in covars.values():
        src_full[f"{n}_within_lag1"] = (
            src_full.groupby(["ID", "segment_id"])[f"{n}_within"].shift(1)
        )
    lagged = ["ID", "quarter"] + [f"{n}_within_lag1" for n in covars.values()]
    return out.merge(src_full[lagged], on=["ID", "quarter"], how="left")


def fit_bayesian_varx1(model_df, unique_ids, id_map,
                       X_person=None, include_agesex=True,
                       include_sp=True, include_ps=True,
                       moderator_direction="both",
                       X_tv=None,
                       idata_kwargs=None,
                       cores=None,
                       progressbar=True,
                       fit_id=None, out_dir=None):
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
    moderator_direction : {"both", "sp", "ps", "none"}, default "both"
        Which coupling slope(s) the external moderator X_person is
        attached to. Independent of ``include_sp`` / ``include_ps``,
        which control whether each coupling direction exists at all.
        Use "sp" for Krause/ACC-style hypotheses (X moderates only
        sleep->pain), "ps" for Lynch-style hypotheses (X moderates
        only pain->sleep), "both" to attach X to both slopes (legacy
        behavior), and "none" to ignore X entirely. Requires
        ``include_sp=True`` when "sp" or "both"; ``include_ps=True``
        when "ps" or "both". Has no effect when ``X_person is None``.
    X_tv : sequence of str, optional
        Column names of TIME-VARYING covariates to adjust for. Each gets a
        ``theta_p_<col>`` and a ``theta_s_<col>``, both ``N(0, 5)``, entering
        the pain and sleep linear predictors as main effects. They do NOT touch
        the coupling slopes: this asks whether lambda survives adjustment, which
        is a different question from whether the covariate moderates lambda
        (that is ``X_person``). Columns must exist in ``model_df`` and be
        complete on every row passed; both are checked, because PyMC would
        otherwise propagate a NaN into the likelihood and return a NaN
        log-probability instead of failing.
        Build the columns with ``build_timevarying``.
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
    # Validate moderator_direction against include_sp / include_ps.
    # ------------------------------------------------------------------
    if moderator_direction not in ("both", "sp", "ps", "none"):
        raise ValueError(
            f"moderator_direction must be one of 'both','sp','ps','none'; "
            f"got {moderator_direction!r}"
        )
    if X_person is not None and moderator_direction in ("sp", "both") \
            and not include_sp:
        raise ValueError(
            "moderator_direction='sp' (or 'both') requires include_sp=True"
        )
    if X_person is not None and moderator_direction in ("ps", "both") \
            and not include_ps:
        raise ValueError(
            "moderator_direction='ps' (or 'both') requires include_ps=True"
        )

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
        # Attached only to the direction(s) specified by
        # moderator_direction, independent of include_sp/include_ps.
        attach_sp = has_moderator and moderator_direction in ("sp", "both")
        attach_ps = has_moderator and moderator_direction in ("ps", "both")
        if attach_sp:
            gamma_sp = pm.Normal("gamma_sp", mu=0, sigma=1)
        if attach_ps:
            gamma_ps = pm.Normal("gamma_ps", mu=0, sigma=1)

        # --- Person-specific random coupling slopes ---
        # tau controls the between-person heterogeneity in coupling.
        # Random effects are only sampled for directions that are
        # present in the model.
        if include_sp:
            tau_sp = pm.HalfCauchy("tau_sp", beta=1)
            # NON-CENTERED. u = z * tau with z ~ N(0,1) is exactly u | tau ~ N(0, tau^2),
            # and the HalfCauchy(1) prior on tau is untouched, so this is a
            # reparameterization and not a model change. It is the remedy for the funnel
            # that left BFMI at 0.09-0.16 and bulk ESS at ~370 no matter how many draws
            # were taken. `u_sp` remains in the posterior as a Deterministic, so every
            # downstream reader of idata.posterior["u_sp"] is unaffected.
            #
            # The offset is named `u_sp_z` deliberately: it contains the substring
            # "u_sp", so az.summary(var_names=["~u_sp", "~u_ps"], filter_vars="like")
            # still excludes every person-level parameter. Naming it `z_sp` would
            # silently turn "worst population-level parameter" into a max over 458
            # person-level offsets.
            u_sp_z = pm.Normal("u_sp_z", mu=0, sigma=1, shape=n_persons)
            u_sp = pm.Deterministic("u_sp", u_sp_z * tau_sp)
        if include_ps:
            tau_ps = pm.HalfCauchy("tau_ps", beta=1)
            u_ps_z = pm.Normal("u_ps_z", mu=0, sigma=1, shape=n_persons)
            u_ps = pm.Deterministic("u_ps", u_ps_z * tau_ps)

        # --- Assemble person-varying coupling slopes ---
        # If the corresponding direction is disabled, a2_i / b1_i stay
        # at zero and the corresponding cross-lag term drops out of the
        # linear predictor below.
        if include_sp:
            a2_i = a2 + u_sp[idx]
            if include_agesex:
                a2_i = a2_i + g_sp_age * age_z + g_sp_sex * sex_c
            if attach_sp:
                a2_i = a2_i + gamma_sp * X_obs
        if include_ps:
            b1_i = b1 + u_ps[idx]
            if include_agesex:
                b1_i = b1_i + g_ps_age * age_z + g_ps_sex * sex_c
            if attach_ps:
                b1_i = b1_i + gamma_ps * X_obs

        # --- Linear predictors ---
        mu_pain = a0 + a1 * pain_lag + a3 * contrast_lag

        # --- Time-varying covariates, main effects only ---------------------
        # One theta per covariate per equation, N(0, 5) as for the localization
        # terms. They enter the LINEAR PREDICTORS only, never the coupling
        # slopes, so lambda_sp and lambda_ps keep the primary model's meaning
        # and Table S4 answers "does the coupling survive adjustment?" rather
        # than estimating a different quantity.
        tv_cols = list(X_tv) if X_tv else []
        for col in tv_cols:
            if col not in model_df.columns:
                raise KeyError(
                    f"X_tv names {col!r}, which is not a column of the model "
                    f"frame. Build the covariate columns with "
                    f"`build_timevarying` and pass the subset that is complete."
                )
            if model_df[col].isna().any():
                raise ValueError(
                    f"X_tv column {col!r} has "
                    f"{int(model_df[col].isna().sum())} missing value(s). "
                    f"PyMC would propagate the NaN into the likelihood and the "
                    f"fit would return NaN log-probabilities rather than fail; "
                    f"restrict the frame to complete rows first."
                )
        for col in tv_cols:
            x_tv = model_df[col].values.astype("float64")
            mu_pain = mu_pain + pm.Normal(f"theta_p_{col}", mu=0, sigma=5) * x_tv
        if include_sp:
            mu_pain = mu_pain + a2_i * sleep_lag + a4 * sleep_x_contrast

        mu_sleep = b0 + b2 * sleep_lag + b3 * contrast_lag
        for col in tv_cols:
            x_tv = model_df[col].values.astype("float64")
            mu_sleep = mu_sleep + pm.Normal(f"theta_s_{col}", mu=0, sigma=5) * x_tv
        if include_ps:
            mu_sleep = mu_sleep + b1_i * pain_lag + b4 * pain_x_contrast

        # --- Correlated innovations via Cholesky trick ---
        add_bivariate_innovations_likelihood(
            mu_pain=mu_pain, mu_sleep=mu_sleep,
            y_pain=y_pain, y_sleep=y_sleep,
        )

        # --- MCMC sampling ---
        # Delegated so that this function does not decide sampler settings and does
        # not have to remember to write diagnostics; `run_fit` owns both.
        extra = {}
        if cores is not None:
            extra["cores"] = cores
        if idata_kwargs is not None:
            extra["idata_kwargs"] = idata_kwargs
        idata = run_fit(model, fit_id=fit_id, out_dir=out_dir,
                        progressbar=progressbar, **extra)

    return idata, sub_df, valid_ids


# ===================================================================
# LOO-CV Model Comparison
# ===================================================================

#: which posterior names carry the population slope and the person deviations
#: for each coupling direction. Naming this once is what lets the ROI loop below
#: be written for "a direction" rather than twice, once per direction.
_DIRECTION_VARS = {
    "sp": {"slope": "a2", "person": "u_sp", "gamma": "gamma_sp"},
    "ps": {"slope": "b1", "person": "u_ps", "gamma": "gamma_ps"},
}


def fit_roi_moderation_set(roi_df, model_df, unique_ids, id_map,
                           direction="sp", rois=None,
                           fit_id_prefix="roi", out_dir=None,
                           include_agesex=True, progressbar=True,
                           verbose=True):
    """Fit one moderation model per ROI and tabulate them.

    The canonical form of the per-ROI loop that steps 14, 17, 19 and 21 all
    need: take a long ROI frame, fit ``fit_bayesian_varx1`` once per ROI with
    that ROI's z-scored value as the person-level moderator, and return one
    tidy row per ROI plus the draws a Johnson-Neyman step needs.

    Parameters
    ----------
    roi_df : DataFrame
        Long, one row per (participant, ROI). Required columns: ``ID``, ``ROI``,
        ``z_value``. Optional and carried through when present: ``label``,
        ``framework``, ``mask_type``, ``modality``, ``raw_mean``, ``raw_sd``,
        ``expected_sign_sp``, ``expected_sign_ps``.
    model_df, unique_ids, id_map
        As for ``fit_bayesian_varx1``.
    direction : {"sp", "ps"}, default "sp"
        Which coupling slope the ROI moderates. Passed straight through as
        ``moderator_direction``, so the OTHER direction's gamma is absent from
        the posterior and its columns come back NaN -- which is the honest
        result, not a missing one.
    rois : sequence of str, optional
        Fit only these, in this order. Default is every ROI in ``roi_df``, in
        first-appearance order. A name that is not present raises rather than
        being skipped: a silently dropped ROI silently deletes a published
        number.
    fit_id_prefix : str, default "roi"
        Each fit is recorded as ``<prefix>_<ROI>``, so ``write_diagnostics``
        leaves one record per ROI and step 22 can find them.
    out_dir : str, optional
        Where the per-fit diagnostics go. None writes none.

    Returns
    -------
    (fitted, draws) : (DataFrame, dict)
        ``fitted`` has one row per ROI: ROI, label, framework, mask_type,
        modality, direction, n_persons, n_obs, rhat_max, the passthrough
        columns, and for BOTH directions gamma_<d>_mean / _sd / _ci_lo /
        _ci_hi / _prob_neg / _p.

        ``draws`` is keyed for ``np.savez``: for each ROI,
        ``<ROI>_<slope>_draws``, ``<ROI>_gamma_<d>_draws``, ``<ROI>_X_vals``,
        ``<ROI>_u_<d>_mean``, ``<ROI>_raw_mean``, ``<ROI>_raw_sd`` -- the six
        arrays a Johnson-Neyman curve needs to be redrawn without refitting.
    """
    if direction not in _DIRECTION_VARS:
        raise ValueError(f"direction must be 'sp' or 'ps'; got {direction!r}")
    names = _DIRECTION_VARS[direction]

    for col in ("ID", "ROI", "z_value"):
        if col not in roi_df.columns:
            raise KeyError(f"roi_df has no {col!r} column")

    available = list(dict.fromkeys(roi_df["ROI"]))
    if rois is None:
        rois = available
    else:
        absent = [r for r in rois if r not in set(available)]
        if absent:
            raise KeyError(f"ROI(s) {absent} absent from roi_df; present: {available}")

    passthrough = [c for c in ("label", "framework", "mask_type", "modality",
                               "raw_mean", "raw_sd",
                               "expected_sign_sp", "expected_sign_ps")
                   if c in roi_df.columns]

    rows, draws = [], {}
    for roi in rois:
        block = roi_df[roi_df["ROI"] == roi]
        X_person = dict(zip(block["ID"].astype(str), block["z_value"].values))

        if verbose:
            label = block["label"].iloc[0] if "label" in block.columns else roi
            print(f"\n    Fitting: {label}...")

        idata, sub_df, valid_ids = fit_bayesian_varx1(
            model_df, unique_ids, id_map,
            X_person=X_person, include_agesex=include_agesex,
            moderator_direction=direction,
            progressbar=progressbar,
            fit_id=f"{fit_id_prefix}_{roi}", out_dir=out_dir,
        )

        res = extract_results(idata, moderator_name=roi)
        row = {"ROI": roi, "direction": direction,
               "n_persons": int(len(valid_ids)), "n_obs": int(len(sub_df)),
               "rhat_max": float(res.get("rhat_max", np.nan))}
        for col in passthrough:
            row[col] = block[col].iloc[0]
        for d in ("sp", "ps"):
            g = _DIRECTION_VARS[d]["gamma"]
            row[f"{g}_mean"] = res.get(f"{g}_mean", np.nan)
            row[f"{g}_sd"] = res.get(f"{g}_sd", np.nan)
            row[f"{g}_ci_lo"] = res.get(f"{g}_ci_lo", np.nan)
            row[f"{g}_ci_hi"] = res.get(f"{g}_ci_hi", np.nan)
            row[f"{g}_prob_neg"] = res.get(f"{g}_prob_neg", np.nan)
            row[f"{g}_p"] = two_tail_p(res.get(f"{g}_prob_neg", np.nan))
        rows.append(row)

        post = idata.posterior
        draws[f"{roi}_{names['slope']}_draws"] = post[names["slope"]].values.reshape(-1)
        draws[f"{roi}_{names['gamma']}_draws"] = post[names["gamma"]].values.reshape(-1)
        draws[f"{roi}_X_vals"] = np.array(
            [X_person[sid] for sid in valid_ids if sid in X_person])
        draws[f"{roi}_{names['person']}_mean"] = (
            post[names["person"]].values.reshape(-1, len(valid_ids)).mean(axis=0))
        for stat in ("raw_mean", "raw_sd"):
            if stat in block.columns:
                draws[f"{roi}_{stat}"] = np.array([block[stat].iloc[0]])

        if verbose:
            g = names["gamma"]
            print(f"      N={row['n_persons']}, obs={row['n_obs']}")
            print(f"      {g} = {row[f'{g}_mean']:+.4f} "
                  f"[{row[f'{g}_ci_lo']:+.4f}, {row[f'{g}_ci_hi']:+.4f}], "
                  f"p={row[f'{g}_p']:.3f}")
            print(f"      R-hat max: {row['rhat_max']:.3f}")

        del idata

    return pd.DataFrame(rows), draws


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

    has_gamma_sp = "gamma_sp" in trace.posterior
    has_gamma_ps = "gamma_ps" in trace.posterior
    has_moderator = has_gamma_sp or has_gamma_ps
    if has_gamma_sp:
        scalar_vars.append("gamma_sp")
    if has_gamma_ps:
        scalar_vars.append("gamma_ps")

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
        results[f"{var}_p_twotail"] = two_tail_p(
            results.get(f"{var}_prob_neg", 0.5)
        )

    for var in ("gamma_sp", "gamma_ps"):
        if var in scalar_vars:
            results[f"{var}_p_twotail"] = two_tail_p(
                results.get(f"{var}_prob_neg", 0.5)
            )

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
