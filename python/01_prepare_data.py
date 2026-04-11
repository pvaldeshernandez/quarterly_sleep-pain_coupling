#!/usr/bin/env python3
"""
01 — Data Preparation for the Bidirectional Sleep-Pain Coupling VARX(1) Model
==============================================================================

This script transforms raw quarterly survey data into the analysis-ready dataset
required by the Bayesian VARX(1) model in script 02. The pipeline has six stages:

  1. **Factor analysis** (real data only):
     A two-factor principal axis factoring (PAF) model is calibrated on the
     polychoric correlation matrix of 8 pain items (4 knee, 4 body-site).
     Factor retention follows Horn's (1965) parallel analysis criterion.

       Factor 1 (General Pain): All 8 items load positively (~0.80). Captures
         overall pain severity irrespective of location. ~66% of variance.
       Factor 2 (Contrast): Knee items load positively, body items negatively.
         Captures WHERE pain is concentrated (knee-dominant vs. body-dominant),
         orthogonal to severity. ~16% of variance.

     Scoring uses the Bartlett (1937) method, which produces BLUE (best linear
     unbiased) factor scores that are uncorrelated across factors even when
     items are missing, by weighting each item inversely by its uniqueness.

  2. **Sleep scoring**:
     Sleep quality (q13, "How well did you sleep?", 0=Very poorly, 10=Very well)
     is z-scored using the grand mean and SD across all person-quarters.
     Higher values = BETTER sleep. This is a single-indicator construct.

  3. **Segment filtering**:
     Retain only runs of >= 3 consecutive quarters where both pain and sleep
     constructs are non-missing. Isolated data points and short runs (1-2 quarters)
     cannot contribute lagged observations to the VARX model.

  4. **Within-person decomposition** (Curran & Bauer, 2011):
     Each variable Y is decomposed as:
       Y_it = Y_bar_i + Y^w_it
     where Y_bar_i is person i's time-average and Y^w_it is the within-person
     deviation. The VARX(1) model operates exclusively on the within-person
     deviations (Y^w), which by construction are free of all stable between-
     person confounders (personality, comorbidities, chronic pain level, etc.).

  5. **Lag creation**:
     Within each person's continuous segment, create t-1 lagged versions of
     the three within-person variables (pain, sleep, contrast). Lags are
     NaN'd at segment boundaries to prevent spurious cross-segment associations.

  6. **Interaction terms**:
     Create the products needed for the contrast moderation terms in the VARX:
       sleep_x_contrast_lag1 = sleep_within_lag1 * contrast_within_lag1
       pain_x_contrast_lag1  = pain_within_lag1  * contrast_within_lag1
     These capture whether the pain localization pattern moderates the
     strength of the cross-lagged coupling.

Synthetic Data Mode
-------------------
When --synthetic is passed, the factor analysis step is SKIPPED because the
synthetic data already contains pre-computed factor scores (pain_severity and
contrast_factor). The script loads from data/synthetic/ and proceeds directly
to stages 3-6. Output is saved to data/synthetic/processed_data.csv.

References
----------
Horn, J. L. (1965). A rationale and test for the number of factors in factor
  analysis. Psychometrika, 30, 179-185.

Bartlett, M. S. (1937). The statistical conception of mental factors. British
  Journal of Psychology, 28, 97-104.

Curran, P. J., & Bauer, D. J. (2011). The disaggregation of within-person
  and between-person effects in longitudinal models of change. Annual Review
  of Psychology, 62, 583-619.

Author: Pedro Valdes-Hernandez
"""

import os
import sys
import argparse
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup: this script lives in python/, data is in data/
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")

# Minimum number of consecutive quarters with both pain and sleep data
# required to retain a segment. Shorter runs cannot contribute lagged
# observations to the VARX(1) model (they produce only NaN lags).
MIN_SEGMENT = 3

# Quarterly pain items from the UPLOAD study (all 0-10 ordinal scales).
# Knee items (q2-q5) load positively on Factor 2; body items (q7-q10)
# load negatively.
KNEE_ITEMS = ["q2_knee_pain", "q3_knee_pain", "q4_knee_pain", "q5_knee_pain"]
BODY_ITEMS = ["q7_body_pain", "q8_body_pain", "q9_body_pain", "q10_body_pain"]
PAIN_ITEMS = KNEE_ITEMS + BODY_ITEMS

# Sleep quality item (0-10): "How well did you sleep?" Higher = better.
SLEEP_ITEM = "q13_sleep"


# ===================================================================
# Polychoric Correlation (for real data factor analysis)
# ===================================================================

def polychoric_corr_pair(x, y):
    """Estimate the polychoric correlation between two ordinal variables.

    The polychoric correlation is the MLE estimate of the latent bivariate
    normal correlation that would produce the observed contingency table,
    given that each ordinal variable is a discretized version of its
    underlying continuous normal variate (Olsson, 1979).

    This is more appropriate than Pearson's r for ordinal 0-10 scales
    because it accounts for the discretization (threshold model) rather
    than treating ordinal categories as interval-scaled.

    Parameters
    ----------
    x, y : ndarray
        Ordinal values (integers). NaN/non-finite values are pairwise deleted.

    Returns
    -------
    rho : float
        Estimated polychoric correlation on [-0.999, 0.999].
    """
    from scipy import stats, optimize

    # Pairwise deletion of missing values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask].astype(int), y[mask].astype(int)

    # Identify unique categories for each variable
    cats_x, cats_y = sorted(np.unique(x)), sorted(np.unique(y))

    # Compute thresholds from marginal cumulative proportions.
    # Under the threshold model, category boundaries in the latent normal
    # are at the z-values corresponding to cumulative marginal proportions.
    cum_x = np.array([np.mean(x <= c) for c in cats_x[:-1]])
    cum_y = np.array([np.mean(y <= c) for c in cats_y[:-1]])
    thresh_x = stats.norm.ppf(cum_x)
    thresh_y = stats.norm.ppf(cum_y)

    # Extend thresholds with +/- infinity for the boundary categories
    tx = np.concatenate([[-np.inf], thresh_x, [np.inf]])
    ty = np.concatenate([[-np.inf], thresh_y, [np.inf]])

    # Build the observed contingency table
    cat_map_x = {c: i for i, c in enumerate(cats_x)}
    cat_map_y = {c: i for i, c in enumerate(cats_y)}
    table = np.zeros((len(cats_x), len(cats_y)))
    for xi, yi in zip(x, y):
        table[cat_map_x[xi], cat_map_y[yi]] += 1

    def neg_loglik(rho_arr):
        """Negative log-likelihood of the polychoric model."""
        rho = rho_arr[0]
        if abs(rho) >= 0.999:
            return 1e10
        ll = 0.0
        cov = [[1, rho], [rho, 1]]
        for i in range(len(cats_x)):
            for j in range(len(cats_y)):
                if table[i, j] == 0:
                    continue
                # Cell probability = integral of bivariate normal over the
                # rectangle defined by adjacent thresholds
                p = (
                    stats.multivariate_normal.cdf(
                        [tx[i + 1], ty[j + 1]], mean=[0, 0], cov=cov
                    )
                    - stats.multivariate_normal.cdf(
                        [tx[i + 1], ty[j]], mean=[0, 0], cov=cov
                    )
                    - stats.multivariate_normal.cdf(
                        [tx[i], ty[j + 1]], mean=[0, 0], cov=cov
                    )
                    + stats.multivariate_normal.cdf(
                        [tx[i], ty[j]], mean=[0, 0], cov=cov
                    )
                )
                ll += table[i, j] * np.log(max(p, 1e-15))
        return -ll

    # Initialize at the Pearson correlation (good starting point)
    r_init = np.corrcoef(x.astype(float), y.astype(float))[0, 1]
    result = optimize.minimize(
        neg_loglik, [r_init], method="L-BFGS-B", bounds=[(-0.999, 0.999)]
    )
    return result.x[0]


def polychoric_corr_matrix(data, items):
    """Compute the full polychoric correlation matrix for a set of ordinal items.

    Parameters
    ----------
    data : DataFrame
        Must contain columns named in ``items``.
    items : list of str
        Column names of ordinal variables.

    Returns
    -------
    R : DataFrame
        Symmetric polychoric correlation matrix (items x items).
    """
    k = len(items)
    R = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            x = data[items[i]].values.astype(float)
            y = data[items[j]].values.astype(float)
            rho = polychoric_corr_pair(x, y)
            R[i, j] = rho
            R[j, i] = rho
    return pd.DataFrame(R, index=items, columns=items)


# ===================================================================
# Factor Analysis: 2-Factor PAF (for real data)
# ===================================================================

def calibrate_pain_factors(df, use_polychoric=False):
    """Calibrate a 2-factor PAF model on the 8 quarterly pain items.

    Principal Axis Factoring (PAF) iteratively estimates communalities by
    replacing the diagonal of the correlation matrix with current communality
    estimates, then extracting eigenvalues/eigenvectors. This is preferred
    over PCA for factor analysis because it models common variance only,
    not total variance (Fabrigar et al., 1999).

    Factor rotation is not applied because the two factors have a natural
    interpretation: Factor 1 (general pain) and Factor 2 (knee-vs-body
    contrast) emerge directly from the eigenstructure due to the item
    structure (4 knee + 4 body items).

    Parameters
    ----------
    df : DataFrame
        Must contain the 8 pain items as columns.
    use_polychoric : bool, default False
        If True, compute polychoric correlations (appropriate for ordinal
        data). If False, use Pearson correlations on z-scored data.

    Returns
    -------
    model : dict
        Keys: items, loadings (8x2), uniquenesses, communalities,
        item_means, item_sds, eigenvalues (2,).
    """
    print("\n" + "=" * 70)
    print("CALIBRATING 2-FACTOR PAIN MODEL (8 items, PAF)")
    print("=" * 70)

    available = [c for c in PAIN_ITEMS if c in df.columns]
    data = df[available].copy()

    # Store raw item means and SDs for Bartlett scoring
    # (needed to z-score new observations against the calibration sample)
    item_means = {item: float(data[item].mean()) for item in available}
    item_sds = {item: float(data[item].std()) for item in available}

    # Compute the correlation matrix
    if use_polychoric:
        print("  Using polychoric correlation matrix (MLE for ordinal data)")
        R = polychoric_corr_matrix(data, available)
    else:
        # Pearson on z-scored data (faster, acceptable approximation for
        # 11-point 0-10 scales which are approximately continuous)
        data_z = (data - data.mean()) / data.std()
        R = data_z.corr()

    off_diag = (R.values.sum() - len(available)) / (
        len(available) ** 2 - len(available)
    )
    print(f"  Mean off-diagonal correlation: {off_diag:.3f}")

    # ------------------------------------------------------------------
    # Iterative PAF: update communalities until convergence
    # ------------------------------------------------------------------
    n_factors = 2
    k = len(available)
    # Initialize communalities at 0.5 (standard starting point)
    communalities = np.ones(k) * 0.5

    for iteration in range(100):
        # Replace diagonal with current communality estimates
        # (reduced correlation matrix — models common variance only)
        R_reduced = R.values.copy()
        np.fill_diagonal(R_reduced, communalities)

        # Eigendecomposition of the reduced correlation matrix
        eigenvalues_all, eigenvectors_all = np.linalg.eigh(R_reduced)

        # Sort eigenvalues descending (eigh returns ascending order)
        idx_sort = np.argsort(eigenvalues_all)[::-1]
        eigenvalues_all = eigenvalues_all[idx_sort]
        eigenvectors_all = eigenvectors_all[:, idx_sort]

        # Extract the top 2 factors
        eigenvalues = eigenvalues_all[:n_factors]
        eigenvectors = eigenvectors_all[:, :n_factors]

        # Factor loadings = eigenvector * sqrt(eigenvalue)
        loadings = np.zeros((k, n_factors))
        for f in range(n_factors):
            loadings[:, f] = eigenvectors[:, f] * np.sqrt(max(eigenvalues[f], 0))

        # Update communalities (sum of squared loadings per item)
        new_communalities = np.sum(loadings ** 2, axis=1)
        change = np.max(np.abs(new_communalities - communalities))
        communalities = new_communalities

        if change < 1e-6:
            print(f"  Converged after {iteration + 1} iterations")
            break
    else:
        print(f"  WARNING: did not converge after 100 iterations")

    # ------------------------------------------------------------------
    # Ensure interpretable factor orientation
    # ------------------------------------------------------------------
    # Factor 1: all loadings positive (general pain severity)
    if np.sum(loadings[:, 0] < 0) > np.sum(loadings[:, 0] > 0):
        loadings[:, 0] *= -1

    # Factor 2: knee items positive, body items negative (contrast)
    # Knee items are indices 0-3, body items are indices 4-7
    if np.mean(loadings[:4, 1]) < np.mean(loadings[4:, 1]):
        loadings[:, 1] *= -1

    uniquenesses = 1 - communalities

    # Report factor loadings
    print(f"  Eigenvalues: {eigenvalues[0]:.3f}, {eigenvalues[1]:.3f}")
    print(
        f"  Variance explained: {eigenvalues[0] / k * 100:.1f}%, "
        f"{eigenvalues[1] / k * 100:.1f}%"
    )
    print(f"\n  {'Item':<20s} {'F1 (General)':>12s} {'F2 (Contrast)':>14s}")
    print(f"  {'-' * 50}")
    for i, item in enumerate(available):
        print(f"  {item:<20s} {loadings[i, 0]:>12.4f} {loadings[i, 1]:>14.4f}")

    # ------------------------------------------------------------------
    # Unreduced eigenvalues of the raw correlation matrix
    # ------------------------------------------------------------------
    # The iterative PAF loop above returns the eigenvalues of the
    # *reduced* correlation matrix (diagonal replaced by communalities),
    # which describe the common-variance structure after PAF extraction.
    # Horn's parallel analysis and the convention followed in the
    # manuscript Results (Table 1 + Figure S1) report the eigenvalues
    # of the *unreduced* correlation matrix (diagonal = 1). These
    # differ, so we compute the unreduced eigenvalues here as well and
    # return them separately for the parallel analysis step.
    eig_unreduced = np.linalg.eigvalsh(R.values)[::-1]

    return {
        "items": available,
        "loadings": loadings,
        "uniquenesses": uniquenesses,
        "communalities": communalities,
        "item_means": item_means,
        "item_sds": item_sds,
        "eigenvalues": eigenvalues,
        "eigenvalues_unreduced": eig_unreduced,
        "correlation_matrix": R.values,
        "n_obs_used": int(data.dropna().shape[0]),
    }


def parallel_analysis(n_obs, n_vars, n_replications=1000, seed=42):
    """Horn's (1965) parallel analysis for factor-retention decisions.

    Simulates ``n_replications`` uncorrelated random datasets of the
    same dimensions as the observed data (n_obs rows, n_vars columns),
    computes the Pearson correlation matrix and its unreduced
    eigenvalues for each, and returns the 95th-percentile and mean
    reference eigenvalue distributions. A factor is retained if its
    observed eigenvalue exceeds the 95th-percentile reference value
    for the corresponding component index.

    Parameters
    ----------
    n_obs : int
        Number of observations in the random datasets (should match
        the usable sample size the observed correlations were computed
        on).
    n_vars : int
        Number of variables (items). For this study, 8 pain items.
    n_replications : int, default 1000
        Number of random datasets to simulate. The manuscript reports
        1000 replications.
    seed : int, default 42
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        random_95th : ndarray shape (n_vars,)
            95th percentile of simulated eigenvalues per component.
        random_mean : ndarray shape (n_vars,)
            Mean of simulated eigenvalues per component.
        all_random : ndarray shape (n_replications, n_vars)
            Raw simulated eigenvalues (kept in memory for plotting).
        n_replications : int
        n_obs : int
        n_vars : int

    References
    ----------
    Horn, J. L. (1965). A rationale and test for the number of
    factors in factor analysis. Psychometrika, 30, 179-185.
    """
    rng = np.random.default_rng(seed)
    random_eigs = np.zeros((n_replications, n_vars))
    for rep in range(n_replications):
        random_data = rng.standard_normal(size=(n_obs, n_vars))
        R_random = np.corrcoef(random_data.T)
        random_eigs[rep] = np.linalg.eigvalsh(R_random)[::-1]

    return {
        "random_95th": np.percentile(random_eigs, 95, axis=0),
        "random_mean": np.mean(random_eigs, axis=0),
        "all_random": random_eigs,
        "n_replications": n_replications,
        "n_obs": n_obs,
        "n_vars": n_vars,
    }


def report_parallel_analysis(actual_eigs, pa_results):
    """Print the parallel-analysis comparison table.

    Mirrors the formatting used in the manuscript Methods §2.3
    description of factor retention.
    """
    n_vars = pa_results["n_vars"]
    n_obs = pa_results["n_obs"]
    n_rep = pa_results["n_replications"]
    r95 = pa_results["random_95th"]
    rmn = pa_results["random_mean"]

    print("\n" + "=" * 70)
    print(f"HORN'S PARALLEL ANALYSIS "
          f"(N={n_obs}, p={n_vars}, {n_rep} replications)")
    print("=" * 70)
    print(f"  {'Component':<12} {'Actual':>8} {'Random 95th':>12} "
          f"{'Random Mean':>12} {'Retain?':>8}")
    print("  " + "-" * 55)
    for i in range(n_vars):
        retain = "YES" if actual_eigs[i] > r95[i] else "no"
        print(f"  {i + 1:<10} {actual_eigs[i]:>8.4f} "
              f"{r95[i]:>12.4f} {rmn[i]:>12.4f} {retain:>8}")


def compute_bartlett_scores_2f(df, model):
    """Compute 2-factor Bartlett scores at each person-quarter.

    The Bartlett (1937) scoring method produces BLUE (best linear unbiased
    estimator) factor scores by solving:

      f = (L'_k Psi_k^{-1} L_k)^{-1} L_k' Psi_k^{-1} z_k

    where L_k is the loading matrix for the k available items, Psi_k is
    the diagonal uniqueness matrix, and z_k is the vector of standardized
    item responses. This naturally handles missing items by restricting to
    the available subset, and produces scores that are uncorrelated across
    factors (orthogonal).

    Requires at least 2 items per observation (the minimum to identify a
    2-factor model).

    Parameters
    ----------
    df : DataFrame
        Must contain the pain item columns.
    model : dict
        Output of calibrate_pain_factors().

    Returns
    -------
    pain_scores : ndarray
        Factor 1 (general pain) Bartlett scores.
    contrast_scores : ndarray
        Factor 2 (knee-vs-body contrast) Bartlett scores.
    n_items_used : ndarray
        Number of items available for each observation.
    """
    items = model["items"]
    loadings = model["loadings"]       # (k, 2) factor loading matrix
    uniquenesses = model["uniquenesses"]  # (k,) diagonal of Psi
    item_means = model["item_means"]
    item_sds = model["item_sds"]

    n = len(df)
    pain_scores = np.full(n, np.nan)
    contrast_scores = np.full(n, np.nan)
    n_items_used = np.full(n, 0)

    for i in range(n):
        # Identify which items are available for this observation
        available_mask = []
        z_vals = []
        for j, item in enumerate(items):
            val = df.iloc[i][item] if item in df.columns else np.nan
            if pd.notna(val):
                available_mask.append(j)
                # Standardize using calibration-sample mean and SD
                z_vals.append((val - item_means[item]) / item_sds[item])

        # Need at least 2 items for a 2-factor model to be identified
        if len(available_mask) < 2:
            continue

        # Subset the loading matrix and uniqueness vector to available items
        L_k = loadings[available_mask, :]  # (k_avail, 2)
        z_k = np.array(z_vals)             # (k_avail,)

        # Bartlett scoring formula:
        #   f = (L' Psi^{-1} L)^{-1} L' Psi^{-1} z
        psi_inv = np.diag(1.0 / uniquenesses[available_mask])
        precision = L_k.T @ psi_inv @ L_k  # (2, 2) information matrix

        try:
            precision_inv = np.linalg.inv(precision)
        except np.linalg.LinAlgError:
            continue  # singular matrix (degenerate item subset)

        scores = precision_inv @ L_k.T @ psi_inv @ z_k  # (2,)

        pain_scores[i] = scores[0]
        contrast_scores[i] = scores[1]
        n_items_used[i] = len(available_mask)

    return pain_scores, contrast_scores, n_items_used


# ===================================================================
# Construct Scoring
# ===================================================================

def score_constructs(df, pain_model):
    """Compute pain factor scores, contrast factor scores, and z-scored sleep.

    Parameters
    ----------
    df : DataFrame
        Must contain pain items and the sleep item.
    pain_model : dict
        Output of calibrate_pain_factors().

    Returns
    -------
    df : DataFrame
        With added columns: pain_factor, contrast_factor, sleep_factor.
    sleep_params : dict
        Grand mean and SD of sleep (for reference/back-transformation).
    """
    print("\n  Scoring constructs (2-factor Bartlett + z-scored sleep)...")

    # Compute 2-factor Bartlett scores
    pain_scores, contrast_scores, n_items = compute_bartlett_scores_2f(
        df, pain_model
    )

    df = df.copy()
    df["pain_factor"] = pain_scores
    df["contrast_factor"] = contrast_scores
    df["pain_n_items"] = n_items

    valid_pain = np.isfinite(pain_scores)
    valid_contrast = np.isfinite(contrast_scores)
    print(f"    Pain factor: {valid_pain.sum()} scored ({valid_pain.mean():.1%})")
    print(f"    Contrast factor: {valid_contrast.sum()} scored")

    # Orthogonality check (should be |r| < 0.05 for Bartlett scores)
    both = valid_pain & valid_contrast
    if both.sum() > 10:
        r = np.corrcoef(pain_scores[both], contrast_scores[both])[0, 1]
        print(f"    Orthogonality: r(F1, F2) = {r:.4f}")

    # Sleep: z-score q13 using the grand mean and SD
    sleep_raw = df[SLEEP_ITEM].values.astype(float)
    sleep_mean = float(np.nanmean(sleep_raw))
    sleep_sd = float(np.nanstd(sleep_raw))
    df["sleep_factor"] = (sleep_raw - sleep_mean) / sleep_sd

    valid_sleep = df["sleep_factor"].notna()
    print(f"    Sleep: {valid_sleep.sum()} scored (raw mean={sleep_mean:.2f})")

    return df, {"sleep_mean": sleep_mean, "sleep_sd": sleep_sd}


# ===================================================================
# Segment Filter
# ===================================================================

def segment_filter(df):
    """Retain only segments of >= MIN_SEGMENT consecutive quarters.

    A "segment" is a maximal run of consecutive quarters where both
    pain_factor and sleep_factor are non-missing for a given person.
    Short segments (< MIN_SEGMENT quarters) are dropped because they
    cannot contribute enough lagged observations to the VARX model.

    For MIN_SEGMENT = 3: a segment of length 3 yields 2 usable transitions
    (t-1 -> t), which is the minimum for the model to separate the AR
    effect from the cross-lag.

    Parameters
    ----------
    df : DataFrame
        Must contain ID, quarter, pain_factor, sleep_factor.

    Returns
    -------
    df : DataFrame
        Filtered to retained segments, with segment_id column added.
    """
    print(f"\n  Segment filter (>= {MIN_SEGMENT} consecutive quarters)...")

    df = df.copy()
    df["has_both"] = df["pain_factor"].notna() & df["sleep_factor"].notna()
    df = df.sort_values(["ID", "quarter"])

    n_before = df["has_both"].sum()
    n_persons_before = df.loc[df["has_both"], "ID"].nunique()

    df["segment_id"] = np.nan
    rows_to_keep = []

    for pid, grp in df.groupby("ID"):
        # Find quarters where both constructs are present
        quarters_with_data = sorted(
            grp.loc[grp["has_both"], "quarter"].values
        )
        if len(quarters_with_data) == 0:
            continue

        # Identify contiguous runs of consecutive quarters
        segments = []
        current = [quarters_with_data[0]]
        for q in quarters_with_data[1:]:
            if q == current[-1] + 1:
                # Consecutive: extend the current segment
                current.append(q)
            else:
                # Gap: close the current segment and start a new one
                segments.append(current)
                current = [q]
        segments.append(current)

        # Retain only segments meeting the minimum length requirement
        seg_id = 0
        for seg in segments:
            if len(seg) >= MIN_SEGMENT:
                seg_id += 1
                idx = grp.index[grp["quarter"].isin(seg) & grp["has_both"]]
                df.loc[idx, "segment_id"] = seg_id
                rows_to_keep.extend(idx.tolist())

    df = df.loc[rows_to_keep].copy()
    df.drop(columns="has_both", inplace=True)

    n_after = len(df)
    n_persons_after = df["ID"].nunique()
    print(
        f"    Before: {n_persons_before} subjects, {n_before} observations"
    )
    print(
        f"    After:  {n_persons_after} subjects, {n_after} observations"
    )
    print(
        f"    Dropped: {n_persons_before - n_persons_after} subjects, "
        f"{n_before - n_after} observations"
    )

    return df


# ===================================================================
# Within-Between Decomposition (Curran & Bauer, 2011)
# ===================================================================

def decompose_within_between(df):
    """Decompose each variable into between- and within-person components.

    For each variable Y:
      Y_bar_i = mean of Y across all time points for person i
      Y^w_it  = Y_it - Y_bar_i  (within-person deviation from own mean)

    This decomposition is the foundation of the within-person modeling
    approach (Curran & Bauer, 2011). By operating on the deviations Y^w,
    the VARX model automatically removes all stable between-person
    confounders. Any variable that is constant within a person (e.g.,
    baseline BMI, chronic pain level, genotype) is absorbed into Y_bar_i
    and does not affect the within-person dynamics.

    This eliminates the need for random intercepts and between-person
    covariates in the model, substantially simplifying the Bayesian
    specification and improving MCMC convergence.

    Parameters
    ----------
    df : DataFrame
        Must contain ID, pain_factor, contrast_factor, sleep_factor.

    Returns
    -------
    df : DataFrame
        With added columns: {pain,contrast,sleep}_person_mean,
        {pain,contrast,sleep}_between, {pain,contrast,sleep}_within.
    """
    print("\n  Within-between decomposition (Curran & Bauer, 2011)...")

    df = df.copy()

    # Map from raw column names to short labels for new columns
    var_map = {
        "pain_factor": "pain",
        "contrast_factor": "contrast",
        "sleep_factor": "sleep",
    }

    for raw_var, label in var_map.items():
        # Compute person-level means (Y_bar_i)
        person_means = df.groupby("ID")[raw_var].mean().reset_index()
        person_means.columns = ["ID", f"{label}_person_mean"]

        # Grand mean (for the between-person component)
        grand_mean = person_means[f"{label}_person_mean"].mean()

        # Between-person deviation: Y_bar_i - grand_mean
        person_means[f"{label}_between"] = (
            person_means[f"{label}_person_mean"] - grand_mean
        )

        # Merge person means back and compute within-person deviation
        df = df.merge(
            person_means[["ID", f"{label}_person_mean", f"{label}_between"]],
            on="ID",
            how="left",
        )
        # Within-person deviation: Y_it - Y_bar_i
        df[f"{label}_within"] = df[raw_var] - df[f"{label}_person_mean"]

        # Report decomposition statistics
        between_var = person_means[f"{label}_between"].var()
        within_var = df[f"{label}_within"].var()
        icc = (
            between_var / (between_var + within_var)
            if (between_var + within_var) > 0
            else np.nan
        )
        print(
            f"    {label}: ICC = {icc:.3f}  "
            f"(between SD = {np.sqrt(between_var):.3f}, "
            f"within SD = {np.sqrt(within_var):.3f})"
        )

    return df


# ===================================================================
# Lag Creation and Interaction Terms
# ===================================================================

def create_lagged_variables(df):
    """Create t-1 lagged variables and interaction terms within segments.

    Lags are computed within (ID, segment_id) groups, so segment boundaries
    are respected. An additional safety check NaN's out any lags where the
    quarters are not strictly consecutive (q_t - q_{t-1} != 1), which would
    indicate a data integrity issue.

    Interaction terms for the contrast moderation in the VARX:
      sleep_x_contrast_lag1 = sleep_within_lag1 * contrast_within_lag1
        -> enters the pain equation as omega_sp * (S^w_{t-1} * K^w_{t-1})
      pain_x_contrast_lag1 = pain_within_lag1 * contrast_within_lag1
        -> enters the sleep equation as omega_ps * (P^w_{t-1} * K^w_{t-1})

    Parameters
    ----------
    df : DataFrame
        Must contain ID, segment_id, quarter, and all within-person columns.

    Returns
    -------
    df : DataFrame
        With added lag and interaction columns.
    """
    print("\n  Creating lagged variables and interaction terms...")

    df = df.sort_values(["ID", "segment_id", "quarter"])

    grp_key = ["ID", "segment_id"]

    # Lag-1 of each within-person variable
    df["pain_within_lag1"] = df.groupby(grp_key)["pain_within"].shift(1)
    df["sleep_within_lag1"] = df.groupby(grp_key)["sleep_within"].shift(1)
    df["contrast_within_lag1"] = df.groupby(grp_key)["contrast_within"].shift(1)

    # Safety check: NaN out lags where quarters are not consecutive.
    # This should not happen if the segment filter is correct, but
    # protects against data issues.
    quarter_lag1 = df.groupby(grp_key)["quarter"].shift(1)
    non_consecutive = (df["quarter"] - quarter_lag1) != 1
    n_invalidated = (non_consecutive & ~df.groupby(grp_key).cumcount().eq(0)).sum()
    if n_invalidated > 0:
        print(f"    WARNING: invalidated {n_invalidated} non-consecutive lags")
    df.loc[non_consecutive, "pain_within_lag1"] = np.nan
    df.loc[non_consecutive, "sleep_within_lag1"] = np.nan
    df.loc[non_consecutive, "contrast_within_lag1"] = np.nan

    # Interaction terms: coupling_lag * contrast_lag
    # These capture whether the pain localization pattern (contrast)
    # moderates the cross-lagged coupling strength
    df["sleep_x_contrast_lag1"] = (
        df["sleep_within_lag1"] * df["contrast_within_lag1"]
    )
    df["pain_x_contrast_lag1"] = (
        df["pain_within_lag1"] * df["contrast_within_lag1"]
    )

    # Report usable observations for the full VARX model
    full_cols = [
        "pain_within", "sleep_within",
        "pain_within_lag1", "sleep_within_lag1",
        "contrast_within_lag1",
        "sleep_x_contrast_lag1", "pain_x_contrast_lag1",
    ]
    full_usable = df[full_cols].notna().all(axis=1).sum()
    n_persons = df.loc[df[full_cols].notna().all(axis=1), "ID"].nunique()
    print(f"    Full model usable: {full_usable} obs from {n_persons} subjects")

    return df


# ===================================================================
# Data Loading (Real vs. Synthetic)
# ===================================================================

def load_real_data(data_dir=None):
    """Load real quarterly data with gateway imputation.

    Gateway imputation: When the gateway question indicates no pain at a
    body site (q1=0 for knee, q6=0 for body), downstream intensity items
    that are missing are set to 0 (the respondent skipped them because
    they had no pain to rate). This is structurally missing data with a
    known value, not imputation of unknown values.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing ``quarterly_data_long.csv``. Defaults to
        :data:`DATA_DIR`. Baseline demographics (Age, Sex) are read
        from the quarterly CSV itself — this step does not read
        ``participants_wideformat.xlsx``. That file is only needed
        downstream by ``06_generate_figures.py`` for Figure S2.

    Returns
    -------
    df_raw : DataFrame
        Raw quarterly data (pre-gateway, no imputation). Used by the
        factor analysis so the correlation matrix is computed only on
        person-quarters where all 8 items were independently answered.
    df_gated : DataFrame
        Gateway-imputed quarterly data, restricted to Q1-Q11 and
        joined with Age/Sex. Used by all downstream stages (Bartlett
        scoring, within-person decomposition, VARX fitting).
    """
    if data_dir is None:
        data_dir = DATA_DIR
    print("  Loading real quarterly data...")

    quarterly_raw = pd.read_csv(os.path.join(data_dir, "quarterly_data_long.csv"))

    # --------------------------------------------------------------
    # Keep an untouched copy for the factor analysis. The factor
    # analysis uses only rows where all 8 items were independently
    # answered -- gateway-imputed 0s would artificially inflate the
    # within-region (knee-knee, body-body) correlations and bias the
    # second (contrast) eigenvalue upward.
    # --------------------------------------------------------------
    df_raw = quarterly_raw.copy()

    # --------------------------------------------------------------
    # Gateway imputation for downstream stages (Bartlett scoring,
    # decomposition, VARX). q1=0 (no knee pain) -> missing q2/q3/q4
    # are structurally 0; q6=0 (no body pain) -> missing q7/q8/q9
    # are structurally 0.
    # --------------------------------------------------------------
    quarterly = quarterly_raw.copy()
    knee_gate = quarterly["q1_knee_pain"] == 0
    for item in ["q2_knee_pain", "q3_knee_pain", "q4_knee_pain"]:
        quarterly.loc[knee_gate & quarterly[item].isna(), item] = 0.0

    body_gate = quarterly["q6_body_pain"] == 0
    for item in ["q7_body_pain", "q8_body_pain", "q9_body_pain"]:
        quarterly.loc[body_gate & quarterly[item].isna(), item] = 0.0

    # Extract demographics from baseline (quarter 0)
    baseline_info = quarterly[quarterly["quarter"] == 0][
        ["ID", "age", "gender"]
    ].drop_duplicates()
    # Backfill missing baseline demographics from later quarters
    all_ids = quarterly["ID"].unique()
    missing_ids = set(all_ids) - set(baseline_info["ID"])
    for mid in missing_ids:
        pdata = (
            quarterly[quarterly["ID"] == mid][["ID", "age", "gender"]]
            .dropna()
            .head(1)
        )
        if len(pdata) > 0:
            baseline_info = pd.concat([baseline_info, pdata], ignore_index=True)
    baseline_info = baseline_info.drop_duplicates(subset="ID")
    baseline_info.columns = ["ID", "Age", "Sex"]

    quarterly = quarterly.merge(baseline_info, on="ID", how="left")

    # Restrict to Q1-Q11 (Q0 is the baseline visit with no quarterly items)
    quarterly = quarterly[quarterly["quarter"] >= 1].copy()

    print(
        f"    {quarterly['ID'].nunique()} subjects, "
        f"{len(quarterly)} observations "
        f"(raw: {len(df_raw)})"
    )
    return df_raw, quarterly


def load_synthetic_data(data_dir=None):
    """Load synthetic quarterly data (factor scores already computed).

    The synthetic dataset has columns: subject_id, quarter, pain_severity,
    contrast_factor, sleep_quality. These correspond to the pain_factor,
    contrast_factor, and sleep_factor that would be produced by the factor
    analysis on real data.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing ``synthetic/quarterly_data_long.csv`` and
        ``synthetic/participants_wideformat.csv``. If this points directly
        at a folder containing those CSVs (with or without the
        ``synthetic/`` suffix), both conventions are tried. Defaults to
        :data:`DATA_DIR`.

    Returns
    -------
    df : DataFrame
        With columns renamed to match the real data pipeline.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    print("  Loading synthetic data (factor analysis pre-computed)...")

    # Accept either `data/` (with a synthetic/ subfolder) or a directory
    # that directly contains the synthetic CSVs.
    candidate = os.path.join(data_dir, "synthetic")
    synth_dir = candidate if os.path.isdir(candidate) else data_dir

    # Load quarterly data
    quarterly = pd.read_csv(os.path.join(synth_dir, "quarterly_data_long.csv"))
    # Load demographics
    participants = pd.read_csv(os.path.join(synth_dir, "participants_wideformat.csv"))

    # Rename columns to match the real data pipeline convention
    quarterly = quarterly.rename(columns={
        "subject_id": "ID",
        "pain_severity": "pain_factor",     # already a factor score
        "sleep_quality": "sleep_factor",    # already z-scored
        # contrast_factor keeps its name
    })

    participants = participants.rename(columns={
        "subject_id": "ID",
        "age": "Age",
        "sex": "Sex",
    })

    # Merge demographics into quarterly data
    quarterly = quarterly.merge(participants[["ID", "Age", "Sex"]], on="ID", how="left")

    print(f"    {quarterly['ID'].nunique()} subjects, {len(quarterly)} observations")
    return quarterly


# ===================================================================
# Main Pipeline
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare quarterly data for VARX(1) coupling analysis."
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (skips factor analysis, loads pre-computed scores)"
    )
    parser.add_argument(
        "--polychoric", action="store_true",
        help="Use polychoric correlations for PAF (real data only, slower but "
             "more rigorous for ordinal scales)"
    )
    parser.add_argument(
        "--interpolate", action="store_true",
        help="Linearly interpolate single interior gaps in factor scores"
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override input data directory (default: data/). The raw "
             "quarterly_data_long.csv is read from here for real data, "
             "and synthetic/*.csv for --synthetic."
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory for the processed CSV and factor-"
             "model JSON. Default: data/ for real, data/synthetic/ for "
             "--synthetic. Downstream steps must be pointed at this same "
             "directory via their own --data-dir flag."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("DATA PREPARATION: Bidirectional Sleep-Pain Coupling VARX(1)")
    print("=" * 70)
    if args.synthetic:
        print("  Mode: SYNTHETIC (factor analysis skipped)")
    else:
        print("  Mode: REAL DATA (full factor analysis pipeline)")

    # Resolve input data directory. For steps 1-5 this is where the
    # raw quarterly items and (for step 6's Figure S2 only) the
    # participants_wideformat file live.
    data_dir = args.data_dir if args.data_dir else DATA_DIR

    # ------------------------------------------------------------------
    # Stage 1: Load data and compute factor scores
    # ------------------------------------------------------------------
    if args.synthetic:
        # Synthetic data already has factor scores — skip factor analysis
        df = load_synthetic_data(data_dir)
    else:
        # Real data: load raw items and run full factor analysis.
        # load_real_data returns two frames: the raw (pre-gateway)
        # quarterly items for the factor analysis + parallel analysis,
        # and the gateway-imputed frame for all downstream stages.
        df_raw, df = load_real_data(data_dir)

        # Calibrate the 2-factor PAF model on the 8 pain items using
        # the RAW (pre-gateway) items. Gateway imputation replaces
        # missing q2/q3/q4 with 0 when q1=0 and similarly for body
        # items, which artificially correlates the within-region
        # items and biases the contrast eigenvalue upward. See
        # Methods §2.3.
        pain_model = calibrate_pain_factors(df_raw, use_polychoric=args.polychoric)

        # Horn's parallel analysis for factor retention (Methods §2.3).
        # Uses the unreduced eigenvalues of the same correlation matrix
        # fit by calibrate_pain_factors (polychoric if --polychoric was
        # passed, Pearson otherwise) and compares them to the 95th
        # percentile of eigenvalues from 1000 random datasets of the
        # same dimensions. The retention decision is reported in Table
        # 1 and Figure S1 of the manuscript.
        pa_results = parallel_analysis(
            n_obs=pain_model["n_obs_used"],
            n_vars=len(pain_model["items"]),
            n_replications=1000,
            seed=42,
        )
        report_parallel_analysis(
            pain_model["eigenvalues_unreduced"], pa_results
        )

        # Compute Bartlett scores and z-scored sleep
        df, sleep_params = score_constructs(df, pain_model)

    # ------------------------------------------------------------------
    # Stage 2: Optional interpolation (real data only)
    # ------------------------------------------------------------------
    if args.interpolate and not args.synthetic:
        print("\n  Interpolating single interior gaps...")
        all_quarters = range(1, 12)
        rows = []
        for pid, grp in df.groupby("ID"):
            full = pd.DataFrame({"quarter": list(all_quarters)})
            full = full.merge(grp.drop(columns="ID"), on="quarter", how="left")
            for col in ["Age", "Sex"]:
                if col in full.columns:
                    val = (
                        grp[col].dropna().iloc[0] if grp[col].notna().any()
                        else np.nan
                    )
                    full[col] = val
            for var in ["pain_factor", "contrast_factor", "sleep_factor"]:
                full[var] = full[var].interpolate(
                    method="linear", limit_area="inside", limit=1
                )
            full["ID"] = pid
            rows.append(full)
        df = pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------------
    # Stage 3: Segment filter (>= 3 consecutive quarters with both)
    # ------------------------------------------------------------------
    df = segment_filter(df)

    # ------------------------------------------------------------------
    # Stage 4: Within-between decomposition
    # ------------------------------------------------------------------
    df = decompose_within_between(df)

    # ------------------------------------------------------------------
    # Stage 5: Create lagged variables
    # ------------------------------------------------------------------
    df = create_lagged_variables(df)

    # ------------------------------------------------------------------
    # Stage 6: Select output columns and save
    # ------------------------------------------------------------------
    output_cols = [
        "ID", "quarter", "segment_id",
        "pain_factor", "contrast_factor", "sleep_factor",
        "pain_person_mean", "pain_between", "pain_within",
        "contrast_person_mean", "contrast_between", "contrast_within",
        "sleep_person_mean", "sleep_between", "sleep_within",
        "pain_within_lag1", "sleep_within_lag1", "contrast_within_lag1",
        "sleep_x_contrast_lag1", "pain_x_contrast_lag1",
        "Age", "Sex",
    ]
    # Only include columns that actually exist (treatment cols may be absent
    # in synthetic data)
    output_cols = [c for c in output_cols if c in df.columns]

    # Resolve output directory. Semantics:
    #   - --output-dir wins if set (write directly into that folder)
    #   - else synthetic writes to data/synthetic/
    #   - else real writes to data/
    if args.output_dir:
        out_dir = args.output_dir
        out_path = os.path.join(out_dir, "processed_data.csv" if args.synthetic
                                else "processed_data_contrast.csv")
    elif args.synthetic:
        out_path = os.path.join(DATA_DIR, "synthetic", "processed_data.csv")
    else:
        out_path = os.path.join(DATA_DIR, "processed_data_contrast.csv")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df[output_cols].to_csv(out_path, index=False)
    print(f"\n  Output saved: {out_path}")
    print(f"  Columns: {output_cols}")
    print(f"  Shape: {df[output_cols].shape}")

    # Save factor model parameters (real data only)
    if not args.synthetic:
        # Unreduced eigenvalues of the 8-item correlation matrix --
        # these are the values reported in the manuscript Results
        # (Factor 1 ~ 5.56, Factor 2 ~ 0.89) and compared to the
        # parallel analysis reference distribution. The reduced PAF
        # eigenvalues (from iterative communality updates) are also
        # stored for completeness as ``pain_eigenvalue_reduced_*``.
        eig_unreduced = pain_model["eigenvalues_unreduced"]
        k = len(pain_model["items"])
        params = {
            "correlation_type": "polychoric" if args.polychoric else "pearson",
            "pain_items": pain_model["items"],
            "pain_loadings_f1": pain_model["loadings"][:, 0].tolist(),
            "pain_loadings_f2": pain_model["loadings"][:, 1].tolist(),
            "pain_uniquenesses": pain_model["uniquenesses"].tolist(),
            "pain_communalities": pain_model["communalities"].tolist(),
            "pain_item_means": pain_model["item_means"],
            "pain_item_sds": pain_model["item_sds"],
            # Unreduced eigenvalues (what the manuscript reports)
            "pain_eigenvalues_unreduced": eig_unreduced.tolist(),
            "pain_eigenvalue_f1": float(eig_unreduced[0]),
            "pain_eigenvalue_f2": float(eig_unreduced[1]),
            "pain_variance_explained_f1_pct":
                float(eig_unreduced[0]) / k * 100.0,
            "pain_variance_explained_f2_pct":
                float(eig_unreduced[1]) / k * 100.0,
            # Reduced (iterative PAF) eigenvalues, for completeness
            "pain_eigenvalue_reduced_f1": float(pain_model["eigenvalues"][0]),
            "pain_eigenvalue_reduced_f2": float(pain_model["eigenvalues"][1]),
            # Horn's parallel analysis reference distribution
            "parallel_analysis": {
                "n_obs": pa_results["n_obs"],
                "n_vars": pa_results["n_vars"],
                "n_replications": pa_results["n_replications"],
                "random_95th_percentile": pa_results["random_95th"].tolist(),
                "random_mean": pa_results["random_mean"].tolist(),
                "retain_factors": int(
                    (eig_unreduced > pa_results["random_95th"]).sum()
                ),
            },
            "sleep_mean": sleep_params["sleep_mean"],
            "sleep_sd": sleep_params["sleep_sd"],
        }
        params_dir = args.output_dir if args.output_dir else DATA_DIR
        params_file = os.path.join(params_dir, "factor_model_params_contrast.json")
        os.makedirs(params_dir, exist_ok=True)
        with open(params_file, "w") as f:
            json.dump(params, f, indent=2)
        print(f"  Factor model parameters saved: {params_file}")

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
