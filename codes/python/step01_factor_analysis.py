"""
Step 01 — Factor analysis and interpolation on the extracted long data.
======================================================================

Input:  new_organization/data/extracted_long.csv
Output: new_organization/data/scored_long.csv
        new_organization/data/factor_model.json

This step runs the 2-factor principal-axis factor analysis on the
eight pain items (q2-q5 knee, q7-q10 body) using a polychoric
correlation matrix, scores every person-quarter via Bartlett
factor scoring, z-scores the q13 sleep item, and then linearly
interpolates single interior gaps in the three factor scores. It
is the direct successor to Step 00 (data extraction) and the
predecessor to Step 02 (segment filtering + within-between
decomposition + lag creation + Figure 1).

The code in this file is a direct port of the factor-analysis and
interpolation stages of the legacy
``scripts/prepare_data_contrast.py`` pipeline that produced the
earlier draft of the paper. The only analytical change from that
pipeline is the gating convention: Step 00 in
``new_organization/`` applies 4+4 gating (q2/q3/q4/q5 for knee,
q7/q8/q9/q10 for body) instead of the legacy 3+3 gating. The
factor model and Bartlett scoring operate on the same eight
items in both cases.

Functions in this file (modular, one responsibility each):

  polychoric_corr_pair / polychoric_corr_matrix
      — build the polychoric correlation matrix of the 8 pain items
  calibrate_factor_model
      — iterative PAF extraction, returns loadings, eigenvalues,
        item statistics, and the correlation matrix
  run_parallel_analysis / report_parallel_analysis
      — Horn's parallel analysis for factor retention
  bartlett_scores_2f
      — per-row Bartlett factor scoring on partial item sets
  score_all_rows
      — top-level factor + sleep scoring for the long frame
  interpolate_factor_scores
      — linear interpolation of single interior gaps in the three
        factor scores, within subjects

The script is runnable directly::

    python new_organization/codes/step01_factor_analysis.py

It reads Step 00's output and writes the two output files.

Author: Pedro Valdes-Hernandez (with Claude Opus 4.6)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =====================================================================
# Paths
# =====================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repo root
DATA_DIR = os.path.join(ROOT, "data")

DERIV_DIR = os.path.join(ROOT, "derivatives")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step01_factor_analysis")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step01_factor_analysis")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

IN_LONG_CSV = os.path.join(DATA_DIR, "step00_extracted_long.csv")
OUT_SCORED_CSV = os.path.join(STEP_DERIV_DIR, "step01_scored_long.csv")
OUT_MODEL_JSON = os.path.join(STEP_DERIV_DIR, "step01_factor_model.json")


# =====================================================================
# Constants
# =====================================================================

# The item lists live in lib/descriptives.py, which is their one home: step 02
# and step 05 describe the same items and must not be able to disagree with the
# measurement model about which eight items "the pain items" are.
from descriptives import (  # noqa: E402
    BODY_ITEMS, KNEE_ITEMS, PAIN_ITEMS, SLEEP_ITEM,
)

# Deterministic bivariate normal CDF. See its docstring for why the SciPy one is
# unusable inside a likelihood that an optimizer descends.
from measurement import bvn_cdf  # noqa: E402

N_FACTORS = 2
QUARTERS = list(range(1, 12))  # 1..11


# =====================================================================
# Polychoric correlation
# =====================================================================

def polychoric_corr_pair(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate the polychoric correlation between two ordinal variables.

    Polychoric correlation is the MLE estimate of the latent bivariate
    normal correlation that would produce the observed contingency
    table, under the threshold model in which each ordinal variable
    is a discretized version of a latent continuous normal variate
    (Olsson, 1979). Appropriate for 0-10 ordinal scales.

    Pairwise NaN deletion is applied before estimation.
    """
    from scipy import stats, optimize

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask].astype(int), y[mask].astype(int)

    cats_x = sorted(np.unique(x))
    cats_y = sorted(np.unique(y))

    # Thresholds from marginal cumulative proportions
    cum_x = np.array([np.mean(x <= c) for c in cats_x[:-1]])
    cum_y = np.array([np.mean(y <= c) for c in cats_y[:-1]])
    thresh_x = stats.norm.ppf(cum_x)
    thresh_y = stats.norm.ppf(cum_y)

    tx = np.concatenate([[-np.inf], thresh_x, [np.inf]])
    ty = np.concatenate([[-np.inf], thresh_y, [np.inf]])

    cat_map_x = {c: i for i, c in enumerate(cats_x)}
    cat_map_y = {c: i for i, c in enumerate(cats_y)}
    table = np.zeros((len(cats_x), len(cats_y)))
    for xi, yi in zip(x, y):
        table[cat_map_x[xi], cat_map_y[yi]] += 1

    def neg_loglik(rho_arr):
        rho = rho_arr[0]
        if abs(rho) >= 0.999:
            return 1e10
        ll = 0.0
        for i in range(len(cats_x)):
            for j in range(len(cats_y)):
                if table[i, j] == 0:
                    continue
                # bvn_cdf, not stats.multivariate_normal.cdf: the latter integrates by
                # randomized QMC, which makes THIS log-likelihood noisy and the
                # optimizer below land somewhere slightly different on every run.
                p = (
                    bvn_cdf(tx[i + 1], ty[j + 1], rho)
                    - bvn_cdf(tx[i + 1], ty[j], rho)
                    - bvn_cdf(tx[i], ty[j + 1], rho)
                    + bvn_cdf(tx[i], ty[j], rho)
                )
                ll += table[i, j] * np.log(max(p, 1e-15))
        return -ll

    r_init = np.corrcoef(x.astype(float), y.astype(float))[0, 1]
    result = optimize.minimize(
        neg_loglik, [r_init], method="L-BFGS-B", bounds=[(-0.999, 0.999)]
    )
    return float(result.x[0])


def polychoric_corr_matrix(
    data: pd.DataFrame, items: List[str],
) -> pd.DataFrame:
    """Compute the full polychoric correlation matrix for ordinal items."""
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


# =====================================================================
# 2-factor PAF model calibration
# =====================================================================

def calibrate_factor_model(
    quarterly_items: pd.DataFrame, use_polychoric: bool = True,
    verbose: bool = True,
) -> Dict:
    """Iterative principal axis factoring with 2 factors.

    Parameters
    ----------
    quarterly_items : DataFrame
        Must contain the 8 pain-item columns (missing rows allowed).
    use_polychoric : bool, default True
        If True, compute the polychoric correlation matrix. If False,
        use Pearson on z-scored data (faster, but the published
        analysis used polychoric).

    Returns
    -------
    model : dict
        Keys:
          ``items``                 list of 8 item names (same order
                                    as PAIN_ITEMS)
          ``loadings``              (8, 2) factor loading matrix
          ``uniquenesses``          (8,) 1 - communality per item
          ``communalities``         (8,) final communality estimates
          ``item_means``            dict {item: mean}
          ``item_sds``              dict {item: SD}
          ``eigenvalues``           (2,) reduced-matrix PAF eigenvalues
                                    for the retained factors
          ``eigenvalues_unreduced`` (8,) all eight eigenvalues of the
                                    unreduced correlation matrix
                                    (diagonal = 1). Horn's parallel
                                    analysis operates on these.
          ``correlation_matrix``    (8, 8) numpy array
          ``n_obs_used``            number of fully non-missing rows
                                    across the 8 items
    """
    if verbose:
        print("\n" + "=" * 70)
    if verbose:
        print("2-FACTOR PAF CALIBRATION (8 pain items)")
    if verbose:
        print("=" * 70)

    available = [c for c in PAIN_ITEMS if c in quarterly_items.columns]
    if len(available) < len(PAIN_ITEMS):
        missing = set(PAIN_ITEMS) - set(available)
        raise KeyError(
            f"Missing pain items in input: {sorted(missing)}"
        )

    data = quarterly_items[available].copy()
    item_means = {c: float(data[c].mean()) for c in available}
    item_sds = {c: float(data[c].std()) for c in available}

    if use_polychoric:
        if verbose:
            print("  Using polychoric correlation matrix (MLE for ordinal data)")
        R = polychoric_corr_matrix(data, available)
    else:
        if verbose:
            print("  Using Pearson correlation matrix (z-scored)")
        data_z = (data - data.mean()) / data.std()
        R = data_z.corr()

    off_diag = (R.values.sum() - len(available)) / (
        len(available) ** 2 - len(available)
    )
    if verbose:
        print(f"  Mean off-diagonal correlation: {off_diag:.3f}")

    # Iterative PAF: update communalities until convergence
    k = len(available)
    communalities = np.ones(k) * 0.5

    for iteration in range(100):
        R_reduced = R.values.copy()
        np.fill_diagonal(R_reduced, communalities)

        eigenvalues_all, eigenvectors_all = np.linalg.eigh(R_reduced)
        idx_sort = np.argsort(eigenvalues_all)[::-1]
        eigenvalues_all = eigenvalues_all[idx_sort]
        eigenvectors_all = eigenvectors_all[:, idx_sort]

        eigenvalues = eigenvalues_all[:N_FACTORS]
        eigenvectors = eigenvectors_all[:, :N_FACTORS]

        loadings = np.zeros((k, N_FACTORS))
        for f in range(N_FACTORS):
            loadings[:, f] = eigenvectors[:, f] * np.sqrt(
                max(eigenvalues[f], 0)
            )

        new_communalities = np.sum(loadings ** 2, axis=1)
        change = np.max(np.abs(new_communalities - communalities))
        communalities = new_communalities

        if change < 1e-6:
            if verbose:
                print(f"  Converged after {iteration + 1} iterations")
            break
    else:
        if verbose:
            print("  WARNING: PAF did not converge after 100 iterations")

    # Factor orientation: F1 all positive (general pain severity),
    # F2 knee-positive / body-negative (contrast).
    if np.sum(loadings[:, 0] < 0) > np.sum(loadings[:, 0] > 0):
        loadings[:, 0] *= -1
    if np.mean(loadings[:4, 1]) < np.mean(loadings[4:, 1]):
        loadings[:, 1] *= -1

    uniquenesses = 1 - communalities

    # Unreduced eigenvalues (for parallel analysis reporting)
    eig_unreduced = np.linalg.eigvalsh(R.values)[::-1]

    if verbose:
        print(
        f"  Reduced eigenvalues (PAF): "
        f"{eigenvalues[0]:.3f}, {eigenvalues[1]:.3f}"
    )
    if verbose:
        print(
        f"  Unreduced eigenvalues (for PA): "
        f"{eig_unreduced[0]:.3f}, {eig_unreduced[1]:.3f}"
    )
    if verbose:
        print(
        f"  Variance explained (unreduced): "
        f"{eig_unreduced[0] / k * 100:.1f}%, "
        f"{eig_unreduced[1] / k * 100:.1f}%"
    )
    if verbose:
        print(f"\n  {'Item':<20s} {'F1 (Severity)':>14s} {'F2 (Contrast)':>14s}")
    if verbose:
        print(f"  {'-' * 52}")
    for i, item in enumerate(available):
        if verbose:
            print(
            f"  {item:<20s} "
            f"{loadings[i, 0]:>14.4f} {loadings[i, 1]:>14.4f}"
        )

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


# =====================================================================
# Horn's parallel analysis
# =====================================================================

def run_parallel_analysis(
    n_obs: int, n_vars: int, n_replications: int = 1000, seed: int = 42,
) -> Dict:
    """Horn's (1965) parallel analysis for factor-retention decisions.

    Simulates ``n_replications`` datasets of shape ``(n_obs, n_vars)``
    of uncorrelated standard-normal noise, computes each one's
    Pearson correlation matrix, and returns the 95th percentile and
    mean of the eigenvalue distribution per component.
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
        "n_replications": n_replications,
        "n_obs": n_obs,
        "n_vars": n_vars,
    }


def report_parallel_analysis(
    actual_eigs: np.ndarray, pa_results: Dict,
) -> None:
    """Print the parallel-analysis comparison table."""
    n_vars = pa_results["n_vars"]
    n_obs = pa_results["n_obs"]
    n_rep = pa_results["n_replications"]
    r95 = pa_results["random_95th"]
    rmn = pa_results["random_mean"]

    print("\n" + "=" * 70)
    print(
        f"HORN'S PARALLEL ANALYSIS "
        f"(N={n_obs}, p={n_vars}, {n_rep} replications)"
    )
    print("=" * 70)
    print(
        f"  {'Component':<12} {'Actual':>8} {'Random 95th':>12} "
        f"{'Random Mean':>12} {'Retain?':>8}"
    )
    print("  " + "-" * 55)
    for i in range(n_vars):
        retain = "YES" if actual_eigs[i] > r95[i] else "no"
        print(
            f"  {i + 1:<10} {actual_eigs[i]:>8.4f} "
            f"{r95[i]:>12.4f} {rmn[i]:>12.4f} {retain:>8}"
        )


# =====================================================================
# Bartlett factor scoring
# =====================================================================

def bartlett_scores_2f(
    df: pd.DataFrame, model: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2-factor Bartlett scores per row, handling partial items.

    Bartlett (1937) scoring:

        f = (L' Psi^{-1} L)^{-1} L' Psi^{-1} z

    where ``L`` is the loading submatrix for items available on
    this row, ``Psi`` is the corresponding diagonal of uniquenesses,
    and ``z`` is the vector of item values z-scored using the
    calibration-sample means and SDs. Rows with fewer than 2
    available items are left as NaN (a 2-factor model is not
    identified with fewer items).
    """
    items = model["items"]
    loadings = model["loadings"]
    uniquenesses = model["uniquenesses"]
    item_means = model["item_means"]
    item_sds = model["item_sds"]

    n = len(df)
    pain_scores = np.full(n, np.nan)
    contrast_scores = np.full(n, np.nan)
    n_items_used = np.zeros(n, dtype=int)

    rows = df[items].reset_index(drop=True)

    for i in range(n):
        available_idx = []
        z_vals = []
        for j, item in enumerate(items):
            val = rows.iat[i, j]
            if pd.notna(val):
                available_idx.append(j)
                z_vals.append((val - item_means[item]) / item_sds[item])

        if len(available_idx) < 2:
            continue

        L_k = loadings[available_idx, :]
        z_k = np.array(z_vals)
        psi_inv = np.diag(1.0 / uniquenesses[available_idx])
        precision = L_k.T @ psi_inv @ L_k

        try:
            precision_inv = np.linalg.inv(precision)
        except np.linalg.LinAlgError:
            continue

        scores = precision_inv @ L_k.T @ psi_inv @ z_k
        pain_scores[i] = scores[0]
        contrast_scores[i] = scores[1]
        n_items_used[i] = len(available_idx)

    return pain_scores, contrast_scores, n_items_used


def score_all_rows(
    long: pd.DataFrame, model: Dict,
) -> Tuple[pd.DataFrame, float, float]:
    """Score pain factor, contrast factor, and z-sleep on every row.

    Only quarter >= 1 rows are scored; quarter-0 rows stay NaN on
    the factor columns (they carry baseline variables only).
    """
    print("\n  Scoring constructs on quarter >= 1 rows...")

    # Only quarter >= 1 rows participate in scoring
    is_q = long["quarter"] >= 1
    q_rows = long.loc[is_q].copy()

    pain_scores, contrast_scores, n_items_used = bartlett_scores_2f(
        q_rows, model
    )

    long = long.copy()
    long["pain_factor"] = np.nan
    long["contrast_factor"] = np.nan
    long["pain_n_items"] = 0

    q_idx = long.index[is_q]
    long.loc[q_idx, "pain_factor"] = pain_scores
    long.loc[q_idx, "contrast_factor"] = contrast_scores
    long.loc[q_idx, "pain_n_items"] = n_items_used

    valid_pain = long["pain_factor"].notna()
    valid_contrast = long["contrast_factor"].notna()
    print(
        f"    Pain factor: {valid_pain.sum()} scored "
        f"({valid_pain.sum() / is_q.sum():.1%} of quarterly rows)"
    )
    print(f"    Contrast factor: {valid_contrast.sum()} scored")

    both = valid_pain & valid_contrast
    if both.sum() > 10:
        r = np.corrcoef(
            long.loc[both, "pain_factor"].values,
            long.loc[both, "contrast_factor"].values,
        )[0, 1]
        print(f"    Orthogonality: r(F1, F2) = {r:+.4f}")

    # Sleep: z-score q13 over quarterly rows
    sleep_vals = long.loc[is_q, SLEEP_ITEM].astype(float)
    sleep_mean = float(np.nanmean(sleep_vals))
    sleep_sd = float(np.nanstd(sleep_vals))

    long["sleep_factor"] = np.nan
    long.loc[q_idx, "sleep_factor"] = (
        (long.loc[q_idx, SLEEP_ITEM].astype(float) - sleep_mean) / sleep_sd
    )
    valid_sleep = long["sleep_factor"].notna()
    print(
        f"    Sleep factor: {valid_sleep.sum()} scored "
        f"(raw mean={sleep_mean:.2f}, SD={sleep_sd:.2f})"
    )

    return long, sleep_mean, sleep_sd


# =====================================================================
# Interpolation
# =====================================================================

#: True  -> fill ONLY gaps of exactly one quarter
#: False -> fill the first quarter of any interior gap  <-- THE COMMITTED RULE
#:
#: DECIDED (Aug 8 2026): False. The analysis is the submitted one, N = 229 /
#: 1,818 transitions. Both arms were run end to end through the identical
#: pipeline and compared on every conclusion the paper draws
#: (`tools/compare_interpolation_arms.py`); the choice was made on that
#: comparison, not left to whichever branch happened to be in the file.
#:
#: The switch stays because the alternative rule is a genuine sensitivity
#: question and step 09 reports one, not because the choice is still open.
#: Flipping it changes the analytic sample and moves three reported findings,
#: so it is not a free parameter -- change it only deliberately, and re-run the
#: whole pipeline when you do.
SINGLETON_GAPS_ONLY = False


def interpolate_factor_scores(long: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate single interior gaps in factor scores.

    For each subject, fill single missing quarters (gap length = 1)
    in ``pain_factor``, ``contrast_factor``, and ``sleep_factor``
    by linear interpolation between the two neighboring quarters.
    Gaps of length 2 or longer are left as NaN, as are all
    boundary-adjacent gaps (``limit_area="inside"``).

    Subjects must have at least 11 quarter-1..11 rows for
    interpolation to work as expected. Step 00 guarantees this:
    every subject has exactly 12 rows (quarter 0..11).

    This function also adds an ``interpolated`` boolean column
    to the output, True on rows where at least one of the three
    factor scores was newly filled by interpolation.
    """
    print("\n  Interpolating single interior gaps in factor scores...")

    out = long.copy()
    out["interpolated"] = False
    # Cells actually FILLED on this row. Kept per-row because the summary below used
    # to count every non-missing cell on an interpolated row instead, which counts
    # values that were observed, not imputed.
    out["n_cells_filled"] = 0

    factor_cols = ["pain_factor", "contrast_factor", "sleep_factor"]

    n_filled_total = 0
    n_rows_affected = 0

    for pid, grp in out.groupby("ID"):
        q_mask = grp["quarter"] >= 1
        qgrp = grp.loc[q_mask].sort_values("quarter")

        for col in factor_cols:
            before = qgrp[col].copy()
            filled = qgrp[col].interpolate(
                method="linear", limit_area="inside", limit=1,
            )
            # `limit=1` caps how many CONSECUTIVE NaNs pandas fills, so it fills the
            # first quarter of a gap of ANY length -- a 3-quarter gap gets its first
            # value invented from neighbours 4 quarters apart. The Methods describes
            # gaps of length one ("provided both neighbors were present"), which is a
            # statement about the gap, not about how many cells to fill.
            #
            # SINGLETON_GAPS_ONLY selects between the two. True makes the code match the
            # Methods sentence; False reproduces the submitted behaviour. It is a switch
            # rather than an edit because the choice changes the analytic sample
            # (229/1,818 -> 227/1,793) and therefore several reported findings, so both
            # arms have to stay runnable while that is being decided.
            if not SINGLETON_GAPS_ONLY:
                newly_filled = before.isna() & filled.notna()
                if newly_filled.any():
                    out.loc[qgrp.index[newly_filled], col] = filled[newly_filled]
                    out.loc[qgrp.index[newly_filled], "interpolated"] = True
                    out.loc[qgrp.index[newly_filled], "n_cells_filled"] += 1
                    n_filled_total += int(newly_filled.sum())
                continue
            isna = before.isna().values
            singleton = np.zeros(len(isna), dtype=bool)
            for i in range(len(isna)):
                if isna[i] and (i == 0 or not isna[i - 1]) and \
                        (i == len(isna) - 1 or not isna[i + 1]):
                    singleton[i] = True
            filled = filled.where(pd.Series(singleton, index=filled.index), before)
            newly_filled = before.isna() & filled.notna()
            if newly_filled.any():
                out.loc[qgrp.index[newly_filled], col] = filled[newly_filled]
                out.loc[qgrp.index[newly_filled], "interpolated"] = True
                out.loc[qgrp.index[newly_filled], "n_cells_filled"] += 1
                n_filled_total += int(newly_filled.sum())

    n_rows_affected = int(out["interpolated"].sum())
    print(
        f"    Interpolation filled {n_filled_total} factor-score cells "
        f"across {n_rows_affected} person-quarter rows"
    )

    return out


# =====================================================================
# Top-level entry point
# =====================================================================

def run_step01(verbose: bool = True, refit: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """Full Step 01 pipeline: load, factor-analyze, score, interpolate.

    Returns
    -------
    scored : DataFrame
        The long-format table with factor scores and interpolated
        values (written to ``scored_long.csv``).
    model : dict
        The factor model parameters (written to ``factor_model.json``).
    """
    if verbose:
        print("=" * 70)
        print("STEP 01 — Factor analysis and interpolation")
        print("=" * 70)

    # Without --refit, load the saved solution instead of re-estimating it.
    #
    # This is not just a time saving. `calibrate_factor_model` is NOT bit-reproducible:
    # `polychoric_corr_pair` builds its log-likelihood from
    # `scipy.stats.multivariate_normal.cdf`, which integrates by randomized
    # quasi-Monte-Carlo, so L-BFGS-B optimizes a noisy objective and the loadings move a
    # little on every call. Re-estimating on a replot therefore silently rewrites the
    # factor scores that every later step reads, and reported numbers drift between runs
    # that were supposed to change nothing -- the ANOVA F of step 03 moved 17.00 -> 16.52
    # exactly this way. Recomputing is now something you ask for.
    if not refit and os.path.exists(OUT_SCORED_CSV) and os.path.exists(OUT_MODEL_JSON):
        scored = pd.read_csv(OUT_SCORED_CSV)
        with open(OUT_MODEL_JSON) as fh:
            model = json.load(fh)
        if verbose:
            print(f"  Loaded saved solution: {OUT_SCORED_CSV}")
            print(f"                         {OUT_MODEL_JSON}")
            print("  (pass --refit to re-estimate; the polychoric EFA is not "
                  "bit-reproducible)")
        return scored, model

    if verbose:
        print(f"  Input:  {IN_LONG_CSV}")

    long = pd.read_csv(IN_LONG_CSV)
    if verbose:
        n_subjects = long["ID"].nunique()
        n_q = (long["quarter"] >= 1).sum()
        print(
            f"  Loaded: {len(long):,} rows, {n_subjects} subjects, "
            f"{n_q} quarterly rows"
        )

    # Calibrate the factor model on the quarterly rows (pain items are
    # all-NaN on quarter-0 rows by construction).
    q_rows = long.loc[long["quarter"] >= 1]
    model = calibrate_factor_model(q_rows, use_polychoric=True)

    # Parallel analysis on the same correlation matrix shape
    pa_results = run_parallel_analysis(
        n_obs=model["n_obs_used"],
        n_vars=len(model["items"]),
        n_replications=1000,
        seed=42,
    )
    report_parallel_analysis(model["eigenvalues_unreduced"], pa_results)

    # Score every row
    scored, sleep_mean, sleep_sd = score_all_rows(long, model)

    # Interpolate single gaps
    scored = interpolate_factor_scores(scored)

    # Persist outputs
    os.makedirs(DATA_DIR, exist_ok=True)
    scored.to_csv(OUT_SCORED_CSV, index=False)
    if verbose:
        print(f"\n  Saved: {OUT_SCORED_CSV}")

    # ---- Complete factor-analysis output as JSON --------------------
    # Save the full analysis record, not just what the manuscript
    # cites. Anything a reviewer or Xiaohan would want to inspect
    # about the factor analysis should be here.
    k = len(model["items"])
    eig_u = model["eigenvalues_unreduced"]

    # Item-level descriptive statistics
    q_scored = scored.loc[scored["quarter"] >= 1]
    item_descriptives = {}
    for item in model["items"]:
        vals = q_scored[item].dropna()
        item_descriptives[item] = {
            "n_nonmissing": int(vals.count()),
            "n_missing": int(q_scored[item].isna().sum()),
            "mean": float(vals.mean()) if len(vals) > 0 else None,
            "sd": float(vals.std()) if len(vals) > 0 else None,
            "min": float(vals.min()) if len(vals) > 0 else None,
            "max": float(vals.max()) if len(vals) > 0 else None,
        }

    # Factor score descriptives
    factor_descriptives = {}
    for col in ["pain_factor", "contrast_factor", "sleep_factor"]:
        vals = q_scored[col].dropna()
        factor_descriptives[col] = {
            "n_scored": int(vals.count()),
            "n_missing": int(q_scored[col].isna().sum()),
            "mean": float(vals.mean()),
            "sd": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    # Factor orthogonality
    both_scored = (
        q_scored["pain_factor"].notna() & q_scored["contrast_factor"].notna()
    )
    r_f1_f2 = float(np.corrcoef(
        q_scored.loc[both_scored, "pain_factor"].values,
        q_scored.loc[both_scored, "contrast_factor"].values,
    )[0, 1]) if both_scored.sum() > 10 else None

    # Interpolation summary
    n_interpolated_rows = int(scored["interpolated"].sum())
    # Cells FILLED by interpolation. The previous expression counted every non-missing
    # factor score on an interpolated row -- including the two that were observed --
    # so it over-reported (354 where 338 were imputed).
    n_interpolated_cells = int(scored["n_cells_filled"].sum())

    model_json = {
        # Correlation matrix
        "correlation_type": "polychoric",
        "correlation_matrix": model["correlation_matrix"].tolist(),

        # Items
        "pain_items": model["items"],
        "item_descriptives": item_descriptives,
        "item_means_calibration": model["item_means"],
        "item_sds_calibration": model["item_sds"],

        # Factor model (PAF)
        "n_factors": N_FACTORS,
        "loadings_f1": model["loadings"][:, 0].tolist(),
        "loadings_f2": model["loadings"][:, 1].tolist(),
        "uniquenesses": model["uniquenesses"].tolist(),
        "communalities": model["communalities"].tolist(),

        # Eigenvalues — unreduced (what PA and the manuscript report)
        "eigenvalues_unreduced_all": eig_u.tolist(),
        "eigenvalue_f1_unreduced": float(eig_u[0]),
        "eigenvalue_f2_unreduced": float(eig_u[1]),
        "variance_explained_f1_pct": float(eig_u[0]) / k * 100.0,
        "variance_explained_f2_pct": float(eig_u[1]) / k * 100.0,
        "total_variance_explained_pct":
            float(eig_u[0] + eig_u[1]) / k * 100.0,

        # Eigenvalues — reduced (from iterative PAF)
        "eigenvalue_f1_reduced": float(model["eigenvalues"][0]),
        "eigenvalue_f2_reduced": float(model["eigenvalues"][1]),

        # Horn's parallel analysis
        "parallel_analysis": {
            "n_obs": pa_results["n_obs"],
            "n_vars": pa_results["n_vars"],
            "n_replications": pa_results["n_replications"],
            "random_95th_percentile": pa_results["random_95th"].tolist(),
            "random_mean": pa_results["random_mean"].tolist(),
            "retain_factors_by_95th": int(
                (eig_u > pa_results["random_95th"]).sum()
            ),
            "retain_factors_by_mean": int(
                (eig_u > pa_results["random_mean"]).sum()
            ),
        },

        # Sleep scoring
        "sleep_item": SLEEP_ITEM,
        "sleep_grand_mean": sleep_mean,
        "sleep_grand_sd": sleep_sd,

        # Scoring summary
        "n_obs_fully_nonmissing_8items": model["n_obs_used"],
        "n_total_quarterly_rows": int((scored["quarter"] >= 1).sum()),
        "factor_descriptives": factor_descriptives,
        "factor_orthogonality_r_f1_f2": r_f1_f2,

        # Interpolation summary
        "interpolation": {
            "method": "linear",
            "limit": 1,
            "limit_area": "inside",
            "n_rows_interpolated": n_interpolated_rows,
            "n_cells_interpolated": n_interpolated_cells,
        },
    }
    os.makedirs(DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    with open(OUT_MODEL_JSON, "w") as f:
        json.dump(model_json, f, indent=2)
    if verbose:
        print(f"  Saved: {OUT_MODEL_JSON}")

    # ---- Results CSV: numbers that fill the manuscript text -----------
    # Every number from Results §3.1 "Factor analysis and pain
    # localization contrast" that is stated in the text, plus the
    # loadings table (Table 1) and PA thresholds.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)
    OUT_RESULTS_CSV = os.path.join(STEP_DERIV_DIR, "step01_factor_results.csv")

    result_rows = []

    def _r(metric, value, note=""):
        result_rows.append({
            "metric": metric, "value": value, "note": note,
        })

    _r("eigenvalue_f1", f"{eig_u[0]:.4f}", "unreduced")
    _r("eigenvalue_f2", f"{eig_u[1]:.4f}", "unreduced")
    _r("variance_f1_pct", f"{eig_u[0]/k*100:.1f}")
    _r("variance_f2_pct", f"{eig_u[1]/k*100:.1f}")
    _r("total_variance_2factor_pct", f"{(eig_u[0]+eig_u[1])/k*100:.1f}")
    _r("pa_95th_comp1", f"{pa_results['random_95th'][0]:.4f}")
    _r("pa_95th_comp2", f"{pa_results['random_95th'][1]:.4f}")
    _r("pa_mean_comp1", f"{pa_results['random_mean'][0]:.4f}")
    _r("pa_mean_comp2", f"{pa_results['random_mean'][1]:.4f}")
    _r("pa_n_replications", str(pa_results["n_replications"]))
    _r("pa_n_obs", str(pa_results["n_obs"]))
    _r("f1_passes_pa_95th", "YES" if eig_u[0] > pa_results["random_95th"][0] else "no")
    _r("f2_passes_pa_95th", "YES" if eig_u[1] > pa_results["random_95th"][1] else "no")
    _r("r_f1_f2", f"{r_f1_f2:.4f}" if r_f1_f2 is not None else "NA")

    # Loadings per item
    for i, item in enumerate(model["items"]):
        _r(f"loading_f1_{item}", f"{model['loadings'][i,0]:.4f}")
        _r(f"loading_f2_{item}", f"{model['loadings'][i,1]:.4f}")

    # Loading ranges (stated in text)
    _r("f1_loadings_range",
       f"{min(model['loadings'][:,0]):.2f} to {max(model['loadings'][:,0]):.2f}")
    _r("f2_knee_loadings_range",
       f"+{min(model['loadings'][:4,1]):.2f} to +{max(model['loadings'][:4,1]):.2f}")
    _r("f2_body_loadings_range",
       f"{max(model['loadings'][4:,1]):.2f} to {min(model['loadings'][4:,1]):.2f}")

    _r("n_interpolated_rows", str(n_interpolated_rows))
    _r("n_interpolated_cells", str(n_interpolated_cells))

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(OUT_RESULTS_CSV, index=False)
    if verbose:
        print(f"  Saved: {OUT_RESULTS_CSV}")


    if verbose:
        print("=" * 70)

    return scored, model_json


def main():
    parser = argparse.ArgumentParser(
        description="Step 01 — factor analysis and interpolation on the "
                    "Step 00 long-format data."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--refit", action="store_true",
        help="Re-run computation from scratch instead of loading saved derivatives",
    )
    args = parser.parse_args()
    run_step01(verbose=not args.quiet, refit=args.refit)


if __name__ == "__main__":
    main()
