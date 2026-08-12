#!/usr/bin/env python3
"""
Step 02 — Measurement checks on the published two-factor pain solution.

Two questions, both about the eight pain items and neither about the coupling model:

1. Does the two-factor structure depend on pooling repeated measures?
   The published solution (step 01) is calibrated on every person-quarter row treated
   as independent. This step re-estimates the SAME two-factor model under four
   samplings of the SAME eight items —

       POOLED    all complete-case quarterly rows, as published (estimator control)
       WITHIN    person-mean-centered items (pure within-person covariation)
       BETWEEN   person means, one row per participant (pure between-person)
       BASELINE  one row per participant, first observed quarter (no clustering)

   — and reports Tucker congruence of each against the published loadings. WITHIN is
   the relevant row, because the VARX model uses only within-person deviations.

2. Do knee and body items differ in range and dominance?
   Participants with substantially greater pain in a non-knee region were excluded at
   screening, so the body-dominant end of the contrast may be truncated by design.
   The knee-minus-body difference is described at the person-quarter and the
   participant level, against the theoretical span of the difference score.

Feeds Supplementary Table S2 (congruence by sampling) and Supplementary Section S3.
No prose, no table notes and no captions are generated here (fork Decision 3): the CSVs
carry the table data and `numbers.json` carries every scalar the text quotes.

Three deliberate departures from the sandbox script (`archive/revision_20260811/a04b_efa_clustering.py`),
each of which moves numbers and so must not be silent:

  * ESTIMATOR. The sandbox carried a FORK of the project's EFA — a plain unrotated
    eigendecomposition of a Pearson matrix with unit diagonal. This step calls the
    canonical estimator, `step01_factor_analysis.calibrate_factor_model`, which is
    iterative principal-axis factoring with communalities on the diagonal, with the
    F1-all-positive / F2-knee-positive sign convention already built in. One estimator,
    one copy, and the comparison isolates the sampling rather than the code path.

  * CORRELATION TYPE. `use_polychoric=False` for all four samplings, uniformly.
    Polychoric is only defined for the raw ordinal items: `polychoric_corr_pair` casts
    its inputs to int (step01_factor_analysis.py:122), so the mean-centered WITHIN data
    and the fractional BETWEEN person means would be silently truncated to integers.
    Uniform Pearson keeps the contrast about the sampling, not the estimator. The cost
    of the substitution is measurable and negligible: the POOLED Pearson re-estimate
    reproduces the published polychoric solution at congruence ~1.000 (F1) and ~0.9998
    (F2), which is why POOLED is kept as a control row rather than dropped.

  * REFERENCE. The sandbox compared each sampling against its own locally re-estimated
    POOLED solution. Table S2's caption says "with the published solution", so the
    reference here is the published loadings loaded from step 01's saved factor model.
    The sandbox's pooled-reference values are retained as extra rows of the congruence
    CSV so the change of reference stays auditable.

Caveat worth stating rather than absorbing: the published loadings were estimated from
a polychoric matrix built with PAIRWISE deletion (step01_factor_analysis.py:178-184),
whereas all four samplings here are complete-case. Congruence therefore compares
solutions fitted to slightly different row sets. That is inherent to the design.

Input:  <step01_factor_analysis.IN_LONG_CSV>   (step 00's long-format extract)
        derivatives/step01_factor_analysis/step01_factor_model.json  (published solution)
Output: derivatives/step02_measurement_checks/step02_efa_loadings.csv
        derivatives/step02_measurement_checks/step02_congruence.csv      -> Table S2
        derivatives/step02_measurement_checks/step02_item_descriptives.csv
        derivatives/step02_measurement_checks/step02_knee_body_dominance.csv
        results/step02_measurement_checks/numbers.json

Usage:
    python step02_measurement_checks.py [--refit] [--quiet]

The default path loads the saved CSVs and re-emits numbers.json without re-estimating
anything; --refit recomputes the four solutions. This step fits no MCMC model and does
not import lib/coupling_model.py.

Author: Pedro Valdes-Hernandez (with Claude Opus 5)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)
ROOT = os.path.dirname(os.path.dirname(HERE))

DERIV_DIR = os.path.join(ROOT, "derivatives")
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step02_measurement_checks")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step02_measurement_checks")

# step02 sits beside step01 in codes/python/, so a bare import resolves.
import step01_factor_analysis as s1  # noqa: E402

# Both input paths are taken from step 01's own constants rather than rebuilt here.
# step00's long file currently lives in data/ (read-only); if it is relocated to
# derivatives/step00_extract_data/, step01.IN_LONG_CSV changes and this step follows
# with no edit — one constant, one place.
IN_LONG_CSV = s1.IN_LONG_CSV
IN_MODEL_JSON = s1.OUT_MODEL_JSON

OUT_LOADINGS_CSV = os.path.join(STEP_DERIV_DIR, "step02_efa_loadings.csv")
OUT_CONGRUENCE_CSV = os.path.join(STEP_RESULTS_DIR, "step02_congruence.csv")
OUT_ITEM_DESC_CSV = os.path.join(STEP_DERIV_DIR, "step02_item_descriptives.csv")
OUT_DOMINANCE_CSV = os.path.join(STEP_DERIV_DIR, "step02_knee_body_dominance.csv")

# Canonical item constants — imported, never re-typed.
PAIN_ITEMS = s1.PAIN_ITEMS
KNEE_ITEMS = s1.KNEE_ITEMS
BODY_ITEMS = s1.BODY_ITEMS
N_FACTORS = s1.N_FACTORS
REGION = {c: ("knee" if c in KNEE_ITEMS else "body") for c in PAIN_ITEMS}

#: order in which the four samplings are estimated and reported
SAMPLINGS = ("POOLED", "WITHIN", "BETWEEN", "BASELINE")

#: the three rows of Table S2. POOLED is the estimator control, not a table row.
TABLE_S2_SAMPLINGS = ("WITHIN", "BETWEEN", "BASELINE")

#: reference labels used in the congruence CSV
PUBLISHED_REF = "published"
POOLED_REF = "pooled_reestimate"

#: conventional "essentially identical factor" threshold for Tucker congruence
CONGRUENCE_THRESHOLD = 0.95

#: the pain items are 0-10 numeric rating scales, so a knee-minus-body mean
#: difference can theoretically span the full width of the scale in either direction
ITEM_MIN, ITEM_MAX = 0.0, 10.0

#: stamped into the item-descriptives CSV so this table cannot be mistaken for the
#: analytic-sample table that step 05 builds on a different (smaller) row set
SAMPLE_LABEL = "step02_calibration_complete_case_quarterly"


# =====================================================================
# Inputs
# =====================================================================

def load_published_solution(path=IN_MODEL_JSON):
    """The published two-factor solution, loaded rather than re-estimated.

    Re-estimating it would mean a second polychoric fit that could diverge from the
    one the manuscript reports (and would cost minutes). Returns the item order and
    the (8, 2) loading matrix in that order.
    """
    with open(path) as fh:
        model_json = json.load(fh)
    items = list(model_json["pain_items"])
    loadings = np.column_stack([
        np.asarray(model_json["loadings_f1"], dtype=float),
        np.asarray(model_json["loadings_f2"], dtype=float),
    ])
    if loadings.shape != (len(items), N_FACTORS):
        raise ValueError(
            f"published loadings are {loadings.shape}, expected "
            f"({len(items)}, {N_FACTORS}) in {path}"
        )
    return items, loadings


def calibration_sample(long):
    """Complete-case quarterly rows on all eight pain items.

    Stated explicitly (`quarter >= 1`, then dropna) rather than relying on quarter-0
    pain items happening to be all-NaN. This mirrors step01_factor_analysis.run_step01,
    which calibrates on `long.loc[long["quarter"] >= 1]`.
    """
    quarterly = long.loc[long["quarter"] >= 1]
    return quarterly.dropna(subset=PAIN_ITEMS).copy()


# =====================================================================
# The four samplings
# =====================================================================

def build_samplings(df):
    """Four samplings of the same eight items, each as a DataFrame.

    DataFrames, not arrays: `calibrate_factor_model` selects its items by column name.
    Each frame carries ID so rows and participants can both be counted — Table S2's
    "Rows" column is person-quarters for WITHIN but participants for BETWEEN and
    BASELINE, and neither number should be typed by hand.
    """
    grouped = df.groupby("ID")[PAIN_ITEMS]
    within = df[PAIN_ITEMS] - grouped.transform("mean")
    first = df.sort_values(["ID", "quarter"]).groupby("ID").head(1)

    return {
        "POOLED": df[["ID"] + PAIN_ITEMS].copy(),
        "WITHIN": pd.concat([df[["ID"]], within], axis=1),
        "BETWEEN": grouped.mean().reset_index()[["ID"] + PAIN_ITEMS],
        "BASELINE": first[["ID"] + PAIN_ITEMS].copy(),
    }


def align_loadings(model, target_items):
    """Reorder a fitted loading matrix into `target_items` order.

    Never assume the estimator returned items in the published order; a silent
    permutation would corrupt every congruence coefficient without failing.
    """
    order = [model["items"].index(item) for item in target_items]
    return np.asarray(model["loadings"], dtype=float)[order, :]


def estimate_two_factor(sampling, frame, target_items):
    """Two-factor solution for one sampling, through the canonical estimator.

    Returns (loadings in `target_items` order, unreduced eigenvalues, reduced
    eigenvalues, n_rows, n_persons).
    """
    X = frame[PAIN_ITEMS]

    if X.isna().any().any():
        raise ValueError(f"{sampling}: missing values reached the estimator")
    sds = X.std(axis=0)
    if not np.all(np.isfinite(sds)) or float(sds.min()) <= 0.0:
        # A raise, not a `continue`: silently skipping a sampling would silently
        # drop a row of Table S2.
        dead = [c for c in PAIN_ITEMS if not np.isfinite(sds[c]) or sds[c] <= 0]
        raise ValueError(f"{sampling}: zero-variance item(s) {dead}")

    # use_polychoric=False for every sampling — see the module docstring.
    model = s1.calibrate_factor_model(X, use_polychoric=False, verbose=False)

    return (
        align_loadings(model, target_items),
        np.asarray(model["eigenvalues_unreduced"], dtype=float),
        np.asarray(model["eigenvalues"], dtype=float),
        int(len(frame)),
        int(frame["ID"].nunique()),
    )


# =====================================================================
# Knee-versus-body range and dominance
# =====================================================================

def _dominance_row(level, diff):
    """Descriptives of one knee-minus-body difference distribution."""
    return {
        "level": level,
        "n": int(diff.size),
        "mean": float(diff.mean()),
        "sd": float(diff.std()),
        "median": float(diff.median()),
        "p25": float(diff.quantile(0.25)),
        "p75": float(diff.quantile(0.75)),
        "min": float(diff.min()),
        "max": float(diff.max()),
        "pct_knee_dominant": 100.0 * float((diff > 0).mean()),
        "pct_body_dominant": 100.0 * float((diff < 0).mean()),
        "pct_equal": 100.0 * float((diff == 0).mean()),
        "theoretical_min": ITEM_MIN - ITEM_MAX,
        "theoretical_max": ITEM_MAX - ITEM_MIN,
    }


def knee_body_dominance(df):
    """Knee-mean minus body-mean, per person-quarter and per participant.

    Section S3 quotes eight numbers from this; the sandbox only printed them.
    """
    diff_pq = df[KNEE_ITEMS].mean(axis=1) - df[BODY_ITEMS].mean(axis=1)
    person = df.groupby("ID")[PAIN_ITEMS].mean()
    diff_person = person[KNEE_ITEMS].mean(axis=1) - person[BODY_ITEMS].mean(axis=1)
    return pd.DataFrame([
        _dominance_row("person_quarter", diff_pq),
        _dominance_row("participant", diff_person),
    ])


# =====================================================================
# Compute
# =====================================================================

def compute(verbose=True):
    """Re-estimate the four solutions and build the four tables."""
    from measurement import tucker_congruence
    from descriptives import item_descriptives

    long = pd.read_csv(IN_LONG_CSV, usecols=["ID", "quarter"] + PAIN_ITEMS)
    df = calibration_sample(long)
    if verbose:
        print(f"  calibration sample: {len(df):,} person-quarters, "
              f"{df['ID'].nunique()} participants")

    published_items, published_loadings = load_published_solution()

    # Item descriptives on this sample. Shared with step 05, which computes the same
    # thing on the analytic sample; the `sample` column keeps the two apart.
    item_desc = item_descriptives(
        df, PAIN_ITEMS, region_map=REGION, sample=SAMPLE_LABEL, ceiling=ITEM_MAX,
    )

    dominance = knee_body_dominance(df)

    samplings = build_samplings(df)
    loading_rows = []
    fitted = {}
    for name in SAMPLINGS:
        loadings, eig_unreduced, eig_reduced, n_rows, n_persons = estimate_two_factor(
            name, samplings[name], published_items,
        )
        fitted[name] = {"loadings": loadings, "n_rows": n_rows, "n_persons": n_persons}
        for i, item in enumerate(published_items):
            loading_rows.append({
                "sampling": name,
                "item": item,
                "region": REGION[item],
                "F1": float(loadings[i, 0]),
                "F2": float(loadings[i, 1]),
                "eig1_unreduced": float(eig_unreduced[0]),
                "eig2_unreduced": float(eig_unreduced[1]),
                "eig1_reduced": float(eig_reduced[0]),
                "eig2_reduced": float(eig_reduced[1]),
                "n_rows": n_rows,
                "n_persons": n_persons,
            })
        if verbose:
            print(f"  {name:<9} rows {n_rows:>5,}  participants {n_persons:>4}  "
                  f"eigenvalues {eig_unreduced[0]:.2f}, {eig_unreduced[1]:.2f}")
    loadings_df = pd.DataFrame(loading_rows)

    # Congruence, long format: every sampling against the published solution (what
    # Table S2 reports), and the three non-pooled samplings against the pooled
    # re-estimate as well (what the sandbox reported, kept for auditability).
    pooled = fitted["POOLED"]["loadings"]
    cong_rows = []
    for name in SAMPLINGS:
        entry = fitted[name]
        for ref_label, ref in ((PUBLISHED_REF, published_loadings), (POOLED_REF, pooled)):
            if name == "POOLED" and ref_label == POOLED_REF:
                continue  # congruence of a solution with itself carries no information
            cong_rows.append({
                "sampling": name,
                "reference": ref_label,
                "n_rows": entry["n_rows"],
                "n_persons": entry["n_persons"],
                "congruence_F1": tucker_congruence(ref[:, 0], entry["loadings"][:, 0]),
                "congruence_F2": tucker_congruence(ref[:, 1], entry["loadings"][:, 1]),
                "threshold": CONGRUENCE_THRESHOLD,
            })
    congruence_df = pd.DataFrame(cong_rows)

    return loadings_df, congruence_df, item_desc, dominance


# =====================================================================
# Numbers
# =====================================================================

def assemble_numbers(loadings_df, congruence_df, dominance_df):
    """Every scalar Section S3 and Table S2 quote, derived from the saved tables.

    Both the refit path and the load path go through this one function, so a reloaded
    run cannot report different numbers from the run that produced the CSVs.
    """
    nums = {}

    for name in SAMPLINGS:
        block = loadings_df[loadings_df["sampling"] == name]
        if block.empty:
            raise ValueError(f"no loadings recorded for sampling {name}")
        key = name.lower()
        nums[f"n_rows_{key}"] = int(block["n_rows"].iloc[0])
        nums[f"n_persons_{key}"] = int(block["n_persons"].iloc[0])
        # Unreduced eigenvalues (correlation matrix with unit diagonal), explicitly
        # named: the reduced PAF series is a different quantity and both are saved.
        nums[f"eigenvalue_f1_unreduced_{key}"] = float(block["eig1_unreduced"].iloc[0])
        nums[f"eigenvalue_f2_unreduced_{key}"] = float(block["eig2_unreduced"].iloc[0])

    nums["n_calibration_person_quarters"] = nums["n_rows_pooled"]
    nums["n_calibration_participants"] = nums["n_persons_pooled"]

    # Sign pattern, checked rather than asserted in prose.
    knee = loadings_df[loadings_df["region"] == "knee"]
    body = loadings_df[loadings_df["region"] == "body"]
    nums["f1_positive_on_all_items_all_samplings"] = bool((loadings_df["F1"] > 0).all())
    nums["f2_knee_positive_body_negative_all_samplings"] = bool(
        (knee["F2"] > 0).all() and (body["F2"] < 0).all()
    )

    published = congruence_df[congruence_df["reference"] == PUBLISHED_REF]
    published = published.set_index("sampling")
    for name in SAMPLINGS:
        key = name.lower()
        nums[f"congruence_{key}_f1"] = float(published.loc[name, "congruence_F1"])
        nums[f"congruence_{key}_f2"] = float(published.loc[name, "congruence_F2"])
    # The "X or above" sentence covers Table S2's three rows only; POOLED is the
    # estimator control and is reported separately as congruence_pooled_f1/f2.
    table_s2 = published.loc[list(TABLE_S2_SAMPLINGS), ["congruence_F1", "congruence_F2"]]
    nums["congruence_min_across_samplings"] = float(np.min(table_s2.values))
    nums["congruence_threshold"] = float(CONGRUENCE_THRESHOLD)

    dom = dominance_df.set_index("level")
    pq, pp = dom.loc["person_quarter"], dom.loc["participant"]
    nums.update({
        "knee_minus_body_mean": float(pq["mean"]),
        "knee_minus_body_sd": float(pq["sd"]),
        "knee_minus_body_min": float(pq["min"]),
        "knee_minus_body_max": float(pq["max"]),
        "knee_minus_body_theoretical_min": float(pq["theoretical_min"]),
        "knee_minus_body_theoretical_max": float(pq["theoretical_max"]),
        "pct_person_quarters_knee_dominant": float(pq["pct_knee_dominant"]),
        "pct_person_quarters_body_dominant": float(pq["pct_body_dominant"]),
        "pct_person_quarters_equal": float(pq["pct_equal"]),
        "pct_participants_knee_dominant": float(pp["pct_knee_dominant"]),
        "pct_participants_body_dominant": float(pp["pct_body_dominant"]),
        "pct_participants_equal": float(pp["pct_equal"]),
    })
    return nums


# =====================================================================
# Top-level entry point
# =====================================================================

def run_step02(verbose=True, refit=False):
    """Measurement checks on the two-factor solution (Table S2, Section S3).

    Default path loads the saved CSVs and re-emits numbers.json. `--refit`
    re-estimates the four solutions from the item data.
    """
    from registry import write_numbers

    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("STEP 02 — Measurement checks (Table S2, Section S3)")
        print("=" * 70)

    saved = [OUT_LOADINGS_CSV, OUT_CONGRUENCE_CSV, OUT_ITEM_DESC_CSV, OUT_DOMINANCE_CSV]
    saved_exist = all(os.path.exists(p) for p in saved)
    if not refit and not saved_exist:
        if verbose:
            print("  Saved derivatives not found — forcing refit.")
        refit = True

    if refit:
        loadings_df, congruence_df, item_desc, dominance_df = compute(verbose=verbose)
        # %.17g so a float survives the CSV round-trip bit-for-bit: numbers.json is
        # assembled from these tables on both paths, and a load-path run must not
        # report a value that differs from the run that produced the CSVs.
        loadings_df.to_csv(OUT_LOADINGS_CSV, index=False, float_format="%.17g")
        congruence_df.to_csv(OUT_CONGRUENCE_CSV, index=False, float_format="%.17g")
        item_desc.to_csv(OUT_ITEM_DESC_CSV, index=False, float_format="%.17g")
        dominance_df.to_csv(OUT_DOMINANCE_CSV, index=False, float_format="%.17g")
    else:
        if verbose:
            print("  Loading saved derivatives (re-run with --refit to re-estimate).")
        # float_precision="round_trip": pandas' default float parser is not
        # correctly rounded and lands 1 ULP away, which would make a reloaded run
        # report values that differ from the run that wrote the CSVs.
        loadings_df = pd.read_csv(OUT_LOADINGS_CSV, float_precision="round_trip")
        congruence_df = pd.read_csv(OUT_CONGRUENCE_CSV, float_precision="round_trip")
        dominance_df = pd.read_csv(OUT_DOMINANCE_CSV, float_precision="round_trip")

    nums = assemble_numbers(loadings_df, congruence_df, dominance_df)
    write_numbers(STEP_RESULTS_DIR, nums, prefix="step02")

    if verbose:
        print(f"\n  congruence with the published solution "
              f"(threshold {CONGRUENCE_THRESHOLD:.2f}):")
        for name in SAMPLINGS:
            key = name.lower()
            tag = "" if name in TABLE_S2_SAMPLINGS else "   (estimator control)"
            print(f"    {name:<9} F1 {nums[f'congruence_{key}_f1']:.4f}   "
                  f"F2 {nums[f'congruence_{key}_f2']:.4f}"
                  f"   n={nums[f'n_rows_{key}']:,} rows / "
                  f"{nums[f'n_persons_{key}']} participants{tag}")
        print(f"  minimum across Table S2 rows: "
              f"{nums['congruence_min_across_samplings']:.4f}")
        print(f"  knee minus body: mean {nums['knee_minus_body_mean']:+.2f}, "
              f"SD {nums['knee_minus_body_sd']:.2f}, range "
              f"{nums['knee_minus_body_min']:+.2f} to {nums['knee_minus_body_max']:+.2f} "
              f"(theoretical {nums['knee_minus_body_theoretical_min']:+.0f} to "
              f"{nums['knee_minus_body_theoretical_max']:+.0f})")
        print(f"  wrote {os.path.relpath(OUT_CONGRUENCE_CSV, ROOT)}")
        print("=" * 70)

    return loadings_df, congruence_df, dominance_df, nums


def main():
    parser = argparse.ArgumentParser(
        description="Step 02 — measurement checks on the two-factor pain solution."
    )
    parser.add_argument("--refit", action="store_true",
                        help="Re-estimate the four solutions instead of loading "
                             "saved derivatives")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output.")
    args = parser.parse_args()
    run_step02(verbose=not args.quiet, refit=args.refit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
