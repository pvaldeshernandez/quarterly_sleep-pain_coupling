"""
Step 07 — Build the VARX(1) frame: centering, lags, interaction terms.
======================================================================

Input:  derivatives/step03_curation/step03_curated_long.csv
Output: derivatives/step07_varx_data/step07_processed_long.csv
        derivatives/step07_varx_data/step07_timepoint_summary.csv

The curation half of this step -- the segment filter, Figure 1, Table 3 and the
timepoint summary -- moved to step 03 on 11 Aug 2026. What is left is model
preparation, and it runs immediately before the fit that consumes it.

Operations, in order:

  1. Within-between decomposition -- for each of pain_factor, contrast_factor and
     sleep_factor: person-mean, between-person deviation, within-person deviation.
  2. Lag creation -- shift each within-person variable by 1 within (ID, segment_id).
  3. Interaction terms -- sleep_x_contrast_lag1 and pain_x_contrast_lag1 for the
     contrast-moderation terms.

Row count is unchanged by all three: the frame keeps every curated person-quarter and
the lag columns are simply missing on the first row of each segment. 2,056 rows, of
which 1,818 carry a lag-1 value.

Author: Pedro Valdes-Hernandez (with Claude Opus 4.6)
"""
from __future__ import annotations

import argparse
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
ROOT = os.path.dirname(os.path.dirname(HERE))  # repo root
DATA_DIR = os.path.join(ROOT, "data")

# lib/ on the path, as every other step that imports from it does. Without this the
# registry import below raises ModuleNotFoundError when the step is run on its own --
# `python step07_prepare_varx_data.py`, the command its own main() documents -- and it
# dies AFTER writing its two derivative CSVs, so the step half-completes and its seven
# named values never reach results/. It survived the cold run only because
# run_pipeline.py inserts lib/ before importing any step.
sys.path.insert(0, os.path.join(HERE, "lib"))
from registry import write_numbers  # noqa: E402

DERIV_DIR = os.path.join(ROOT, "derivatives")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step07_varx_data")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step07_varx_data")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

#: The curated analytic sample, from step 03.
IN_CURATED_CSV = os.path.join(DERIV_DIR, "step03_curation", "step03_curated_long.csv")

OUT_PROCESSED_CSV = os.path.join(STEP_DERIV_DIR, "step07_processed_long.csv")
#: Counts the lag-1 transitions, so it can only be computed once the lags exist.
OUT_SUMMARY_CSV = os.path.join(STEP_DERIV_DIR, "step07_timepoint_summary.csv")


# =====================================================================
# Constants
# =====================================================================

FACTOR_COLS = ["pain_factor", "contrast_factor", "sleep_factor"]

# =====================================================================
# Within-between decomposition
# =====================================================================

def decompose_within_between(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose each factor score into between- and within-person parts.

    For each variable Y in {pain_factor, contrast_factor, sleep_factor}:
      Y_bar_i         = person i's time-average across retained quarters
      Y_between_i     = Y_bar_i - grand_mean (between-person deviation)
      Y_within_it     = Y_it - Y_bar_i       (within-person deviation)

    The VARX(1) model operates on the within-person deviations, which
    by construction remove all stable between-person confounders
    (Curran & Bauer, 2011).
    """
    print("\n  Within-between decomposition (Curran & Bauer, 2011)...")

    df = df.copy()
    labels = {
        "pain_factor": "pain",
        "contrast_factor": "contrast",
        "sleep_factor": "sleep",
    }

    for raw_var, label in labels.items():
        grand_mean = df[raw_var].mean()

        person_means = (
            df.groupby("ID")[raw_var]
            .mean()
            .reset_index()
            .rename(columns={raw_var: f"{label}_person_mean"})
        )
        person_means[f"{label}_between"] = (
            person_means[f"{label}_person_mean"] - grand_mean
        )

        df = df.merge(
            person_means[["ID", f"{label}_person_mean", f"{label}_between"]],
            on="ID",
            how="left",
        )
        df[f"{label}_within"] = df[raw_var] - df[f"{label}_person_mean"]

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


# =====================================================================
# Lag creation and interaction terms
# =====================================================================

def create_lagged_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create t-1 lags and VARX interaction terms within segments.

    Lags are computed within (ID, segment_id) groups. An additional
    safety check NaN's out any lags where the quarters are not
    strictly consecutive.

    Interaction terms:
      sleep_x_contrast_lag1 = sleep_within_lag1 * contrast_within_lag1
      pain_x_contrast_lag1  = pain_within_lag1  * contrast_within_lag1
    """
    print("\n  Creating lagged variables and interaction terms...")

    df = df.sort_values(["ID", "segment_id", "quarter"])
    grp_key = ["ID", "segment_id"]

    df["pain_within_lag1"] = df.groupby(grp_key)["pain_within"].shift(1)
    df["sleep_within_lag1"] = df.groupby(grp_key)["sleep_within"].shift(1)
    df["contrast_within_lag1"] = df.groupby(grp_key)["contrast_within"].shift(1)

    # Safety: NaN out lags across non-consecutive quarters
    quarter_lag1 = df.groupby(grp_key)["quarter"].shift(1)
    non_consecutive = (df["quarter"] - quarter_lag1) != 1
    first_in_group = df.groupby(grp_key).cumcount() == 0
    invalid = non_consecutive & ~first_in_group
    n_invalidated = int(invalid.sum())
    if n_invalidated > 0:
        print(f"    WARNING: invalidated {n_invalidated} non-consecutive lags")
    df.loc[non_consecutive, "pain_within_lag1"] = np.nan
    df.loc[non_consecutive, "sleep_within_lag1"] = np.nan
    df.loc[non_consecutive, "contrast_within_lag1"] = np.nan

    # Interaction terms
    df["sleep_x_contrast_lag1"] = (
        df["sleep_within_lag1"] * df["contrast_within_lag1"]
    )
    df["pain_x_contrast_lag1"] = (
        df["pain_within_lag1"] * df["contrast_within_lag1"]
    )

    usable = df["pain_within_lag1"].notna() & df["sleep_within_lag1"].notna()
    print(
        f"    Full model usable: {usable.sum()} obs "
        f"from {df.loc[usable, 'ID'].nunique()} subjects"
    )

    return df


# =====================================================================
# Timepoint summary
# =====================================================================

def compute_timepoint_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the timepoint summary matching manuscript claims.

    Returns a single-row DataFrame with all reported counts so they
    can be saved to CSV and compared against the manuscript.
    """
    usable = df["pain_within_lag1"].notna() & df["sleep_within_lag1"].notna()

    n_subjects = df["ID"].nunique()
    n_retained_points = len(df)
    n_lag_transitions = int(usable.sum())

    # Per-person lag counts
    per_person = df.loc[usable].groupby("ID").size()
    median_lags = int(per_person.median())
    min_lags = int(per_person.min())
    max_lags = int(per_person.max())

    # Interpolated retained points
    n_interpolated = int(df["interpolated"].sum()) if "interpolated" in df.columns else 0

    summary = {
        "n_subjects_analytic": n_subjects,
        "n_retained_points": n_retained_points,
        "n_interpolated_retained": n_interpolated,
        "n_lag_transitions": n_lag_transitions,
        "median_lags_per_person": median_lags,
        "min_lags_per_person": min_lags,
        "max_lags_per_person": max_lags,
    }

    print("\n  Timepoint summary:")
    print(f"    Analytic sample:      {n_subjects} subjects")
    print(f"    Retained points:      {n_retained_points}")
    print(f"    Interpolated:         {n_interpolated} of {n_retained_points}")
    print(f"    Lag transitions:      {n_lag_transitions}")
    print(f"    Lags per person:      median={median_lags}, range=[{min_lags}, {max_lags}]")

    return pd.DataFrame([summary])



# =====================================================================
# Pipeline
# =====================================================================

def run_step07(verbose: bool = True, refit: bool = False):
    """Build the VARX(1)-ready frame from the curated sample."""
    if verbose:
        print("=" * 70)
        print("STEP 07 — Prepare VARX data")
        print("=" * 70)
        print(f"  Input: {IN_CURATED_CSV}")

    if not os.path.exists(IN_CURATED_CSV):
        raise FileNotFoundError(
            f"{IN_CURATED_CSV} does not exist. Run step 03 first -- it defines the "
            f"analytic sample this step centers and lags.")

    # float_precision="round_trip": pandas' DEFAULT csv parser is a fast one that can be
    # off by 1 ULP. That did not matter before the split, when the frame was built in a
    # single pass in memory; now the curated sample makes an extra trip through disk, and
    # a 4.4e-16 difference on eleven rows was enough for NUTS to take a different
    # trajectory and move step10 and step15 in the third decimal.
    df = pd.read_csv(IN_CURATED_CSV, float_precision="round_trip")
    if verbose:
        print(f"  Loaded: {len(df):,} rows, {df['ID'].nunique()} subjects")

    df = decompose_within_between(df)
    df = create_lagged_variables(df)

    output_cols = [
        "ID", "quarter", "segment_id",
        "pain_factor", "contrast_factor", "sleep_factor",
        "interpolated",
        "pain_person_mean", "pain_between", "pain_within",
        "contrast_person_mean", "contrast_between", "contrast_within",
        "sleep_person_mean", "sleep_between", "sleep_within",
        "pain_within_lag1", "sleep_within_lag1", "contrast_within_lag1",
        "sleep_x_contrast_lag1", "pain_x_contrast_lag1",
        "Age", "Sex",
    ]
    df_out = df[[c for c in output_cols if c in df.columns]].copy()

    df_out.to_csv(OUT_PROCESSED_CSV, index=False)

    summary = compute_timepoint_summary(df_out)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    # Publish them under names. "median 9 per participant, range 2-10" is in the
    # Results, and it lived only in this wide-format CSV in derivatives/, which the
    # value collector cannot read -- it looks for a metric,value schema. Three
    # reported numbers with nothing to check them against.
    write_numbers(STEP_RESULTS_DIR,
                  {k: (int(v) if float(v).is_integer() else float(v))
                   for k, v in summary.iloc[0].items()},
                  prefix="step07")

    if verbose:
        print(f"\n  Saved: {OUT_PROCESSED_CSV}")
        print(f"    Shape: {df_out.shape}")
        print("\n" + "=" * 70)
        print("STEP 07 COMPLETE")
        print("=" * 70)
    return df_out


def main():
    ap = argparse.ArgumentParser(description="Step 07 — Prepare VARX data.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()
    run_step07(verbose=not args.quiet, refit=args.refit)


if __name__ == "__main__":
    main()
