"""
Step 03 — Segment filter, within-between decomposition, lag creation,
         Figure 1, Table 3, and timepoint summary.
======================================================================

Input:  new_organization/data/step01_scored_long.csv
Output: new_organization/data/step02_processed_long.csv
        new_organization/data/step02_figure1.png
        new_organization/data/step02_table3_demographics.csv
        new_organization/data/step02_timepoint_summary.csv

This step takes the factor-scored long table from Step 01 and
produces the VARX(1)-ready dataset. Operations in order:

  1. Segment filter — identify maximal runs of consecutive quarters
     where both pain_factor and sleep_factor are non-missing; assign
     segment_id; discard segments shorter than 3 quarters.
  2. Within-between decomposition — for each of pain_factor,
     contrast_factor, sleep_factor: compute person-mean, between-
     person deviation, within-person deviation.
  3. Lag creation — shift each within-person variable by 1 within
     (ID, segment_id) groups.
  4. Interaction terms — sleep_x_contrast_lag1 and
     pain_x_contrast_lag1 for the VARX contrast-moderation terms.
  5. Figure 1 — data-availability grid showing observed,
     interpolated, and retained person-quarters.
  6. Table 3 — demographic and baseline clinical characteristics
     of the analytic sample (subjects surviving the segment filter).
  7. Timepoint summary — N subjects, N retained points, N lag
     transitions, median/range of transitions per person, etc.

The code reuses the segment-filter, decomposition, and lag-creation
functions from the legacy ``python/01_prepare_data.py`` pipeline.

Author: Pedro Valdes-Hernandez (with Claude Opus 4.6)
"""
from __future__ import annotations

import argparse
import os
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

DERIV_DIR = os.path.join(ROOT, "derivatives")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step03_varx_data")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step03_varx_data")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

IN_SCORED_CSV = os.path.join(DERIV_DIR, "step01_factor_analysis", "step01_scored_long.csv")

OUT_PROCESSED_CSV = os.path.join(STEP_DERIV_DIR, "step03_processed_long.csv")
OUT_FIGURE1 = os.path.join(STEP_RESULTS_DIR, "step03_figure1.png")
OUT_TABLE3_CSV = os.path.join(STEP_RESULTS_DIR, "step03_table3_demographics.csv")
OUT_SUMMARY_CSV = os.path.join(STEP_DERIV_DIR, "step03_timepoint_summary.csv")


# =====================================================================
# Constants
# =====================================================================

MIN_SEGMENT = 3
FACTOR_COLS = ["pain_factor", "contrast_factor", "sleep_factor"]


# =====================================================================
# Segment filter
# =====================================================================

def segment_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Identify and retain segments of >= MIN_SEGMENT consecutive quarters.

    A segment is a maximal run of consecutive quarters within a
    subject where both ``pain_factor`` and ``sleep_factor`` are
    non-missing. Each retained row gets a ``segment_id`` (integer
    per subject); rows outside retained segments are dropped.

    Parameters
    ----------
    df : DataFrame
        Must contain ``ID``, ``quarter``, ``pain_factor``,
        ``sleep_factor``. Only rows with ``quarter >= 1`` are
        considered; quarter-0 rows are dropped from the output.

    Returns
    -------
    df : DataFrame
        With ``segment_id`` column added, rows restricted to
        retained segments only.
    """
    print("\n  Segment filter (>= 3 consecutive quarters)...")

    # Only quarter >= 1 rows participate
    df = df[df["quarter"] >= 1].copy()

    n_before = len(df)
    n_subj_before = df["ID"].nunique()

    df["has_both"] = df["pain_factor"].notna() & df["sleep_factor"].notna()
    df["segment_id"] = np.nan

    retained_indices = []
    for pid, grp in df.groupby("ID"):
        grp = grp.sort_values("quarter")
        quarters = grp["quarter"].values
        has_both = grp["has_both"].values
        idx = grp.index.values

        # Find maximal runs of consecutive quarters with both present
        seg_id = 0
        run_start = None
        for i in range(len(quarters)):
            if has_both[i]:
                if run_start is None:
                    run_start = i
                # Check if next quarter is consecutive
                if i + 1 < len(quarters) and quarters[i + 1] == quarters[i] + 1 and has_both[i + 1]:
                    continue
                # End of run
                run_end = i
                run_len = run_end - run_start + 1
                if run_len >= MIN_SEGMENT:
                    for j in range(run_start, run_end + 1):
                        df.loc[idx[j], "segment_id"] = seg_id
                        retained_indices.append(idx[j])
                    seg_id += 1
                run_start = None
            else:
                run_start = None

    df = df.loc[retained_indices].copy()
    df["segment_id"] = df["segment_id"].astype(int)
    df = df.drop(columns=["has_both"])

    n_after = len(df)
    n_subj_after = df["ID"].nunique()
    n_dropped_subj = n_subj_before - n_subj_after
    n_dropped_obs = n_before - n_after

    print(f"    Before: {n_subj_before} subjects, {n_before} observations")
    print(f"    After:  {n_subj_after} subjects, {n_after} observations")
    print(f"    Dropped: {n_dropped_subj} subjects, {n_dropped_obs} observations")

    return df


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
# Table 3 — Demographics of the analytic sample
# =====================================================================

def compute_table3(
    df_processed: pd.DataFrame, df_full: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Table 3 demographics for the analytic sample.

    The analytic sample is the set of subjects in ``df_processed``
    (post-segment-filter). Baseline variables are pulled from
    ``df_full`` (which has the quarter-0 rows with baseline data).

    Returns a DataFrame with one row per variable, columns for
    the statistic name, value, and formatting.
    """
    print("\n  Computing Table 3 demographics...")

    analytic_ids = set(df_processed["ID"].unique())
    baseline = df_full[df_full["quarter"] == 0].copy()
    baseline = baseline[baseline["ID"].isin(analytic_ids)]

    n = len(baseline)
    rows = []

    def _add(variable, level, value):
        rows.append({"Variable": variable, "Level": level, "Value": value})

    # Age
    age = baseline["age__s1"].dropna()
    _add("Age, years, mean (SD) [range]", "",
         f"{age.mean():.1f} ({age.std():.1f}) [{age.min():.0f}-{age.max():.0f}]")

    # Sex
    sex = baseline["gender__s1"]
    n_female = int((sex == 2).sum())  # raw coding: 1=male, 2=female
    n_male = int((sex == 1).sum())
    _add("Female sex, N (%)", "", f"{n_female} ({100*n_female/n:.1f})")
    _add("Male sex, N (%)", "", f"{n_male} ({100*n_male/n:.1f})")

    # Race/ethnicity — from multi-select checkbox indicators
    race_map = [
        ("qst_bedside_race_ethno___2__s1", "Black/African American"),
        ("qst_bedside_race_ethno___white__s1", "White/Caucasian"),
        ("qst_bedside_race_ethno___5__s1", "Hispanic/Latino"),
    ]
    other_cols = [
        "qst_bedside_race_ethno___1__s1",
        "qst_bedside_race_ethno___3__s1",
        "qst_bedside_race_ethno___4__s1",
        "qst_bedside_race_ethno___88__s1",
    ]
    for col, label in race_map:
        if col in baseline.columns:
            count = int(baseline[col].fillna(0).astype(int).sum())
            _add("Race/ethnicity, N (%)", label,
                 f"{count} ({100*count/n:.1f})")
    # "Other" = anyone who checked any of the remaining categories
    # and none of the three main ones
    if all(c in baseline.columns for c in other_cols):
        main_any = baseline[[c for c, _ in race_map]].fillna(0).max(axis=1)
        other_any = baseline[other_cols].fillna(0).max(axis=1)
        n_other = int(((other_any == 1) & (main_any == 0)).sum())
        _add("Race/ethnicity, N (%)", "Other",
             f"{n_other} ({100*n_other/n:.1f})")

    # BMI
    bmi = baseline["pe_bmi__s1"].dropna()
    _add("BMI, kg/m2, mean (SD) [range]", "",
         f"{bmi.mean():.1f} ({bmi.std():.1f}) [{bmi.min():.1f}-{bmi.max():.1f}]")

    # WOMAC
    for col, label, scale in [
        ("womac_pain__s1", "WOMAC Pain (0-20)", ""),
        ("womac_stiffness__s1", "WOMAC Stiffness (0-8)", ""),
        ("womac_phys_function__s1", "WOMAC Physical Function (0-68)", ""),
        ("total_womac__s1", "WOMAC Total (0-96)", ""),
    ]:
        vals = baseline[col].dropna()
        _add(f"{label}, mean (SD)", "",
             f"{vals.mean():.1f} ({vals.std():.1f})")

    # PHQ
    for col, label in [
        ("phq_knee_pain_days__s1", "PHQ knee pain days per week"),
        ("phq_percent_pain__s1", "PHQ % waking day in knee pain"),
    ]:
        vals = baseline[col].dropna()
        _add(f"{label}, mean (SD)", "",
             f"{vals.mean():.1f} ({vals.std():.1f})")

    # QST knee pain rating
    vals = baseline["qst_knee_pain_rating__s1"].dropna()
    _add("Knee pain rating (0-100), mean (SD)", "",
         f"{vals.mean():.1f} ({vals.std():.1f})")

    # KL grade
    kl = baseline["KL_Index__s1"].dropna()
    n_kl = len(kl)
    n_missing_kl = n - n_kl
    for grade in sorted(kl.unique()):
        count = int((kl == grade).sum())
        _add("Kellgren-Lawrence grade, N (%)", str(int(grade)),
             f"{count} ({100*count/n_kl:.1f})")
    if n_missing_kl > 0:
        _add("Kellgren-Lawrence grade", "missing", str(n_missing_kl))

    table3 = pd.DataFrame(rows)
    print(f"    {len(rows)} rows generated for N = {n}")

    return table3


# =====================================================================
# Figure 1 — Data availability grid
# =====================================================================

def generate_figure1(
    df_processed: pd.DataFrame,
    df_full: pd.DataFrame,
    out_path: str,
) -> None:
    """Generate Figure 1: data availability grid.

    Shows every subject (row) x quarter (column) with dots colored
    by status:
      - Blue: observed and retained
      - Red: interpolated and retained
      - Grey: available but discarded (segment < 3)
    Horizontal lines connect consecutive quarters within retained
    segments. Subjects are sorted by number of retained points.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    print("\n  Generating Figure 1 (data availability grid)...")

    # Build a full grid of all subjects x all quarters (1..11)
    all_ids = sorted(df_full["ID"].unique())
    quarters = list(range(1, 12))

    # Which (ID, quarter) pairs have factor scores in the full table
    full_q = df_full[df_full["quarter"] >= 1].copy()
    full_q["has_data"] = full_q["pain_factor"].notna() & full_q["sleep_factor"].notna()
    full_q["is_interpolated"] = full_q.get("interpolated", False).fillna(False)

    # Which (ID, quarter) pairs survived the segment filter
    retained_set = set(
        zip(df_processed["ID"], df_processed["quarter"])
    )

    # Build per-subject segment info from df_processed
    person_segments = {}
    for pid, grp in df_processed.groupby("ID"):
        segs = []
        for sid, sgrp in grp.groupby("segment_id"):
            segs.append(sorted(sgrp["quarter"].values))
        person_segments[pid] = segs

    # Sort subjects by number of retained points
    retained_counts = df_processed.groupby("ID").size().reindex(all_ids, fill_value=0)
    id_order = retained_counts.sort_values().index.tolist()

    n_subjects = len(id_order)
    n_excluded = int((retained_counts == 0).sum())
    n_ret_interp = int(
        df_processed["interpolated"].sum()
        if "interpolated" in df_processed.columns else 0
    )

    # Plot
    fig_height = max(6, n_subjects * 0.055 + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    for yi, pid in enumerate(id_order):
        # Draw segment connectors
        for seg_quarters in person_segments.get(pid, []):
            if len(seg_quarters) >= 2:
                ax.plot(
                    [seg_quarters[0], seg_quarters[-1]], [yi, yi],
                    color="#90CAF9", linewidth=2.5, alpha=0.5, zorder=1,
                    solid_capstyle="round",
                )

        # Draw dots
        pdata = full_q[full_q["ID"] == pid]
        for _, row in pdata.iterrows():
            q = row["quarter"]
            if not row["has_data"]:
                continue
            is_retained = (pid, q) in retained_set
            is_interp = bool(row["is_interpolated"])

            if is_retained:
                color = "#D32F2F" if is_interp else "#1565C0"
                alpha = 0.85
                zorder = 3
            else:
                color = "#BDBDBD"
                alpha = 0.5
                zorder = 2
            ax.plot(q, yi, "o", color=color, markersize=5,
                    alpha=alpha, zorder=zorder)

    if n_excluded > 0:
        ax.axhline(
            y=n_excluded - 0.5, color="black", linewidth=0.8,
            linestyle="--", alpha=0.5,
        )
        ax.text(
            0.7, n_excluded - 0.8, f"{n_excluded} excluded",
            fontsize=14, va="top", ha="left", color="#666666",
        )

    ax.set_xlim(0.5, 11.5)
    ax.set_ylim(-1, n_subjects)
    ax.set_xticks(quarters)
    ax.set_xticklabels([f"Q{q}" for q in quarters], fontsize=16)
    ax.set_xlabel("Quarter", fontsize=20)
    ax.set_ylabel(f"Participant (N = {n_subjects})", fontsize=18)
    ax.set_yticks([])

    legend_handles = [
        Line2D([0], [0], marker="o", color="#1565C0", markersize=10,
               linestyle="", label="Observed, retained"),
        Line2D([0], [0], marker="o", color="#BDBDBD", markersize=10,
               linestyle="", alpha=0.5, label="Discarded (segment < 3)"),
        Line2D([0], [0], color="#90CAF9", linewidth=4, alpha=0.5,
               label="Retained segment"),
    ]
    if n_ret_interp > 0:
        legend_handles.insert(
            1,
            Line2D([0], [0], marker="o", color="#D32F2F", markersize=10,
                   linestyle="", label="Interpolated, retained"),
        )
    ax.legend(
        handles=legend_handles, loc="lower right", fontsize=15,
        framealpha=0.9, edgecolor="#CCCCCC",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# =====================================================================
# Top-level entry point
# =====================================================================

def run_step03(verbose: bool = True, refit: bool = False):
    """Full Step 03 pipeline."""
    if verbose:
        print("=" * 70)
        print("STEP 03 — Prepare VARX data")
        print("=" * 70)
        print(f"  Input: {IN_SCORED_CSV}")

    df_full = pd.read_csv(IN_SCORED_CSV)
    if verbose:
        n_subj = df_full["ID"].nunique()
        n_q = (df_full["quarter"] >= 1).sum()
        print(f"  Loaded: {len(df_full):,} rows, {n_subj} subjects")

    # 1. Segment filter
    df = segment_filter(df_full)

    # 2. Within-between decomposition
    df = decompose_within_between(df)

    # 3. Lags + interactions
    df = create_lagged_variables(df)

    # 4. Select output columns
    output_cols = [
        "ID", "quarter", "segment_id",
        "pain_factor", "contrast_factor", "sleep_factor",
        "interpolated",
        "pain_person_mean", "pain_between", "pain_within",
        "contrast_person_mean", "contrast_between", "contrast_within",
        "sleep_person_mean", "sleep_between", "sleep_within",
        "pain_within_lag1", "sleep_within_lag1", "contrast_within_lag1",
        "sleep_x_contrast_lag1", "pain_x_contrast_lag1",
    ]
    # Add baseline columns that downstream steps need
    baseline_cols = ["age__s1", "gender__s1"]
    # Forward-fill baseline from the full table's quarter-0 row
    baseline_q0 = df_full[df_full["quarter"] == 0][["ID"] + baseline_cols]
    baseline_q0 = baseline_q0.rename(columns={
        "age__s1": "Age", "gender__s1": "Sex",
    })
    df = df.merge(baseline_q0, on="ID", how="left")
    output_cols += ["Age", "Sex"]

    output_cols = [c for c in output_cols if c in df.columns]
    df_out = df[output_cols].copy()

    # Save processed data (derivative)
    os.makedirs(DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    df_out.to_csv(OUT_PROCESSED_CSV, index=False)
    if verbose:
        print(f"\n  Saved: {OUT_PROCESSED_CSV}")
        print(f"    Shape: {df_out.shape}")
        print(f"    Columns: {output_cols}")

    # 5. Timepoint summary (derivative)
    summary = compute_timepoint_summary(df_out)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)
    if verbose:
        print(f"  Saved: {OUT_SUMMARY_CSV}")

    # 6. Results: Table 3 + Figure 1 + text numbers
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    table3 = compute_table3(df_out, df_full)
    table3.to_csv(OUT_TABLE3_CSV, index=False)
    if verbose:
        print(f"  Saved: {OUT_TABLE3_CSV}")

    generate_figure1(df_out, df_full, OUT_FIGURE1)

    # Numbers stated in the manuscript text (Results §3.1, Figure 1
    # caption, Table 3 note, Abstract, Methods §2.1)
    usable = df_out["pain_within_lag1"].notna() & df_out["sleep_within_lag1"].notna()
    per_person = df_out.loc[usable].groupby("ID").size()
    n_excluded = df_full["ID"].nunique() - df_out["ID"].nunique()
    n_interp_retained = int(df_out["interpolated"].sum()) if "interpolated" in df_out.columns else 0

    text_numbers = pd.DataFrame([
        {"metric": "n_parent_study", "value": df_full["ID"].nunique()},
        {"metric": "n_analytic_sample", "value": df_out["ID"].nunique()},
        {"metric": "n_excluded", "value": n_excluded},
        {"metric": "n_retained_points", "value": len(df_out)},
        {"metric": "n_interpolated_retained", "value": n_interp_retained},
        {"metric": "n_lag_transitions", "value": int(usable.sum())},
        {"metric": "median_lags_per_person", "value": int(per_person.median())},
        {"metric": "min_lags_per_person", "value": int(per_person.min())},
        {"metric": "max_lags_per_person", "value": int(per_person.max())},
    ])
    OUT_TEXT_CSV = os.path.join(STEP_RESULTS_DIR, "step03_text_numbers.csv")
    text_numbers.to_csv(OUT_TEXT_CSV, index=False)
    if verbose:
        print(f"  Saved: {OUT_TEXT_CSV}")

    generate_text_paragraphs(verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 03 COMPLETE")
        print("=" * 70)


def generate_text_paragraphs(verbose: bool = True) -> None:
    """Generate step03_text.md with the manuscript paragraphs that cite
    analytic-sample counts, populated from step03_text_numbers.csv.

    Covers: Results section 2.1 (analytic sample description) and
    the Figure 1 caption.
    """
    OUT_TEXT_MD = os.path.join(STEP_RESULTS_DIR, "step03_text.md")
    IN_TEXT_CSV = os.path.join(STEP_RESULTS_DIR, "step03_text_numbers.csv")

    if not os.path.exists(IN_TEXT_CSV):
        if verbose:
            print("  SKIP: step03_text_numbers.csv not found — run step03 first")
        return

    if verbose:
        print("  Generating step03_text.md ...")

    df = pd.read_csv(IN_TEXT_CSV)
    v = dict(zip(df["metric"], df["value"].astype(str)))

    n_parent = v["n_parent_study"]
    n_analytic = v["n_analytic_sample"]
    n_excluded = v["n_excluded"]
    n_retained = v["n_retained_points"]
    n_interp = v["n_interpolated_retained"]
    n_lags = v["n_lag_transitions"]
    med_lags = v["median_lags_per_person"]
    min_lags = v["min_lags_per_person"]
    max_lags = v["max_lags_per_person"]

    para_sample = (
        f"Of the {n_parent} participants in the parent study, "
        f"{n_excluded} were excluded because they lacked three or more "
        f"consecutive quarters with both pain and sleep data, yielding an "
        f"analytic sample of $N$ = {n_analytic}. Across these participants, "
        f"{n_retained} person-quarter observations were retained within "
        f"valid segments (\u22653 consecutive quarters), of which "
        f"{n_interp} ({float(n_interp)/float(n_retained)*100:.1f}%) were "
        f"single-gap interpolated values. After applying the one-quarter "
        f"lag, {n_lags} usable lag transitions remained "
        f"(median = {med_lags} per person, range = {min_lags}\u2013{max_lags})."
    )

    fig1_caption = (
        f"**Figure 1.** Data availability grid (participants x quarters). "
        f"Blue dots indicate observed retained data points; red dots indicate "
        f"retained points for which scores were interpolated "
        f"({n_interp} of {n_retained} retained points); grey dots indicate "
        f"observations discarded due to segment length < 3. Horizontal lines "
        f"connect consecutive quarters within retained segments. Participants "
        f"are sorted by number of retained points (bottom = fewest). "
        f"Participants below the dashed line ($N$ = {n_excluded}) were excluded "
        f"for lacking any segment of three or more consecutive quarters."
    )

    text = (
        "## Results > 2.1 Analytic sample\n"
        "### Paragraph (analytic sample description)\n\n"
        f"{para_sample}\n\n"
        "### Figure 1 caption\n\n"
        f"{fig1_caption}\n"
    )

    with open(OUT_TEXT_MD, "w") as f:
        f.write(text)

    if verbose:
        print(f"    Saved: {OUT_TEXT_MD}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 03 — segment filter, decomposition, lags, "
                    "Figure 1, Table 3, timepoint summary."
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
    run_step03(verbose=not args.quiet, refit=args.refit)


if __name__ == "__main__":
    main()
