"""
Step 03 — Data curation: who is in the analytic sample.
======================================================================

Input:  derivatives/step01_factor_analysis/step01_scored_long.csv
Output: derivatives/step03_curation/step03_curated_long.csv
        results/step03_curation/step03_figure1.png
        results/step03_curation/step03_table3_demographics.csv

SPLIT OUT OF the old prepare_varx_data on 11 Aug 2026. That step did two unrelated
jobs -- deciding who is in the study, and building the VARX frame -- and three steps
that only needed the FIRST (raw descriptives, contrast validation, sleep-measure
correlates) had to reach forward to a step that had not run. On a clean checkout the
numeric order failed; on a rerun it silently used the previous run's sample, which is
how the ANOVA F drifted before the determinism fix.

Curation is one operation: keep the maximal runs of consecutive quarters where both
pain and sleep are scored, discard runs shorter than MIN_SEGMENT, and report what that
left. Everything downstream that asks "who is in the analytic sample" reads this step.
The VARX frame -- centering and lags -- is built in step 07, immediately before the fit
that consumes it.

Operations, in order:

  1. Segment filter -- maximal runs of consecutive quarters with both factors scored;
     assign segment_id; discard segments shorter than 3 quarters.
  2. Baseline merge -- Age and Sex forward-filled from the quarter-0 row.
  3. Figure 1 -- data-availability grid: observed, interpolated, discarded.
  4. Table 3 -- demographics of the participants surviving the filter.

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from registry import write_numbers  # noqa: E402

warnings.filterwarnings("ignore")


# =====================================================================
# Paths
# =====================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repo root
DATA_DIR = os.path.join(ROOT, "data")

DERIV_DIR = os.path.join(ROOT, "derivatives")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step03_curation")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step03_curation")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

IN_SCORED_CSV = os.path.join(DERIV_DIR, "step01_factor_analysis", "step01_scored_long.csv")

#: The analytic sample. Every step that asks "who is in the study" reads THIS file.
OUT_CURATED_CSV = os.path.join(STEP_DERIV_DIR, "step03_curated_long.csv")
OUT_FIGURE1 = os.path.join(STEP_RESULTS_DIR, "step03_figure1.png")
OUT_TABLE3_CSV = os.path.join(STEP_RESULTS_DIR, "step03_table3_demographics.csv")


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

    # Race group — Race__s1 (= Race_Group from screening form, 1=NHB, 2=NHW)
    if "Race__s1" in baseline.columns:
        race = baseline["Race__s1"]
        n_nhb = int((race == 1).sum())
        n_nhw = int((race == 2).sum())
        _add("Race group, N (%)", "Non-Hispanic Black",
             f"{n_nhb} ({100*n_nhb/n:.1f})")
        _add("Race group, N (%)", "Non-Hispanic White",
             f"{n_nhw} ({100*n_nhw/n:.1f})")

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
# Pipeline
# =====================================================================

def run_step03(verbose: bool = True, refit: bool = False):
    """Curate the analytic sample and report it."""
    if verbose:
        print("=" * 70)
        print("STEP 03 — Data curation")
        print("=" * 70)
        print(f"  Input: {IN_SCORED_CSV}")

    df_full = pd.read_csv(IN_SCORED_CSV, float_precision="round_trip")
    if verbose:
        n_subj = df_full["ID"].nunique()
        print(f"  Loaded: {len(df_full):,} rows, {n_subj} subjects")

    df = segment_filter(df_full)

    # Baseline columns downstream steps need, forward-filled from quarter 0.
    baseline_cols = ["age__s1", "gender__s1"]
    baseline_q0 = df_full[df_full["quarter"] == 0][["ID"] + baseline_cols]
    baseline_q0 = baseline_q0.rename(columns={"age__s1": "Age", "gender__s1": "Sex"})
    df = df.merge(baseline_q0, on="ID", how="left")

    keep = ["ID", "quarter", "segment_id",
            "pain_factor", "contrast_factor", "sleep_factor",
            "interpolated",
            # carried so the per-score interpolation counts can be taken AFTER curation
            "interpolated_pain_factor", "interpolated_contrast_factor",
            "interpolated_sleep_factor",
            "Age", "Sex"]
    df_out = df[[c for c in keep if c in df.columns]].copy()

    df_out.to_csv(OUT_CURATED_CSV, index=False)
    if verbose:
        print(f"\n  Saved: {OUT_CURATED_CSV}")
        print(f"    Shape: {df_out.shape}")

    table3 = compute_table3(df_out, df_full)
    table3.to_csv(OUT_TABLE3_CSV, index=False)
    if verbose:
        print(f"  Saved: {OUT_TABLE3_CSV}")

    generate_figure1(df_out, df_full, OUT_FIGURE1)
    if verbose:
        print(f"  Saved: {OUT_FIGURE1}")

    # The curation counts the Results state directly. They were previously readable only
    # from step 07's timepoint summary, which is a different step describing a different
    # frame -- so "229 participants" had no named value of its own to be checked against.
    n_before = int(df_full["ID"].nunique())
    n_after = int(df_out["ID"].nunique())
    nums = {
        "n_participants_scored": n_before,
        "n_participants_retained": n_after,
        "n_participants_excluded": n_before - n_after,
        "n_person_quarters_retained": int(len(df_out)),
        "n_segments": int(df_out.groupby(["ID", "segment_id"]).ngroups),
        "min_segment_length": int(MIN_SEGMENT),
    }
    if "interpolated" in df_out.columns:
        nums["n_interpolated_rows_retained"] = int(df_out["interpolated"].sum())
    # Per-score cell counts, RETAINED sample only. The Results give all three ("105 pain
    # intensity, 105 knee-body contrast, and 113 self-reported sleep quality") and their
    # total. Step 01 counts across the whole cohort and gets five more per score, because
    # curation has not happened yet there -- only this step knows who was retained.
    per_score = {}
    for col, label in (("pain_factor", "pain"),
                       ("contrast_factor", "contrast"),
                       ("sleep_factor", "sleep")):
        flag = f"interpolated_{col}"
        if flag in df_out.columns:
            per_score[label] = int(df_out[flag].sum())
            nums[f"n_interpolated_cells_retained_{label}"] = per_score[label]
    if per_score:
        nums["n_interpolated_cells_retained"] = int(sum(per_score.values()))
    path = write_numbers(STEP_RESULTS_DIR, nums, prefix="step03")
    if verbose:
        print(f"  Wrote {len(nums)} numbers: {path}")
        print("\n" + "=" * 70)
        print("STEP 03 COMPLETE")
        print("=" * 70)
    return df_out


def main():
    ap = argparse.ArgumentParser(description="Step 03 — Data curation.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()
    run_step03(verbose=not args.quiet, refit=args.refit)


if __name__ == "__main__":
    main()
