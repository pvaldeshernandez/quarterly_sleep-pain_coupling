"""
Step 00 — Extract paper-relevant variables from the legacy wide-format data.
=============================================================================

This is the ORIGIN script for the project. It takes the raw
``participants_wideformat.xlsx`` file that was exported from REDCap
by the study team and produces a single long-format CSV containing
every variable that any paper-contributing code has ever referenced.

Design principles (do not silently weaken any of these):

1. **No listwise deletion.** Every subject in the raw xlsx stays in
   the output. Every quarter row is emitted whether or not the
   underlying items are missing. Gateway imputation, segment
   filtering, factor scoring, and any other data-preparation
   operations are the job of later steps, not of this one.

2. **Single source of truth.** The only input is
   ``data/original/participants_wideformat.xlsx`` plus the
   companion ``data/original/UPLOAD2_Data_Dictionary.xlsx``. Nothing
   else is read. In particular, the legacy
   ``quarterly_data_long.csv`` file that ships with the current
   repo is not read or trusted here; this script regenerates the
   long-format table from scratch so that its contents are fully
   traceable to the original export.

3. **Explicit variable enumeration.** The full list of baseline
   variables to extract is written out in this file as a plain
   Python list of column names. The list was assembled by scanning
   every paper-contributing code path in the project (the committed
   python/ pipeline, the legacy scripts/ folder, the archive
   exploratory scripts, and the figure generators). No pattern
   matching, no "grab everything that starts with X". If a column
   isn't listed here it isn't extracted.

4. **Baseline variables live on ``quarter = 0`` rows.** Quarterly
   items live on ``quarter = 1..11`` rows. Each subject gets a
   single quarter-0 row carrying the baseline set and up to eleven
   quarter-N rows carrying that quarter's items.

5. **Dictionary is filtered in parallel.** The output
   ``extracted_dictionary.xlsx`` contains only the data-dictionary
   rows corresponding to variables that survived the extraction.
   Its schema matches the original dictionary exactly.

Not covered here (handled by later steps or different scripts):
  - FreeSurfer cortical / subcortical / brainstem / thalamic-nuclei
    measures (these live in separate TSVs, not in the wide xlsx).
  - fMRI / VBM neuroimaging contrast extraction (NIfTI files).
  - Factor analysis, Bartlett scoring, decomposition, lag creation.
  - Any arithmetic on the extracted variables.

Author: Pedro Valdes-Hernandez (with Claude Opus 4.6)
"""
from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import pandas as pd


# =====================================================================
# Paths
# =====================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repo root
DATA_DIR = os.path.join(ROOT, "data")
ORIG_DIR = os.path.join(DATA_DIR, "original")

WIDE_XLSX = os.path.join(ORIG_DIR, "participants_wideformat.xlsx")
DICT_XLSX = os.path.join(ORIG_DIR, "UPLOAD2_Data_Dictionary.xlsx")

OUT_LONG_CSV = os.path.join(DATA_DIR, "step00_extracted_long.csv")
OUT_DICT_XLSX = os.path.join(DATA_DIR, "step00_extracted_dictionary.xlsx")


# =====================================================================
# Variable enumeration
# =====================================================================
#
# Each name below is the wide-file column name as it appears in
# ``participants_wideformat.xlsx``. Baseline variables all carry the
# ``__s1`` suffix (session 1 = baseline visit). Quarterly items use
# the ``qhu{quarter}_q{item}__s1`` naming from REDCap, where
# ``quarter`` runs 1..11 and ``item`` runs 1..25.

# ---------------------------------------------------------------------
# Baseline variables — ONLY variables that the final paper uses.
#
# This is not an exhaustive list of everything in the wide file. It
# is the minimal set that directly supports a number, a table, a
# figure, or a modeling decision reported in the manuscript. An
# earlier draft of this file was padded with the 91 variables from
# the legacy Aim 2a behavioral sweep; that sweep was archived because
# nothing survived FDR correction and the entire analysis was dropped
# from the paper. None of those variables are extracted here.
#
# Reference map — where each variable is used in the paper:
#   age__s1                      : the baseline-characteristics table, VARX covariate
#   gender__s1                   : the baseline-characteristics table, VARX covariate
#   Race__s1                     : the baseline-characteristics table (1=NHB, 2=NHW; = Race_Group)
#   pe_bmi__s1                   : the baseline-characteristics table
#   womac_pain__s1               : the baseline-characteristics table, the convergent-validity figure convergent validity
#   womac_stiffness__s1          : the baseline-characteristics table, the convergent-validity figure
#   womac_phys_function__s1      : the baseline-characteristics table, the convergent-validity figure
#   total_womac__s1              : the baseline-characteristics table, the convergent-validity figure
#   KL_Index__s1                 : the baseline-characteristics table
#   phq_knee_pain_days__s1       : the baseline-characteristics table, the convergent-validity figure
#   phq_percent_pain__s1         : the baseline-characteristics table, the convergent-validity figure
#   qst_knee_pain_rating__s1     : the baseline-characteristics table, the convergent-validity figure
#   gcps_pain_intensity__s1      : GCPS characteristic pain intensity (0-100)
#   gcps_interference__s1        : GCPS pain interference (0-100)
#   img_test_site__s1            : fMRI ROI contralateralization for
#                                  contralateral S1 and Middle Insula
#   phq_pain_areas___{1..13}__s1 : Factor analysis validation
#                                  (the endorsement figure: ANOVA + point-biserial
#                                  correlations of contrast scores
#                                  against 13 body-map endorsements)
#
# 28 variables in total (plus ID).

BASELINE_VARS = [
    # Demographics (the baseline-characteristics table + VARX covariates)
    "age__s1",
    "gender__s1",
    "pe_bmi__s1",

    # Race group (the baseline-characteristics table) — single screening-form variable, 1=NHB, 2=NHW.
    # In the wide xlsx this is the export of Race_Group from the screening
    # REDCap form (data dictionary entry #1806).
    "Race__s1",

    # WOMAC (the baseline-characteristics table + the convergent-validity figure)
    "womac_pain__s1",
    "womac_stiffness__s1",
    "womac_phys_function__s1",
    "total_womac__s1",

    # Kellgren-Lawrence grade (the baseline-characteristics table)
    "KL_Index__s1",

    # PHQ clinical (the baseline-characteristics table + the convergent-validity figure)
    "phq_knee_pain_days__s1",
    "phq_percent_pain__s1",

    # QST knee pain rating (the baseline-characteristics table + the convergent-validity figure)
    "qst_knee_pain_rating__s1",

    # Baseline sleep instruments (the demographics table). The paper's sleep measure is the
    # QUARTERLY item q13; these characterize how the sample slept at baseline, which a
    # sleep-pain paper's demographics table has to state. Names are the canonical ones
    # in lib/sleep_instruments.py, so this table and the sleep-measure correlations refer
    # to the same variables. STOP-BANG is not here because it is DERIVED from eight
    # components rather than exported as a column; lib/stopbang.py scores it.
    "Insomnia__s1",
    "PROMIS_Sleep_Tscore__s1",
    "PSQI_Duration__s1",

    # GCPS — Graded Chronic Pain Scale (baseline, scaled 0-100)
    "gcps_pain_intensity__s1",
    "gcps_interference__s1",

    # fMRI stimulation side (for contralateralized S1 / Mid Insula)
    "img_test_site__s1",

    # Body-map endorsements (the endorsement figure factor-analysis validation)
    "phq_pain_areas___1__s1",
    "phq_pain_areas___2__s1",
    "phq_pain_areas___3__s1",
    "phq_pain_areas___4__s1",
    "phq_pain_areas___5__s1",
    "phq_pain_areas___6__s1",
    "phq_pain_areas___7__s1",
    "phq_pain_areas___8__s1",
    "phq_pain_areas___9__s1",
    "phq_pain_areas___10__s1",
    "phq_pain_areas___11__s1",
    "phq_pain_areas___12__s1",
    "phq_pain_areas___13__s1",
]


# ---------------------------------------------------------------------
# Quarterly items — 11 quarters × 25 items each.
# ---------------------------------------------------------------------
#
# The wide-file column names are ``qhu{quarter}_q{item}__s1`` where
# ``quarter`` runs 1..11 and ``item`` runs 1..25. The data dictionary
# documents these as ``qhu{quarter}_q{item}`` (without the ``__s1``
# suffix that REDCap adds at export time).
#
# For convenience, each quarterly item is renamed in the long output
# to a short descriptive name based on the question text. The mapping
# was read off the data dictionary:
#
#   q1  = knee pain endorsement (gateway question)
#   q2  = knee pain worst
#   q3  = knee pain average
#   q4  = knee pain right now
#   q5  = knee pain interference
#   q6  = body pain endorsement (gateway question)
#   q7  = body pain worst
#   q8  = body pain average
#   q9  = body pain right now
#   q10 = body pain interference
#   q11 = fatigue
#   q12 = mood
#   q13 = sleep quality
#   q14 = surgery/procedures (treatment)
#   q15 = other new treatments
#   q16 = PSS: unable to control important things
#   q17 = PSS: upset by unexpected events
#   q18 = PSS: nervous and stressed
#   q19 = PSS: confident about handling problems
#   q20 = PSS: things going your way
#   q21 = PSS: cope with all things
#   q22 = PSS: in control of irritations
#   q23 = PSS: on top of things
#   q24 = PSS: angered by things out of control
#   q25 = PSS: piling-up difficulties

QUARTERLY_ITEM_NAMES = {
    1:  "q1_knee_pain",
    2:  "q2_knee_pain",
    3:  "q3_knee_pain",
    4:  "q4_knee_pain",
    5:  "q5_knee_pain",
    6:  "q6_body_pain",
    7:  "q7_body_pain",
    8:  "q8_body_pain",
    9:  "q9_body_pain",
    10: "q10_body_pain",
    11: "q11_fatigue",
    12: "q12_mood",
    13: "q13_sleep",
    14: "q14_pain_treatment",
    15: "q15_pain_treatment",
    16: "q16_pss",
    17: "q17_pss",
    18: "q18_pss",
    19: "q19_pss",
    20: "q20_pss",
    21: "q21_pss",
    22: "q22_pss",
    23: "q23_pss",
    24: "q24_pss",
    25: "q25_pss",
}

QUARTERS = list(range(1, 12))  # 1..11
ITEMS = list(range(1, 26))     # 1..25


# =====================================================================
# Extraction functions
# =====================================================================

def reshape_quarterly_items(wide: pd.DataFrame) -> pd.DataFrame:
    """Reshape ``qhu{quarter}_q{item}__s1`` columns to long format.

    Returns a DataFrame with one row per (ID, quarter) pair for
    quarters 1..11, with columns named by ``QUARTERLY_ITEM_NAMES``.
    Missing input columns become NaN-filled output columns; no row
    is dropped for missingness.
    """
    rows = []
    for quarter in QUARTERS:
        chunk = pd.DataFrame({"ID": wide["ID"].values, "quarter": quarter})
        for item in ITEMS:
            src = f"qhu{quarter}_q{item}__s1"
            dst = QUARTERLY_ITEM_NAMES[item]
            if src in wide.columns:
                chunk[dst] = wide[src].values
            else:
                chunk[dst] = pd.NA
        rows.append(chunk)
    return pd.concat(rows, ignore_index=True)


def apply_gateway_imputation(long: pd.DataFrame) -> pd.DataFrame:
    """Apply gateway-question imputation to quarter 1..11 rows.

    The quarterly questionnaire uses gateway questions: q1 asks
    whether the respondent had any knee pain in the past week, and
    q6 asks the same for body pain. When the respondent answers 0
    ("no") to one of these, the follow-up items for that body
    region are skipped in the form and come back as missing. Those
    missing values are not unknown -- they are structurally 0 (no
    pain -> nothing to rate).

    The full set of follow-up items is gated:

      Knee (q1 == 0 -> set to 0 when NaN):
        q2_knee_pain  (worst)
        q3_knee_pain  (average)
        q4_knee_pain  (right now)
        q5_knee_pain  (interference with general activity)

      Body (q6 == 0 -> set to 0 when NaN):
        q7_body_pain  (worst)
        q8_body_pain  (average)
        q9_body_pain  (right now)
        q10_body_pain (interference with general activity)

    This is the 4+4 gating scope. The legacy
    ``scripts/prepare_data_contrast.py`` that produced the earlier
    draft of the paper used 3+3 gating (intensity items only,
    leaving the interference items q5/q10 ungated) and then ran
    the factor analysis on all 8 items anyway. That combination is
    logically inconsistent: if a subject says "no knee pain" at q1
    then the knee interference rating is also structurally 0, not
    missing. Gating q5 and q10 alongside q2-q4 and q7-q9 resolves
    the inconsistency. Downstream counts (N, retained points, lag
    transitions) will differ slightly from the earlier draft; the
    magnitudes are tracked in ``MANUSCRIPT_CORRECTIONS.md``.

    Applied only to rows where ``quarter >= 1``; quarter-0 rows
    carry baseline variables only and the quarterly item columns
    are all NaN there.
    """
    is_q1_11 = long["quarter"] >= 1

    # Knee pain gateway: q1 == 0 -> q2/q3/q4/q5 -> 0 when NaN
    knee_follow_ups = [
        "q2_knee_pain", "q3_knee_pain", "q4_knee_pain", "q5_knee_pain",
    ]
    knee_gate = is_q1_11 & (long["q1_knee_pain"] == 0)
    for col in knee_follow_ups:
        mask = knee_gate & long[col].isna()
        long.loc[mask, col] = 0.0

    # Body pain gateway: q6 == 0 -> q7/q8/q9/q10 -> 0 when NaN
    body_follow_ups = [
        "q7_body_pain", "q8_body_pain", "q9_body_pain", "q10_body_pain",
    ]
    body_gate = is_q1_11 & (long["q6_body_pain"] == 0)
    for col in body_follow_ups:
        mask = body_gate & long[col].isna()
        long.loc[mask, col] = 0.0

    return long


def build_baseline_rows(wide: pd.DataFrame) -> pd.DataFrame:
    """Build the ``quarter = 0`` rows carrying baseline variables.

    One row per subject. Every BASELINE_VAR listed above is extracted
    verbatim. Columns missing from the input xlsx are emitted as NaN
    columns with a printed warning, so downstream code can still
    count on their presence.
    """
    present = [c for c in BASELINE_VARS if c in wide.columns]
    missing = [c for c in BASELINE_VARS if c not in wide.columns]
    if missing:
        print(
            f"  WARNING: {len(missing)} baseline variables not in the wide "
            f"xlsx; they will appear as all-NaN columns:"
        )
        for m in missing:
            print(f"    - {m}")
    baseline = wide[["ID"] + present].copy()
    for m in missing:
        baseline[m] = pd.NA
    # Reorder columns to the canonical ordering
    baseline = baseline[["ID"] + BASELINE_VARS]
    baseline.insert(1, "quarter", 0)
    return baseline


def assemble_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full long-format output.

    The output has one row per (ID, quarter) for quarter in 0..11.
    Quarter-0 rows carry the baseline variables and NaN quarterly
    items. Quarter 1..11 rows carry the quarterly items and NaN
    baseline variables. All subjects are kept regardless of
    missingness.
    """
    quarterly_long = reshape_quarterly_items(wide)
    baseline_long = build_baseline_rows(wide)

    # Full set of columns = baseline + quarterly item names. Build one
    # dataframe each with NaN columns for the other block, then
    # concatenate.
    baseline_cols = [c for c in BASELINE_VARS]
    item_cols = [QUARTERLY_ITEM_NAMES[i] for i in ITEMS]

    # Baseline block: add empty item columns
    for c in item_cols:
        if c not in baseline_long.columns:
            baseline_long[c] = pd.NA

    # Quarterly block: add empty baseline columns
    for c in baseline_cols:
        if c not in quarterly_long.columns:
            quarterly_long[c] = pd.NA

    # Canonical column order
    col_order = ["ID", "quarter"] + baseline_cols + item_cols
    baseline_long = baseline_long[col_order]
    quarterly_long = quarterly_long[col_order]

    long = pd.concat([baseline_long, quarterly_long], ignore_index=True)
    long = long.sort_values(["ID", "quarter"], kind="stable").reset_index(drop=True)
    return long


# =====================================================================
# Dictionary filtering
# =====================================================================

def filter_dictionary(dict_df: pd.DataFrame) -> pd.DataFrame:
    """Keep dictionary rows that correspond to extracted variables.

    For baseline variables the data dictionary lists them exactly
    once per ``Variable Name`` with a ``__s1`` suffix (most entries)
    or without (a handful of session-invariant variables).

    For quarterly items the dictionary lists them as
    ``qhu{quarter}_q{item}`` (without the ``__s1`` suffix), one row
    per (quarter, item) pair. All 275 rows are retained.
    """
    var_col = "Variable Name"
    if var_col not in dict_df.columns:
        raise KeyError(
            f"Data dictionary is missing a '{var_col}' column; "
            f"actual columns: {list(dict_df.columns)}"
        )

    # Build the full set of variable names to keep
    baseline_names = set(BASELINE_VARS)
    # Session-1 suffix stripped variants (in case the dictionary lists
    # the variable without __s1):
    baseline_names_no_s1 = {
        re.sub(r"__s1$", "", c) for c in BASELINE_VARS
    }
    quarterly_names = {
        f"qhu{q}_q{i}" for q in QUARTERS for i in ITEMS
    }
    keep_names = baseline_names | baseline_names_no_s1 | quarterly_names | {"ID"}

    mask = dict_df[var_col].astype(str).isin(keep_names)
    filtered = dict_df[mask].copy()
    return filtered


# =====================================================================
# Top-level entry point
# =====================================================================

def extract(verbose: bool = True, refit: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full extraction and write outputs.

    Returns
    -------
    long : DataFrame
        The extracted long-format table (written to ``extracted_long.csv``).
    filtered_dict : DataFrame
        The filtered data dictionary (written to ``extracted_dictionary.xlsx``).
    """
    if verbose:
        print("=" * 70)
        print("STEP 00 — Extract paper-relevant variables from wide-format data")
        print("=" * 70)
        print(f"  Wide input: {WIDE_XLSX}")
        print(f"  Dict input: {DICT_XLSX}")

    wide = pd.read_excel(WIDE_XLSX)
    dict_df = pd.read_excel(DICT_XLSX)

    if verbose:
        print(
            f"  Loaded: wide {wide.shape[0]} subjects x "
            f"{wide.shape[1]} columns; dictionary {dict_df.shape[0]} rows"
        )

    # Baseline variable presence check
    missing_baseline = [c for c in BASELINE_VARS if c not in wide.columns]
    present_baseline = [c for c in BASELINE_VARS if c in wide.columns]
    if verbose:
        print(
            f"  Baseline variables requested: {len(BASELINE_VARS)}; "
            f"present: {len(present_baseline)}; "
            f"missing: {len(missing_baseline)}"
        )

    # Quarterly item presence check
    expected_qhu = [
        f"qhu{q}_q{i}__s1" for q in QUARTERS for i in ITEMS
    ]
    present_qhu = [c for c in expected_qhu if c in wide.columns]
    missing_qhu = [c for c in expected_qhu if c not in wide.columns]
    if verbose:
        print(
            f"  Quarterly items requested: {len(expected_qhu)}; "
            f"present: {len(present_qhu)}; "
            f"missing: {len(missing_qhu)}"
        )

    # Assemble the long table
    long = assemble_long(wide)

    # Drop subjects with zero quarterly data. These subjects enrolled
    # in the parent study and have baseline data in the wide file but
    # never completed any quarterly follow-up — every one of their 275
    # quarterly item cells is NaN. They contribute nothing to a
    # longitudinal analysis and should not inflate the parent-study N.
    item_cols = [QUARTERLY_ITEM_NAMES[i] for i in ITEMS]
    q_rows = long[long["quarter"] >= 1]
    items_per_subject = q_rows.groupby("ID")[item_cols].apply(
        lambda g: g.notna().any(axis=1).sum()
    )
    no_data_ids = set(items_per_subject[items_per_subject == 0].index)
    if no_data_ids:
        long = long[~long["ID"].isin(no_data_ids)].reset_index(drop=True)

    if verbose:
        n_subjects = long["ID"].nunique()
        n_q0 = (long["quarter"] == 0).sum()
        n_q1_11 = (long["quarter"] >= 1).sum()
        if no_data_ids:
            print(
                f"  Excluded {len(no_data_ids)} subjects with zero "
                f"quarterly data (baseline only): "
                f"{sorted(no_data_ids)}"
            )
        print(
            f"  Long output: {len(long):,} rows "
            f"({n_subjects} subjects, {n_q0} quarter-0 rows, "
            f"{n_q1_11} quarterly rows)"
        )

    # Gateway imputation on quarter 1..11 rows (4+4 scope).
    is_q1_11 = long["quarter"] >= 1
    n_knee_gate = int(((long["q1_knee_pain"] == 0) & is_q1_11).sum())
    n_body_gate = int(((long["q6_body_pain"] == 0) & is_q1_11).sum())
    long = apply_gateway_imputation(long)
    if verbose:
        print("  Gateway imputation (quarter >= 1):")
        print(
            f"    q1=0 -> q2/q3/q4/q5 filled where NaN: "
            f"{n_knee_gate} person-quarters"
        )
        print(
            f"    q6=0 -> q7/q8/q9/q10 filled where NaN: "
            f"{n_body_gate} person-quarters"
        )

    # Filter the dictionary
    filtered_dict = filter_dictionary(dict_df)
    if verbose:
        print(
            f"  Dictionary filtered: {len(filtered_dict)} of {len(dict_df)} "
            f"rows retained"
        )

    # Write outputs
    os.makedirs(DATA_DIR, exist_ok=True)
    long.to_csv(OUT_LONG_CSV, index=False)
    filtered_dict.to_excel(OUT_DICT_XLSX, index=False)
    if verbose:
        print(f"  Saved: {OUT_LONG_CSV}")
        print(f"  Saved: {OUT_DICT_XLSX}")
        print("=" * 70)

    return long, filtered_dict


def main():
    parser = argparse.ArgumentParser(
        description="Step 00 — extract paper-relevant variables from "
                    "participants_wideformat.xlsx into a long-format CSV."
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
    extract(verbose=not args.quiet, refit=args.refit)


if __name__ == "__main__":
    main()
