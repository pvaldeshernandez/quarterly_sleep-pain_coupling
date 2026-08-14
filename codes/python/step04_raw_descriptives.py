#!/usr/bin/env python3
"""
Step 04 — Raw quarterly descriptive statistics.
======================================================================

The raw data behind the model: per-item pooled descriptives, the same items broken
out by quarter, the variance decomposition (ICC / between-SD / within-SD) of both the
raw items and the three factor scores, and the zero-pain floor structure at both the
person-quarter and the participant level.

Input:  data/step00_extracted_long.csv                          (READ ONLY — raw q2-q13 items)
        derivatives/step03_curation/step03_curated_long.csv  (analytic row set + factors)
Output:
  results/step04_raw_descriptives/
    step04_item_descriptives.csv       — per item, pooled over the described person-quarters
    step04_by_quarter.csv              — per item per quarter
    step04_variance_decomposition.csv  — ICC, between-SD, within-SD, within-person SD spread
    numbers.json                       — every scalar this step computed
  derivatives/step04_raw_descriptives/
    step04_zero_pain.csv               — zero/no-pain proportions (audit trail)

The CSVs are named for their CONTENT, not for the supplement table they currently feed.
As of this writing the binding is: item_descriptives -> Panel A, by_quarter -> Panel B,
variance_decomposition -> Panel C of the raw-descriptives table (the raw-descriptives table). Table and panel numbering is authored
prose and moves; filenames must not follow it.

NO PROSE, NO FIGURE.
  - Prose: table notes, panel captions and the section text are authored by hand from
    numbers.json. Nothing here writes a `*_text.md`. No manuscript number is ever read
    off stdout — the stdout block is a human sanity check, numbers.json is the source.
  - Figure: a mean-plus-band trajectory plot over quarters shows BETWEEN-person spread
    at each timepoint, not the within-person variability the coupling model consumes, so
    it does not answer the question it appears to answer. The per-quarter table plus the
    ICC / within-SD numbers answer it properly. This is a settled decision, defended in
    reply R2.20; step04 deliberately emits no PNG.

This step fits no model, so it never touches lib.coupling_model.run_fit.

Usage:
    python step04_raw_descriptives.py [--refit] [--quiet]

Author: Pedro Valdes-Hernandez (with Claude Opus 5)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

# =====================================================================
# Paths
# =====================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repo root

DERIV_DIR = os.path.join(ROOT, "derivatives")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step04_raw_descriptives")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step04_raw_descriptives")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

#: raw items, one row per (ID, quarter), quarter 0..11. Written by step00; read only.
IN_LONG_CSV = os.path.join(ROOT, "data", "step00_extracted_long.csv")
#: the VARX-ready frame; supplies the analytic ID set, the retained (ID, quarter)
#: pairs, and pain_factor / contrast_factor / sleep_factor.
IN_PROCESSED_CSV = os.path.join(DERIV_DIR, "step03_curation",
                                "step03_curated_long.csv")

OUT_ITEM_CSV = os.path.join(STEP_RESULTS_DIR, "step04_item_descriptives.csv")
OUT_BY_QUARTER_CSV = os.path.join(STEP_RESULTS_DIR, "step04_by_quarter.csv")
OUT_VARCOMP_CSV = os.path.join(STEP_RESULTS_DIR, "step04_variance_decomposition.csv")
OUT_ZERO_PAIN_CSV = os.path.join(STEP_DERIV_DIR, "step04_zero_pain.csv")
OUT_NUMBERS_JSON = os.path.join(STEP_RESULTS_DIR, "numbers.json")

NUMBERS_PREFIX = "step04"


# =====================================================================
# Constants
# =====================================================================

# ---------------------------------------------------------------------
# ROW-SET RULE — UNDECIDED, PEDRO'S CALL. DO NOT FLIP WITHOUT HIM.
# ---------------------------------------------------------------------
# Which raw person-quarters are described?
#
#   "byID"    every raw row belonging to an analytic participant (2,748 rows,
#             including quarter-0 rows and the 463 person-quarters the step 03
#             segment filter dropped), each item summarized on its own complete
#             cases. This is what the sandbox did and what the ACCEPTED text of
#             manuscript.docx, supplementary_materials.docx and the response
#             letter currently quote. It is the DEFAULT so that running this step
#             cannot silently change a published number.
#
#   "bypair"  only the (ID, quarter) pairs retained by step 03 (2,056 rows), i.e.
#             exactly the person-quarters the coupling model consumes.
#
# Switching to "bypair" is defensible — the described base would then be the
# analytic base — but it moves NINE numbers that are already in the documents:
#
#     q13 n                          1,966  ->  1,943
#     q13 SD                          2.69  ->   2.70   (2.694525 -> 2.695841)
#     q13 ceiling %                    9.9  ->    9.8   (9.868 -> 9.830)
#     quarterly sleep mean, max       6.39  ->   6.43
#     pain item mean, min             2.17  ->   2.14
#     pain item mean, max             4.20  ->   4.18
#     pain per-item zero %, min       25.7  ->   25.8
#     pain per-item zero %, max       49.0  ->   49.2
#     all-eight-pain-items-zero %     12.9  ->   13.0
#
# (q13 mean stays 6.01, median 6, IQR 4-8; the two participant-level floor
# figures, 68.6% never pain-free and 2.2% pain-free throughout, are unchanged.)
# Those nine appear in the Results raw-descriptives paragraph, replies R2.15 and
# R2.17, and the raw-descriptives table note, plus every Panel A/B/C cell. There
# is no internal inconsistency in the live text forcing the change: the accepted
# Results sentence names no denominator ("Across the person-quarters, ... n =
# 1,966 ratings"). So this is a judgement call about which base to describe, not
# a bug fix, and it has not been made.
#
# Both bases are published in numbers.json regardless of the rule in force
# (n_person_quarters_byid / n_person_quarters_bypair), so the denominator can be
# stated accurately and is never borrowed from another step again.
ROW_SET = "byID"

#: row-set rule -> the key columns of the processed frame used to restrict the raw frame
ROW_SET_KEYS = {"byID": ("ID",), "bypair": ("ID", "quarter")}

#: An (item, quarter) cell with fewer than this many observations is suppressed from the
#: by-quarter table. Explicit and printed: no cell is suppressed today (the smallest cell
#: has n = 113), but if attrition deepened, a quarter could vanish from the table with no
#: warning. run_step04 prints every cell it drops.
MIN_CELL_N = 20

#: response scale of every q2-q13 item; defines "floor" and "ceiling"
ITEM_SCALE = (0.0, 10.0)

#: the factor scores, taken from the processed frame (invariant to ROW_SET)
FACTOR_COLS = ["pain_factor", "contrast_factor", "sleep_factor"]

#: the single sleep item; the Results paragraph quotes its pooled descriptives
SLEEP_ITEM = "q13_sleep"


# =====================================================================
# Loading
# =====================================================================

def load_frames(verbose=False):
    """Return (raw_long, processed) as (DataFrame, DataFrame), IDs typed as str.

    The processed frame goes through ``lib.coupling_model.load_varx_frame`` so the ID
    typing and the row set are defined once and match every fitting step. Its FIRST
    return value (the full long frame) is used, never ``model_df``: model_df drops
    lag-incomplete rows, which would bias the variance decomposition.
    """
    from coupling_model import load_varx_frame

    processed = load_varx_frame(IN_PROCESSED_CSV, verbose=verbose)[0].copy()
    raw_long = pd.read_csv(IN_LONG_CSV, float_precision="round_trip")
    raw_long["ID"] = raw_long["ID"].astype(str)
    processed["ID"] = processed["ID"].astype(str)
    return raw_long, processed


# =====================================================================
# Computation
# =====================================================================

def _compute(verbose=True):
    """Everything this step knows. Returns (frames, numbers, suppressed)."""
    from descriptives import (ALL_ITEMS, KNEE_ITEMS, PAIN_ITEMS,
                              icc_varcomp, item_descriptives,
                              restrict_to_analytic)

    raw_long, processed = load_frames(verbose=False)

    n_participants = int(processed["ID"].nunique())
    by_id = restrict_to_analytic(raw_long, processed, by=ROW_SET_KEYS["byID"])
    by_pair = restrict_to_analytic(raw_long, processed, by=ROW_SET_KEYS["bypair"])
    described = {"byID": by_id, "bypair": by_pair}[ROW_SET]

    if verbose:
        print(f"  analytic participants:        {n_participants}")
        print(f"  described person-quarters:    {len(described)}  (rule: {ROW_SET})")
        print(f"    by-ID rows:                 {len(by_id)}")
        print(f"    retained (ID, quarter):     {len(by_pair)}")

    # ---- Panel A: per-item descriptives, pooled ------------------------
    desc = item_descriptives(described, ALL_ITEMS, scale=ITEM_SCALE)

    # ---- Panel B: per item per quarter ---------------------------------
    qframes = []
    for q, grp in described.groupby("quarter"):
        cell = item_descriptives(grp, ALL_ITEMS, scale=ITEM_SCALE)
        cell.insert(1, "quarter", int(q))
        qframes.append(cell)
    by_quarter_all = pd.concat(qframes, ignore_index=True)
    keep = by_quarter_all["n"] >= MIN_CELL_N
    suppressed = by_quarter_all.loc[~keep & (by_quarter_all["n"] > 0),
                                    ["item", "quarter", "n"]]
    # Cells with n == 0 are dropped by the same filter but were counted nowhere: the
    # suppressed frame requires n > 0, so an item with no data at all in a quarter left
    # Panel B silently one row shorter. A cell absent because it was suppressed and a
    # cell absent because it was never collected are different facts about the study.
    n_empty_cells = int((by_quarter_all["n"] == 0).sum())
    by_quarter = by_quarter_all.loc[keep].reset_index(drop=True)

    # ---- zero-pain / floor structure -----------------------------------
    # Person-quarters with all eight pain items present (list-wise complete). This
    # denominator is NOT the per-item n of Panel A, which is per-item complete-case.
    pain_complete = described.dropna(subset=PAIN_ITEMS)
    all_zero = (pain_complete[PAIN_ITEMS] == 0).all(axis=1)
    knee_zero = (pain_complete[KNEE_ITEMS] == 0).all(axis=1)
    # A participant counts as "pain-free throughout" if every one of their COMPLETE
    # pain quarters is all-zero; quarters with a missing pain item do not veto it.
    per_person = pain_complete.assign(_z=all_zero).groupby("ID")["_z"]
    person_any_zero = per_person.any()
    person_all_zero = per_person.all()

    zero_rows = [
        {"metric": "n_person_quarters_complete_pain", "value": float(len(pain_complete))},
        {"metric": "person_quarters_all_eight_zero", "value": 100 * all_zero.mean()},
        {"metric": "person_quarters_all_knee_zero", "value": 100 * knee_zero.mean()},
        {"metric": "participants_with_>=1_zero_quarter",
         "value": 100 * person_any_zero.mean()},
        {"metric": "participants_never_zero", "value": 100 * (~person_any_zero).mean()},
        {"metric": "participants_zero_throughout", "value": 100 * person_all_zero.mean()},
        {"metric": "n_participants_never_zero", "value": float((~person_any_zero).sum())},
        {"metric": "n_participants_zero_throughout", "value": float(person_all_zero.sum())},
    ]
    zero_pain = pd.DataFrame(zero_rows)

    # ---- Panel C: variance decomposition -------------------------------
    varcomp_rows = []
    for col, frame in ([(c, described) for c in ALL_ITEMS]
                       + [(c, processed) for c in FACTOR_COLS if c in processed.columns]):
        icc, sd_b, sd_w = icc_varcomp(frame, col)
        person_sd = frame.groupby("ID")[col].std().dropna()
        varcomp_rows.append({
            "variable": col, "ICC": icc, "between_SD": sd_b, "within_SD": sd_w,
            "median_person_SD": person_sd.median(),
            "p25_person_SD": person_sd.quantile(.25),
            "p75_person_SD": person_sd.quantile(.75),
            "min_person_SD": person_sd.min(), "max_person_SD": person_sd.max(),
        })
    varcomp = pd.DataFrame(varcomp_rows)

    frames = {
        "item_descriptives": desc,
        "by_quarter": by_quarter,
        "variance_decomposition": varcomp,
        "zero_pain": zero_pain,
    }
    numbers = _numbers(frames, n_participants, len(described), len(by_id), len(by_pair),
                       len(processed), len(suppressed), PAIN_ITEMS)
    return frames, numbers, suppressed


def _numbers(frames, n_participants, n_described, n_by_id, n_by_pair,
             n_retained, n_suppressed, pain_items):
    """Every scalar the documents quote, or could quote, from this step."""
    desc = frames["item_descriptives"].set_index("item")
    byq = frames["by_quarter"]
    varc = frames["variance_decomposition"].set_index("variable")
    zero = frames["zero_pain"].set_index("metric")["value"]

    sleep = desc.loc[SLEEP_ITEM]
    sleep_q = byq[byq["item"] == SLEEP_ITEM]
    pain = desc.loc[pain_items]
    pain_q = byq[byq["item"].isin(pain_items)]

    nums = {
        # ---- sample -----------------------------------------------------
        "n_participants": int(n_participants),
        "n_person_quarters_described": int(n_described),
        "n_person_quarters_byid": int(n_by_id),
        "n_person_quarters_bypair": int(n_by_pair),
        "n_person_quarters_retained": int(n_retained),
        "n_quarters": int(byq["quarter"].nunique()),
        "n_person_quarters_complete_pain": int(zero["n_person_quarters_complete_pain"]),

        # ---- the sleep item, pooled -------------------------------------
        "sleep_item_n": int(sleep["n"]),
        "sleep_item_mean": float(sleep["mean"]),
        "sleep_item_sd": float(sleep["sd"]),
        "sleep_item_median": float(sleep["median"]),
        "sleep_item_p25": float(sleep["p25"]),
        "sleep_item_p75": float(sleep["p75"]),
        "sleep_item_pct_floor": float(sleep["pct_at_floor"]),
        "sleep_item_pct_ceiling": float(sleep["pct_at_ceiling"]),

        # ---- the sleep item, by quarter ---------------------------------
        "sleep_quarter_mean_min": float(sleep_q["mean"].min()),
        "sleep_quarter_mean_max": float(sleep_q["mean"].max()),
        # Unambiguous names. The claim being supported is "the full 0-10 range was
        # present in EVERY quarter", which needs the WORST per-quarter floor and the
        # WORST per-quarter ceiling — not the pooled min/max, which are trivially 0
        # and 10 and would support nothing.
        "sleep_quarter_max_of_per_quarter_min": float(sleep_q["min"].max()),
        "sleep_quarter_min_of_per_quarter_max": float(sleep_q["max"].min()),

        # ---- the eight pain items, pooled -------------------------------
        "pain_item_mean_min": float(pain["mean"].min()),
        "pain_item_mean_max": float(pain["mean"].max()),
        "pain_item_pct_zero_min": float(pain["pct_at_floor"].min()),
        "pain_item_pct_zero_max": float(pain["pct_at_floor"].max()),
        "pain_item_min_observed": float(pain["min"].min()),
        "pain_item_max_observed": float(pain["max"].max()),

        # ---- does every item reach both ends in every quarter? ----------
        # It does NOT: 6 of 121 (item, quarter) cells top out at 9. Any claim that
        # "every item spanned the full 0-10 range at every quarter" is false as
        # stated and has to be narrowed to the sleep item, which does.
        "n_item_quarter_cells_total": int(len(byq)),
        "n_item_quarter_cells_full_range": int(((byq["min"] == ITEM_SCALE[0])
                                                & (byq["max"] == ITEM_SCALE[1])).sum()),
        "min_item_quarter_max": float(byq["max"].min()),
        "max_item_quarter_min": float(byq["min"].max()),
        "n_item_quarter_cells_suppressed": int(n_suppressed),
        # Derived from the table itself rather than passed in, so the reload path
        # reports it too. A cell missing because the item had NO data that quarter
        # was counted nowhere -- `suppressed` requires n > 0 -- so Panel B could
        # come up a row short with nothing saying why.
        "n_item_quarter_cells_empty": int(
            byq["item"].nunique() * byq["quarter"].nunique()
            - len(byq) - int(n_suppressed)),
        "min_item_quarter_n": int(byq["n"].min()),
        "pain_item_quarter_pct_zero_max": float(pain_q["pct_at_floor"].max()),

        # ---- floor structure --------------------------------------------
        "pct_person_quarters_all_eight_pain_zero": float(zero["person_quarters_all_eight_zero"]),
        "pct_person_quarters_all_knee_pain_zero": float(zero["person_quarters_all_knee_zero"]),
        "pct_participants_any_pain_free_quarter": float(zero["participants_with_>=1_zero_quarter"]),
        "pct_participants_never_pain_free": float(zero["participants_never_zero"]),
        "pct_participants_pain_free_throughout": float(zero["participants_zero_throughout"]),
        "n_participants_never_pain_free": int(zero["n_participants_never_zero"]),
        "n_participants_pain_free_throughout": int(zero["n_participants_zero_throughout"]),
    }

    # ---- variance decomposition ------------------------------------------
    for col in FACTOR_COLS:
        if col not in varc.index:
            continue
        nums[f"icc_{col}"] = float(varc.loc[col, "ICC"])
        nums[f"between_sd_{col}"] = float(varc.loc[col, "between_SD"])
        nums[f"within_sd_{col}"] = float(varc.loc[col, "within_SD"])
        nums[f"median_person_sd_{col}"] = float(varc.loc[col, "median_person_SD"])
    if SLEEP_ITEM in varc.index:
        nums["icc_sleep_item_q13"] = float(varc.loc[SLEEP_ITEM, "ICC"])
        nums["between_sd_sleep_item_q13"] = float(varc.loc[SLEEP_ITEM, "between_SD"])
        nums["within_sd_sleep_item_q13"] = float(varc.loc[SLEEP_ITEM, "within_SD"])
    return nums


# =====================================================================
# I/O
# =====================================================================

def _write(frames, numbers, verbose=True):
    from registry import write_numbers

    frames["item_descriptives"].to_csv(OUT_ITEM_CSV, index=False)
    frames["by_quarter"].to_csv(OUT_BY_QUARTER_CSV, index=False)
    frames["variance_decomposition"].to_csv(OUT_VARCOMP_CSV, index=False)
    frames["zero_pain"].to_csv(OUT_ZERO_PAIN_CSV, index=False)
    write_numbers(STEP_RESULTS_DIR, numbers, prefix=NUMBERS_PREFIX)
    if verbose:
        for path in (OUT_ITEM_CSV, OUT_BY_QUARTER_CSV, OUT_VARCOMP_CSV,
                     OUT_ZERO_PAIN_CSV, OUT_NUMBERS_JSON):
            print(f"  wrote {os.path.relpath(path, ROOT)}")


SAVED_OUTPUTS = (OUT_ITEM_CSV, OUT_BY_QUARTER_CSV, OUT_VARCOMP_CSV,
                 OUT_ZERO_PAIN_CSV, OUT_NUMBERS_JSON)


def _load_saved():
    """Read back what a previous run wrote. Returns (frames, numbers)."""
    frames = {
        "item_descriptives": pd.read_csv(OUT_ITEM_CSV, float_precision="round_trip"),
        "by_quarter": pd.read_csv(OUT_BY_QUARTER_CSV, float_precision="round_trip"),
        "variance_decomposition": pd.read_csv(OUT_VARCOMP_CSV, float_precision="round_trip"),
        "zero_pain": pd.read_csv(OUT_ZERO_PAIN_CSV, float_precision="round_trip"),
    }
    with open(OUT_NUMBERS_JSON) as fh:
        raw = json.load(fh)
    numbers = {k.split(".", 1)[-1]: v for k, v in raw.items()}
    return frames, numbers


# =====================================================================
# Entry point
# =====================================================================

def run_step04(verbose=True, refit=False):
    """Compute (or load) the raw quarterly descriptives.

    The DEFAULT path loads the saved CSVs and numbers.json and reports them; nothing is
    recomputed and no file is touched. ``--refit`` recomputes from step00's raw long
    table and step 03's curated frame and rewrites every output. There is no MCMC in
    this step — ``refit`` is accepted for pipeline uniformity and costs about a second.

    Returns a dict with the four frames under ``frames`` and every scalar under
    ``numbers``.
    """
    if verbose:
        print("=" * 70)
        print("STEP 04 — Raw quarterly descriptive statistics")
        print("=" * 70)

    saved_exist = all(os.path.exists(p) for p in SAVED_OUTPUTS)
    if not refit and not saved_exist:
        if verbose:
            print("  Saved derivatives not found — forcing recomputation.")
        refit = True

    if not refit:
        frames, numbers = _load_saved()
        suppressed = pd.DataFrame(columns=["item", "quarter", "n"])
        if verbose:
            print("  WARNING: Running in load mode -- reading saved derivatives.")
            print("  If you have changed upstream data or code, re-run with --refit.")
    else:
        for path in (IN_LONG_CSV, IN_PROCESSED_CSV):
            if not os.path.exists(path):
                raise FileNotFoundError(f"required input missing: {path}")
        frames, numbers, suppressed = _compute(verbose=verbose)
        _write(frames, numbers, verbose=verbose)

    if verbose:
        _report(frames, numbers, suppressed)
        print("=" * 70)
    return {"frames": frames, "numbers": numbers}


def _report(frames, numbers, suppressed):
    """Human sanity check on stdout. NOT a source for any manuscript number."""
    n = numbers
    print("\n  === item descriptives (0-10) ===")
    cols = ["item", "n", "mean", "sd", "median", "min", "max",
            "pct_at_floor", "pct_at_ceiling"]
    print(frames["item_descriptives"][cols]
          .to_string(index=False, float_format=lambda x: f"{x:7.2f}"))

    print(f"\n  === {SLEEP_ITEM} by quarter ===")
    sub = frames["by_quarter"]
    sub = sub[sub["item"] == SLEEP_ITEM][["quarter", "n", "mean", "sd", "min", "max"]]
    print(sub.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

    if len(suppressed):
        print(f"\n  === (item, quarter) cells suppressed at n < {MIN_CELL_N} ===")
        print(suppressed.to_string(index=False))

    print("\n  === zero-pain structure (%) ===")
    for _, r in frames["zero_pain"].iterrows():
        print(f"    {r['metric']:<38} {r['value']:7.2f}")

    print("\n  === variance decomposition ===")
    print(frames["variance_decomposition"][
        ["variable", "ICC", "between_SD", "within_SD", "median_person_SD"]
    ].to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    full = n.get("n_item_quarter_cells_full_range")
    total = n.get("n_item_quarter_cells_total")
    if full is not None and total is not None and full < total:
        print(f"\n  NOTE: {total - full} of {total} (item, quarter) cells do NOT span "
              f"the full {ITEM_SCALE[0]:.0f}-{ITEM_SCALE[1]:.0f} range "
              f"(lowest cell maximum = {n['min_item_quarter_max']:.0f}). "
              "An 'every item, every quarter' range claim is false as stated.")


def main():
    ap = argparse.ArgumentParser(
        description="Step 04 — Raw quarterly descriptive statistics."
    )
    ap.add_argument("--refit", action="store_true",
                    help="Recompute from the raw data instead of loading saved outputs "
                         "(no MCMC in this step; takes about a second)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run_step04(verbose=not args.quiet, refit=args.refit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
