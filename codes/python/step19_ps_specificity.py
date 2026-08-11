#!/usr/bin/env python3
"""
Step 17 — Directional-specificity control for the sleep-to-pain fMRI ROIs.
======================================================================

Reviewer 1's only analytic request: test the NAcc and dACC/MCC ROIs as
moderators of the PAIN-TO-SLEEP pathway, "to round out the findings by
verifying if the moderation is directionally specific."

This is a SPECIFICITY CONTROL, not a hypothesis test. The Krause (sleep
deprivation) and Sardi (sleep restriction) frameworks motivate moderation of
the SLEEP-TO-PAIN path only; neither predicts moderation of pain-to-sleep.
The expected result is therefore NULL, and the null is the point: if these
ROIs moderated BOTH directions, the published sleep-to-pain moderation would
indicate generally stronger coupling in those individuals rather than a
direction-specific effect.

Because it is a control with a pre-stated expected null, it does not belong to
the discovery family and must not be pooled into the multiplicity correction
for the Aim 2 tests. Downstream steps that assemble a corrected-p family must
not glob this step's summary CSV.

R1 names the left NAcc and the ACC; both hemispheres of both regions are
fitted so the control is symmetric and cannot be accused of reporting only the
side that gives the convenient answer. That makes four ROIs — a deliberate
subset of step 13's eight — so this step's table is NOT row-comparable with
Table 5.

Method: identical to step16_fit_sp_moderation.py except that
``moderator_direction`` is "ps" instead of "sp". Same ROI values, same masking
rule, same priors, same sample. step16 is not touched, so the published
gamma_sp estimates cannot drift.

The four fits go through ``lib.coupling_model.run_fit`` (via
``fit_roi_moderation_set`` -> ``fit_bayesian_varx1``) with an explicit
``fit_id``/``out_dir``, so each writes a diagnostics record. Those four are
counted in the supplement's model-diagnostics fit inventory ("four
pain-to-sleep fMRI ROI moderation models") and folded into the across-fits
aggregate by step 22, which discovers them by ``fit_id``. Dropping the
diagnostics here would silently change a published count.

Input:  derivatives/step07_varx_data/step07_processed_long.csv
        derivatives/step14_sp_roi_values/step14_sp_roi_values.csv
        results/step16_sp_moderation/step16_table5_sp_moderation.csv
            (read-only cross-check of the analytic sample; optional)
Output:
  derivatives/step19_ps_specificity/
    step19_ps_specificity_draws.npz   — per-ROI posterior draws
    diagnostics_step19_ps_<ROI>.json  — written by run_fit, one per fit
  results/step19_ps_specificity/
    step19_ps_specificity.csv         — table DATA, one row per ROI
    numbers.json                      — the registry entries

No figure: there is no Johnson-Neyman curve for this control. No prose.

Usage:
    python step19_ps_specificity.py [--refit] [--quiet]

Author: Pedro Valdes-Hernandez (with Claude Opus 5)
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DERIV_DIR = os.path.join(ROOT, "derivatives")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step19_ps_specificity")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step19_ps_specificity")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

IN_PROCESSED_CSV = os.path.join(DERIV_DIR, "step07_varx_data",
                                "step07_processed_long.csv")
IN_ROI_CSV = os.path.join(DERIV_DIR, "step14_sp_roi_values",
                          "step14_sp_roi_values.csv")
#: step 14's table, used only to verify that this control ran on the same
#: persons as the sleep-to-pain fits it is a control for. Never written.
IN_STEP14_TABLE = os.path.join(RESULTS_DIR, "step16_sp_moderation",
                               "step16_table5_sp_moderation.csv")

OUT_DRAWS_NPZ = os.path.join(STEP_DERIV_DIR, "step19_ps_specificity_draws.npz")
OUT_SUMMARY_CSV = os.path.join(STEP_RESULTS_DIR, "step19_ps_specificity.csv")

#: Row order is the sentence order in the Results paragraph. Kept explicit so a
#: change in step 13's ROI ordering cannot reorder the published numbers.
TARGET_ROIS = ["Left_NAcc", "Right_NAcc", "Left_dACC_MCC", "Right_dACC_MCC"]

#: fit_id = f"{FIT_ID_PREFIX}_{roi}"; step 22 groups on this prefix.
FIT_ID_PREFIX = "step19_ps"

#: Columns of the summary CSV, in order. Table DATA only — no note, no
#: formatting, no caption.
SUMMARY_COLUMNS = [
    "ROI", "label", "framework", "mask_type", "direction",
    "gamma_ps_mean", "gamma_ps_sd", "gamma_ps_ci_lo", "gamma_ps_ci_hi",
    "prob_neg", "two_tail_p", "credible", "n_persons", "n_obs", "rhat_max",
]


def _build_summary(fitted):
    """Select and rename ``fit_roi_moderation_set``'s columns into the
    published table's schema, in TARGET_ROIS order.

    The posterior summaries themselves are computed in lib (extract_results +
    two_tail_p), not here, so this step cannot drift from step 14 and step 19.
    """
    f = fitted.set_index("ROI").loc[TARGET_ROIS].reset_index()
    lo = f["gamma_ps_ci_lo"].astype(float)
    hi = f["gamma_ps_ci_hi"].astype(float)
    summary = pd.DataFrame({
        "ROI": f["ROI"],
        "label": f["label"],
        "framework": f["framework"],
        "mask_type": f["mask_type"],
        "direction": f["direction"],
        "gamma_ps_mean": f["gamma_ps_mean"].astype(float),
        "gamma_ps_sd": f["gamma_ps_sd"].astype(float),
        "gamma_ps_ci_lo": lo,
        "gamma_ps_ci_hi": hi,
        "prob_neg": f["gamma_ps_prob_neg"].astype(float),
        "two_tail_p": f["gamma_ps_p"].astype(float),
        "credible": (lo > 0) | (hi < 0),
        "n_persons": f["n_persons"].astype(int),
        "n_obs": f["n_obs"].astype(int),
        "rhat_max": f["rhat_max"].astype(float),
    })
    return summary[SUMMARY_COLUMNS]


def _step16_n_persons():
    """Number of persons step 14 fitted for these same four ROIs, or None.

    None means the question cannot be answered right now (step 14's table is
    absent, or it does not itself use one sample for the four), not that the
    samples agree.
    """
    if not os.path.exists(IN_STEP14_TABLE):
        return None
    table = pd.read_csv(IN_STEP14_TABLE)
    if "ROI" not in table.columns or "N" not in table.columns:
        return None
    rows = table[table["ROI"].isin(TARGET_ROIS)]
    values = sorted({int(v) for v in rows["N"].dropna()})
    if len(values) != 1:
        return None
    return values[0]


def _collect_numbers(summary, verbose=True):
    """Every quantity this step publishes, under its registry key.

    The four posterior means and CrIs are quoted inline in the Results
    paragraph with no table to fall back on, so they are keyed per ROI.
    ``n_credible`` exists so that the paragraph's claim ("none was credible")
    is machine-checkable rather than assumed: its expected value is 0.
    """
    n_persons = sorted({int(v) for v in summary["n_persons"]})
    n_obs = sorted({int(v) for v in summary["n_obs"]})
    if len(n_persons) != 1 or len(n_obs) != 1:
        raise ValueError(
            "the four ROIs were fitted on different samples "
            f"(n_persons={n_persons}, n_obs={n_obs}); the Results paragraph "
            "says 'the same four regions', so a single scalar sample size "
            "cannot be published. Re-extract step 13's ROI values."
        )

    nums = {}
    for _, row in summary.iterrows():
        roi = row["ROI"]
        nums[f"ps_specificity_{roi}_gamma_ps"] = float(row["gamma_ps_mean"])
        nums[f"ps_specificity_{roi}_gamma_ps_ci"] = [
            float(row["gamma_ps_ci_lo"]), float(row["gamma_ps_ci_hi"]),
        ]
        nums[f"ps_specificity_{roi}_gamma_ps_p"] = float(row["two_tail_p"])
        nums[f"ps_specificity_{roi}_rhat_max"] = float(row["rhat_max"])

    nums["ps_specificity_n_rois_tested"] = int(len(summary))
    nums["ps_specificity_n_credible"] = int(summary["credible"].astype(bool).sum())
    nums["ps_specificity_n_persons"] = n_persons[0]
    nums["ps_specificity_n_transitions"] = n_obs[0]
    # Scalar over four fits, which differ: the worst one, so the key has a
    # defined value and bounds the others.
    nums["ps_specificity_rhat_max"] = float(summary["rhat_max"].max())

    step16_n = _step16_n_persons()
    nums["ps_specificity_n_persons_step16"] = step16_n
    nums["ps_specificity_sample_matches_step16"] = (
        None if step16_n is None else bool(step16_n == n_persons[0])
    )
    if verbose and step16_n is not None and step16_n != n_persons[0]:
        print("  WARNING: sample differs from step 14 for the same four ROIs: "
              f"{n_persons[0]} persons here vs {step16_n} there. The Results "
              "paragraph says 'the same four regions'. Re-run step 14 with "
              "--refit if its table is stale; otherwise step 13's ROI values "
              "changed and both steps must be refitted.")
    return nums


def run_step19(verbose=True, refit=False):
    """Fit (or reload) the four pain-to-sleep specificity fits and publish
    the table data and the registry numbers.

    Default path is load-and-summarize: it reads the saved summary CSV and
    posterior draws and rewrites numbers.json without sampling, so a
    re-registration costs seconds. ``refit=True`` re-runs the four MCMC fits.
    """
    if verbose:
        print("=" * 70)
        print("STEP 17 — Directional-specificity control (pain-to-sleep)")
        print("=" * 70)

    saved_exist = os.path.exists(OUT_SUMMARY_CSV) and os.path.exists(OUT_DRAWS_NPZ)
    if not refit and not saved_exist:
        if verbose:
            print("  Saved derivatives not found — forcing refit.")
        refit = True

    if not refit:
        # ------ LOAD MODE: re-summarize saved derivatives ------
        if verbose:
            print("  WARNING: Running in load mode -- using saved derivatives.")
            print("  If you have changed upstream data or code, re-run with --refit.")
        summary = pd.read_csv(OUT_SUMMARY_CSV)
        missing = [c for c in SUMMARY_COLUMNS if c not in summary.columns]
        if missing:
            raise ValueError(
                f"{OUT_SUMMARY_CSV} is missing columns {missing}; it predates the "
                "current schema. Re-run this step with --refit."
            )
        summary["credible"] = summary["credible"].astype(bool)
        if verbose:
            print(f"  Loaded: {os.path.relpath(OUT_SUMMARY_CSV, ROOT)}")
            print(f"  Loaded: {os.path.relpath(OUT_DRAWS_NPZ, ROOT)}")
    else:
        # ------ FULL MCMC FIT (4 fits) ------
        from coupling_model import load_varx_frame, fit_roi_moderation_set

        df_full, model_df, unique_ids, id_map = load_varx_frame(
            IN_PROCESSED_CSV, verbose=verbose)

        roi_df = pd.read_csv(IN_ROI_CSV)
        available = set(roi_df["ROI"].unique())
        missing = [r for r in TARGET_ROIS if r not in available]
        if missing:
            # Never skip: a silently dropped ROI silently deletes a number that
            # is quoted inline in the manuscript.
            raise KeyError(
                f"{missing} absent from {IN_ROI_CSV}. All four ROIs are quoted "
                "in the Results; re-run step 13 before this step."
            )

        if verbose:
            print(f"  {len(TARGET_ROIS)} ROIs, moderator_direction='ps'")

        fitted, draws_dict = fit_roi_moderation_set(
            roi_df, model_df, unique_ids, id_map,
            direction="ps",
            rois=TARGET_ROIS,
            fit_id_prefix=FIT_ID_PREFIX,
            out_dir=STEP_DERIV_DIR,
            progressbar=verbose,
            verbose=verbose,
        )

        summary = _build_summary(fitted)
        summary.to_csv(OUT_SUMMARY_CSV, index=False)
        np.savez(OUT_DRAWS_NPZ, **draws_dict)
        if verbose:
            print(f"\n  Saved table data: {os.path.relpath(OUT_SUMMARY_CSV, ROOT)}")
            print(f"  Saved draws: {os.path.relpath(OUT_DRAWS_NPZ, ROOT)}")

    nums = _collect_numbers(summary, verbose=verbose)

    from registry import write_numbers
    numbers_path = write_numbers(STEP_RESULTS_DIR, nums, prefix="step19")

    if verbose:
        for _, row in summary.iterrows():
            print(f"  {row['ROI']:<16s} gamma_ps = {row['gamma_ps_mean']:+.4f} "
                  f"[{row['gamma_ps_ci_lo']:+.4f}, {row['gamma_ps_ci_hi']:+.4f}]  "
                  f"p = {row['two_tail_p']:.3f}  "
                  f"R-hat {row['rhat_max']:.4f}")
        print(f"  credible: {nums['ps_specificity_n_credible']} / "
              f"{nums['ps_specificity_n_rois_tested']}   "
              f"N = {nums['ps_specificity_n_persons']}, "
              f"transitions = {nums['ps_specificity_n_transitions']}")
        print(f"  Saved numbers: {os.path.relpath(numbers_path, ROOT)}")
        print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Step 17 — pain-to-sleep specificity control (4 fMRI ROIs)."
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--refit", action="store_true",
                        help="Re-run the four MCMC fits instead of loading "
                             "saved derivatives")
    args = parser.parse_args()
    run_step19(verbose=not args.quiet, refit=args.refit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
