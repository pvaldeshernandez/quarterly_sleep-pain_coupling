#!/usr/bin/env python3
"""
Step 24 — Collect the diagnostics of every fit in the paper.

Runs LAST, and that is the whole reason it is a separate step. Per-fit diagnostics are
written by `lib.coupling_model.run_fit` at each fit, so there is nothing to compute in
the middle of the pipeline. But Tables S3 and S4 and the "across all fits" sentence
summarize EVERY model the paper reports, including step 21's, so the aggregation can
only happen after the last one has run.

This step recomputes nothing. It reads what the fits already recorded.

Input:  derivatives/step*/diagnostics_*.json            (one per fit)
        derivatives/step*/diagnostics_*_by_param.csv    (per-parameter, per fit)
Output: derivatives/step24_diagnostics/step24_all_fits.csv
        results/step24_diagnostics/step24_by_family.csv     -> Table S3
        results/step24_diagnostics/step24_sampler.csv       -> Table S4
        results/step24_diagnostics/numbers.json

Two fits are ALIASES of the primary model rather than separate models: the full model of
the LOO comparison, and the "all transitions" arm of the interpolation sensitivity. They
are counted once. This is why the paper says 52 distinct fits and not 54 — a count nobody
adds up is a count nobody checks.

Usage:
    python step24_diagnostics_summary.py [--refit] [--quiet]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)
ROOT = os.path.dirname(os.path.dirname(HERE))

DERIV_DIR = os.path.join(ROOT, "derivatives")
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step24_diagnostics")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step24_diagnostics")

OUT_ALL = os.path.join(STEP_DERIV_DIR, "step24_all_fits.csv")
OUT_FAMILY = os.path.join(STEP_RESULTS_DIR, "step24_by_family.csv")
OUT_SAMPLER = os.path.join(STEP_RESULTS_DIR, "step24_sampler.csv")

#: the primary model, whose per-parameter table becomes Table S3
PRIMARY = "step08_primary"

#: fits that are the primary model under another name. Counted once.
ALIASES = {"step08_loo_full", "step11_all_transitions"}

#: parameter -> family, for Table S3. Order is the table's row order.
FAMILIES = [
    ("Pain equation: intercept, autoregression, coupling, direct and interaction terms",
     lambda n: n in {"a0", "a1", "a2", "a3", "a4"}),
    ("Sleep equation: intercept, coupling, autoregression, direct and interaction terms",
     lambda n: n in {"b0", "b1", "b2", "b3", "b4"}),
    ("Age and sex moderation of both coupling directions",
     lambda n: n.startswith("g_")),
    ("Between-person SDs of the person-specific slopes",
     lambda n: n in {"tau_sp", "tau_ps"}),
    ("Innovation SDs and their correlation",
     lambda n: n in {"sigma_pain", "sigma_sleep", "rho_innov", "rho_raw"}),
    ("Person-specific sleep-to-pain slopes", lambda n: n.startswith("u_sp[")),
    ("Person-specific pain-to-sleep slopes", lambda n: n.startswith("u_ps[")),
]


def collect(verbose=True):
    """Every per-fit diagnostics record the pipeline wrote."""
    rows = []
    for path in sorted(glob.glob(os.path.join(DERIV_DIR, "step*", "diagnostics_*.json"))):
        with open(path) as fh:
            rec = json.load(fh)
        rec["source"] = os.path.relpath(path, ROOT)
        rec["is_alias"] = rec.get("fit_id") in ALIASES
        rows.append(rec)
    if verbose:
        print(f"  found {len(rows)} diagnostics record(s)")
    return pd.DataFrame(rows)


def by_family(verbose=True):
    """Table S3: convergence by parameter family, for the primary model."""
    hits = glob.glob(os.path.join(DERIV_DIR, "step*",
                                  f"diagnostics_{PRIMARY}_by_param.csv"))
    if not hits:
        if verbose:
            print(f"  no per-parameter table for {PRIMARY}; run step08 --refit first")
        return None
    d = pd.read_csv(hits[0], index_col=0, float_precision="round_trip")
    rows, used = [], []
    for label, belongs in FAMILIES:
        names = [n for n in d.index if belongs(str(n))]
        if not names:
            continue
        used.extend(names)
        s = d.loc[names]
        rows.append({"family": label, "n": len(names),
                     "rhat_max": s["r_hat"].max(),
                     "ess_bulk_min": s["ess_bulk"].min(),
                     "ess_tail_min": s["ess_tail"].min()})
    s = d.loc[used]
    rows.append({"family": "All parameters", "n": len(used),
                 "rhat_max": s["r_hat"].max(),
                 "ess_bulk_min": s["ess_bulk"].min(),
                 "ess_tail_min": s["ess_tail"].min()})
    missed = [n for n in d.index if n not in used]
    if missed and verbose:
        # Loud on purpose: an unclassified parameter would silently drop out of the
        # "All parameters" row and understate the worst diagnostic in the model.
        print(f"  WARNING: {len(missed)} parameter(s) matched no family, e.g. {missed[:5]}")
    return pd.DataFrame(rows)


def run_step24(verbose=True, refit=False):
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("STEP 24 — Diagnostics summary (aggregates; recomputes nothing)")
        print("=" * 70)

    allfits = collect(verbose)
    if allfits.empty:
        if verbose:
            print("  no fits have written diagnostics yet — run the pipeline with --refit")
        return None

    allfits.to_csv(OUT_ALL, index=False)
    distinct = allfits[~allfits["is_alias"]]

    fam = by_family(verbose)
    if fam is not None:
        fam.to_csv(OUT_FAMILY, index=False)

    primary = allfits[allfits["fit_id"] == PRIMARY]
    if not primary.empty:
        primary.to_csv(OUT_SAMPLER, index=False)

    nums = {
        "n_fit_records": int(len(allfits)),
        "n_distinct_fits": int(len(distinct)),
        "rhat_max_all_fits": float(distinct["rhat_max"].max()),
        "ess_bulk_min_all_fits": float(distinct["ess_bulk_min"].min()),
        "ess_tail_min_all_fits": float(distinct["ess_tail_min"].min()),
        "bfmi_min_all_fits": float(distinct["bfmi_min"].min()),
        "divergences_total_all_fits": int(distinct["divergences"].sum()),
        "max_tree_depth_all_fits": int(distinct["max_tree_depth"].max()),
    }

    # Per-GROUP minima, because six table notes quote them and none of them had a name.
    # Each note says "Across the sixteen adjusted models, R-hat did not exceed X, bulk and
    # tail effective sample sizes were at least Y and Z" -- numbers computed by hand from
    # this frame and typed in. Table 5's note was still quoting the previous run's pair
    # months later, and nothing could have caught it: with no name there is nothing for a
    # checker to compare against. Published here, one set per note.
    for prefix, tag in (("step10", "timevarying_six"),
                        ("step11", "nointerp"),
                        ("step16", "sp_moderation_eight"),
                        ("step18", "nuisance_sixteen"),
                        ("step19", "ps_specificity_four"),
                        ("step21", "arousal_ten"),
                        ("step23", "severity_three")):
        grp = distinct[distinct["fit_id"].astype(str).str.startswith(prefix)]
        if grp.empty:
            continue
        nums.update({
            f"n_fits_{tag}": int(len(grp)),
            f"rhat_max_{tag}": float(grp["rhat_max"].max()),
            f"ess_bulk_min_{tag}": float(grp["ess_bulk_min"].min()),
            f"ess_tail_min_{tag}": float(grp["ess_tail_min"].min()),
            f"divergences_total_{tag}": int(grp["divergences"].sum()),
        })
    if not primary.empty:
        r = primary.iloc[0]
        nums.update({
            "primary_rhat_max": float(r["rhat_max"]),
            "primary_ess_bulk_min": float(r["ess_bulk_min"]),
            "primary_ess_tail_min": float(r["ess_tail_min"]),
            "primary_divergences": int(r["divergences"]),
            "primary_max_tree_depth": int(r["max_tree_depth"]),
            "primary_mean_accept": float(r["mean_accept"]),
            "primary_bfmi_min": float(r["bfmi_min"]),
            "primary_n_draws_total": int(r["n_draws_total"]),
        })

    from registry import write_numbers
    write_numbers(STEP_RESULTS_DIR, nums, prefix="step24")

    if verbose:
        print(f"\n  {nums['n_distinct_fits']} distinct fit(s) "
              f"({nums['n_fit_records']} records, "
              f"{nums['n_fit_records'] - nums['n_distinct_fits']} alias(es))")
        print(f"  R-hat <= {nums['rhat_max_all_fits']:.4f}   "
              f"bulk ESS >= {nums['ess_bulk_min_all_fits']:.0f}   "
              f"tail ESS >= {nums['ess_tail_min_all_fits']:.0f}")
        print(f"  BFMI >= {nums['bfmi_min_all_fits']:.3f}   "
              f"divergences {nums['divergences_total_all_fits']}   "
              f"deepest tree {nums['max_tree_depth_all_fits']}")
        print(f"\n  wrote {os.path.relpath(OUT_ALL, ROOT)}")
    return allfits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true",
                    help="accepted for pipeline uniformity; this step never fits")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run_step24(verbose=not args.quiet, refit=args.refit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
