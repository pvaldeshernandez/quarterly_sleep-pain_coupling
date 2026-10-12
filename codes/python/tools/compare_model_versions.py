#!/usr/bin/env python3
"""Compare a model version's checks with the published (observed) version.

    python tools/compare_model_versions.py --model latent

Reads only what the two pipeline runs saved; computes nothing new. Writes to
``results_<model>/version_comparison/``:

    diagnostics_by_fit.csv    every fit both versions ran: R-hat max, bulk and
                              tail ESS min, BFMI min, divergences, deepest tree,
                              % of draws at the depth cap, published vs <model>
    diagnostics_summary.csv   the same, worst case over each analysis family and
                              over all fits (the paper's "Model diagnostics"
                              sentence)
    ppc.csv                   posterior predictive check (step 09) side by side
    loo_pareto.csv            LOO Pareto k-hat summary (step 08) side by side

R-hat and ESS are over every parameter each run recorded, person-level ones
included; the latent model has 458 more (each person's usual pain and sleep
level), so its worst case is taken over a larger set.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CODE))
sys.path.insert(0, os.path.join(CODE, "lib"))

import model_version  # noqa: E402

#: diagnostic -> (how the worst case aggregates, column label)
METRICS = {
    "rhat_max": ("max", "R-hat (max)"),
    "ess_bulk_min": ("min", "Bulk ESS (min)"),
    "ess_tail_min": ("min", "Tail ESS (min)"),
    "bfmi_min": ("min", "BFMI (min)"),
    "divergences": ("sum", "Divergences"),
    "max_tree_depth": ("max", "Tree depth (max)"),
    "pct_draws_at_cap": ("max", "% draws at depth cap"),
}

#: fit-id prefix -> analysis family (row label in the summary)
FAMILIES = [
    ("step08_primary", "Primary coupling model (step 08)"),
    ("step08_loo", "LOO comparison models (step 08)"),
    ("step10", "Time-varying covariates (step 10)"),
    ("step11", "Interpolation-free refit (step 11)"),
    ("step16", "Sleep-to-pain fMRI moderation (step 16)"),
    ("step18", "Nuisance-adjusted moderation (step 18)"),
    ("step19", "Pain-to-sleep specificity (step 19)"),
    ("step21", "Pain-arousal relay moderation (step 21)"),
    ("step23", "Severity moderation (step 23)"),
]


def read_diagnostics(deriv_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(deriv_dir, "step*", "diagnostics_*.json"))):
        with open(path) as fh:
            rows.append(json.load(fh))
    return pd.DataFrame(rows).set_index("fit_id") if rows else pd.DataFrame()


def family_of(fit_id):
    for prefix, label in FAMILIES:
        if fit_id.startswith(prefix):
            return label
    return "Other"


def by_fit(pub, new, name):
    ids = sorted(set(pub.index) | set(new.index))
    out = pd.DataFrame(index=pd.Index(ids, name="fit_id"))
    out["family"] = [family_of(i) for i in ids]
    for m, (_, label) in METRICS.items():
        out[f"{label} | published"] = pub[m].reindex(ids) if m in pub else None
        out[f"{label} | {name}"] = new[m].reindex(ids) if m in new else None
    return out


def summarize(fits, name):
    def agg(frame, how, col):
        s = pd.to_numeric(frame[col], errors="coerce").dropna()
        if s.empty:
            return None
        return {"max": s.max, "min": s.min, "sum": s.sum}[how]()

    rows = []
    groups = [(label, fits[fits["family"] == label]) for _, label in FAMILIES]
    groups.append(("All fits", fits))
    for label, frame in groups:
        if frame.empty:
            continue
        row = {"family": label,
               "fits | published": int(frame[f"{METRICS['rhat_max'][1]} | published"].notna().sum()),
               f"fits | {name}": int(frame[f"{METRICS['rhat_max'][1]} | {name}"].notna().sum())}
        for m, (how, lab) in METRICS.items():
            for v in ("published", name):
                row[f"{lab} | {v}"] = agg(frame, how, f"{lab} | {v}")
        rows.append(row)
    return pd.DataFrame(rows)


def side_by_side(path_pub, path_new, keys, value_cols, name, read=None):
    read = read or (lambda p: pd.read_csv(p, float_precision="round_trip"))
    if not (os.path.exists(path_pub) and os.path.exists(path_new)):
        return None
    a, b = read(path_pub), read(path_new)
    return a[keys + value_cols].merge(b[keys + value_cols], on=keys, how="outer",
                                      suffixes=(" | published", f" | {name}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(set(model_version.MODELS) - {"observed"}))
    args = ap.parse_args()
    name = args.model

    os.environ[model_version.ENV_VAR] = "observed"
    pub_d, pub_r = model_version.versioned(ROOT, "derivatives"), model_version.versioned(ROOT, "results")
    os.environ[model_version.ENV_VAR] = name
    new_d, new_r = model_version.versioned(ROOT, "derivatives"), model_version.versioned(ROOT, "results")
    out_dir = os.path.join(new_r, "version_comparison")
    os.makedirs(out_dir, exist_ok=True)

    fits = by_fit(read_diagnostics(pub_d), read_diagnostics(new_d), name)
    fits.to_csv(os.path.join(out_dir, "diagnostics_by_fit.csv"))
    summary = summarize(fits, name)
    summary.to_csv(os.path.join(out_dir, "diagnostics_summary.csv"), index=False)
    print(f"  {len(fits)} fits -> {os.path.relpath(out_dir, ROOT)}/diagnostics_*.csv")

    ppc_rel = os.path.join("step09_posterior_predictive_check", "step09_ppc_summary.csv")
    ppc = side_by_side(os.path.join(pub_r, ppc_rel), os.path.join(new_r, ppc_rel),
                       ["variable", "statistic"], ["observed", "rep_mean", "ppp"], name)
    if ppc is not None:
        ppc.to_csv(os.path.join(out_dir, "ppc.csv"), index=False)

    loo_rel = os.path.join("step08_coupling_model", "step08_text_numbers.csv")
    read_tn = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)  # noqa: E731
    loo = side_by_side(os.path.join(pub_r, loo_rel), os.path.join(new_r, loo_rel),
                       ["metric"], ["value"], name, read=read_tn)
    if loo is not None:
        loo = loo[loo["metric"].str.startswith(("pareto_", "loo_"))]
        loo.to_csv(os.path.join(out_dir, "loo_pareto.csv"), index=False)


if __name__ == "__main__":
    main()
