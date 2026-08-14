#!/usr/bin/env python3
"""
Step 10 — Time-varying covariates: fatigue/mood and treatment activity.
======================================================================

Does the coupling survive adjustment for things that move quarter to quarter?
Two covariate sets are tested, each against its OWN baseline on its OWN
complete-case sample:

  fatigue/mood   M0  primary specification, no time-varying covariates
                 C1  + fatigue(t-1), mood(t-1)   in both equations
                 C2  + fatigue(t),   mood(t)     in both equations

  treatment      M0  primary specification, no time-varying covariates
                 T1  + surgery(t-1), new-treatment(t-1)  in both equations
                 T2  + surgery(t),   new-treatment(t)    in both equations

Six fits, one loop. The two sets are NOT merged into one sample: their
missingness differs, so each block carries its own M0 and its own n. Covariates
enter as MAIN EFFECTS only (``fit_bayesian_varx1(X_tv=...)``, one N(0,5) theta
per covariate per equation, in the linear predictors), never as moderators of
the coupling slopes, so lambda keeps the primary model's meaning.

C2/T2 caveat (interpretation, not code): a same-quarter covariate sits AFTER the
predictor at t-1, so attenuation there is ambiguous between confounding and
mediation. C1/T1 are the clean tests.

Inputs
------
  derivatives/step07_varx_data/step07_processed_long.csv
      primary analytic frame — ID, quarter, segment_id, the within/lag columns,
      Age, Sex. Supplies the primary transition count and the person index.
  data/step00_extracted_long.csv            (read-only; preferred location
  derivatives/step00_extract_data/step00_extracted_long.csv is used when it
  exists, see RAW_LONG_CANDIDATES)
      raw quarterly items q11_fatigue, q12_mood, q14_pain_treatment,
      q15_pain_treatment.

Outputs
-------
  derivatives/step10_timevarying_covariates/
      step10_<set>_<model>_idata.nc      six posterior archives
      step10_posterior_summary.csv       tidy: covariate_set x model x param
      step10_timevarying_design.csv      ID, quarter, the eight covariate
                                         columns, per-set complete-case flags
      step10_fit_diagnostics.csv         six rows, aggregated from the
                                         diagnostics_*.json that lib.run_fit
                                         writes beside each fit
  results/step10_timevarying_covariates/
      step10_coupling_by_adjustment.csv    Panel A (data only)
      step10_covariate_coefficients.csv  Panel B (data only)
      numbers.json                          the registry (Decision 4)

No prose is generated: no *_text.md, no table notes, no captions. the two panels'
note and the surrounding prose are authored by hand from numbers.json.

Default path is LOAD-AND-EMIT: the two panels and numbers.json are rebuilt from
the saved summary + design frame without sampling. ``--refit`` re-runs all six
fits through lib.coupling_model.run_fit, which is the only place in the
repository that calls pm.sample.

Usage:
    python step10_timevarying_covariates.py            # rebuild from saved
    python step10_timevarying_covariates.py --refit     # six MCMC fits

Author: Pedro Valdes-Hernandez (with Claude Opus 5)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DERIV_DIR = os.path.join(ROOT, "derivatives")
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step10_timevarying_covariates")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step10_timevarying_covariates")

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

IN_PROCESSED_CSV = os.path.join(DERIV_DIR, "step07_varx_data",
                                "step07_processed_long.csv")

#: Where the raw quarterly items live. The fork declares data/ read-only and
#: step 00 is to write its long frame into derivatives/, but today it still
#: writes into data/. Both are accepted, derivatives/ first, so this step does
#: not have to be edited when step 00 migrates. Nothing here writes to data/.
RAW_LONG_CANDIDATES = [
    os.path.join(DERIV_DIR, "step00_extract_data", "step00_extracted_long.csv"),
    os.path.join(ROOT, "data", "step00_extracted_long.csv"),
]

OUT_SUMMARY_CSV = os.path.join(STEP_DERIV_DIR, "step10_posterior_summary.csv")
OUT_DESIGN_CSV = os.path.join(STEP_DERIV_DIR, "step10_timevarying_design.csv")
OUT_DIAG_CSV = os.path.join(STEP_DERIV_DIR, "step10_fit_diagnostics.csv")
OUT_PANEL_A_CSV = os.path.join(STEP_RESULTS_DIR,
                               "step10_coupling_by_adjustment.csv")
OUT_PANEL_B_CSV = os.path.join(STEP_RESULTS_DIR,
                               "step10_covariate_coefficients.csv")

# ---------------------------------------------------------------------------
# The two covariate sets. One list, one loop — a09_mood_fatigue.py and
# a09b_treatment.py differed only in these five fields.
#
#   standardize : person-mean centre, then z-score the pooled deviations
#                 (continuous items). False leaves the centred deviation on its
#                 natural "treated vs not, relative to own average" scale.
#   binary      : round back to {0,1} after interpolation, before centring.
# ---------------------------------------------------------------------------
BLOCKS = [
    {
        "key": "fm",
        "set": "fatiguemood",
        "covars": {"q11_fatigue": "fatigue", "q12_mood": "mood"},
        "standardize": True,
        "binary": False,
        # (model label, which covariate timing enters X_tv)
        "models": [("M0", None), ("C1", "lag"), ("C2", "cur")],
        "endorsement": False,
    },
    {
        "key": "tx",
        "set": "treatment",
        "covars": {"q14_pain_treatment": "surgery",
                   "q15_pain_treatment": "newtx"},
        "standardize": False,
        "binary": True,
        "models": [("M0", None), ("T1", "lag"), ("T2", "cur")],
        # q14/q15 are endorsements; the surrounding prose quotes how often they occur.
        "endorsement": True,
    },
]

#: posterior variable name -> the manuscript quantity it is Panel A's column for
COUPLING_PARAMS = [("b1", "lambda_ps"), ("a2", "lambda_sp"),
                   ("rho_innov", "rho_innov")]


# ===================================================================
# Small helpers
# ===================================================================

def resolve_raw_long():
    """First existing path in RAW_LONG_CANDIDATES."""
    for path in RAW_LONG_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "raw quarterly long frame not found; tried: "
        + ", ".join(RAW_LONG_CANDIDATES)
    )


def _cols(block, timing):
    """Covariate column names for one block at one timing.

    timing is "cur" (same quarter) or "lag" (preceding quarter); these are the
    names ``build_timevarying`` attaches.
    """
    suffix = "_within" if timing == "cur" else "_within_lag1"
    return [f"{name}{suffix}" for name in block["covars"].values()]


def _stem_parts(stem):
    """('fatigue_within_lag1') -> ('fatigue', '_lag1');  ('mood_within') -> ('mood', '')."""
    if stem.endswith("_within_lag1"):
        return stem[: -len("_within_lag1")], "_lag1"
    return stem[: -len("_within")], ""


def _theta_key(block_key, model, equation, stem):
    """numbers.json key for one theta, e.g. fm_C1_theta_pain_fatigue_lag1."""
    name, lag = _stem_parts(stem)
    return f"{block_key}_{model}_theta_{equation}_{name}{lag}"


def _row(summary, cov_set, model, param):
    """One row of the tidy summary, or None if that parameter is not in the fit."""
    hit = summary[(summary["covariate_set"] == cov_set)
                  & (summary["model"] == model)
                  & (summary["param"] == param)]
    return None if hit.empty else hit.iloc[0]


# ===================================================================
# Design: the eight covariate columns and the two complete-case samples
# ===================================================================

def build_design(model_df, df_full, raw_long, verbose=True):
    """Attach every block's covariates to the primary frame, once.

    One row per primary transition. Each block contributes its covariates at t
    and at t-1 plus a ``complete_<set>`` flag marking the transitions on which
    that block's three fits are estimated. Persisting the flags is what makes
    the analytic-sample counts (and therefore "eleven fewer than the primary
    analysis") reproducible without refitting.

    The per-column construction — single-interior-gap interpolation, person-mean
    centring, optional z-scoring, optional rounding for binary items, and the
    within-segment lag — is `lib.coupling_model.build_timevarying`, which is the
    merged canonical form of a09's and a09b's two near-identical builders.
    """
    from coupling_model import build_timevarying

    design = model_df[["ID", "quarter"]].copy()
    for block in BLOCKS:
        md = build_timevarying(
            model_df, df_full, raw_long, block["covars"],
            standardize=block["standardize"], binary=block["binary"],
        )
        need = _cols(block, "lag") + _cols(block, "cur")
        add = md[["ID", "quarter"] + need].copy()
        add[f"complete_{block['set']}"] = add[need].notna().all(axis=1)
        design = design.merge(add, on=["ID", "quarter"], how="left")

        if verbose:
            n_ok = int(design[f"complete_{block['set']}"].sum())
            print(f"  {block['set']}: complete on all covariates "
                  f"{n_ok} / {len(design)} transitions "
                  f"({100 * n_ok / max(len(design), 1):.1f}%)")
            for c in need:
                print(f"     {c:<28} missing {int(design[c].isna().sum()):>4}")
    return design


def block_sample(model_df, design, block):
    """The transitions, ids and rebuilt person index for one block's three fits.

    ``pid_idx`` MUST be rebuilt on the subset: fit_bayesian_varx1 reads
    ``model_df['pid_idx']`` directly when X_person is None and sizes the random
    effects from the id list, so carrying the parent frame's index would
    silently mis-index every person-specific slope.
    """
    flag = design.set_index(["ID", "quarter"])[
        f"complete_{block['set']}"].astype(bool)
    keys = list(zip(model_df["ID"], model_df["quarter"]))
    keep = np.array([bool(flag.get(k, False)) for k in keys])

    cov_cols = _cols(block, "lag") + _cols(block, "cur")
    sub = model_df[keep].merge(
        design[["ID", "quarter"] + cov_cols], on=["ID", "quarter"], how="left"
    )
    sub_ids = sorted(sub["ID"].unique())
    id_map = {p: i for i, p in enumerate(sub_ids)}
    sub["pid_idx"] = sub["ID"].map(id_map)
    return sub, sub_ids, id_map


# ===================================================================
# The six fits
# ===================================================================

def fit_blocks(model_df, design, verbose=True):
    """Fit M0/C1/C2 and M0/T1/T2 and return one tidy posterior summary.

    Every fit goes through ``fit_bayesian_varx1`` with a ``fit_id`` and
    ``out_dir``, so ``lib.run_fit`` samples at the project settings and writes
    this fit's diagnostics beside it. This step names no sampler settings.
    """
    from coupling_model import fit_bayesian_varx1, tidy_posterior_summary

    frames = []
    for block in BLOCKS:
        sub, sub_ids, id_map = block_sample(model_df, design, block)
        if verbose:
            print(f"\n  === {block['set']}: {len(sub)} transitions, "
                  f"{len(sub_ids)} persons")

        for label, timing in block["models"]:
            tv = None if timing is None else _cols(block, timing)
            fit_id = f"step10_{block['set']}_{label}"
            if verbose:
                print(f"  --- fitting {label} "
                      + (f"(X_tv={tv})" if tv else "(primary spec)"))

            idata, sub_df, valid_ids = fit_bayesian_varx1(
                sub, sub_ids, id_map,
                X_tv=tv,
                include_agesex=True,
                progressbar=verbose,
                fit_id=fit_id, out_dir=STEP_DERIV_DIR,
            )

            frames.append(tidy_posterior_summary(
                idata,
                labels={"covariate_set": block["set"], "model": label},
                n_obs=len(sub_df), n_persons=len(valid_ids),
            ))

            nc_path = os.path.join(STEP_DERIV_DIR, f"{fit_id}_idata.nc")
            idata.to_netcdf(nc_path)
            if verbose:
                print(f"      saved {os.path.relpath(nc_path, ROOT)}")
            del idata

    return pd.concat(frames, ignore_index=True)


def collect_diagnostics(verbose=True):
    """The six per-fit diagnostics records lib.run_fit wrote, as one frame.

    Read from ``diagnostics_<fit_id>.json`` rather than recomputed: R-hat and
    ESS need the chain structure, which the saved summary no longer carries.
    Falls back to a previously written CSV so the default path works after a
    cleanup of the JSONs.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(STEP_DERIV_DIR,
                                              "diagnostics_step10_*.json"))):
        with open(path) as fh:
            rows.append(json.load(fh))
    if rows:
        return pd.DataFrame(rows)
    if os.path.exists(OUT_DIAG_CSV):
        return pd.read_csv(OUT_DIAG_CSV, float_precision="round_trip")
    if verbose:
        print("  WARNING: no per-fit diagnostics found; the six-fit convergence "
              "numbers will be omitted from numbers.json. Run with --refit.")
    return pd.DataFrame()


# ===================================================================
# the time-varying covariate tables — the two panels, as DATA
# ===================================================================

def panel_a(summary):
    """Panel A: coupling and innovation correlation under each of the six models."""
    rows = []
    for block in BLOCKS:
        for label, _ in block["models"]:
            row = {"covariate_set": block["set"], "model": label}
            first = _row(summary, block["set"], label, "b1")
            row["n_obs"] = int(first["n_obs"]) if first is not None else np.nan
            row["n_persons"] = int(first["n_persons"]) if first is not None else np.nan
            for param, name in COUPLING_PARAMS:
                r = _row(summary, block["set"], label, param)
                row[f"{name}_mean"] = np.nan if r is None else float(r["mean"])
                row[f"{name}_ci_lo"] = np.nan if r is None else float(r["ci_lo_2.5"])
                row[f"{name}_ci_hi"] = np.nan if r is None else float(r["ci_hi_97.5"])
                row[f"{name}_P_neg"] = np.nan if r is None else float(r["P_neg"])
            rows.append(row)
    return pd.DataFrame(rows)


def panel_b(summary):
    """Panel B: the sixteen covariate coefficients (theta), pain and sleep equations."""
    rows = []
    for block in BLOCKS:
        for label, timing in block["models"]:
            if timing is None:
                continue
            for stem in _cols(block, timing):
                name, lag = _stem_parts(stem)
                row = {"covariate_set": block["set"], "model": label,
                       "covariate": name,
                       "timing": "t-1" if lag else "t",
                       "column": stem}
                for eq, prefix in (("pain", "theta_p_"), ("sleep", "theta_s_")):
                    r = _row(summary, block["set"], label, f"{prefix}{stem}")
                    row[f"{eq}_mean"] = np.nan if r is None else float(r["mean"])
                    row[f"{eq}_ci_lo"] = np.nan if r is None else float(r["ci_lo_2.5"])
                    row[f"{eq}_ci_hi"] = np.nan if r is None else float(r["ci_hi_97.5"])
                    row[f"{eq}_P_neg"] = np.nan if r is None else float(r["P_neg"])
                rows.append(row)
    return pd.DataFrame(rows)


# ===================================================================
# The numbers registry
# ===================================================================

def endorsement_counts(raw_long_df, block):
    """How often a binary item was endorsed, on the RAW (uninterpolated) column."""
    out = {}
    for src in block["covars"]:
        stem = src.split("_")[0]          # q14_pain_treatment -> q14
        col = raw_long_df[src]
        observed = int(col.notna().sum())
        endorsed = int(np.nansum(col.values))
        out[f"{stem}_endorsed_n"] = endorsed
        out[f"{stem}_observed_n"] = observed
        out[f"{stem}_endorsed_pct"] = (
            float(100.0 * endorsed / observed) if observed else float("nan")
        )
    return out


def assemble_numbers(summary, design, raw_long_df, diagnostics):
    """Every quantity the time-varying covariate tables quote, under a stable key.

    Derived entirely from the persisted summary, the persisted design frame and
    the raw item columns, so it is reproducible without refitting.
    """
    nums = {}

    n_primary = int(len(design))
    nums["n_transitions_primary"] = n_primary
    nums["n_persons_primary"] = int(design["ID"].nunique())

    for block in BLOCKS:
        bset, bkey = block["set"], block["key"]
        flag = design[f"complete_{bset}"].astype(bool)
        n_set = int(flag.sum())
        nums[f"n_transitions_{bset}"] = n_set
        nums[f"n_persons_{bset}"] = int(design.loc[flag, "ID"].nunique())
        nums[f"n_transitions_dropped_{bset}"] = n_primary - n_set

        # Counted on ALL primary transitions, before the complete-case filter:
        # these are the per-column missingness a09/a09b only printed to stdout.
        for col in _cols(block, "cur") + _cols(block, "lag"):
            nums[f"n_missing_{col}"] = int(design[col].isna().sum())

        if block["endorsement"]:
            nums.update(endorsement_counts(raw_long_df, block))

        for label, timing in block["models"]:
            for param, name in COUPLING_PARAMS:
                r = _row(summary, bset, label, param)
                if r is None:
                    continue
                stub = f"{bkey}_{label}_{name}"
                nums[stub] = float(r["mean"])
                nums[f"{stub}_ci"] = [float(r["ci_lo_2.5"]), float(r["ci_hi_97.5"])]
                if name != "rho_innov":
                    nums[f"{stub}_pneg"] = float(r["P_neg"])

            if timing is None:
                continue
            for stem in _cols(block, timing):
                for eq, prefix in (("pain", "theta_p_"), ("sleep", "theta_s_")):
                    r = _row(summary, bset, label, f"{prefix}{stem}")
                    if r is None:
                        continue
                    key = _theta_key(bkey, label, eq, stem)
                    nums[key] = float(r["mean"])
                    nums[f"{key}_ci"] = [float(r["ci_lo_2.5"]),
                                         float(r["ci_hi_97.5"])]

        # The "concurrent adjustment roughly halved the innovation correlation"
        # claim is a derived ratio, so it gets its own key: if a refit stops it
        # being roughly half, the document checker sees it.
        base, conc = block["models"][0][0], block["models"][2][0]
        r0 = _row(summary, bset, base, "rho_innov")
        r2 = _row(summary, bset, conc, "rho_innov")
        if r0 is not None and r2 is not None and float(r0["mean"]) != 0.0:
            nums[f"rho_innov_reduction_pct_{bset}_{base}_to_{conc}"] = float(
                100.0 * (1.0 - float(r2["mean"]) / float(r0["mean"]))
            )

    if not diagnostics.empty:
        nums["rhat_max_six_fits"] = float(diagnostics["rhat_max"].max())
        nums["ess_bulk_min_six_fits"] = float(diagnostics["ess_bulk_min"].min())
        nums["ess_tail_min_six_fits"] = float(diagnostics["ess_tail_min"].min())
        nums["divergences_total_six_fits"] = int(diagnostics["divergences"].sum())

    return nums


# ===================================================================
# Entry point
# ===================================================================

def run_step10(verbose=True, refit=False):
    """Rebuild the time-varying covariate tables and numbers.json; ``refit`` re-runs the six fits."""
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("STEP 10 — Time-varying covariates (the time-varying covariate tables)")
        print("=" * 70)

    raw_long_path = resolve_raw_long()
    raw_cols = ["ID", "quarter"]
    for block in BLOCKS:
        raw_cols += list(block["covars"])
    raw_long_df = pd.read_csv(raw_long_path, usecols=raw_cols, float_precision="round_trip")
    if verbose:
        print(f"  raw items: {os.path.relpath(raw_long_path, ROOT)}")

    saved_exist = (os.path.exists(OUT_SUMMARY_CSV)
                   and os.path.exists(OUT_DESIGN_CSV))
    if not refit and not saved_exist:
        if verbose:
            print("  Saved derivatives not found — forcing refit.")
        refit = True

    if not refit:
        # ------ DEFAULT: load saved derivatives, emit tables and numbers ------
        if verbose:
            print("  Loading saved derivatives (no sampling).")
            print("  If upstream data or code changed, re-run with --refit.")
        summary = pd.read_csv(OUT_SUMMARY_CSV, float_precision="round_trip")
        design = pd.read_csv(OUT_DESIGN_CSV, float_precision="round_trip")
    else:
        # ------ FULL: six MCMC fits ------
        from coupling_model import load_varx_frame

        if verbose:
            print(f"  Input: {os.path.relpath(IN_PROCESSED_CSV, ROOT)}")
        df_full, model_df, unique_ids, id_map = load_varx_frame(
            IN_PROCESSED_CSV, verbose=verbose)
        if verbose:
            print(f"  primary analytic set: {len(model_df)} transitions, "
                  f"{len(unique_ids)} persons")

        design = build_design(model_df, df_full, raw_long_df, verbose=verbose)
        design.to_csv(OUT_DESIGN_CSV, index=False)
        if verbose:
            print(f"  Saved design: {os.path.relpath(OUT_DESIGN_CSV, ROOT)}")

        summary = fit_blocks(model_df, design, verbose=verbose)
        summary.to_csv(OUT_SUMMARY_CSV, index=False)
        if verbose:
            print(f"\n  Saved posterior summary: "
                  f"{os.path.relpath(OUT_SUMMARY_CSV, ROOT)}")

    diagnostics = collect_diagnostics(verbose=verbose)
    if not diagnostics.empty:
        diagnostics.to_csv(OUT_DIAG_CSV, index=False)

    pa = panel_a(summary)
    pa.to_csv(OUT_PANEL_A_CSV, index=False)
    pb = panel_b(summary)
    pb.to_csv(OUT_PANEL_B_CSV, index=False)

    nums = assemble_numbers(summary, design, raw_long_df, diagnostics)
    from registry import write_numbers
    numbers_path = write_numbers(STEP_RESULTS_DIR, nums, prefix="step10")

    if verbose:
        print(f"\n  Saved Panel A ({len(pa)} rows): "
              f"{os.path.relpath(OUT_PANEL_A_CSV, ROOT)}")
        print(f"  Saved Panel B ({len(pb)} rows): "
              f"{os.path.relpath(OUT_PANEL_B_CSV, ROOT)}")
        print(f"  Saved numbers ({len(nums)} keys): "
              f"{os.path.relpath(numbers_path, ROOT)}")
        for block in BLOCKS:
            bkey, bset = block["key"], block["set"]
            print(f"\n  {bset}: n = {nums.get(f'n_transitions_{bset}')} "
                  f"transitions, {nums.get(f'n_persons_{bset}')} persons "
                  f"({nums.get(f'n_transitions_dropped_{bset}')} fewer than primary)")
            for label, _ in block["models"]:
                ps = nums.get(f"{bkey}_{label}_lambda_ps")
                sp = nums.get(f"{bkey}_{label}_lambda_sp")
                rho = nums.get(f"{bkey}_{label}_rho_innov")
                if ps is None:
                    continue
                print(f"    {label}: lambda_ps={ps:+.3f} "
                      f"P(<0)={nums.get(f'{bkey}_{label}_lambda_ps_pneg'):.3f}   "
                      f"lambda_sp={sp:+.3f}   rho_innov={rho:+.3f}")
        if "rhat_max_six_fits" in nums:
            print(f"\n  six fits: R-hat <= {nums['rhat_max_six_fits']:.4f}, "
                  f"bulk ESS >= {nums['ess_bulk_min_six_fits']:.0f}, "
                  f"tail ESS >= {nums['ess_tail_min_six_fits']:.0f}, "
                  f"divergences {nums['divergences_total_six_fits']}")
        print("\n" + "=" * 70)
        print("STEP 10 COMPLETE")
        print("=" * 70)

    return summary, design, nums


def main():
    parser = argparse.ArgumentParser(
        description="Step 10 — time-varying covariate sensitivity ."
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--refit", action="store_true",
                        help="Re-run all six fits instead of loading saved derivatives")
    args = parser.parse_args()
    run_step10(verbose=not args.quiet, refit=args.refit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
