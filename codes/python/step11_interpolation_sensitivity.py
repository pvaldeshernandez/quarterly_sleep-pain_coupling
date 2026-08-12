#!/usr/bin/env python3
"""
Step 11 — Sensitivity of the coupling estimates to interpolated observations.
======================================================================

Feeds Table S7 and supplement Section S7.

Step 01 fills isolated gaps in the scored items before the VARX frame is built, so a
transition can carry an interpolated value at its outcome end (t), at its predictor end
(t-1), or both. This step refits the PRIMARY specification to the transitions that carry
NO interpolated value at EITHER endpoint, and places that fit beside the primary fit.

The primary ("all transitions") column is READ from step 08, never refitted here. Two
executions of the same code with the same seed do not land on the same third decimal, and
a local refit of the primary model is how a supplement table drifts away from Table 4
without anyone noticing. One fit runs in this step: the interpolation-free arm.

A transition is excluded when EITHER endpoint is interpolated. `interpolated` is a
ROW-level OR across pain, contrast and sleep factor scores, so a transition is dropped if
any of the three was filled at either end. That is deliberately stricter than "the two
modeled series were interpolated" and matches the supplement's wording.

Input:
  derivatives/step07_varx_data/step07_processed_long.csv
      ID, quarter, interpolated, the within/lag columns, Age, Sex
  derivatives/step08_coupling_model/step08_posterior_summary.csv   (preferred)
      the primary fit's tidy summary, if step 08 publishes one
  results/step08_coupling_model/step08_table4_coupling.csv         (fallback estimates)
  derivatives/step08_coupling_model/diagnostics_step08_primary_by_param.csv
      per-parameter R-hat / bulk ESS of the primary fit, written by lib.write_diagnostics
  results/step08_coupling_model/numbers.json                        (optional)
      n_transitions / n_persons of the primary fit, for the equality assertion

Output:
  derivatives/step11_interpolation_sensitivity/
      step11_nointerp_idata.nc          posterior draws of the interpolation-free fit
      step11_posterior_summary.csv      arm x param tidy summary (both arms)
      step11_transition_flags.csv       per-transition audit trail behind the counts
      diagnostics_step11_nointerp.json  written by lib.coupling_model.run_fit
      diagnostics_step11_nointerp_by_param.csv
  results/step11_interpolation_sensitivity/
      step11_tableS7_interpolation.csv  Table S7 DATA (row label + numeric cells only)
      step11_figure_interpolation_sensitivity.png
      numbers.json

No prose is generated. Table notes, column headers and captions are authored by hand from
numbers.json.

Usage:
    python step11_interpolation_sensitivity.py            # load and plot (default)
    python step11_interpolation_sensitivity.py --refit    # re-run the one MCMC fit

Author: Pedro Valdes-Hernandez (with Claude Opus 5)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =====================================================================
# Paths
# =====================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repo root

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

DERIV_DIR = os.path.join(ROOT, "derivatives")
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step11_interpolation_sensitivity")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step11_interpolation_sensitivity")

# --- inputs (data/ and data/original/ are never touched by this step) ---
IN_PROCESSED_CSV = os.path.join(DERIV_DIR, "step07_varx_data",
                                "step07_processed_long.csv")
STEP08_DERIV_DIR = os.path.join(DERIV_DIR, "step08_coupling_model")
STEP08_RESULTS_DIR = os.path.join(RESULTS_DIR, "step08_coupling_model")
#: preferred source for the primary arm — one tidy frame carrying estimates AND
#: diagnostics. Read it if step 08 publishes it; otherwise the two files below.
IN_STEP08_SUMMARY = os.path.join(STEP08_DERIV_DIR, "step08_posterior_summary.csv")
IN_STEP08_TABLE4 = os.path.join(STEP08_RESULTS_DIR, "step08_table4_coupling.csv")
IN_STEP08_BYPARAM = os.path.join(STEP08_DERIV_DIR,
                                 "diagnostics_step08_primary_by_param.csv")
IN_STEP08_NUMBERS = os.path.join(STEP08_RESULTS_DIR, "numbers.json")
IN_STEP08_PERSON = os.path.join(STEP08_DERIV_DIR, "step08_person_coupling.csv")

# --- outputs ---
OUT_NI_IDATA = os.path.join(STEP_DERIV_DIR, "step11_nointerp_idata.nc")
OUT_SUMMARY_CSV = os.path.join(STEP_DERIV_DIR, "step11_posterior_summary.csv")
OUT_FLAGS_CSV = os.path.join(STEP_DERIV_DIR, "step11_transition_flags.csv")
OUT_TABLE_S7_CSV = os.path.join(STEP_RESULTS_DIR,
                                "step11_tableS7_interpolation.csv")
OUT_FIGURE = os.path.join(STEP_RESULTS_DIR,
                          "step11_figure_interpolation_sensitivity.png")
OUT_NUMBERS = os.path.join(STEP_RESULTS_DIR, "numbers.json")

#: the fit this step runs. `lib.coupling_model.run_fit` writes its diagnostics under
#: this id, and step 24 aggregates them.
FIT_ID = "step11_nointerp"
IN_NI_DIAGNOSTICS = os.path.join(STEP_DERIV_DIR, f"diagnostics_{FIT_ID}.json")

#: the fit whose published output supplies the "all transitions" column. Recorded in
#: numbers.json so the provenance of that column is not folklore.
ALL_ARM_FIT_ID = "step08_primary"

# =====================================================================
# Schema
# =====================================================================

#: One row per arm x population-level parameter. `rhat` and `ess_bulk` describe the fit
#: that produced the row: for `nointerp` they come from this step's fit, for `all` from
#: step 08's per-parameter diagnostics table.
SUMMARY_COLUMNS = ["arm", "param", "mean", "sd", "ci_lo_2.5", "ci_hi_97.5",
                   "P_neg", "rhat", "ess_bulk", "n_obs", "n_persons"]

ARM_ALL = "all"
ARM_NI = "nointerp"

#: (manuscript name, script name, Table S7 row label) in the table's row order.
#: Script-internal names (a1/a2/b1/...) are never renamed in code; the manuscript name
#: is what numbers.json keys on, so a value typed into the supplement traces back.
MANUSCRIPT_PARAMS = [
    ("phi_p", "a1", "Pain autoregression"),
    ("lambda_sp", "a2", "Sleep-to-pain coupling"),
    ("phi_s", "b2", "Sleep autoregression"),
    ("lambda_ps", "b1", "Pain-to-sleep coupling"),
    ("delta_s", "b3", "Localization-to-sleep direct effect"),
    ("tau_sp", "tau_sp", "Between-person SD of sleep-to-pain coupling"),
    ("tau_ps", "tau_ps", "Between-person SD of pain-to-sleep coupling"),
    ("rho_innov", "rho_innov", "Innovation correlation"),
]

#: parameters whose probability of direction the supplement quotes
PNEG_PARAMS = ("lambda_sp", "lambda_ps")


# =====================================================================
# The transition filter
# =====================================================================

def flag_interpolated_transitions(df_full, model_df):
    """Audit every transition for interpolated endpoints and return the clean subset.

    Concept, not format: a transition is identified by (person, quarter) and is
    contaminated when the row at `quarter` or the row at `quarter - 1` carries an
    interpolated value. Nothing here assumes a fixed number of quarters, contiguous
    quarters, or a fixed row order.

    Rebuilding `pid_idx` on the subset is MANDATORY and not cosmetic:
    ``fit_bayesian_varx1`` reads ``model_df["pid_idx"]`` directly when ``X_person`` is
    None and ignores ``id_map``, so a subset that kept the parent frame's indices would
    silently mis-index every person-level random effect.

    Parameters
    ----------
    df_full : DataFrame
        The full long frame from ``load_varx_frame`` — every retained (ID, quarter),
        including the rows that have no lagged predecessor.
    model_df : DataFrame
        The transitions actually modeled (``df_full`` minus rows with NaN lags).

    Returns
    -------
    flags : DataFrame
        ID, quarter, interp_at_t, interp_at_t_minus_1, contaminated — one row per
        transition, in ``model_df`` order.
    clean : DataFrame
        ``model_df`` restricted to uncontaminated transitions, with ``pid_idx`` rebuilt.
    clean_ids : list
        Sorted person IDs surviving in ``clean``.
    """
    if "interpolated" not in df_full.columns:
        raise RuntimeError(
            f"no 'interpolated' flag in {IN_PROCESSED_CSV}; step 07 must carry it "
            "forward from step 01 or this sensitivity analysis cannot be defined"
        )

    keyed = df_full.set_index(["ID", "quarter"])["interpolated"].astype(bool)
    if keyed.index.has_duplicates:
        raise RuntimeError(
            "(ID, quarter) is not unique in the step 07 frame; the interpolation flag "
            "cannot be attributed to a transition endpoint"
        )
    lookup = keyed.to_dict()

    ids = model_df["ID"].to_numpy()
    quarters = model_df["quarter"].to_numpy()
    at_t = np.array([bool(lookup.get((i, q), False))
                     for i, q in zip(ids, quarters)])
    at_prev = np.array([bool(lookup.get((i, q - 1), False))
                        for i, q in zip(ids, quarters)])

    flags = pd.DataFrame({
        "ID": ids,
        "quarter": quarters,
        "interp_at_t": at_t,
        "interp_at_t_minus_1": at_prev,
        "contaminated": at_t | at_prev,
    })

    clean = model_df.loc[~flags["contaminated"].to_numpy()].copy()
    clean_ids = sorted(clean["ID"].unique())
    clean["pid_idx"] = clean["ID"].map({p: i for i, p in enumerate(clean_ids)})
    return flags, clean, clean_ids


# =====================================================================
# The primary ("all transitions") arm — read, never refitted
# =====================================================================

def _conform(df, arm, n_obs, n_persons):
    """Force a summary frame onto SUMMARY_COLUMNS so both arms share one schema."""
    out = df.copy()
    out["arm"] = arm
    out["n_obs"] = n_obs
    out["n_persons"] = n_persons
    for col in SUMMARY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[SUMMARY_COLUMNS].reset_index(drop=True)


def _lookup_number(numbers, name):
    """Read one value out of a registry numbers.json, whatever prefix it carries."""
    if name in numbers:
        return numbers[name]
    hits = [v for k, v in numbers.items() if k.split(".")[-1] == name]
    return hits[0] if len(hits) == 1 else None


def primary_counts(model_df, unique_ids, verbose=True):
    """The primary fit's transition and person counts, cross-checked against step 08.

    The counts are recomputed here from step 08's frame and then asserted equal to what
    step 08 published. If they ever disagree, the "all transitions" column is describing
    a different analytic sample than the one being compared against, and the step stops
    rather than publishing a comparison of two different datasets.
    """
    n_obs, n_persons = int(len(model_df)), int(len(unique_ids))

    published = {}
    if os.path.exists(IN_STEP08_NUMBERS):
        with open(IN_STEP08_NUMBERS) as fh:
            published = json.load(fh)
    ref_obs = _lookup_number(published, "n_transitions")
    ref_persons = _lookup_number(published, "n_persons")

    if ref_persons is None and os.path.exists(IN_STEP08_PERSON):
        # step 08 writes one row per person in the primary fit; a weaker check than
        # its numbers.json, but it is a real one and it exists today.
        ref_persons = int(len(pd.read_csv(IN_STEP08_PERSON, float_precision="round_trip")))
        if verbose:
            print(f"  step08 numbers.json absent — person count cross-checked "
                  f"against {os.path.relpath(IN_STEP08_PERSON, ROOT)}")

    mismatches = []
    if ref_obs is not None and int(ref_obs) != n_obs:
        mismatches.append(f"transitions: step07 frame {n_obs} vs step08 {int(ref_obs)}")
    if ref_persons is not None and int(ref_persons) != n_persons:
        mismatches.append(f"persons: step07 frame {n_persons} vs step08 "
                          f"{int(ref_persons)}")
    if mismatches:
        raise RuntimeError(
            "the primary fit and this step do not describe the same analytic sample "
            "(" + "; ".join(mismatches) + "). Re-run step08 --refit before comparing."
        )
    if ref_obs is None and verbose:
        print("  WARNING: step08 published no n_transitions; the all-transitions "
              "column header is taken from step 08's frame without cross-check.")
    return n_obs, n_persons


def load_primary_arm(n_obs, n_persons, verbose=True):
    """The primary fit's posterior summary, from step 08's published outputs.

    Preference order:
      1. step08_posterior_summary.csv — the tidy summary, estimates and diagnostics
         in one frame (this is what step 08 should publish; see lib_changes_needed).
      2. step08_table4_coupling.csv (estimates) merged with
         diagnostics_step08_primary_by_param.csv (per-parameter R-hat and bulk ESS).
         The per-parameter diagnostics are recoverable only from that file: step 08's
         saved draws are flattened with no chain dimension, so R-hat cannot be
         recomputed downstream.

    A missing diagnostics source leaves `rhat`/`ess_bulk` NaN for the `all` arm and says
    so out loud — it never triggers a local refit of the primary model.
    """
    if os.path.exists(IN_STEP08_SUMMARY):
        summ = pd.read_csv(IN_STEP08_SUMMARY, float_precision="round_trip")
        if "arm" in summ.columns:
            summ = summ.drop(columns=["arm"])
        if verbose:
            print(f"  all-transitions arm read from "
                  f"{os.path.relpath(IN_STEP08_SUMMARY, ROOT)}")
        return _conform(summ, ARM_ALL, n_obs, n_persons)

    if not os.path.exists(IN_STEP08_TABLE4):
        raise FileNotFoundError(
            f"neither {IN_STEP08_SUMMARY} nor {IN_STEP08_TABLE4} exists; run "
            "step08_fit_coupling_model.py --refit before this step"
        )
    tab = pd.read_csv(IN_STEP08_TABLE4, float_precision="round_trip").rename(columns={
        "Parameter": "param", "Estimate": "mean", "SD": "sd",
        "CrI_lo": "ci_lo_2.5", "CrI_hi": "ci_hi_97.5",
    })[["param", "mean", "sd", "ci_lo_2.5", "ci_hi_97.5", "P_neg"]]

    if os.path.exists(IN_STEP08_BYPARAM):
        diag = pd.read_csv(IN_STEP08_BYPARAM, index_col=0, float_precision="round_trip")
        diag = diag.rename(columns={"r_hat": "rhat"})
        diag = diag[[c for c in ("rhat", "ess_bulk") if c in diag.columns]]
        diag.index.name = "param"
        tab = tab.merge(diag.reset_index(), on="param", how="left")
    elif verbose:
        print("  WARNING: no per-parameter diagnostics for the primary fit "
              f"({os.path.relpath(IN_STEP08_BYPARAM, ROOT)} missing); the "
              "all-transitions arm carries estimates only.")

    if verbose:
        print(f"  all-transitions arm read from "
              f"{os.path.relpath(IN_STEP08_TABLE4, ROOT)}")
    return _conform(tab, ARM_ALL, n_obs, n_persons)


# =====================================================================
# The interpolation-free fit
# =====================================================================

def fit_nointerp_arm(clean, clean_ids, verbose=True):
    """Fit the PRIMARY specification to the interpolation-free transitions.

    Same call as step 08's primary fit — same model graph, same priors, same sampler.
    Sampling is delegated to ``lib.coupling_model.run_fit`` through
    ``fit_bayesian_varx1``; this step names no sampler setting anywhere, so it cannot
    drift from the settings the Methods describes, and its diagnostics are written for
    step 24 without this step remembering to do it.
    """
    from coupling_model import fit_bayesian_varx1, summarize_posterior

    id_map = {pid: i for i, pid in enumerate(clean_ids)}
    idata, _, _ = fit_bayesian_varx1(
        clean, clean_ids, id_map,
        include_agesex=True,
        progressbar=verbose,
        fit_id=FIT_ID, out_dir=STEP_DERIV_DIR,
    )

    try:
        idata.to_netcdf(OUT_NI_IDATA)
        if verbose:
            print(f"  Saved posterior draws: {OUT_NI_IDATA}")
    except Exception as exc:  # keep the summary, but never lose the failure
        print(f"  WARNING: could not write {OUT_NI_IDATA}: {exc}")

    summ = summarize_posterior(idata)
    return _conform(summ, ARM_NI, len(clean), len(clean_ids))


# =====================================================================
# Numbers
# =====================================================================

def _cell(summary, arm, param):
    """One parameter's row for one arm, or None if the arm does not carry it."""
    hit = summary[(summary["arm"] == arm) & (summary["param"] == param)]
    return hit.iloc[0] if len(hit) else None


def estimate_numbers(summary, verbose=True):
    """Every Table S7 cell, plus the two derived ratios the supplement states."""
    nums = {}
    missing = []
    for name, param, _label in MANUSCRIPT_PARAMS:
        for arm, suffix in ((ARM_ALL, "all"), (ARM_NI, "nointerp")):
            row = _cell(summary, arm, param)
            if row is None:
                missing.append(f"{arm}/{param}")
                continue
            nums[f"{name}_{suffix}"] = float(row["mean"])
            nums[f"{name}_{suffix}_ci"] = [float(row["ci_lo_2.5"]),
                                           float(row["ci_hi_97.5"])]
            if name in PNEG_PARAMS and pd.notna(row["P_neg"]):
                nums[f"{name}_{suffix}_pneg"] = float(row["P_neg"])
    if missing:
        raise RuntimeError(
            "Table S7 cannot be assembled; missing parameter(s): "
            f"{sorted(missing)}"
        )

    # Discussion: the pain autoregression roughly halves without interpolated endpoints.
    all_phi, ni_phi = nums["phi_p_all"], nums["phi_p_nointerp"]
    nums["phi_p_ratio_nointerp_over_all"] = (
        float(ni_phi / all_phi) if all_phi else float("nan"))

    # Section S7: the pain-to-sleep estimate barely moves. Stated as a fraction of the
    # primary fit's posterior SD, so the claim is checkable and not eyeballed.
    shift = nums["lambda_ps_nointerp"] - nums["lambda_ps_all"]
    nums["lambda_ps_shift_nointerp_minus_all"] = float(shift)
    all_sd = _cell(summary, ARM_ALL, "b1")["sd"]
    nums["lambda_ps_shift_in_posterior_sd"] = (
        float(shift / all_sd) if pd.notna(all_sd) and all_sd else float("nan"))

    if verbose:
        print(f"  phi_p ratio (nointerp / all): "
              f"{nums['phi_p_ratio_nointerp_over_all']:.3f}")
        print(f"  lambda_ps shift: {shift:+.4f} "
              f"({nums['lambda_ps_shift_in_posterior_sd']:+.3f} posterior SD)")
    return nums


def diagnostic_numbers(verbose=True):
    """The interpolation-free fit's diagnostics, as `run_fit` recorded them."""
    if not os.path.exists(IN_NI_DIAGNOSTICS):
        if verbose:
            print(f"  WARNING: {os.path.relpath(IN_NI_DIAGNOSTICS, ROOT)} missing; "
                  "diagnostics for the interpolation-free fit are not reported.")
        return {}
    with open(IN_NI_DIAGNOSTICS) as fh:
        rec = json.load(fh)
    out = {}
    for key, source in (("rhat_max_nointerp", "rhat_max"),
                        ("rhat_max_param_nointerp", "rhat_max_param"),
                        ("ess_bulk_min_nointerp", "ess_bulk_min"),
                        ("ess_tail_min_nointerp", "ess_tail_min"),
                        ("divergences_nointerp", "divergences"),
                        ("bfmi_min_nointerp", "bfmi_min"),
                        ("max_tree_depth_nointerp", "max_tree_depth")):
        if rec.get(source) is not None:
            out[key] = rec[source]
    return out


def count_numbers(flags, n_obs, n_persons, n_interp_rows):
    """The 'X of Y transitions' block of Section S7, all of it derived, none hardcoded."""
    n_bad = int(flags["contaminated"].sum())
    clean_flags = flags[~flags["contaminated"]]
    n_clean_persons = int(clean_flags["ID"].nunique())
    return {
        # the primary fit's sample, named the same way step 08 and step 10 name it
        "n_transitions_primary": int(n_obs),
        "n_persons_primary": int(n_persons),
        # deliberately NOT `n_rows_interpolated`: step 01 publishes that name for the
        # interpolated rows of the FULL scored frame, a larger number. This one counts
        # only the rows that survived into the analytic frame.
        "n_interpolated_rows_in_analytic_frame": int(n_interp_rows),
        "n_transitions_interp_at_t": int(flags["interp_at_t"].sum()),
        "n_transitions_interp_at_t_minus_1": int(flags["interp_at_t_minus_1"].sum()),
        "n_transitions_contaminated": n_bad,
        "pct_transitions_contaminated": float(100.0 * n_bad / len(flags)),
        "n_transitions_nointerp": int(len(clean_flags)),
        "n_persons_nointerp": n_clean_persons,
        "n_persons_dropped": int(n_persons - n_clean_persons),
        "all_arm_source_fit_id": ALL_ARM_FIT_ID,
    }


def _saved_counts():
    """The count block of a previous run, read back from its numbers.json."""
    with open(OUT_NUMBERS) as fh:
        saved = json.load(fh)
    keys = ["n_transitions_primary", "n_persons_primary",
            "n_interpolated_rows_in_analytic_frame",
            "n_transitions_interp_at_t", "n_transitions_interp_at_t_minus_1",
            "n_transitions_contaminated", "pct_transitions_contaminated",
            "n_transitions_nointerp", "n_persons_nointerp", "n_persons_dropped",
            "all_arm_source_fit_id"]
    out = {}
    for key in keys:
        val = _lookup_number(saved, key)
        if val is not None:
            out[key] = val
    return out


# =====================================================================
# Table data and figure (both regenerated on every run, from the summary)
# =====================================================================

def write_table_s7(summary, path):
    """Table S7 DATA: one row per parameter, the two arms side by side.

    Numbers and row labels only. The column headers ('1,818 transitions,
    229 participants'), the note and the caption are prose and are authored by hand
    from numbers.json.
    """
    rows = []
    for name, param, label in MANUSCRIPT_PARAMS:
        a = _cell(summary, ARM_ALL, param)
        n = _cell(summary, ARM_NI, param)
        rows.append({
            "row_label": label,
            "param_manuscript": name,
            "param_script": param,
            "all_mean": None if a is None else float(a["mean"]),
            "all_ci_lo": None if a is None else float(a["ci_lo_2.5"]),
            "all_ci_hi": None if a is None else float(a["ci_hi_97.5"]),
            "nointerp_mean": None if n is None else float(n["mean"]),
            "nointerp_ci_lo": None if n is None else float(n["ci_lo_2.5"]),
            "nointerp_ci_hi": None if n is None else float(n["ci_hi_97.5"]),
        })
    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table.to_csv(path, index=False)
    return table


def plot_arms(table, out_path):
    """Forest plot of the two arms, drawn from the table DATA only.

    Reads nothing but the frame it is handed, so restyling it never touches the fit.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(table)
    y = np.arange(n)[::-1].astype(float)
    fig, ax = plt.subplots(figsize=(9, 0.72 * n + 2.2))

    specs = [
        ("all_mean", "all_ci_lo", "all_ci_hi", +0.17, "#1565C0",
         "All transitions"),
        ("nointerp_mean", "nointerp_ci_lo", "nointerp_ci_hi", -0.17, "#D32F2F",
         "No interpolated endpoint"),
    ]
    for col_mean, col_lo, col_hi, offset, color, _label in specs:
        m = table[col_mean].to_numpy(dtype=float)
        lo = table[col_lo].to_numpy(dtype=float)
        hi = table[col_hi].to_numpy(dtype=float)
        ax.hlines(y + offset, lo, hi, color=color, linewidth=2.0, alpha=0.85,
                  zorder=2)
        ax.plot(m, y + offset, marker="o", linestyle="", color=color,
                markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                zorder=3)

    ax.axvline(0, color="black", linewidth=1.0, alpha=0.55, zorder=1)
    for k in range(n):
        if k % 2 == 0:
            ax.axhspan(y[k] - 0.5, y[k] + 0.5, color="#000000", alpha=0.035,
                       zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(table["row_label"], fontsize=11)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Posterior mean and 95% credible interval", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [Line2D([0], [0], color=c, marker="o", linewidth=2.0,
                      markersize=7, markeredgecolor="white", label=lab)
               for *_ignored, c, lab in specs]
    ax.legend(handles=handles, loc="lower right", fontsize=11, framealpha=0.9,
              edgecolor="#CCCCCC")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Entry point
# =====================================================================

def run_step11(verbose: bool = True, refit: bool = False):
    """Compare the primary fit with its interpolation-free refit (Table S7)."""
    from registry import write_numbers

    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("STEP 11 — Interpolation sensitivity of the coupling estimates")
        print("=" * 70)

    saved_exist = (os.path.exists(OUT_SUMMARY_CSV)
                   and os.path.exists(OUT_FLAGS_CSV)
                   and os.path.exists(OUT_NUMBERS))
    if not refit and not saved_exist:
        if verbose:
            print("  Saved derivatives not found — forcing refit.")
        refit = True

    if not refit:
        # -------- DEFAULT: load and plot; nothing is sampled --------
        if verbose:
            print("  Loading saved derivatives (replot mode).")
            print("  If upstream data or code changed, re-run with --refit.")
        summary = pd.read_csv(OUT_SUMMARY_CSV, float_precision="round_trip")
        counts = _saved_counts()
        if verbose:
            print(f"  Loaded {os.path.relpath(OUT_SUMMARY_CSV, ROOT)}")
    else:
        # -------- REFIT: one MCMC fit, the interpolation-free arm --------
        from coupling_model import load_varx_frame

        if verbose:
            print(f"  Input: {IN_PROCESSED_CSV}")
        df_full, model_df, unique_ids, _id_map = load_varx_frame(
            IN_PROCESSED_CSV, verbose=verbose)

        n_obs, n_persons = primary_counts(model_df, unique_ids, verbose=verbose)
        flags, clean, clean_ids = flag_interpolated_transitions(df_full, model_df)
        flags.to_csv(OUT_FLAGS_CSV, index=False)

        n_interp_rows = int(df_full["interpolated"].astype(bool).sum())
        counts = count_numbers(flags, n_obs, n_persons, n_interp_rows)

        if verbose:
            print(f"\n  interpolated rows in the analytic frame : "
                  f"{counts['n_interpolated_rows_in_analytic_frame']}")
            print(f"  transitions with interpolated t         : "
                  f"{counts['n_transitions_interp_at_t']}")
            print(f"  transitions with interpolated t-1       : "
                  f"{counts['n_transitions_interp_at_t_minus_1']}")
            print(f"  contaminated (either endpoint)          : "
                  f"{counts['n_transitions_contaminated']} / "
                  f"{counts['n_transitions_primary']} "
                  f"({counts['pct_transitions_contaminated']:.1f}%)")
            print(f"  interpolation-free set                  : "
                  f"{counts['n_transitions_nointerp']} transitions, "
                  f"{counts['n_persons_nointerp']} persons "
                  f"(lost {counts['n_persons_dropped']} persons)")
            print(f"  Saved transition flags: {OUT_FLAGS_CSV}")

        # Read the all-transitions arm BEFORE sampling: if step 08's outputs are
        # missing, fail in a second rather than after an hour of MCMC.
        all_arm = load_primary_arm(n_obs, n_persons, verbose=verbose)

        if verbose:
            print("\n  Fitting the interpolation-free arm "
                  "(the all-transitions arm is READ from step 08, never refitted)...")
        ni = fit_nointerp_arm(clean, clean_ids, verbose=verbose)
        summary = pd.concat([all_arm, ni], ignore_index=True)
        summary.to_csv(OUT_SUMMARY_CSV, index=False)
        if verbose:
            print(f"  Saved posterior summary: {OUT_SUMMARY_CSV}")

    # -------- everything below runs in both modes --------
    nums = dict(counts)
    nums.update(estimate_numbers(summary, verbose=verbose))
    nums.update(diagnostic_numbers(verbose=verbose))
    write_numbers(STEP_RESULTS_DIR, nums, prefix="step11")

    table = write_table_s7(summary, OUT_TABLE_S7_CSV)
    plot_arms(table, OUT_FIGURE)

    if verbose:
        print(f"\n  Saved Table S7 data: {OUT_TABLE_S7_CSV}")
        print(f"  Saved figure: {OUT_FIGURE}")
        print(f"  Saved numbers: {OUT_NUMBERS}")
        print("\n  Key comparison (posterior mean [95% CrI]):")
        for name, _param, label in MANUSCRIPT_PARAMS:
            a_ci = nums.get(f"{name}_all_ci", [float("nan")] * 2)
            n_ci = nums.get(f"{name}_nointerp_ci", [float("nan")] * 2)
            print(f"    {label:<44s} "
                  f"all {nums.get(f'{name}_all', float('nan')):+.4f} "
                  f"[{a_ci[0]:+.4f}, {a_ci[1]:+.4f}]   "
                  f"NI {nums.get(f'{name}_nointerp', float('nan')):+.4f} "
                  f"[{n_ci[0]:+.4f}, {n_ci[1]:+.4f}]")
        print("\n" + "=" * 70)
        print("STEP 11 COMPLETE")
        print("=" * 70)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Step 11 — interpolation sensitivity of the coupling estimates "
                    "(Table S7, Section S7)."
    )
    parser.add_argument("--refit", action="store_true",
                        help="Re-run the interpolation-free MCMC fit instead of "
                             "loading saved derivatives")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output.")
    args = parser.parse_args()
    run_step11(verbose=not args.quiet, refit=args.refit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
