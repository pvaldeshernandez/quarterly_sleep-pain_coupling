"""
Extension — the primary coupling model refit with LATENT mean centering.
=========================================================================

Not a pipeline step (no ``stepNN_`` prefix, so ``run_pipeline.py`` never runs it) and
it writes nothing under ``derivatives/`` or ``results/``. Every output goes to
``extensions/latent_centering/`` at the repository root.

Why
---
Step 07 subtracts each person's OBSERVED mean (about nine quarters) before step 08
fits the VARX(1). With a lagged outcome and short series that biases the
autoregressions down (Nickell 1981). ``fit_bayesian_varx1(centering="latent")``
estimates each person's usual pain and sleep levels inside the model instead
(the RI-CLPM random intercept of Hamaker et al. 2015). This script validates that
by simulation on the real panel, fits it to the real data, and tabulates it beside
the published fit (read from step 08's saved posterior, never refitted).

Subcommands (compute and report are decoupled; ``report`` only reads saved fits)
-----------
    simulate --rep K             write simulated dataset K on the real panel structure
    fit --data real|simK --centering observed|latent [--tune N --draws N]
                                 one fit; saves the InferenceData and diagnostics
    report                       write every CSV the handoff asks for

Run from the repository root:
    python codes/python/extensions/latent_centering.py fit --data real --centering latent
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CODE_DIR))
sys.path.insert(0, os.path.join(CODE_DIR, "lib"))
sys.path.insert(0, CODE_DIR)

from coupling_model import (  # noqa: E402
    fit_bayesian_varx1, load_varx_frame, summarize_posterior,
)
import step07_prepare_varx_data as step07  # noqa: E402
import step25_optimal_lag as step25  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IN_VARX_CSV = os.path.join(ROOT, "derivatives", "step07_varx_data",
                           "step07_processed_long.csv")
IN_PUBLISHED_IDATA = os.path.join(ROOT, "derivatives", "step08_coupling_model",
                                  "step08_primary_idata.nc")
IN_PUBLISHED_NUMBERS = os.path.join(ROOT, "results", "step08_coupling_model",
                                    "numbers.json")

OUT_DIR = os.path.join(ROOT, "extensions", "latent_centering")
SIM_DIR = os.path.join(OUT_DIR, "simulated_data")
FIT_DIR = os.path.join(OUT_DIR, "fits")
OUT_VS_PUBLISHED = os.path.join(OUT_DIR, "latent_vs_published.csv")
OUT_OPTIMAL_LAG = os.path.join(OUT_DIR, "optimal_lag_latent_vs_published.csv")
OUT_RECOVERY = os.path.join(OUT_DIR, "recovery_check.csv")
OUT_DIAGNOSTICS = os.path.join(OUT_DIR, "diagnostics_all_fits.csv")

#: script-internal name -> manuscript symbol, every population parameter reported
PARAMS = {
    "a1": "phi_p", "b2": "phi_s", "a2": "lambda_sp", "b1": "lambda_ps",
    "a3": "delta_p", "b3": "delta_s", "a4": "omega_sp", "b4": "omega_ps",
    "g_sp_age": "gamma_sp_age", "g_sp_sex": "gamma_sp_sex",
    "g_ps_age": "gamma_ps_age", "g_ps_sex": "gamma_ps_sex",
    "tau_sp": "tau_sp", "tau_ps": "tau_ps",
    "sigma_pain": "sigma_pain", "sigma_sleep": "sigma_sleep",
    "rho_innov": "rho_innov",
    "a0": "mu_p (observed-centering intercept)",
    "b0": "mu_s (observed-centering intercept)",
    "mu_P": "mu_P (mean usual pain)", "mu_S": "mu_S (mean usual sleep)",
    "tau_aP": "tau_aP (SD usual pain)", "tau_aS": "tau_aS (SD usual sleep)",
    "rho_alpha": "rho_alpha (corr usual pain, usual sleep)",
}

# ---------------------------------------------------------------------------
# Simulation truth (handoff section 6: near the calibration's expectation)
# ---------------------------------------------------------------------------
TRUTH = dict(phi_p=0.27, phi_s=0.15, lambda_sp=-0.03, lambda_ps=-0.15,
             sigma_pain=0.437, sigma_sleep=0.661, rho_innov=-0.157,
             tau_ps=0.36, tau_sp=0.11,
             delta_p=0.0, delta_s=0.0, omega_sp=0.0, omega_ps=0.0)
SIM_SEED_BASE = 1000
#: reduced sampler for the recovery fits (handoff section 6 allows it); recorded
#: in each fit's diagnostics through run_fit
SIM_SAMPLER = dict(tune=1500, draws=1000)


# ===========================================================================
# Data
# ===========================================================================

def add_raw_lags(df):
    """Raw-score lags ``pain_factor_lag1``, ``sleep_factor_lag1`` by step 07's rule.

    Step 07's ``create_lagged_variables`` is the one implementation of the lag rule
    (within ID x segment, NaN across non-consecutive quarters). It lags the
    ``*_within`` columns, so it is handed the raw scores under those names; the
    returned frame keeps ``df``'s index, so the lags align back by index.
    """
    tmp = df[["ID", "segment_id", "quarter"]].copy()
    tmp["pain_within"] = df["pain_factor"]
    tmp["sleep_within"] = df["sleep_factor"]
    tmp["contrast_within"] = df["contrast_factor"]
    lagged = step07.create_lagged_variables(tmp)
    out = df.copy()
    out["pain_factor_lag1"] = lagged["pain_within_lag1"]
    out["sleep_factor_lag1"] = lagged["sleep_within_lag1"]
    return out


def load_latent_frame(csv_path):
    """``load_varx_frame`` plus raw lags, on exactly the rows step 08 fits.

    Checks, because a silent mismatch here would change the sample: the raw lags
    are missing exactly where step 07's within-person lags are, and raw lag minus
    the person mean reproduces step 07's within-person lag.
    """
    df, model_df, ids, id_map = load_varx_frame(csv_path)
    df = add_raw_lags(df)
    for raw, within, mean in (("pain_factor_lag1", "pain_within_lag1", "pain_person_mean"),
                              ("sleep_factor_lag1", "sleep_within_lag1", "sleep_person_mean")):
        if not (df[raw].isna() == df[within].isna()).all():
            raise ValueError(f"{raw} is missing on different rows than {within}")
        gap = np.nanmax(np.abs(df[raw] - df[mean] - df[within]))
        if gap > 1e-12:
            raise ValueError(f"{raw} - {mean} differs from {within} by {gap:.2e}")
    return df, df.loc[model_df.index], ids, id_map


def first_points_of(df):
    """The first retained point of every segment: the rows step 07 gives no lag."""
    return df[df["pain_within_lag1"].isna()].copy()


def simulate_panel(frame, rep, truth=TRUTH):
    """Raw pain and sleep scores generated on the real panel structure.

    Same people, quarters, segment breaks and localization contrast as ``frame``
    (step 07's long frame). Each person's deviation process runs on every quarter
    from 1 to the last observed quarter, starting from its stationary
    distribution, and is read off at the person's retained quarters, so gaps
    behave as in the data. Usual levels are drawn with the observed between-person
    means, SDs and correlation; random slopes with SDs tau_sp, tau_ps. Contrast
    effects and Age/Sex moderation are zero.
    """
    from scipy.linalg import solve_discrete_lyapunov

    rng = np.random.default_rng(SIM_SEED_BASE + rep)
    pm_ = frame.groupby("ID")[["pain_factor", "sleep_factor"]].mean()
    mu = pm_.mean().values
    cov_alpha = np.cov(pm_.values.T)
    sp, ss, r = truth["sigma_pain"], truth["sigma_sleep"], truth["rho_innov"]
    sig = np.array([[sp ** 2, r * sp * ss], [r * sp * ss, ss ** 2]])
    chol = np.linalg.cholesky(sig)

    out = frame.copy()
    for pid, g in frame.groupby("ID"):
        alpha = rng.multivariate_normal(mu, cov_alpha)
        while True:
            phi = np.array([[truth["phi_p"], truth["lambda_sp"] + rng.normal(0, truth["tau_sp"])],
                            [truth["lambda_ps"] + rng.normal(0, truth["tau_ps"]), truth["phi_s"]]])
            if np.max(np.abs(np.linalg.eigvals(phi))) < 0.99:
                break
        q = g["quarter"].values.astype(int)
        x = np.zeros((q.max(), 2))
        x[0] = np.linalg.cholesky(solve_discrete_lyapunov(phi, sig)) @ rng.standard_normal(2)
        for t in range(1, q.max()):
            x[t] = phi @ x[t - 1] + chol @ rng.standard_normal(2)
        y = alpha + x[q - 1]
        out.loc[g.index, "pain_factor"] = y[:, 0]
        out.loc[g.index, "sleep_factor"] = y[:, 1]

    # Step 07's own decomposition and lags, so the observed-centering arm is analyzed
    # exactly as the published data were.
    keep = ["ID", "quarter", "segment_id", "pain_factor", "contrast_factor",
            "sleep_factor", "interpolated", "Age", "Sex"]
    out = step07.decompose_within_between(out[[c for c in keep if c in out.columns]])
    return step07.create_lagged_variables(out)


def data_csv(data):
    return IN_VARX_CSV if data == "real" else os.path.join(SIM_DIR, f"{data}.csv")


# ===========================================================================
# Fitting
# ===========================================================================

def run_one_fit(data, centering, tune=None, draws=None, initial="conditional"):
    """One fit, saved as InferenceData; diagnostics written by run_fit.

    ``initial="stationary"`` (latent only) models each segment's first point from
    the person's stationary distribution instead of conditioning on it.
    """
    os.makedirs(FIT_DIR, exist_ok=True)
    df, model_df, ids, id_map = load_latent_frame(data_csv(data))
    first = first_points_of(df) if initial == "stationary" else None
    overrides = {}
    if tune is not None:
        overrides["tune"] = tune
    if draws is not None:
        overrides["draws"] = draws
    if first is not None:
        # The stationary likelihood is -inf wherever a person's transition matrix is
        # explosive. PyMC's default start jitters every transition-matrix term
        # (autoregressions, couplings, Age/Sex moderators, random-slope SDs near 1) by
        # up to +/-1 on the unconstrained scale, which puts some persons there, so
        # every chain fails to initialize. Start A at zero with small slope SDs and
        # no jitter (all chains share the start; NUTS separates them).
        overrides["init"] = "adapt_diag"
        overrides["initvals"] = {k: 0.0 for k in ("a1", "b2", "a2", "b1", "g_sp_age",
                                                  "g_sp_sex", "g_ps_age", "g_ps_sex")}
        overrides["initvals"].update(tau_sp=0.05, tau_ps=0.05)
    fit_id = f"{data}_{centering}" + ("_stationary" if first is not None else "")
    print(f"  {fit_id}: {len(ids)} persons, {len(model_df)} transitions, "
          f"{0 if first is None else len(first)} modeled first points, "
          f"overrides={overrides or 'none'}")
    idata, _, _ = fit_bayesian_varx1(
        model_df, ids, id_map, include_agesex=True, cores=4,
        progressbar=False, fit_id=fit_id, out_dir=OUT_DIR,
        centering=centering, first_points=first,
        sampler_overrides=overrides or None)
    idata.to_netcdf(os.path.join(FIT_DIR, f"{fit_id}_idata.nc"))
    return idata


# ===========================================================================
# Reporting (reads saved fits only)
# ===========================================================================

def _load_idata(path):
    import arviz as az
    return az.from_netcdf(path)


def standardization_sds():
    """The SDs step 08 standardizes with: within-person pain and sleep, analytic frame."""
    _, model_df, _, _ = load_varx_frame(IN_VARX_CSV)
    return float(model_df["pain_within"].std()), float(model_df["sleep_within"].std())


def tidy(idata):
    """summarize_posterior, plus the two standardized couplings (step 08's formula)."""
    t = summarize_posterior(idata).set_index("param")
    sd_p, sd_s = standardization_sds()
    post = idata.posterior
    for name, var, ratio in (("lambda_ps_standardized", "b1", sd_p / sd_s),
                             ("lambda_sp_standardized", "a2", sd_s / sd_p)):
        d = post[var].values.reshape(-1) * ratio
        t.loc[name, ["mean", "ci_lo_2.5", "ci_hi_97.5", "P_neg"]] = [
            d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d < 0).mean()]
    return t


def table_vs_published(tables):
    """One row per parameter; one block of columns per fit in ``tables`` (tag -> tidy)."""
    rows = []
    names = list(PARAMS) + ["lambda_ps_standardized", "lambda_sp_standardized"]
    for key in names:
        row = {"param": key, "symbol": PARAMS.get(key, key)}
        for tag, t in tables.items():
            for col, out in (("mean", "mean"), ("ci_lo_2.5", "ci_lo"),
                             ("ci_hi_97.5", "ci_hi"), ("P_neg", "P_neg")):
                row[f"{tag}_{out}"] = t.loc[key, col] if key in t.index else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def lag_quantities(phi_p, phi_s, lambda_sp, lambda_ps):
    """Step 25's quantities, from step 25's own functions.

    Reciprocal (Eq. S2): step 25's amplification and optimum formulas with the two
    eigenvalues in place of the two autoregressions; defined only when both are
    positive. Unidirectional (Eq. S3): step 25's ``derive`` unchanged.
    """
    coeffs = dict(phi_p=phi_p, phi_s=phi_s, lambda_sp=lambda_sp, lambda_ps=lambda_ps)
    _, n = step25.derive(coeffs, verbose=False)
    l1, l2 = n["eigenvalue_1"], n["eigenvalue_2"]
    if n["eq_s2_applicable"]:
        w = step25.unidirectional_optimum(l1, l2)
        pct = 100 * step25.amplification(1.0, l1, l2) / step25.amplification(w, l1, l2)
    else:
        w, pct = np.nan, np.nan
    return {
        "eigenvalue_1": l1, "eigenvalue_2": l2,
        "oscillatory_regime (eigenvalue_2 < 0)": float(l2 < 0),
        "reciprocal_optimal_lag_quarters": w,
        "reciprocal_optimal_lag_days": w * step25.DAYS_PER_QUARTER,
        "reciprocal_pct_of_peak_one_quarter": pct,
        "unidirectional_optimal_lag_quarters": n["omega_opt_quarters"],
        "unidirectional_optimal_lag_days": n["omega_opt_days"],
        "unidirectional_pct_of_peak_one_quarter": n["pct_of_peak_captured_quarterly"],
    }


def table_optimal_lag(fits, thin=20):
    """Point values from posterior means (as step 25), plus posterior 95% intervals."""
    rows = {}
    for tag, idata in fits.items():
        post = idata.posterior
        draws = {s: post[k].values.reshape(-1) for k, s in step25.COEFFS.items()}
        point = lag_quantities(**{s: float(v.mean()) for s, v in draws.items()})
        per = pd.DataFrame([lag_quantities(**{s: float(v[i]) for s, v in draws.items()})
                            for i in range(0, len(draws["phi_p"]), thin)])
        for q, v in point.items():
            r = rows.setdefault(q, {"quantity": q})
            r[f"{tag}_at_posterior_mean"] = v
            col = per[q]
            r[f"{tag}_draws_defined_pct"] = 100 * col.notna().mean()
            r[f"{tag}_ci_lo"] = col.quantile(0.025) if col.notna().any() else np.nan
            r[f"{tag}_ci_hi"] = col.quantile(0.975) if col.notna().any() else np.nan
    return pd.DataFrame(list(rows.values()))


#: the fits compared in simulation: fit-id suffix -> label
ARMS = ("observed", "latent", "latent_stationary")


def table_recovery():
    truth = {"a1": TRUTH["phi_p"], "b2": TRUTH["phi_s"], "a2": TRUTH["lambda_sp"],
             "b1": TRUTH["lambda_ps"], "a3": 0.0, "b3": 0.0, "a4": 0.0, "b4": 0.0,
             "g_sp_age": 0.0, "g_sp_sex": 0.0, "g_ps_age": 0.0, "g_ps_sex": 0.0,
             "tau_sp": TRUTH["tau_sp"], "tau_ps": TRUTH["tau_ps"],
             "sigma_pain": TRUTH["sigma_pain"], "sigma_sleep": TRUTH["sigma_sleep"],
             "rho_innov": TRUTH["rho_innov"]}
    rows = []
    reps = sorted({f.split("_")[0] for f in os.listdir(FIT_DIR) if f.startswith("sim")})
    for rep in reps:
        t = {}
        for c in ARMS:
            p = os.path.join(FIT_DIR, f"{rep}_{c}_idata.nc")
            if os.path.exists(p):
                t[c] = summarize_posterior(_load_idata(p)).set_index("param")
        for key, true in truth.items():
            row = {"rep": rep, "param": key, "symbol": PARAMS[key], "true": true}
            for c, s in t.items():
                row[f"{c}_mean"] = s.loc[key, "mean"]
                row[f"{c}_ci_lo"] = s.loc[key, "ci_lo_2.5"]
                row[f"{c}_ci_hi"] = s.loc[key, "ci_hi_97.5"]
                row[f"{c}_covers_truth"] = bool(s.loc[key, "ci_lo_2.5"] <= true
                                                <= s.loc[key, "ci_hi_97.5"])
            rows.append(row)
    rec = pd.DataFrame(rows)
    # mean over replicates, one row per parameter
    num = [c for c in rec.columns if c.endswith("_mean")]
    avg = rec.groupby(["param", "symbol", "true"], sort=False)[num].mean().reset_index()
    avg.insert(0, "rep", "mean_over_reps")
    return pd.concat([rec, avg], ignore_index=True)


def run_report():
    pub_idata = _load_idata(IN_PUBLISHED_IDATA)
    fits = {"published": pub_idata}
    for arm in ARMS[1:]:
        path = os.path.join(FIT_DIR, f"real_{arm}_idata.nc")
        if os.path.exists(path):
            fits[arm] = _load_idata(path)
    tables = {tag: tidy(i) for tag, i in fits.items()}
    pub = tables["published"]

    # The published table must be the published numbers: check against step 08.
    with open(IN_PUBLISHED_NUMBERS) as fh:
        nums = json.load(fh)
    for key, sym in (("a1", "phi_p"), ("b1", "lambda_ps"), ("a2", "lambda_sp")):
        published = nums.get(f"step08.{sym}_mean")
        if published is not None and abs(pub.loc[key, "mean"] - published) > 1e-12:
            raise ValueError(f"{sym}: step 08 posterior {pub.loc[key, 'mean']} "
                             f"!= published {published}")

    table_vs_published(tables).to_csv(OUT_VS_PUBLISHED, index=False)
    table_optimal_lag(fits).to_csv(OUT_OPTIMAL_LAG, index=False)
    if os.path.isdir(FIT_DIR) and any(f.startswith("sim") for f in os.listdir(FIT_DIR)):
        table_recovery().to_csv(OUT_RECOVERY, index=False)

    from coupling_model import read_diagnostics
    read_diagnostics(OUT_DIR).to_csv(OUT_DIAGNOSTICS, index=False)
    print(f"  wrote {OUT_VS_PUBLISHED}\n  wrote {OUT_OPTIMAL_LAG}\n"
          f"  wrote {OUT_RECOVERY}\n  wrote {OUT_DIAGNOSTICS}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--rep", type=int, required=True)
    f = sub.add_parser("fit")
    f.add_argument("--data", required=True, help="real, or simK")
    f.add_argument("--centering", choices=("observed", "latent"), required=True)
    f.add_argument("--tune", type=int)
    f.add_argument("--draws", type=int)
    f.add_argument("--initial", choices=("conditional", "stationary"),
                   default="conditional")
    sub.add_parser("report")
    args = ap.parse_args()

    if args.cmd == "simulate":
        os.makedirs(SIM_DIR, exist_ok=True)
        frame = pd.read_csv(IN_VARX_CSV, float_precision="round_trip")
        sim = simulate_panel(frame, args.rep)
        path = data_csv(f"sim{args.rep}")
        sim.to_csv(path, index=False, float_format="%.17g")
        print(f"  wrote {path}")
    elif args.cmd == "fit":
        run_one_fit(args.data, args.centering, args.tune, args.draws, args.initial)
    else:
        run_report()


if __name__ == "__main__":
    main()
