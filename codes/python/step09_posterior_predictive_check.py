#!/usr/bin/env python3
"""
Step 08 — Posterior predictive check of the primary coupling model (Table S5, Section S5).

Simulates replicated datasets from the posterior step 07 already saved and scores the
discrepancy statistics: for pain and for sleep separately the mean, SD, 5th and 95th
percentile, IQR and excess kurtosis; then the same-quarter correlation of the outcomes
and the realized same-quarter correlation of the innovations. Every posterior predictive
p-value is P(T_rep >= T_obs); 0.5 is ideal, 0 or 1 is misfit.

THIS STEP NEVER FITS. `pm.sample`, `run_fit` and `fit_bayesian_varx1` do not appear here.
The sandbox ancestor (`revision/a02_diagnostics.py:113-125`) refit the primary model purely
to obtain an InferenceData, and mutated the library's sampler constants to do it; that is
deleted. Step 08 reads step 07's saved posterior, so the PPC is guaranteed to describe the
fit the paper reports rather than a same-specification re-run.

THE SIMULATOR IS CHOSEN FROM THE POSTERIOR, NEVER FROM A FILENAME OR A FLAG.
`lib/ppc.py` can replicate Gaussian or Student-t innovations, and the two differ exactly
where this check bites: fed a Student-t posterior, a Gaussian simulator would draw Normal
data from t-SCALE parameters, leaving excess-kurtosis ppp at 0.00 and IQR ppp at 1.00 —
the misfit would look unchanged when in fact the simulator was wrong. So the likelihood is
read off the artifact (`"nu" in idata.posterior.data_vars`) and never inferred from the
path or supplied on the command line. The primary model is Gaussian by Decision 2
(Student-t innovations stay in the sandbox; they retract delta_s), which is asserted here
rather than assumed.

Input:  derivatives/step08_coupling_model/step08_primary_idata.nc
        derivatives/step07_varx_data/step07_processed_long.csv
Output: derivatives/step09_posterior_predictive_check/step09_ppc_replicates.npz
        results/step09_posterior_predictive_check/step09_table_s5_ppc.csv   -> Table S5
        results/step09_posterior_predictive_check/numbers.json

The CSV carries machine names in its `statistic` column (mean/sd/q05/q95/iqr/kurt/
corr(pain,sleep)/corr(innov_pain,innov_sleep)). Display labels, the table note and the
Section S5 sentences are prose and are written by hand, not generated here.

No figure. The replicate arrays, the observed series and the per-replicate statistic
matrices are all persisted, so a standalone `plot_*` script can draw a predictive overlay
in seconds without re-simulating anything.

Usage:
    python step09_posterior_predictive_check.py [--refit] [--quiet]
                                                [--n-rep 200] [--seed 42] [--selftest]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)
ROOT = os.path.dirname(os.path.dirname(HERE))

DERIV_DIR = os.path.join(ROOT, "derivatives")
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step09_posterior_predictive_check")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step09_posterior_predictive_check")

IN_IDATA_NC = os.path.join(DERIV_DIR, "step08_coupling_model",
                           "step08_primary_idata.nc")
IN_PROCESSED_CSV = os.path.join(DERIV_DIR, "step07_varx_data",
                                "step07_processed_long.csv")

OUT_REPLICATES_NPZ = os.path.join(STEP_DERIV_DIR, "step09_ppc_replicates.npz")
OUT_TABLE_CSV = os.path.join(STEP_RESULTS_DIR, "step09_table_s5_ppc.csv")

#: Replicated datasets. Pinned from the sandbox grid (`sandbox/s02_grid.py:116`,
#: N_REP_PPC) rather than retyped, and quoted in the Methods as "200 datasets
#: replicated from the posterior". Note that ppp is therefore quantized to 1/200:
#: "ppp = 0.00" means no replicate reached the observed value, not zero probability.
N_REP = 200

#: `sandbox/s02_grid.py:100` RANDOM_SEED. Fixes which posterior draws are used and the
#: innovations drawn from them, so the table is reproducible to the last decimal.
SEED = 42

#: The analytic sample. Asserted after the model frame is rebuilt, because the frame is
#: rebuilt from the CSV rather than read back out of the fit: without this a PPC computed
#: on a silently different subset would still produce a plausible-looking table.
N_TRANSITIONS_EXPECTED = 1818
N_PERSONS_EXPECTED = 229

#: The rows whose ppp the Section S5 sentence summarizes as a range. Location and spread
#: of both series plus the joint outcome correlation — deliberately excluding the tail
#: statistics (q05, q95) and the two shape statistics (IQR, excess kurtosis), which are
#: the ones that DO misfit and are quoted individually. Without this set fixed in code,
#: `ppp_close_min` / `ppp_close_max` would mean something different in every re-run.
PPP_CLOSE_STATS = (
    ("pain", "mean"), ("pain", "sd"),
    ("sleep", "mean"), ("sleep", "sd"),
    ("joint", "corr(pain,sleep)"),
)

#: Machine statistic name -> registry key stem, for the two joint rows.
JOINT_KEY_STEMS = {
    "corr(pain,sleep)": "corr_pain_sleep",
    "corr(innov_pain,innov_sleep)": "corr_innov",
}

TABLE_COLUMNS = ["variable", "statistic", "observed", "rep_mean",
                 "rep_2.5", "rep_97.5", "ppp", "n_rep"]

#: rep_mean recomputed from the saved arrays must match the table to this tolerance.
#: It is an equality check, not a comparison: the arrays are regenerated with the same
#: seed and the same call order, so any difference at all means they are not the arrays
#: behind the table and the npz must not be trusted by a downstream figure.
ARRAY_TOLERANCE = 1e-9


def _key_stem(variable, statistic):
    """Registry key stem for one table row, e.g. 'pain_kurt', 'joint_corr_innov'."""
    if variable == "joint":
        return f"joint_{JOINT_KEY_STEMS[statistic]}"
    return f"{variable}_{statistic}"


def numbers_from_table(table, meta):
    """Every quantity Table S5 and Section S5 quote, under a stable key.

    Built from the CSV in BOTH paths — after simulating and after loading — so the
    registry cannot drift from the table it is supposed to describe.
    """
    nums = {}
    for _, r in table.iterrows():
        stem = _key_stem(r["variable"], r["statistic"])
        nums[f"ppc_{stem}_observed"] = float(r["observed"])
        nums[f"ppc_{stem}_rep_mean"] = float(r["rep_mean"])
        nums[f"ppc_{stem}_rep_lo95"] = float(r["rep_2.5"])
        nums[f"ppc_{stem}_rep_hi95"] = float(r["rep_97.5"])
        nums[f"ppc_{stem}_ppp"] = float(r["ppp"])

    close = table[[(v, s) in PPP_CLOSE_STATS
                   for v, s in zip(table["variable"], table["statistic"])]]
    if len(close) != len(PPP_CLOSE_STATS):
        raise ValueError(
            f"PPP_CLOSE_STATS matched {len(close)} of {len(PPP_CLOSE_STATS)} rows; "
            f"the table's variable/statistic names have changed")
    nums["ppc_ppp_close_min"] = float(close["ppp"].min())
    nums["ppc_ppp_close_max"] = float(close["ppp"].max())

    nums["ppc_n_replicates"] = int(table["n_rep"].max())
    nums["ppc_n_posterior_draws"] = int(meta["n_posterior_draws"])
    nums["ppc_n_obs"] = int(meta["n_obs"])
    nums["ppc_n_persons"] = int(meta["n_persons"])
    nums["ppc_seed"] = int(meta["seed"])
    nums["ppc_innovations_is_studentt"] = bool(meta["innovations"] == "studentt")
    return nums


def _stat_matrix(reps):
    """(n_rep, 6) matrix of the canonical marginal statistics, replicate by replicate."""
    import ppc
    return np.column_stack(
        [np.array([stat(r) for r in reps], dtype=float) for stat in ppc.MARGINAL_STATS]
    )


def _check_arrays_match_table(table, stat_pain, stat_sleep, rep_corr, rep_ic):
    """The saved arrays must be the arrays the table was scored from.

    `ppc.ppc_table` simulates internally, so the arrays persisted for the figure step are
    regenerated by a second `simulate_replicates` call under the same seed. Same seed,
    same rng call order, same draws — the two are identical, and this asserts it rather
    than asserting it in a comment.
    """
    import ppc

    def _want(variable, statistic):
        sel = table[(table["variable"] == variable) & (table["statistic"] == statistic)]
        return float(sel["rep_mean"].iloc[0])

    worst = 0.0
    for j, stat in enumerate(ppc.MARGINAL_STATS):
        for variable, mat in (("pain", stat_pain), ("sleep", stat_sleep)):
            worst = max(worst, abs(float(mat[:, j].mean()) - _want(variable, stat.__name__)))
    worst = max(worst, abs(float(rep_corr.mean()) - _want("joint", "corr(pain,sleep)")))
    worst = max(worst, abs(float(rep_ic.mean())
                           - _want("joint", "corr(innov_pain,innov_sleep)")))
    if worst > ARRAY_TOLERANCE:
        raise AssertionError(
            f"saved replicates do not reproduce the table (max |delta| = {worst:.3e}); "
            f"the npz would mislead any figure drawn from it")
    return worst


def load_inputs(verbose=True):
    """The model frame the primary fit used, and the posterior it produced."""
    import arviz as az
    from coupling_model import load_varx_frame

    for path, what in ((IN_PROCESSED_CSV, "step 04"), (IN_IDATA_NC, "step 07")):
        if not os.path.exists(path):
            # An ordinary exception, not SystemExit: run_pipeline.py catches
            # Exception per step and records a FAIL row, and SystemExit is not
            # an Exception — it would kill the whole run and every step after it.
            raise FileNotFoundError(
                f"  MISSING INPUT: {path}\n"
                f"  Step 08 recomputes nothing and fits nothing; it needs the {what} "
                f"artifact.\n"
                f"  Re-run that step with --refit. Note that "
                f"step08_posterior_draws.npz is NOT a substitute for the saved "
                f"InferenceData: it stores u_sp/u_ps posterior MEANS and only the a2/a4/"
                f"b1/b4 draws, while replicating a dataset needs per-draw a0..b4, the "
                f"four g_* moderators, u_sp, u_ps, sigma_pain, sigma_sleep and rho_innov.")

    _, md, unique_ids, _ = load_varx_frame(IN_PROCESSED_CSV, verbose=verbose)

    n_obs, n_persons = len(md), int(md["pid_idx"].nunique())
    if n_obs != N_TRANSITIONS_EXPECTED or n_persons != N_PERSONS_EXPECTED:
        raise AssertionError(
            f"model frame is {n_obs} transitions / {n_persons} persons, expected "
            f"{N_TRANSITIONS_EXPECTED} / {N_PERSONS_EXPECTED}; the PPC would be scored "
            f"against a different sample than the fit")

    idata = az.from_netcdf(IN_IDATA_NC)
    post = idata.posterior
    if "u_sp" not in post.data_vars:
        raise AssertionError(f"{IN_IDATA_NC} has no u_sp; this is not the primary fit")
    if post["u_sp"].shape[-1] != n_persons:
        raise AssertionError(
            f"posterior carries {post['u_sp'].shape[-1]} person effects but the model "
            f"frame has {n_persons} persons — the saved fit is not this sample")

    # From the artifact, never from the filename. See the module docstring.
    innovations = "studentt" if "nu" in post.data_vars else "gaussian"
    if innovations != "gaussian":
        raise ValueError(
            f"  {IN_IDATA_NC} carries 'nu': it is a Student-t fit.\n"
            f"  Student-t innovations stay in the sandbox (they retract delta_s) and are "
            f"never the primary model, so this posterior does not belong at step 07's "
            f"primary path.")

    n_draws = int(post.chain.size * post.draw.size)
    if verbose:
        print(f"  posterior: {n_draws} draws, {n_persons} person effects, "
              f"{innovations} innovations (read from the posterior)")
    return md, idata, innovations, n_draws


def simulate_and_score(md, idata, innovations, n_rep, seed, verbose=True):
    """Table S5 plus everything a later figure would need, from one posterior.

    The whole computation is `lib/ppc.py`: `ppc_table` for the 14 discrepancy rows and
    the replicate arrays it scored them on. Neither is reimplemented here, and the
    arrays are the SAME ones the table was built from — `ppc_table(return_arrays=True)`
    hands back what it simulated instead of this step simulating a second set.
    """
    import ppc

    table, (rep_pain, rep_sleep, mu_pain, mu_sleep) = ppc.ppc_table(
        idata, md, n_rep=n_rep, seed=seed, innovations=innovations,
        return_arrays=True)
    table = table[TABLE_COLUMNS]

    obs_pain = md["pain_within"].values
    obs_sleep = md["sleep_within"].values

    stat_pain = _stat_matrix(rep_pain)
    stat_sleep = _stat_matrix(rep_sleep)
    rep_corr = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(rep_pain, rep_sleep)])
    obs_ic = np.array([np.corrcoef(obs_pain - mp, obs_sleep - ms)[0, 1]
                       for mp, ms in zip(mu_pain, mu_sleep)])
    rep_ic = np.array([np.corrcoef(rp - mp, rs - ms)[0, 1]
                       for rp, rs, mp, ms in zip(rep_pain, rep_sleep, mu_pain, mu_sleep)])

    worst = _check_arrays_match_table(table, stat_pain, stat_sleep, rep_corr, rep_ic)
    if verbose:
        print(f"  saved arrays reproduce the table exactly (max |delta| = {worst:.1e})")

    arrays = {
        "rep_pain": rep_pain.astype(np.float32),
        "rep_sleep": rep_sleep.astype(np.float32),
        "mu_pain": mu_pain.astype(np.float32),
        "mu_sleep": mu_sleep.astype(np.float32),
        "obs_pain": obs_pain,
        "obs_sleep": obs_sleep,
        "stat_names": np.array([s.__name__ for s in ppc.MARGINAL_STATS]),
        "stat_pain": stat_pain,
        "stat_sleep": stat_sleep,
        "rep_corr_pain_sleep": rep_corr,
        "obs_corr_innov": obs_ic,
        "rep_corr_innov": rep_ic,
    }
    return table, arrays


def _save(table, arrays, meta):
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)
    # %.17g, not pandas' default 16 significant digits: the registry is built from this
    # CSV in the load path as well as in the simulate path, and at 16 digits the two
    # disagree in the last ULP -- i.e. re-running the step without --refit would rewrite
    # numbers.json with silently different values.
    table.to_csv(OUT_TABLE_CSV, index=False, float_format="%.17g")
    np.savez_compressed(
        OUT_REPLICATES_NPZ,
        n_rep=np.array(meta["n_rep"]),
        seed=np.array(meta["seed"]),
        n_obs=np.array(meta["n_obs"]),
        n_persons=np.array(meta["n_persons"]),
        n_posterior_draws=np.array(meta["n_posterior_draws"]),
        innovations=np.array(meta["innovations"]),
        **arrays,
    )


def _load_saved():
    """The table and the run metadata from a previous run; None if incomplete."""
    if not (os.path.exists(OUT_TABLE_CSV) and os.path.exists(OUT_REPLICATES_NPZ)):
        return None, None
    # float_precision="round_trip" for the same reason `_save` writes %.17g: pandas'
    # default parser is accurate to ~15 digits, so without it the load path and the
    # simulate path publish different last digits for the same quantity.
    table = pd.read_csv(OUT_TABLE_CSV, float_precision="round_trip")
    missing = [c for c in TABLE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(
            f"  {OUT_TABLE_CSV} is missing column(s) {missing} — it predates the current "
            f"format. Re-run with --refit.")
    with np.load(OUT_REPLICATES_NPZ, allow_pickle=False) as z:
        meta = {
            "n_rep": int(z["n_rep"]),
            "seed": int(z["seed"]),
            "n_obs": int(z["n_obs"]),
            "n_persons": int(z["n_persons"]),
            "n_posterior_draws": int(z["n_posterior_draws"]),
            "innovations": str(z["innovations"]),
        }
    return table, meta


def run_step09(verbose=True, refit=False, n_rep=N_REP, seed=SEED):
    """Score the primary model's posterior predictive check.

    Default path is load-and-report: the saved table and replicates are read back and the
    registry re-emitted. `refit=True` re-simulates from step 07's posterior. Nothing here
    ever fits a model.
    """
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("STEP 08 — Posterior predictive check of the primary model (Table S5)")
        print("=" * 70)

    table, meta = (None, None) if refit else _load_saved()

    # --n-rep / --seed have no effect on the load path: what is reported comes from
    # the saved run, whose settings are stored in the NPZ. Say so rather than let the
    # caller believe the flags did something.
    if table is not None and (meta["seed"] != seed or meta["n_rep"] != n_rep):
        print(f"  NOTE: reporting the SAVED run (n_rep={meta['n_rep']}, "
              f"seed={meta['seed']}); the requested n_rep={n_rep}, seed={seed} are "
              f"ignored unless you pass --refit.")

    if table is None:
        if verbose and not refit:
            print("  Saved outputs not found — simulating.")
        md, idata, innovations, n_draws = load_inputs(verbose=verbose)
        if verbose:
            print(f"  simulating {n_rep} replicated datasets "
                  f"(seed {seed}, {innovations} innovations)")
        table, arrays = simulate_and_score(md, idata, innovations, n_rep, seed,
                                           verbose=verbose)
        meta = {
            "n_rep": int(table["n_rep"].max()),
            "seed": int(seed),
            "n_obs": int(len(md)),
            "n_persons": int(md["pid_idx"].nunique()),
            "n_posterior_draws": int(n_draws),
            "innovations": innovations,
        }
        _save(table, arrays, meta)
        if verbose:
            print(f"\n  wrote {os.path.relpath(OUT_TABLE_CSV, ROOT)}")
            print(f"  wrote {os.path.relpath(OUT_REPLICATES_NPZ, ROOT)}")
    elif verbose:
        print("  Loading saved replicates — pass --refit to re-simulate.")

    nums = numbers_from_table(table, meta)

    from registry import write_numbers
    write_numbers(STEP_RESULTS_DIR, nums, prefix="step09")

    if verbose:
        print(f"\n  {meta['n_rep']} replicated datasets from {meta['n_posterior_draws']} "
              f"posterior draws; {meta['n_obs']} transitions, {meta['n_persons']} persons")
        print("  ppp near 0 or 1 indicates misfit; 0.5 is ideal\n")
        print(table.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
        print(f"\n  ppp over location/spread rows: "
              f"{nums['ppc_ppp_close_min']:.3f} to {nums['ppc_ppp_close_max']:.3f}")
        print(f"  wrote {os.path.relpath(os.path.join(STEP_RESULTS_DIR, 'numbers.json'), ROOT)}")
        print("=" * 70)
    return table


def main():
    ap = argparse.ArgumentParser(
        description="Step 08 — posterior predictive check of the primary coupling model.")
    ap.add_argument("--refit", action="store_true",
                    help="re-simulate from step 07's saved posterior instead of loading "
                         "the saved replicates (this step never fits a model)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--n-rep", type=int, default=N_REP, dest="n_rep",
                    help=f"replicated datasets (default {N_REP}; the manuscript quotes it)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"rng seed for draw selection and innovations (default {SEED})")
    ap.add_argument("--selftest", action="store_true",
                    help="run lib/ppc.py's Student-t moment check and exit")
    args = ap.parse_args()

    if args.selftest:
        import ppc
        ppc.verify_studentt_moments(verbose=not args.quiet)
        print("PPC SELF-TEST PASSED")
        return 0

    run_step09(verbose=not args.quiet, refit=args.refit,
               n_rep=args.n_rep, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
