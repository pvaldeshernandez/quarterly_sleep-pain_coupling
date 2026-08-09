"""Posterior predictive simulation for the VARX(1) coupling model.

The canonical home. Consumed by ``step08_posterior_predictive_check.py`` and by
``revision/sandbox/s02_grid.py`` (which now imports this module through a shim).
Writes nothing on import.

WHY THIS FILE EXISTS
--------------------
``revision/a02_diagnostics.py:52`` already has a ``simulate_replicates()``
that re-implements the generative model from the posterior draws. It is
hard-wired to GAUSSIAN innovations:

    ep = rng.normal(0, sigma_p, n)
    es = rng.normal(rho*(sigma_s/sigma_p)*ep, sigma_s*sqrt(1-rho^2))

Fed a Student-t fit it would replicate Gaussian data from t-SCALE
parameters. The excess-kurtosis ppp would stay at 0.00 and the IQR ppp at
1.00 -- i.e. the Student-t fix would look like it had failed when in fact
the simulator was wrong. So the PPC has to understand the likelihood.

This module GENERALIZES a02's function rather than forking it. The
signature is a superset of a02's, the Gaussian path issues the SAME rng
calls in the SAME order, and ``verify_gaussian_identity()`` proves the
output is bit-identical (max |delta| == 0.0) to
``a02_diagnostics.simulate_replicates`` under a shared seed. a02 itself is
NOT edited: it backs numbers already in the manuscript.

THE STUDENT-T GENERATIVE FORM
-----------------------------
A bivariate Student-t is a scale mixture of a bivariate normal with ONE
shared mixing variable per observation VECTOR:

    (eps_p, eps_s) = (z_p, z_s) / sqrt(g),
    (z_p, z_s) ~ N(0, Sigma),  g ~ Gamma(nu/2, scale=2/nu)  [= chi2_nu / nu]

The mixing variable is shared between the pain and the sleep innovation of
the same quarter. Drawing two INDEPENDENT univariate t's instead would
give the right margins, the right tail index and (approximately) the right
correlation, but no tail dependence -- and tail dependence is precisely
what the failing kurtosis check is about. Sigma here is the SCALE matrix:
Var(eps) = Sigma * nu/(nu-2), so the simulated innovation SD is
sigma * sqrt(nu/(nu-2)), not sigma.

Self-test:  python ppc.py
"""
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Discrepancy statistics. Names are kept byte-identical to
# a02_diagnostics.py:196-203 so the two CSVs can be stacked and compared.
# ---------------------------------------------------------------------------


def mean(x):
    return float(np.mean(x))


def sd(x):
    return float(np.std(x, ddof=1))


def q05(x):
    return float(np.percentile(x, 5))


def q95(x):
    return float(np.percentile(x, 95))


def iqr(x):
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def kurt(x):
    """Excess kurtosis (0 for a Gaussian). a02_diagnostics.py:201."""
    z = (x - x.mean()) / x.std(ddof=1)
    return float((z ** 4).mean() - 3)


MARGINAL_STATS = (mean, sd, q05, q95, iqr, kurt)


def simulate_replicates(idata, md, n_rep, rng, innovations="gaussian",
                        return_mu=False):
    """Draw y_rep from the posterior predictive by re-implementing the model.

    P^w_it  = a0 + a1*P_lag + a2_i*S_lag + a3*K_lag + a4*(S_lag*K_lag) + eps_p
    S^w_it  = b0 + b1_i*P_lag + b2*S_lag + b3*K_lag + b4*(P_lag*K_lag) + eps_s
    a2_i    = a2 + u_sp[i] + g_sp_age*Age_z + g_sp_sex*Sex_c
    b1_i    = b1 + u_ps[i] + g_ps_age*Age_z + g_ps_sex*Sex_c

    innovations="gaussian"  (default, a02-identical)
        eps_p       ~ N(0, sigma_p)
        eps_s|eps_p ~ N(rho*(sigma_s/sigma_p)*eps_p, sigma_s*sqrt(1-rho^2))

    innovations="studentt"
        the same Gaussian pair divided by sqrt(g), one g per observation,
        g ~ Gamma(nu/2, 2/nu). This is the exact bivariate-t scale mixture
        matching coupling_model_fork.add_bivariate_innovations_likelihood
        (student_t=True), whose conditional carries df nu+1 and the
        (nu + d1)/(nu + 1) scale inflation.

    Works for BOTH parameterizations without changes: under
    parameterization="noncentered" the fork exposes u_sp / u_ps as
    Deterministics under their original names, so the draws are read the
    same way.

    Parameters
    ----------
    idata : arviz.InferenceData
        Must carry a ``posterior`` group with a0..b4, u_sp, u_ps, the four
        g_* moderators, sigma_pain, sigma_sleep, rho_innov, and -- for
        innovations="studentt" -- nu.
    md : DataFrame
        The model frame actually fit (pain_within_lag1, sleep_within_lag1,
        contrast_within_lag1, sleep_x_contrast_lag1, pain_x_contrast_lag1,
        pid_idx, Age_z, Sex_c).
    n_rep : int
        Number of replicated datasets (posterior draws sampled without
        replacement).
    rng : numpy.random.Generator
    innovations : {"gaussian", "studentt"}
    return_mu : bool, default False
        If True also return the per-replicate linear predictors, so a
        REALIZED discrepancy T(y, theta) -- e.g. the innovation
        correlation -- can be evaluated at the same draw that generated
        the replicate. False reproduces a02's 2-tuple return exactly.

    Returns
    -------
    (rep_pain, rep_sleep) or (rep_pain, rep_sleep, mu_pain, mu_sleep)
        Each of shape (n_used, n_obs).
    """
    if innovations not in ("gaussian", "studentt"):
        raise ValueError(
            f"innovations must be 'gaussian' or 'studentt'; got {innovations!r}"
        )
    student_t = innovations == "studentt"

    post = idata.posterior
    flat = {v: post[v].values.reshape((-1,) + post[v].values.shape[2:])
            for v in post.data_vars}
    if student_t and "nu" not in flat:
        raise KeyError(
            "innovations='studentt' but the posterior has no 'nu'. This idata "
            "was fit with Gaussian innovations."
        )
    n_draws = flat["a0"].shape[0]
    picks = rng.choice(n_draws, size=min(n_rep, n_draws), replace=False)

    pain_lag = md["pain_within_lag1"].values
    sleep_lag = md["sleep_within_lag1"].values
    k_lag = md["contrast_within_lag1"].values
    sxk = md["sleep_x_contrast_lag1"].values
    pxk = md["pain_x_contrast_lag1"].values
    idx = md["pid_idx"].values.astype(int)
    age = md["Age_z"].values
    sex = md["Sex_c"].values

    reps_p, reps_s, mus_p, mus_s = [], [], [], []
    for d in picks:
        a2_i = (flat["a2"][d] + flat["u_sp"][d][idx]
                + flat["g_sp_age"][d] * age + flat["g_sp_sex"][d] * sex)
        b1_i = (flat["b1"][d] + flat["u_ps"][d][idx]
                + flat["g_ps_age"][d] * age + flat["g_ps_sex"][d] * sex)
        mu_p = (flat["a0"][d] + flat["a1"][d] * pain_lag + a2_i * sleep_lag
                + flat["a3"][d] * k_lag + flat["a4"][d] * sxk)
        mu_s = (flat["b0"][d] + b1_i * pain_lag + flat["b2"][d] * sleep_lag
                + flat["b3"][d] * k_lag + flat["b4"][d] * pxk)
        sp, ss = flat["sigma_pain"][d], flat["sigma_sleep"][d]
        rho = flat["rho_innov"][d]
        # Identical rng call order to a02_diagnostics.py:89-90, so the
        # Gaussian path reproduces it exactly under a shared seed.
        ep = rng.normal(0, sp, size=len(mu_p))
        es = rng.normal(rho * (ss / sp) * ep, ss * np.sqrt(1 - rho ** 2))
        if student_t:
            # ONE shared mixing variable per observation vector.
            nu = float(flat["nu"][d])
            g = rng.gamma(nu / 2.0, 2.0 / nu, size=len(mu_p))
            inflate = 1.0 / np.sqrt(g)
            ep = ep * inflate
            es = es * inflate
        reps_p.append(mu_p + ep)
        reps_s.append(mu_s + es)
        if return_mu:
            mus_p.append(mu_p)
            mus_s.append(mu_s)
    if return_mu:
        return (np.array(reps_p), np.array(reps_s),
                np.array(mus_p), np.array(mus_s))
    return np.array(reps_p), np.array(reps_s)


def ppc_table(idata, md, n_rep=200, seed=42, innovations="gaussian",
              return_arrays=False):
    """Full posterior predictive check table for one fit.

    Rows: for pain and for sleep separately -- mean, sd, 5th and 95th
    percentile, IQR and excess kurtosis; then two joint rows.

    ``corr(pain,sleep)`` reproduces a02_diagnostics.py:210-216 exactly (the
    same-quarter correlation of the OUTCOMES, whose only free component is
    the innovation correlation).

    ``corr(innov_pain,innov_sleep)`` is the REALIZED-discrepancy version:
    at each draw the observed residuals y - mu(theta_d) are correlated and
    compared against the replicate's residuals under the SAME theta_d. This
    is the same-quarter innovation correlation proper; the observed side is
    a distribution rather than a single number, so ``observed`` reports its
    mean.

    Every ppp is P(T_rep >= T_obs); 0.5 is ideal, 0 or 1 is misfit.

    Parameters
    ----------
    return_arrays : bool, default False
        When True, return ``(table, (rep_pain, rep_sleep, mu_pain, mu_sleep))``
        instead of the table alone. The replicates are simulated here anyway;
        a caller that also needs them for a figure would otherwise call
        ``simulate_replicates`` a second time, doubling the cost and — because
        that second call starts a fresh generator — plotting replicates that
        are NOT the ones the table was scored on.
    """
    rng = np.random.default_rng(seed)
    rep_p, rep_s, mu_p, mu_s = simulate_replicates(
        idata, md, n_rep, rng, innovations=innovations, return_mu=True)
    obs_p = md["pain_within"].values
    obs_s = md["sleep_within"].values

    rows = []
    for name, obs, rep in (("pain", obs_p, rep_p), ("sleep", obs_s, rep_s)):
        for stat in MARGINAL_STATS:
            o = stat(obs)
            r = np.array([stat(x) for x in rep])
            rows.append({
                "variable": name, "statistic": stat.__name__, "observed": o,
                "rep_mean": float(r.mean()),
                "rep_2.5": float(np.percentile(r, 2.5)),
                "rep_97.5": float(np.percentile(r, 97.5)),
                "ppp": float((r >= o).mean()),
                "n_rep": int(len(r)),
            })

    obs_corr = float(np.corrcoef(obs_p, obs_s)[0, 1])
    rep_corr = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(rep_p, rep_s)])
    rows.append({
        "variable": "joint", "statistic": "corr(pain,sleep)",
        "observed": obs_corr, "rep_mean": float(rep_corr.mean()),
        "rep_2.5": float(np.percentile(rep_corr, 2.5)),
        "rep_97.5": float(np.percentile(rep_corr, 97.5)),
        "ppp": float((rep_corr >= obs_corr).mean()),
        "n_rep": int(len(rep_corr)),
    })

    obs_ic = np.array([
        np.corrcoef(obs_p - mp, obs_s - ms)[0, 1] for mp, ms in zip(mu_p, mu_s)
    ])
    rep_ic = np.array([
        np.corrcoef(rp - mp, rs - ms)[0, 1]
        for rp, rs, mp, ms in zip(rep_p, rep_s, mu_p, mu_s)
    ])
    rows.append({
        "variable": "joint", "statistic": "corr(innov_pain,innov_sleep)",
        "observed": float(obs_ic.mean()), "rep_mean": float(rep_ic.mean()),
        "rep_2.5": float(np.percentile(rep_ic, 2.5)),
        "rep_97.5": float(np.percentile(rep_ic, 97.5)),
        "ppp": float((rep_ic >= obs_ic).mean()),
        "n_rep": int(len(rep_ic)),
    })
    table = pd.DataFrame(rows)
    if return_arrays:
        return table, (rep_p, rep_s, mu_p, mu_s)
    return table


# ---------------------------------------------------------------------------
# Regression test: the Gaussian path must be a02_diagnostics, bit for bit.
# ---------------------------------------------------------------------------

def _toy_idata_and_frame(seed=0, n_obs=137, n_persons=11, n_chains=2,
                         n_draws=25):
    """A tiny fake posterior + model frame. No fitting, no data touched."""
    import arviz as az
    rng = np.random.default_rng(seed)
    scalars = ["a0", "a1", "a2", "a3", "a4", "b0", "b1", "b2", "b3", "b4",
               "g_sp_age", "g_sp_sex", "g_ps_age", "g_ps_sex"]
    post = {v: rng.normal(0, 0.3, size=(n_chains, n_draws)) for v in scalars}
    post["u_sp"] = rng.normal(0, 0.2, size=(n_chains, n_draws, n_persons))
    post["u_ps"] = rng.normal(0, 0.3, size=(n_chains, n_draws, n_persons))
    post["sigma_pain"] = rng.uniform(0.5, 1.2, size=(n_chains, n_draws))
    post["sigma_sleep"] = rng.uniform(0.5, 1.2, size=(n_chains, n_draws))
    post["rho_innov"] = rng.uniform(-0.5, 0.5, size=(n_chains, n_draws))
    post["nu"] = rng.uniform(3.0, 30.0, size=(n_chains, n_draws))
    idata = az.from_dict(posterior=post)

    idx = rng.integers(0, n_persons, size=n_obs)
    age = rng.normal(size=n_persons)[idx]
    sex = rng.choice([-0.35, 0.65], size=n_persons)[idx]
    md = pd.DataFrame({
        "pain_within": rng.normal(size=n_obs),
        "sleep_within": rng.normal(size=n_obs),
        "pain_within_lag1": rng.normal(size=n_obs),
        "sleep_within_lag1": rng.normal(size=n_obs),
        "contrast_within_lag1": rng.normal(size=n_obs),
        "sleep_x_contrast_lag1": rng.normal(size=n_obs),
        "pain_x_contrast_lag1": rng.normal(size=n_obs),
        "pid_idx": idx, "Age_z": age, "Sex_c": sex,
    })
    return idata, md


def verify_gaussian_identity(verbose=True):
    """max |delta| between this module's Gaussian path and a02's must be 0."""
    sys.dont_write_bytecode = True   # do not drop a .pyc into revision/
    # lib/ -> codes/python/ -> codes/python/revision/, where a02 still lives.
    # a02 is NOT edited: it backs numbers already in the manuscript, and it is
    # the reference this identity check exists to compare against.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "revision"))
    import a02_diagnostics as a02

    idata, md = _toy_idata_and_frame()
    n_rep = 17
    ref_p, ref_s = a02.simulate_replicates(
        idata, md, n_rep, np.random.default_rng(2024))
    new_p, new_s = simulate_replicates(
        idata, md, n_rep, np.random.default_rng(2024))
    dp = float(np.max(np.abs(ref_p - new_p)))
    ds = float(np.max(np.abs(ref_s - new_s)))
    if verbose:
        print(f"  gaussian path vs a02_diagnostics.simulate_replicates:")
        print(f"    shapes {ref_p.shape} == {new_p.shape}: "
              f"{ref_p.shape == new_p.shape}")
        print(f"    max |delta| pain  = {dp!r}")
        print(f"    max |delta| sleep = {ds!r}")
    assert ref_p.shape == new_p.shape and ref_s.shape == new_s.shape
    assert dp == 0.0 and ds == 0.0, "generalization changed the Gaussian path"
    return True


def verify_studentt_moments(verbose=True):
    """The Student-t path must inflate variance by nu/(nu-2) and keep rho.

    Simulated innovation SD should be sigma*sqrt(nu/(nu-2)), the same-quarter
    innovation correlation should stay rho, and excess kurtosis should be
    clearly positive (Gaussian path: ~0).
    """
    import arviz as az
    n = 400_000
    nu, sp, ss, rho = 6.0, 0.83, 1.27, -0.42
    post = {v: np.full((1, 1), 0.0) for v in
            ["a0", "a1", "a2", "a3", "a4", "b0", "b1", "b2", "b3", "b4",
             "g_sp_age", "g_sp_sex", "g_ps_age", "g_ps_sex"]}
    post["u_sp"] = np.zeros((1, 1, 1))
    post["u_ps"] = np.zeros((1, 1, 1))
    post["sigma_pain"] = np.full((1, 1), sp)
    post["sigma_sleep"] = np.full((1, 1), ss)
    post["rho_innov"] = np.full((1, 1), rho)
    post["nu"] = np.full((1, 1), nu)
    idata = az.from_dict(posterior=post)
    md = pd.DataFrame({
        "pain_within": np.zeros(n), "sleep_within": np.zeros(n),
        "pain_within_lag1": np.zeros(n), "sleep_within_lag1": np.zeros(n),
        "contrast_within_lag1": np.zeros(n),
        "sleep_x_contrast_lag1": np.zeros(n),
        "pain_x_contrast_lag1": np.zeros(n),
        "pid_idx": np.zeros(n, int), "Age_z": np.zeros(n),
        "Sex_c": np.zeros(n),
    })
    rp, rs = simulate_replicates(idata, md, 1, np.random.default_rng(7),
                                 innovations="studentt")
    gp, gs = simulate_replicates(idata, md, 1, np.random.default_rng(7),
                                 innovations="gaussian")
    infl = np.sqrt(nu / (nu - 2))
    got = (sd(rp[0]), sd(rs[0]), float(np.corrcoef(rp[0], rs[0])[0, 1]),
           kurt(rp[0]), kurt(gp[0]))
    want = (sp * infl, ss * infl, rho, 6.0 / (nu - 4), 0.0)
    if verbose:
        print("  studentt path moments (n=400,000, nu=6):")
        print(f"    SD pain   {got[0]:.4f}  vs sigma*sqrt(nu/(nu-2)) "
              f"= {want[0]:.4f}")
        print(f"    SD sleep  {got[1]:.4f}  vs {want[1]:.4f}")
        print(f"    corr      {got[2]:.4f}  vs rho = {want[2]:.4f}")
        print(f"    excess kurtosis  studentt {got[3]:.3f} "
              f"(theory 6/(nu-4) = {want[3]:.3f}), gaussian {got[4]:.3f}")
    assert abs(got[0] - want[0]) < 0.02 * want[0]
    assert abs(got[1] - want[1]) < 0.02 * want[1]
    assert abs(got[2] - want[2]) < 0.02
    assert got[3] > 1.0, "studentt path is not heavy tailed"
    assert abs(got[4]) < 0.1, "gaussian path drifted away from kurtosis 0"
    return True


if __name__ == "__main__":
    print("ppc.py self-test")
    verify_gaussian_identity()
    verify_studentt_moments()
    print("ALL PPC CHECKS PASSED")
