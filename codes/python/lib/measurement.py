"""Measurement-model comparison helpers.

One concept per function, defined on the quantities themselves (loading vectors,
not "the loadings of step 01"), so the same code compares any two solutions.

Functions
---------
tucker_congruence(x, y)
    Tucker's coefficient of congruence between two loading vectors.
bvn_cdf(h, k, rho)
    Standard bivariate normal CDF, exactly and deterministically.
"""
from __future__ import annotations

import numpy as np
from scipy import special, stats

__all__ = ["tucker_congruence", "bvn_cdf"]


def bvn_cdf(h, k, rho):
    """P(X <= h, Y <= k) for standard bivariate normal with correlation `rho`.

    Why this exists rather than ``scipy.stats.multivariate_normal.cdf``: SciPy
    integrates the multivariate normal by RANDOMIZED quasi-Monte-Carlo, so the same
    call returns a slightly different number every time (spread ~1e-6). That is
    harmless when you want one probability, and corrosive when the value feeds a
    likelihood that ``optimize.minimize`` then descends -- the optimizer is handed a
    noisy objective and lands somewhere slightly different on each run. The polychoric
    correlations, hence the factor loadings, hence every factor score and every number
    derived from one, drifted between runs for exactly this reason.

    The bivariate case does not need numerical integration. Owen (1956) gives it in
    closed form through Owen's T function, which SciPy provides exactly
    (``scipy.special.owens_t``):

        BVN(h, k, r) = [Phi(h) + Phi(k)]/2 - T(h, a_h) - T(k, a_k) - delta

    with a_h = (k - r h) / (h sqrt(1 - r^2)), a_k symmetric, and delta = 1/2 when h and
    k have opposite signs (0 otherwise).

    Parameters
    ----------
    h, k : float
        Upper limits, in standard-normal units.
    rho : float
        Correlation, |rho| < 1. Values at or beyond +-1 are pulled just inside it;
        the caller's optimizer walks to the boundary and a hard error there would
        abort a fit over a point it was going to reject anyway.

    Returns
    -------
    float
        The probability. Deterministic: identical arguments give identical bits.
    """
    h, k, r = float(h), float(k), float(rho)
    if abs(r) >= 1.0:
        r = np.sign(r) * (1.0 - 1e-12)

    # Both limits at the origin: the orthant probability, in closed form.
    if h == 0.0 and k == 0.0:
        return 0.25 + np.arcsin(r) / (2.0 * np.pi)

    den = np.sqrt(1.0 - r * r)

    def _a(num, denom_var):
        # h or k of exactly zero sends the argument to +-infinity, where
        # T(x, +-inf) = +-arctan(inf)/(2 pi) = +-1/4. owens_t handles the infinity,
        # but the ratio has to be formed without dividing by zero.
        if denom_var == 0.0:
            return np.inf if num > 0 else (-np.inf if num < 0 else 0.0)
        return num / (denom_var * den)

    a_h = _a(k - r * h, h)
    a_k = _a(h - r * k, k)

    # delta corrects the quadrant: a half is subtracted when h and k straddle zero.
    prod = h * k
    delta = 0.0 if (prod > 0 or (prod == 0.0 and (h + k) >= 0)) else 0.5

    return float(0.5 * (stats.norm.cdf(h) + stats.norm.cdf(k))
                 - special.owens_t(h, a_h) - special.owens_t(k, a_k) - delta)


def tucker_congruence(x, y):
    """Tucker's coefficient of congruence between two loading vectors.

    phi = sum(x*y) / sqrt(sum(x^2) * sum(y^2))

    This is the cosine of the angle between the two vectors: unlike a Pearson
    correlation it is NOT mean-centred, which is what makes it the right index
    for loadings — a factor whose loadings are uniformly high is congruent with
    another uniformly high one, and centring would destroy exactly that.

    Parameters
    ----------
    x, y : array-like, 1-D, same length
        Loadings of one factor under two solutions, item-aligned. Pairs where
        either entry is NaN are dropped.

    Returns
    -------
    float
        The congruence, in [-1, 1]; NaN if fewer than two usable pairs remain
        or if either vector is all zeros.

    Notes
    -----
    Sign is NOT taken care of here. A factor is identified only up to sign, so a
    reflected solution gives -phi. Callers that do not align signs beforehand
    should compare ``abs(...)`` against their threshold; step 02 aligns signs
    when it estimates each solution, so it compares the signed value.

    Conventional thresholds (Lorenzo-Seva & ten Berge, 2006): >= 0.95 "fair
    similarity", >= 0.98 "equal".
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"vectors must be the same length; got {x.shape} and {y.shape}")

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    x, y = x[ok], y[ok]

    denom = np.sqrt(float(x @ x) * float(y @ y))
    if denom == 0:
        return float("nan")
    return float((x @ y) / denom)
