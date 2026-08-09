"""Measurement-model comparison helpers.

One concept per function, defined on the quantities themselves (loading vectors,
not "the loadings of step 01"), so the same code compares any two solutions.

Functions
---------
tucker_congruence(x, y)
    Tucker's coefficient of congruence between two loading vectors.
"""
from __future__ import annotations

import numpy as np

__all__ = ["tucker_congruence"]


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
