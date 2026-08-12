"""Person-level nuisance covariates: reading them, aligning them, removing them.

The canonical home for the concepts that the sandbox scripts a04 (motion QC),
a06b (site), a06c (evoked pain) and a06d (motion-adjusted) each implemented
their own copy of. Every function here is written around the concept, so the
same code serves a scanner site, an evoked-pain rating and a mean framewise
displacement without a per-covariate variant:

  a person-level covariate is a {ID: value} map,
  aligning is looking several maps up on one ID list,
  adjusting is regressing a moderator on those columns and z-scoring the residual.

Functions
---------
read_spm_design(path)            SPM.mat -> the design matrix as an array
fd_from_design(X, n_motion=6)    design matrix -> per-volume framewise displacement
person_covariate(path, column)   a wide table -> {ID: value}
roi_moderator(roi_df, roi)       a long ROI frame -> {ID: z_value} for one ROI
align(ids, *maps)                ID list + maps -> (len(ids), n_maps) array
residualize(ids, values, covmaps)  adjust a moderator, return z-scored residuals
cohens_d(a, b)                   pooled-SD standardized mean difference
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["read_spm_design", "fd_from_design", "person_covariate",
           "roi_moderator", "align", "residualize", "cohens_d",
           "HEAD_RADIUS_MM", "N_MOTION_REGRESSORS"]

#: Radius of the sphere on which a rotation is converted to a surface
#: displacement, in mm. Power et al. (2012) use 50 mm.
HEAD_RADIUS_MM = 50.0

#: SPM's realignment parameters: 3 translations then 3 rotations, sitting
#: immediately before the constant column of the design matrix.
N_MOTION_REGRESSORS = 6


def read_spm_design(path):
    """The design matrix of an SPM.mat, as a float array (n_volumes, n_columns).

    Raises rather than returning None if the file cannot be read or does not
    carry ``SPM.xX.X``: a participant whose design matrix is unreadable is a
    participant missing from the QC table, and the caller has to be able to
    count them.
    """
    from scipy.io import loadmat

    spm = loadmat(path, struct_as_record=False, squeeze_me=True)["SPM"]
    return np.asarray(spm.xX.X, dtype=float)


def fd_from_design(X, n_motion=N_MOTION_REGRESSORS, radius_mm=HEAD_RADIUS_MM):
    """Framewise displacement per volume, from the motion columns of a design matrix.

    FD_t = sum |d translations| + radius * sum |d rotations|, the standard
    Power et al. (2012) definition. SPM stores rotations in RADIANS, which is
    why multiplying by the radius gives millimetres.

    The motion regressors are taken as the ``n_motion`` columns immediately
    before the constant. That layout is an ASSUMPTION about how the first-level
    model was specified, so it is checked rather than trusted:

    - a design matrix too narrow to hold them raises;
    - a last column that is not constant raises, because then the slice is off
      by at least one column and the "rotations" would be task regressors.

    A wrong FD is worse than a missing one -- it would be averaged into a
    published motion summary and silently shift it -- so this raises where the
    sandbox version returned None.

    Returns
    -------
    ndarray, shape (n_volumes - 1,)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < n_motion + 1:
        raise ValueError(
            f"design matrix is {X.shape}; too narrow to hold {n_motion} motion "
            f"regressors plus a constant")
    if X.shape[0] < 2:
        raise ValueError(f"design matrix has {X.shape[0]} volume(s); FD needs at least 2")

    last = X[:, -1]
    if not np.allclose(last, last[0]) or last[0] == 0:
        raise ValueError(
            "the last design-matrix column is not a nonzero constant, so the "
            "motion regressors are not the columns before it; FD would be "
            "computed from the wrong columns")

    M = X[:, -(n_motion + 1):-1]
    d = np.abs(np.diff(M, axis=0))
    return d[:, :3].sum(axis=1) + radius_mm * d[:, 3:6].sum(axis=1)


def person_covariate(path, column, id_columns=("subject_id", "ID")):
    """``{ID: float}`` for one column of a wide, one-row-per-participant table.

    Reads .xlsx or .csv. IDs are kept as STRINGS exactly as written, because
    they carry re-enrolment suffixes ("2014-2") that any numeric coercion would
    destroy — and a destroyed ID silently drops that participant from every
    later join rather than raising.

    Rows with a missing ID or a non-numeric value are dropped.
    """
    wanted = set(id_columns) | {column}
    if str(path).lower().endswith((".xlsx", ".xls")):
        w = pd.read_excel(path, usecols=lambda c: c in wanted)
    else:
        w = pd.read_csv(path, usecols=lambda c: c in wanted, float_precision="round_trip")

    idcol = next((c for c in id_columns if c in w.columns), None)
    if idcol is None:
        raise KeyError(f"{path} has none of the ID columns {list(id_columns)}")
    if column not in w.columns:
        raise KeyError(f"{path} has no column {column!r}")

    w = w[[idcol, column]].copy()
    w[column] = pd.to_numeric(w[column], errors="coerce")
    w = w.dropna(subset=[idcol, column])
    return dict(zip(w[idcol].astype(str), w[column].astype(float)))


def roi_moderator(roi_df, roi, value_col="z_value"):
    """``{ID: z_value}`` for one ROI of a long ROI-value frame.

    The same map ``fit_bayesian_varx1`` takes as ``X_person``, built in one
    place so a residualized fit and an unadjusted fit cannot disagree about
    which participants an ROI covers.
    """
    for col in ("ID", "ROI", value_col):
        if col not in roi_df.columns:
            raise KeyError(f"roi_df has no {col!r} column")
    block = roi_df[roi_df["ROI"] == roi]
    if block.empty:
        raise KeyError(f"ROI {roi!r} absent; present: {sorted(set(roi_df['ROI']))}")
    v = pd.to_numeric(block[value_col], errors="coerce")
    keep = v.notna()
    return dict(zip(block.loc[keep, "ID"].astype(str), v[keep].astype(float)))


def align(ids, *maps):
    """Look several person-level maps up on one ID list.

    Returns an array of shape ``(len(ids), len(maps))``; an ID absent from a map
    gives NaN in that column, never a dropped row — so every caller sees the
    same rows in the same order and decides for itself what to do with the
    missing ones. IDs are matched as strings.
    """
    if not maps:
        raise ValueError("align needs at least one map")
    keys = [str(i) for i in ids]
    return np.column_stack([
        np.array([m.get(k, np.nan) for k in keys], dtype=float) for m in maps
    ])


def residualize(ids, values, covmaps):
    """Regress a moderator on one or more person-level covariates; z-score the residual.

    The adjusted moderator for the sensitivity fits: what is left of an ROI
    value once site, evoked pain and/or motion are removed. Z-scoring the
    residual keeps it on the same scale as the unadjusted moderator, so the two
    gammas are directly comparable — an unstandardized residual would change
    gamma by a scale factor and look like an effect moving.

    Parameters
    ----------
    ids : sequence
        The participants, in order. Defines the output's row order.
    values : array-like
        The moderator, one value per entry of ``ids``.
    covmaps : sequence of dict
        One ``{ID: value}`` map per covariate. A single map may be passed
        directly instead of a one-element sequence.

    Returns
    -------
    (resid, ok) : (ndarray, ndarray of bool)
        ``resid`` is NaN wherever the moderator or any covariate is missing;
        ``ok`` flags the rows that entered the regression. Both have
        ``len(ids)`` entries.
    """
    if isinstance(covmaps, dict):
        covmaps = [covmaps]
    C = align(ids, *covmaps)
    v = np.asarray(values, dtype=float)
    if v.shape[0] != C.shape[0]:
        raise ValueError(f"{len(v)} values for {C.shape[0]} ids")

    ok = np.isfinite(C).all(axis=1) & np.isfinite(v)
    resid = np.full_like(v, np.nan)
    if ok.sum() < C.shape[1] + 2:
        # Fewer usable rows than free parameters plus one: the fit is degenerate
        # and every "residual" would be an artifact of that.
        return resid, ok

    X = np.column_stack([np.ones(int(ok.sum())), C[ok]])
    beta, *_ = np.linalg.lstsq(X, v[ok], rcond=None)
    r = v[ok] - X @ beta
    sd = r.std(ddof=0)
    resid[ok] = (r - r.mean()) / sd if sd > 0 else np.nan
    return resid, ok


def cohens_d(a, b):
    """Standardized mean difference with the pooled SD (ddof=1 in each group).

    NaNs are dropped within each group; NaN is returned if either group has
    fewer than two usable values or the pooled SD is zero.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(sp2))
