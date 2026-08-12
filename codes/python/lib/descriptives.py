"""Item-level descriptives and the canonical item lists.

This module owns two things that were previously re-typed in several places:

1. **The item lists.** ``KNEE_ITEMS``/``BODY_ITEMS``/``PAIN_ITEMS``/``SLEEP_ITEM``
   used to be defined in ``step01_factor_analysis`` and imported from there by
   step 02, while step 04 held its own copy. One home now; step 01 imports from
   here, so there is exactly one place where "the eight pain items" is decided.

2. **The descriptive concepts** — per-item summaries, the one-way ICC from
   variance components, and the restriction of a raw frame to the analytic row
   set. Each is written around the concept, not around one caller's data: the
   response scale is a parameter, the row-set key is a parameter, and the frame
   is whatever the caller has.

Functions
---------
item_descriptives(df, items, scale=(0., 10.), floor=None, ceiling=None,
                  region_map=None, sample=None)
    One row per item: n, mean, sd, median, IQR, range, % at floor, % at ceiling.
icc_varcomp(df, value, group="ID")
    One-way random-effects ICC and its between/within SDs, unbalanced-corrected.
restrict_to_analytic(raw, analytic, by=("ID",))
    The rows of ``raw`` whose ``by`` key appears in ``analytic``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "KNEE_ITEMS", "BODY_ITEMS", "PAIN_ITEMS", "SLEEP_ITEM", "OTHER_ITEMS",
    "ALL_ITEMS", "ITEM_SCALE",
    "item_descriptives", "icc_varcomp", "restrict_to_analytic",
]

# ---------------------------------------------------------------------
# The item lists — one home
# ---------------------------------------------------------------------

#: the four knee-pain items of the quarterly questionnaire
KNEE_ITEMS = [
    "q2_knee_pain", "q3_knee_pain", "q4_knee_pain", "q5_knee_pain",
]
#: the four body-pain (non-knee) items
BODY_ITEMS = [
    "q7_body_pain", "q8_body_pain", "q9_body_pain", "q10_body_pain",
]
#: the eight items that enter the two-factor pain measurement model
PAIN_ITEMS = KNEE_ITEMS + BODY_ITEMS
#: the single sleep-quality item. 0 = "very poorly", 10 = "very well":
#: HIGHER IS BETTER SLEEP, which is why a negative pain-to-sleep coupling is
#: the intuitive direction.
SLEEP_ITEM = "q13_sleep"
#: the quarterly items that are described but do not enter the pain factors
OTHER_ITEMS = ["q11_fatigue", "q12_mood", SLEEP_ITEM]
#: every item the descriptive tables cover
ALL_ITEMS = PAIN_ITEMS + OTHER_ITEMS

#: the common response scale of every q2-q13 item, as (floor, ceiling)
ITEM_SCALE = (0.0, 10.0)


# ---------------------------------------------------------------------
# Descriptives
# ---------------------------------------------------------------------

def item_descriptives(df, items, scale=ITEM_SCALE, floor=None, ceiling=None,
                      region_map=None, sample=None):
    """Per-item descriptives on whatever rows the caller passes.

    Each item is summarized on its OWN complete cases (``dropna`` per column),
    not on list-wise complete rows — so ``n`` differs between items when an item
    was skipped, and the denominator of every percentage is that item's ``n``.

    Parameters
    ----------
    df : DataFrame
        The rows to describe. Restriction to a sample, a quarter or an analytic
        row set is the caller's business; this function describes what it is
        given.
    items : sequence of str
        Column names, in the order they should appear in the table.
    scale : (float, float), default ``ITEM_SCALE``
        The response scale as ``(floor, ceiling)``. Defines what "at floor" and
        "at ceiling" mean; it is NOT used to clip or validate.
    floor, ceiling : float, optional
        Override either end of ``scale`` individually. Present because a caller
        that only cares about the ceiling should not have to restate the floor.
    region_map : dict, optional
        ``{item: region}``. When given, a ``region`` column is emitted.
    sample : str, optional
        A label for the row set described. When given, a ``sample`` column is
        emitted, so descriptives computed on different samples (step 02's
        calibration sample and step 04's analytic sample) can be concatenated
        without ambiguity.

    Returns
    -------
    DataFrame
        One row per item, ``item`` first so callers can ``insert`` a key column
        at position 1. Columns: item, [region], [sample], n, mean, sd, median,
        p25, p75, min, max, pct_at_floor, pct_at_ceiling.
        An item with no observations gives ``n = 0`` and NaN elsewhere rather
        than being dropped — a vanished item must be visible, not absent.
    """
    lo = scale[0] if floor is None else floor
    hi = scale[1] if ceiling is None else ceiling

    missing = [c for c in items if c not in df.columns]
    if missing:
        raise KeyError(f"item(s) absent from the frame: {missing}")

    rows = []
    for item in items:
        s = pd.to_numeric(df[item], errors="coerce").dropna()
        row = {"item": item}
        if region_map is not None:
            row["region"] = region_map.get(item)
        if sample is not None:
            row["sample"] = sample
        if len(s) == 0:
            row.update({"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan,
                        "p25": np.nan, "p75": np.nan, "min": np.nan, "max": np.nan,
                        "pct_at_floor": np.nan, "pct_at_ceiling": np.nan})
        else:
            row.update({
                "n": int(len(s)),
                "mean": float(s.mean()),
                "sd": float(s.std()),             # ddof=1, pandas default
                "median": float(s.median()),
                "p25": float(s.quantile(0.25)),
                "p75": float(s.quantile(0.75)),
                "min": float(s.min()),
                "max": float(s.max()),
                "pct_at_floor": float(100 * (s == lo).mean()),
                "pct_at_ceiling": float(100 * (s == hi).mean()),
            })
        rows.append(row)

    cols = (["item"]
            + (["region"] if region_map is not None else [])
            + (["sample"] if sample is not None else [])
            + ["n", "mean", "sd", "median", "p25", "p75", "min", "max",
               "pct_at_floor", "pct_at_ceiling"])
    return pd.DataFrame(rows)[cols]


def icc_varcomp(df, value, group="ID"):
    """One-way random-effects ICC from VARIANCE COMPONENTS, not a variance ratio.

    sigma2_between is estimated as ``(MSB - MSW) / k0``, with ``k0`` the
    correction for unbalanced group sizes; the naive "between-group variance of
    the person means" overstates it whenever people contribute different numbers
    of observations, which they do here.

    Parameters
    ----------
    df : DataFrame
        Long frame with one row per observation.
    value : str
        The column to decompose.
    group : str, default "ID"
        The clustering column (person).

    Returns
    -------
    (icc, sd_between, sd_within) : tuple of float
        ``sigma2_between`` is floored at zero (a negative variance estimate is
        reported as zero, the usual convention). All three are NaN when fewer
        than two groups or no within-group degrees of freedom remain.
    """
    d = df[[group, value]].dropna()
    grp = d.groupby(group)[value]
    n_i = grp.size()
    k = len(n_i)
    n_total = int(n_i.sum())
    if k < 2 or n_total <= k:
        return np.nan, np.nan, np.nan

    gmean = d[value].mean()
    msb = float((n_i * (grp.mean() - gmean) ** 2).sum() / (k - 1))
    msw = float(grp.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum() / (n_total - k))
    k0 = float((n_total - (n_i ** 2).sum() / n_total) / (k - 1))

    s2b = max((msb - msw) / k0, 0.0)
    icc = s2b / (s2b + msw) if (s2b + msw) > 0 else np.nan
    return float(icc), float(np.sqrt(s2b)), float(np.sqrt(msw))


def restrict_to_analytic(raw, analytic, by=("ID",)):
    """The rows of ``raw`` whose ``by`` key appears in ``analytic``.

    Two row-set rules are in use and the difference matters, so it is a
    parameter rather than a hard-coded join:

    - ``by=("ID",)`` — every raw row belonging to an analytic PARTICIPANT,
      including person-quarters the modelling frame later dropped.
    - ``by=("ID", "quarter")`` — only the person-quarters the model consumes.

    IDs are compared as strings on both sides, so an integer-typed ID column on
    one side and a string-typed one on the other cannot silently match nothing.

    Returns
    -------
    DataFrame
        A copy, with the original column set and dtypes of ``raw`` except that
        ``ID`` is string-typed.
    """
    by = tuple(by)
    for col in by:
        if col not in raw.columns:
            raise KeyError(f"{col!r} absent from the raw frame")
        if col not in analytic.columns:
            raise KeyError(f"{col!r} absent from the analytic frame")

    r = raw.copy()
    a = analytic
    if "ID" in by:
        r["ID"] = r["ID"].astype(str)
        a = a.assign(ID=a["ID"].astype(str))

    if len(by) == 1:
        keep = r[by[0]].isin(set(a[by[0]]))
    else:
        keys = set(map(tuple, a[list(by)].itertuples(index=False, name=None)))
        keep = pd.Series(
            [k in keys for k in r[list(by)].itertuples(index=False, name=None)],
            index=r.index,
        )
    return r[keep].copy()
