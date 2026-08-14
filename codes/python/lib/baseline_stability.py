"""Stability of a repeated measure's association with a person-level baseline variable.

The concept: one variable is measured ONCE per person (a baseline instrument, an apnea
risk score, anything person-level); another is measured repeatedly (the quarterly sleep
item). Correlating them separately at each occasion shows whether the association is a
stable property of the measure or an artifact of one particular assessment.

Written around that concept rather than around a particular file layout, so the same
function serves the sixteen baseline sleep instruments of the sleep-stability heatmap and the STOP-BANG
score added to it, at any number of occasions, with no per-variable variants.

Consumers:
  a10c_per_quarter_sleep_stability.py    the sixteen baseline sleep instruments
  a10e_stopbang_stability.py             the STOP-BANG apnea risk score
"""
import numpy as np
import pandas as pd
from scipy import stats

#: an occasion contributes a correlation only with at least this many complete pairs
MIN_PAIRS = 5

#: an occasion is skipped outright below this many people in common
MIN_COMMON = 10


def correlate(x, y, min_pairs=MIN_PAIRS, method="pearson"):
    """Correlation r/rho, p and n over the IDs where both Series are finite.

    `method` is "pearson" (default) or "spearman". Both are reported for the
    person-mean column, so the choice is a parameter rather than a second
    function: the pairing, the alignment and the minimum-pairs rule are the
    same, and only the coefficient differs.

    Returns (nan, nan, 0) rather than raising when too few pairs survive, because a
    sparse occasion should leave a blank cell in the figure, not kill the run.
    """
    common = x.index.intersection(y.index)
    if len(common) < MIN_COMMON:
        return np.nan, np.nan, 0
    a = pd.to_numeric(x[common], errors="coerce").values
    b = pd.to_numeric(y[common], errors="coerce").values
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_pairs:
        return np.nan, np.nan, int(m.sum())
    if method == "pearson":
        r, p = stats.pearsonr(a[m], b[m])
    elif method == "spearman":
        r, p = stats.spearmanr(a[m], b[m])
    else:
        raise ValueError(f"method must be 'pearson' or 'spearman'; got {method!r}")
    return float(r), float(p), int(m.sum())


def per_occasion(repeated, baseline, occasions=None):
    """Correlate a person-level `baseline` Series with `repeated` at each occasion.

    `repeated` is a long-format frame with columns ID, occasion, value -- named
    positionally by the caller via `rename`, so this function never assumes a column
    spelling. `baseline` is a Series indexed by ID.

    Returns a DataFrame indexed by occasion with columns r, p, n.
    """
    need = {"ID", "occasion", "value"}
    if not need.issubset(repeated.columns):
        raise KeyError(f"`repeated` needs columns {sorted(need)}, has "
                       f"{sorted(repeated.columns)}")
    rep = repeated.copy()
    rep["ID"] = rep["ID"].astype(str)
    base = baseline.copy()
    base.index = base.index.astype(str)

    if occasions is None:
        occasions = sorted(rep["occasion"].unique())

    rows = {}
    for occ in occasions:
        at = rep[rep["occasion"] == occ].set_index("ID")["value"]
        r, p, n = correlate(at, base)
        rows[occ] = {"r": r, "p": p, "n": n}
    out = pd.DataFrame(rows).T
    out["n"] = out["n"].astype(int)
    out.index.name = "occasion"
    return out


def against_person_mean(repeated, baseline, method="pearson"):
    """Correlate `baseline` with each person's MEAN of the repeated measure.

    This is the "Mean" column of the sleep-stability heatmap: one number summarizing the association
    across all occasions, computed on the person-level average rather than pooled
    across occasions, so each person contributes exactly once.

    `method` selects Pearson (default) or Spearman; the person means, the
    alignment and the n are identical either way, which is the point of it being
    a parameter.
    """
    rep = repeated.copy()
    rep["ID"] = rep["ID"].astype(str)
    means = rep.groupby("ID")["value"].mean()
    base = baseline.copy()
    base.index = base.index.astype(str)
    return correlate(means, base, method=method)
