"""Canonical STOP-BANG obstructive sleep apnea risk score.

ONE definition of how the score is formed, so the number quoted in the supplement, the
number in the Response, and the row drawn in the sleep-stability heatmap can never disagree.

Background: no polysomnography, apnea diagnosis or CPAP record exists in this dataset,
but the four STOP items were administered verbatim as items 5-8 of the baseline sleep
questionnaire and all four BANG components were measured at the same visit, so a
genuine STOP-BANG score is computable rather than a proxy.

    S  sleep_snore           snores loudly
    T  sleep_tired_daytime   often tired, fatigued or sleepy in the daytime
    O  sleep_stop_breathing  observed to stop breathing during sleep
    P  sleep_high_bp         has, or is treated for, high blood pressure
    B  pe_bmi                BMI > 35
    A  age                   > 50 years
    N  pe_neck_circum        neck circumference > 40 cm
    G  gender                male

Risk bands: 0-2 low, 3-4 intermediate, 5-8 high; 3 or higher screens positive.

A participant scores only when all eight components are present -- a partial score is
not comparable across people, since a missing component is indistinguishable from a
negative one.

This module was factored out of `a11_stopbang.py` after the script that produced
`a11_stopbang_summary_N229.csv` -- the file the supplement and the Response quote -- was
found to no longer exist, leaving those numbers unreproducible. `summary()` below
regenerates them exactly; `test_reproduces_published()` asserts it.

Consumers:
  a11_stopbang.py                        distribution and association with sleep quality
  a10b_diary_averaged.py                 person-mean correlation (heatmap Mean column)
  a10c_per_quarter_sleep_stability.py    per-quarter correlations (heatmap, Q1-Q11)
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

# Resolved from THIS file's location: lib/ -> codes/python/ -> codes/ -> UPLOAD2.
# The absolute path this module used to hard-code kept pointing at step05's
# output long after the frame moved to step07, which is exactly the failure a
# hard-coded path produces -- a helper that keeps answering, with stale data.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
IN_WIDE = os.path.join(REPO, "data/original/participants_wideformat.xlsx")
IN_ANALYTIC = os.path.join(REPO, "derivatives/step07_varx_data",
                           "step07_processed_long.csv")

STOP = {"S": "sleep_snore__s1", "T": "sleep_tired_daytime__s1",
        "O": "sleep_stop_breathing__s1", "P": "sleep_high_bp__s1"}
BANG = {"B": "pe_bmi__s1", "A": "age__s1", "N": "pe_neck_circum__s1",
        "G": "gender__s1"}
ITEMS = ["S", "T", "O", "P", "B", "A", "N", "G"]

#: label used wherever the score is displayed alongside the sleep instruments
LABEL = "STOP-BANG apnea risk"

# Thresholds are the published STOP-BANG cut-points, named rather than inlined so a
# reader can check them against the instrument without reading the arithmetic.
BMI_CUT, AGE_CUT, NECK_CUT_CM = 35, 50, 40
MALE_CODE = 1               # this export codes gender 1 = male, 2 = female


def _yes(series):
    """1.0 for an endorsed binary item, 0.0 for a denied one, NaN when absent."""
    return (series == 1).astype(float).where(series.notna())


def components(wide=None):
    """Per-participant STOP-BANG components and total, indexed by ID.

    Columns: S, T, O, P, B, A, N, G, n_missing, score. `score` is NaN unless all
    eight components are present.

    `wide` may be a preloaded participants frame; it is read from disk otherwise.
    """
    cols = list(STOP.values()) + list(BANG.values())
    if wide is None:
        wide = pd.read_excel(IN_WIDE, usecols=lambda c: c in (["ID"] + cols))
    wide = wide.copy()
    wide["ID"] = wide["ID"].astype(str)
    missing = [c for c in cols if c not in wide.columns]
    if missing:
        raise KeyError(f"STOP-BANG components absent from the export: {missing}")
    for c in cols:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")

    # Neck circumference is recorded in cm in this export, but the instrument is
    # specified in inches; convert only if the distribution says the units are inches,
    # rather than trusting either assumption silently.
    neck = wide[BANG["N"]]
    neck_cm = neck if neck.median() > 30 else neck * 2.54

    sb = pd.DataFrame(index=wide["ID"])
    for k in ("S", "T", "O", "P"):
        sb[k] = _yes(wide[STOP[k]]).values
    sb["B"] = (wide[BANG["B"]] > BMI_CUT).astype(float).where(wide[BANG["B"]].notna()).values
    sb["A"] = (wide[BANG["A"]] > AGE_CUT).astype(float).where(wide[BANG["A"]].notna()).values
    sb["N"] = (neck_cm > NECK_CUT_CM).astype(float).where(neck_cm.notna()).values
    sb["G"] = (wide[BANG["G"]] == MALE_CODE).astype(float).where(
        wide[BANG["G"]].notna()).values

    sb["n_missing"] = sb[ITEMS].isna().sum(axis=1)
    sb["score"] = sb[ITEMS].sum(axis=1).where(sb["n_missing"] == 0)
    sb.index.name = "ID"
    return sb


def _default_ids():
    """The analytic sample, for callers that pass no explicit ID set.

    Delegates to ``lib.analytic_sample.analytic_ids`` so there is one definition
    of who is in the sample, and reads it lazily: importing this module must not
    require the pipeline to have run.
    """
    from analytic_sample import analytic_ids
    return analytic_ids(IN_ANALYTIC)


def scores(ids=None, wide=None):
    """STOP-BANG total per participant, restricted to `ids` and to complete scorers.

    Returns a Series indexed by ID. `ids` defaults to the analytic sample, which is
    the population every reported STOP-BANG number refers to; pass an explicit set to
    widen it.
    """
    sb = components(wide)
    keep = _default_ids() if ids is None else set(map(str, ids))
    return sb.loc[sb.index.isin(keep), "score"].dropna()


def summary(q13_person_mean=None, ids=None, wide=None):
    """Every STOP-BANG figure reported in the paper, in one dict.

    `q13_person_mean` is a Series of person-mean sleep quality indexed by ID; when
    given, the association statistics are computed too.
    """
    keep = _default_ids() if ids is None else set(map(str, ids))
    sb = components(wide)
    sb = sb.loc[sb.index.isin(keep)]
    complete = sb[sb["n_missing"] == 0]
    s = complete["score"]

    out = {
        "n_complete": len(complete), "n_analytic": len(keep),
        "mean": s.mean(), "sd": s.std(ddof=1), "median": s.median(),
        "min": s.min(), "max": s.max(),
        "pct_low": 100 * (s <= 2).mean(),
        "pct_int": 100 * ((s >= 3) & (s <= 4)).mean(),
        "pct_high": 100 * (s >= 5).mean(),
        "pct_positive": 100 * (s >= 3).mean(),
    }
    for i in ITEMS:
        out[f"endorse_{i}"] = 100 * complete[i].mean()

    if q13_person_mean is not None:
        q = pd.Series(q13_person_mean).copy()
        q.index = q.index.astype(str)
        j = pd.DataFrame({"score": s}).join(q.rename("q13")).dropna()
        r, p = stats.pearsonr(j["score"], j["q13"])
        hi, lo = j[j["score"] >= 5]["q13"], j[j["score"] <= 2]["q13"]
        _, p_hl = stats.ttest_ind(hi, lo, equal_var=False)
        out.update({"r_q13": r, "p_q13": p, "n_q13": len(j),
                    "hi_mean": hi.mean(), "hi_n": len(hi),
                    "lo_mean": lo.mean(), "lo_n": len(lo), "p_hi_lo": p_hl})
    return out


def test_reproduces_published(q13_person_mean=None, ids=None, wide=None):
    """Assert this module reproduces the numbers the supplement and the Response quote.

    The producing script was lost, so these values were only ever recoverable from a
    CSV. Pinning them here makes the loss non-repeatable: any future change to the
    scoring that would move a published number fails loudly instead of silently.

    All three arguments are optional and default to reading from disk, so the
    module remains runnable standalone (``python lib/stopbang.py``). A caller
    that has already loaded these frames should pass them: the check then runs
    on the SAME data the caller is about to use, which is what makes it a gate
    rather than a separate computation that happens to agree.
    """
    if q13_person_mean is None:
        long = pd.read_csv(os.path.join(REPO, "data/step00_extracted_long.csv"),
                           usecols=["ID", "q13_sleep"], float_precision="round_trip")
        long["ID"] = long["ID"].astype(str)
        q13_person_mean = long.groupby("ID")["q13_sleep"].mean()
    got = summary(q13_person_mean=q13_person_mean, ids=ids, wide=wide)

    want = {"n_complete": 221, "n_analytic": 229, "mean": 3.3122171945701355,
            "sd": 1.6451816887431403, "median": 3.0, "min": 0.0, "max": 8.0,
            "pct_low": 34.38914027149321, "pct_int": 41.6289592760181,
            "pct_high": 23.981900452488688, "pct_positive": 65.61085972850678,
            "r_q13": -0.28574002086362954, "p_q13": 1.6030717383711965e-05,
            "n_q13": 221, "hi_mean": 4.9799797163004715, "hi_n": 53,
            "lo_mean": 6.433765759094706, "lo_n": 76,
            "p_hi_lo": 0.00024093562791044533}
    bad = []
    for k, v in want.items():
        g = got[k]
        if not np.isclose(g, v, rtol=1e-9, atol=1e-12):
            bad.append(f"  {k}: got {g!r}, published {v!r}")
    if bad:
        raise AssertionError("STOP-BANG no longer reproduces the published numbers:\n"
                             + "\n".join(bad))
    return got


if __name__ == "__main__":
    got = test_reproduces_published()
    print("STOP-BANG reproduces every published number.")
    for k, v in got.items():
        print(f"  {k:14s} {v}")
