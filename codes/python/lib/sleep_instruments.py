"""Canonical registry of the baseline sleep instruments used in the R1.1/R2.3 revision.

ONE definition of which variables are in the set, what they are called in print,
and what their long-form descriptions are. Every script that computes or displays
these correlations imports from here, so the instrument set and the labels can
never drift between the table, the figure, and the manuscript prose.

The manuscript reports SIXTEEN baseline sleep variables. The nightly diary form
also carries two PAIN items; they come off the same form and are computed by the
upstream scripts, but they are not sleep measures and are excluded from the
reported set. That exclusion lives in PAIN_DIARY below so it is explicit and
auditable rather than silent.

Consumers:
  step06_sleep_measure_correlates.py     the whole sweep, the grid and Figure S3
  a10b_diary_averaged.py                 person-mean correlations (the Mean column)
  a10c_per_quarter_sleep_stability.py    per-quarter correlations (Q1-Q11)
  a10d_render_heatmap.py                 Figure S3 heatmap + table CSV
"""
import re

import numpy as np
import pandas as pd

# --- diary items -------------------------------------------------------------
# 7 nightly administrations at baseline, averaged over the nights completed
# (minimum 3 of 7). `varname` is the data-dictionary stem; nightly columns are
# <stem><n>__s1, except some items use a bare stem for night 1, so consumers
# resolve columns from the dictionary rather than assuming a numbering scheme.
DIARY = {
    "sleep_quality":            "Diary: sleep quality",
    "sleep_refreshed":          "Diary: restedness on waking",
    "sleep_awakenings_length":  "Diary: minutes awake after onset",
    "sleep_fall_asleep_length": "Diary: sleep-onset latency",
    "sleep_wake_up_times":      "Diary: number of awakenings",
    "sleep_sleep_length":       "Diary: total hours slept",
    "sleep_nap_times":          "Diary: number of naps",
    "sleep_length_nap":         "Diary: nap duration",
    "sleep_sleepmeds":          "Diary: sleep medication use",
    "sleep_alcohol_drinks":     "Diary: alcoholic drinks",
}

# --- single-administration instruments ---------------------------------------
SINGLE = {
    "Insomnia__s1":            "ISI total",
    "sleep_insomnia_q1__s1":   "ISI: difficulty falling asleep",
    "sleep_insomnia_q2__s1":   "ISI: difficulty staying asleep",
    "sleep_insomnia_q3__s1":   "ISI: waking too early",
    "PROMIS_Sleep_Tscore__s1": "PROMIS Sleep-Related Impairment",
    "PSQI_Duration__s1":       "PSQI: sleep duration",
}

# Long-form descriptions for the single-administration instruments, written into
# the `description` column of the a10b correlation table. Kept verbatim from the
# original a10b SINGLE dict so rewiring a10b to this registry changes no output.
SINGLE_DESC = {
    "Insomnia__s1":            "Insomnia Severity Index (total)",
    "sleep_insomnia_q1__s1":   "ISI item: difficulty falling asleep",
    "sleep_insomnia_q2__s1":   "ISI item: difficulty staying asleep",
    "sleep_insomnia_q3__s1":   "ISI item: problems waking too early",
    "PROMIS_Sleep_Tscore__s1": "PROMIS Sleep-Related Impairment (T-score)",
    "PSQI_Duration__s1":       "PSQI sleep-duration item (discretized)",
}

# --- diary items measuring PAIN, not sleep -----------------------------------
# Computed upstream because they sit on the same nightly diary form, but they are
# sleep-pain correlations rather than measurement evidence, so they are not part
# of the reported sleep-instrument set and never appear in Figure S3.
PAIN_DIARY = {
    "sleep_pain_today":  "Diary: any pain that day",
    "sleep_pain_rating": "Diary: average pain that day",
}

# --- measures assembled from several columns ---------------------------------
# Baseline sleep measures like the rest, but SCORED rather than read from a single
# column, so they arrive from their own module instead of from the a10b/a10c sweep.
# They are peers of the sixteen everywhere a reader is concerned: same figure, same
# ordering by strength, same interpretation. Only the plumbing differs.
DERIVED = {
    "stopbang_total": "STOP-BANG apnea risk",
}

# The sixteen COLUMN-BASED instruments. `select()` restricts the frames a10b and
# a10c produce, and those frames contain only these.
LABELS = {**DIARY, **SINGLE}

N_INSTRUMENTS = len(LABELS)          # 16

# Every row of Figure S3 -- the count the manuscript and the reply quote.
ROW_LABELS = {**LABELS, **DERIVED}

N_ROWS = len(ROW_LABELS)             # 17

#: the one derived row's variable name, so no consumer types the string
STOPBANG_ROW = "stopbang_total"

#: a participant needs at least this many of the 7 diary nights to contribute
MIN_NIGHTS = 3

#: an instrument needs at least this many complete pairs with the quarterly item
MIN_N = 50


# --- the two |r| bands the Results quote -------------------------------------
# The gradient the paper describes is by HOW a measure was obtained, not by what
# it covers. That grouping is a registry decision, recorded here rather than
# re-derived in the step, so the two bands and the sentence cannot drift apart.
#
#: subjective RATINGS of sleep quality, insomnia symptoms and daytime impairment
RATING = (
    "sleep_quality", "sleep_refreshed",
    "Insomnia__s1", "sleep_insomnia_q1__s1", "sleep_insomnia_q2__s1",
    "sleep_insomnia_q3__s1", "PROMIS_Sleep_Tscore__s1",
)
#: counts and durations RECORDED in the nightly diary
DIARY_RECORDED = (
    "sleep_awakenings_length", "sleep_fall_asleep_length",
    "sleep_wake_up_times", "sleep_sleep_length",
)
# Deliberately in NEITHER band: the PSQI duration item (a single retrospective
# estimate, not a nightly record, and the sentence places it BETWEEN the bands),
# and naps, sleep medication and alcohol, which are behaviours rather than
# measures of the night's sleep.


def instruments():
    """Variable names of the sixteen sleep instruments, in registry order."""
    return list(LABELS)


def label(varname):
    """Publication label for a variable name; falls back to the raw name."""
    return ROW_LABELS.get(varname, PAIN_DIARY.get(varname, varname))


def labels_for(varnames):
    """Publication labels for a sequence of variable names."""
    return [label(v) for v in varnames]


def is_pain_item(varname):
    """True for the diary items that measure pain rather than sleep."""
    return varname in PAIN_DIARY


def select(df):
    """Restrict a variable-name-indexed frame to the sixteen sleep instruments.

    Preserves the frame's own row order, so a caller that has already sorted
    keeps its sort. Raises if any instrument is missing, because a silently
    short figure is worse than a crash.
    """
    keep = [v for v in df.index if v in LABELS]
    missing = [v for v in LABELS if v not in df.index]
    if missing:
        raise KeyError(f"registry instruments absent from data: {missing}")
    return df.loc[keep]


# =====================================================================
# Assembling the person-level values
# =====================================================================

def diary_stems(dd, form="sleep_diary", n_nights=7):
    """Every item of `form` that has exactly `n_nights` nightly variables.

    Resolved from the DATA DICTIONARY rather than by pattern-matching column
    names, because the nightly naming is not uniform -- some items run 1..7,
    others use a bare stem for night 1 and 2..7 after it -- and a keyword regex
    over descriptions is what caused the original sweep to miss `sleep_quality`,
    the diary item measuring the same construct as the quarterly rating.

    Returns ``{stem: [(variable_name, description), ...]}``.
    """
    for col in ("REDCap Form", "Variable Name", "Description"):
        if col not in dd.columns:
            raise KeyError(f"data dictionary has no {col!r} column")
    sub = dd[dd["REDCap Form"].astype(str) == form]
    groups = {}
    for _, r in sub.iterrows():
        v = str(r["Variable Name"])
        groups.setdefault(re.sub(r"\d+$", "", v), []).append(
            (v, str(r["Description"])))
    return {k: v for k, v in groups.items() if len(v) == n_nights}


def baseline_frame(wide, dd, min_nights=MIN_NIGHTS):
    """Every column-based baseline sleep measure as person-level values, plus metadata.

    One concept -- "a baseline sleep measure, per person" -- covering both kinds:
    a 7-night diary item becomes the mean of the nights a participant completed
    (at least `min_nights` of them), and a single-administration instrument is
    read straight off its column. The caller does not have to know which is which.

    The two PAIN diary items come off the same nightly form and are returned
    here too; `select()` is what removes them from the reported set, so their
    exclusion is one visible decision rather than a silent omission upstream.

    Parameters
    ----------
    wide : DataFrame
        One row per participant, with an ``ID`` column.
    dd : DataFrame
        The REDCap data dictionary, used to resolve the nightly column names.
    min_nights : int, default ``MIN_NIGHTS``
        Nights required before a diary mean is computed; below it the person's
        value is NaN rather than an average of one or two nights.

    Returns
    -------
    (values, meta) : (DataFrame, DataFrame)
        ``values`` is indexed by ID (string), one column per measure.
        ``meta`` is indexed by measure with columns kind, description,
        median_nights (NaN for single-administration instruments).
    """
    if "ID" not in wide.columns:
        raise KeyError("`wide` has no ID column")
    ids = wide["ID"].astype(str).values

    values, meta_rows = {}, {}

    for stem, items in sorted(diary_stems(dd).items()):
        cols = [f"{v}__s1" for v, _ in items if f"{v}__s1" in wide.columns]
        if len(cols) < len(items):
            # An item whose nightly columns are not all present cannot be
            # averaged over "the nights completed" without changing what the
            # mean means, so it is skipped -- and it is not in LABELS, so
            # `select` would drop it anyway.
            continue
        block = wide[cols].apply(pd.to_numeric, errors="coerce")
        n_nights = block.notna().sum(axis=1)
        avg = block.mean(axis=1).where(n_nights >= min_nights)
        values[stem] = pd.Series(avg.values, index=ids)
        meta_rows[stem] = {
            "kind": f"diary ({len(items)}-night mean)",
            "description": items[0][1][:80],
            "median_nights": float(n_nights[n_nights >= min_nights].median()),
        }

    for col, desc in SINGLE_DESC.items():
        if col not in wide.columns:
            raise KeyError(
                f"{col} ({LABELS.get(col, col)}) is a registry instrument but is "
                f"absent from the wide frame; Figure S3 cannot be drawn short")
        values[col] = pd.Series(
            pd.to_numeric(wide[col], errors="coerce").values, index=ids)
        meta_rows[col] = {"kind": "single administration",
                          "description": desc,
                          "median_nights": np.nan}

    frame = pd.DataFrame(values)
    frame.index.name = "ID"
    meta = pd.DataFrame(meta_rows).T
    meta.index.name = "instrument"
    return frame, meta[["kind", "description", "median_nights"]]
