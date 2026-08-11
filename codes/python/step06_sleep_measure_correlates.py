#!/usr/bin/env python3
"""
Step 06 — Correlates of the quarterly sleep-quality item (Figure S3, Section S4).
======================================================================

Every correlate of the quarterly sleep item, computed once, in one sweep:

  * the person-mean correlation of the item with each baseline sleep instrument
    (the "Mean" column of Figure S3),
  * the same correlation recomputed separately at each of the 11 quarters
    (the Q1-Q11 columns),
  * the STOP-BANG apnea-risk score -- its distribution, its item endorsement, and
    its association with the item -- as ONE MORE ROW of the same grid, not a
    special case,
  * the assembled, already-sorted 17 x 12 grid Figure S3 is drawn from.

This collapses five sandbox scripts (a10b, a10c, a10d's data assembly, a10e, a11).
They swept the same instruments over the same sample, which is why `diary_stems`,
`analytic_ids`, `SINGLE` and `MIN_NIGHTS` each existed two or three times. One sweep
means one definition of each, and it means the STOP-BANG row cannot drift from the
sixteen it is ranked against -- it comes out of the same call to
`baseline_stability.per_occasion` inside the same loop.

Every correlation in this step -- including STOP-BANG's -- goes through
`lib.baseline_stability.correlate`. There is no second Pearson in the step.

Compute only. This step draws NOTHING; `plot_step06_sleep_stability.py` reads
`step06_figureS3_grid.csv` and renders the heatmap. The grid CSV carries an explicit
integer `rank` column so the figure's row order (by |Mean r|, the ordering both the
caption and Section S4's ranking claim depend on) is data, not a re-sort.

Restricted UNCONDITIONALLY to the N=229 coupling analytic sample, so no n in any
table exceeds the N reported elsewhere in the paper.

Input:  data/original/participants_wideformat.xlsx          (read-only)
        data/original/UPLOAD2_Data_Dictionary.xlsx          (read-only; the sleep_diary
                                                             form's nightly variable names)
        derivatives/step00_extract_data/step00_extracted_long.csv   (step 00; ID, quarter,
                                                             q13_sleep -- falls back to
                                                             data/step00_extracted_long.csv)
        derivatives/step04_varx_data/step04_processed_long.csv      (step 04; ID only,
                                                             defines the N=229 sample)
Output:
  derivatives/step06_sleep_measure_correlates/
    step06_person_mean_correlations.csv  — one row per column-based instrument
    step06_per_quarter_correlations.csv  — tidy (instrument, quarter, r, p, n, n_q13)
    step06_stopbang_summary.csv          — one row: distribution, endorsement, association
    step06_figureS3_grid.csv             — tidy 17 x 12 grid, sorted, with `rank`
  results/step06_sleep_measure_correlates/
    numbers.json                         — every quantity the documents quote

Aggregates only; no participant-level output. No prose.

Usage:
    python step06_sleep_measure_correlates.py            # load saved CSVs, rewrite numbers
    python step06_sleep_measure_correlates.py --refit    # recompute from data/

Author: Pedro Valdes-Hernandez (with Claude Opus 5)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)
ROOT = os.path.dirname(os.path.dirname(HERE))

# Shared modules. All four are pandas-only by design: this step fits nothing, so it
# must not drag the sampler stack (pymc/pytensor/arviz) in through an import.
import baseline_stability as bs          # noqa: E402
import sleep_instruments as si           # noqa: E402
import stopbang as sbg                   # noqa: E402
from analytic_sample import analytic_ids  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
DERIV_DIR = os.path.join(ROOT, "derivatives")
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step06_sleep_measure_correlates")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step06_sleep_measure_correlates")

IN_WIDE = os.path.join(DATA_DIR, "original", "participants_wideformat.xlsx")
IN_DICT = os.path.join(DATA_DIR, "original", "UPLOAD2_Data_Dictionary.xlsx")
IN_ANALYTIC = os.path.join(DERIV_DIR, "step04_varx_data", "step04_processed_long.csv")

# Step 00's long export. The pipeline is moving it out of the read-only data/ tree into
# derivatives/; accept either while that move is in flight rather than hard-coding the
# one that happens to exist today.
IN_LONG_CANDIDATES = [
    os.path.join(DERIV_DIR, "step00_extract_data", "step00_extracted_long.csv"),
    os.path.join(DATA_DIR, "step00_extracted_long.csv"),
]

OUT_MEAN_CSV = os.path.join(STEP_DERIV_DIR, "step06_person_mean_correlations.csv")
OUT_QUARTER_CSV = os.path.join(STEP_DERIV_DIR, "step06_per_quarter_correlations.csv")
OUT_STOPBANG_CSV = os.path.join(STEP_DERIV_DIR, "step06_stopbang_summary.csv")
OUT_GRID_CSV = os.path.join(STEP_DERIV_DIR, "step06_figureS3_grid.csv")

#: Every supplementary figure lands in one directory, named as the supplement numbers
#: it, so the figure-update tool can pair a file with the caption that guards it.
SUPP_DIR = os.path.join(RESULTS_DIR, "supplementary_materials")
OUT_FIGURE_S3 = os.path.join(SUPP_DIR, "figure_sleep_stability_heatmap.png")

#: the coupling model's quarters. The quarterly item is restricted to these in ONE
#: place; a10b did not filter and a10c/a10e did, which was only harmless because
#: quarter 0 carries no q13 rating at all. `_load_inputs` asserts that, so the day it
#: stops being true this step fails instead of quietly changing the Mean column.
QUARTERS = list(range(1, 12))

#: column order of the grid: the eleven quarters, then the person-mean summary
COL_LABELS = [f"Q{q}" for q in QUARTERS] + ["Mean"]

#: a cell at or below this p is "significant" -- hatched otherwise in Figure S3
ALPHA = 0.05


# ===================================================================
# Inputs
# ===================================================================

def _first_existing(candidates, what):
    """The first path that exists, or a FileNotFoundError naming every candidate."""
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"{what}: none of these exist:\n  " + "\n  ".join(candidates))


def _load_inputs(verbose=True):
    """Read every input and return (wide, dd, rep, ids).

    `rep` is the quarterly sleep item in the long form `baseline_stability` expects
    (ID, occasion, value), restricted to the analytic sample and to QUARTERS. `wide`
    is restricted to the same IDs, so every person-level value derived from it is
    restricted by construction rather than by each consumer remembering to.
    """
    in_long = _first_existing(IN_LONG_CANDIDATES, "step 00 long export")
    for path, what in ((IN_WIDE, "participants export"),
                       (IN_DICT, "data dictionary"),
                       (IN_ANALYTIC, "step 04 analytic sample")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{what}: {path}")

    ids = analytic_ids(IN_ANALYTIC)

    long = pd.read_csv(in_long, usecols=["ID", "quarter", "q13_sleep"])
    long["ID"] = long["ID"].astype(str)
    long = long[long["ID"].isin(ids)]

    # The unification of the quarter filter is a no-op ONLY while quarter 0 carries no
    # rating. Assert it rather than inherit it.
    n_q0 = int(long.loc[long["quarter"] == 0, "q13_sleep"].notna().sum())
    assert n_q0 == 0, (
        f"quarter 0 now carries {n_q0} sleep ratings; restricting to quarters "
        f"{QUARTERS[0]}-{QUARTERS[-1]} is no longer a no-op and the person-mean "
        f"column would change")

    rep = long[long["quarter"].isin(QUARTERS)].rename(
        columns={"quarter": "occasion", "q13_sleep": "value"})

    wide = pd.read_excel(IN_WIDE)
    wide["ID"] = wide["ID"].astype(str)
    wide = wide[wide["ID"].isin(ids)]
    dd = pd.read_excel(IN_DICT)

    if verbose:
        print(f"  analytic sample: {len(ids)} IDs")
        print(f"  long export:     {os.path.relpath(in_long, ROOT)}")
        print(f"  quarters:        {QUARTERS[0]}-{QUARTERS[-1]} "
              f"({rep['value'].notna().sum()} ratings)")
    return wide, dd, rep, ids


# ===================================================================
# The sweep
# ===================================================================

def _person_level_values(wide, dd, ids, verbose=True):
    """Every baseline measure as a person-level Series, plus its print metadata.

    Returns (series, meta):
      series  {instrument -> Series indexed by ID}, the 7-night diary means, the
              single-administration columns, and the STOP-BANG total.
      meta    DataFrame indexed by instrument with kind / description / median_nights.

    STOP-BANG arrives here rather than in a branch of its own so that everything
    downstream -- the correlations, the sort, the grid -- treats it as one more
    baseline measure. Being scored from eight components instead of read from one
    column is plumbing, and plumbing is this function's business, not the figure's.
    """
    values, meta = si.baseline_frame(wide, dd, min_nights=si.MIN_NIGHTS)
    series = {col: values[col] for col in values.columns}

    series[si.STOPBANG_ROW] = sbg.scores(ids=ids, wide=wide)
    meta.loc[si.STOPBANG_ROW] = pd.Series(
        {"kind": "derived score", "description": sbg.LABEL,
         "median_nights": np.nan})

    if verbose:
        n_diary = int((meta["kind"].astype(str).str.startswith("diary")).sum())
        print(f"  baseline measures: {len(series)} "
              f"({n_diary} diary, {len(si.SINGLE)} single administration, 1 derived)")
    return series, meta


def _sweep(series, meta, rep, verbose=True):
    """Correlate every baseline measure with the quarterly item, per quarter and overall.

    ONE loop, ONE correlation function. Returns (person_mean_rows, per_quarter_rows).

    The n >= MIN_N screen a10b/a10c applied silently is kept, but a registry
    instrument that trips it raises by name instead of vanishing from the figure --
    a short grid is worse than a crash, and a KeyError later inside `select` reads
    like a registry bug rather than the data screen it actually is.
    """
    n_q13_by_quarter = (rep.groupby("occasion")["value"]
                        .apply(lambda v: int(v.notna().sum())).to_dict())

    mean_rows, quarter_rows = [], []
    for name, s in series.items():
        kind = str(meta.at[name, "kind"])
        r, p, n = bs.against_person_mean(rep, s)
        rho, p_rho, _ = bs.against_person_mean(rep, s, method="spearman")

        if n < si.MIN_N:
            if name in si.ROW_LABELS:
                raise ValueError(
                    f"{name} ({si.label(name)}) has only {n} complete pairs, below "
                    f"sleep_instruments.MIN_N = {si.MIN_N}, but it is a reported "
                    f"instrument. Figure S3 cannot be drawn short; fix the data or "
                    f"the registry, do not drop the row.")
            if verbose:
                print(f"    dropped {name}: {n} complete pairs < {si.MIN_N}")
            continue

        mean_rows.append({
            "instrument": name,
            "kind": kind,
            "description": meta.at[name, "description"],
            "n": int(n),
            "pearson_r": r,
            "pearson_p": p,
            "spearman_rho": rho,
            "spearman_p": p_rho,
            "median_nights": meta.at[name, "median_nights"],
        })

        grid = bs.per_occasion(rep, s, occasions=QUARTERS)
        for q in QUARTERS:
            quarter_rows.append({
                "instrument": name,
                "quarter": int(q),
                "r": grid.at[q, "r"],
                "p": grid.at[q, "p"],
                "n": int(grid.at[q, "n"]),
                "n_q13": n_q13_by_quarter.get(q, 0),
            })
    return mean_rows, quarter_rows


def _assemble_grid(mean_table, per_quarter, verbose=True):
    """The sorted 17 x 12 grid, tidy, with an explicit integer `rank`.

    STOP-BANG joins BEFORE the sort, not after it. The reader's question is where
    apnea risk ranks against the insomnia, restedness and continuity measures;
    segregating it to the bottom answers a bookkeeping question instead and hides the
    comparison the figure exists to make.

    `rank` is written out because the figure step must not re-sort: a tidy CSV read
    back with `pivot` comes out alphabetical, and both the caption ("ordered by
    strength") and Section S4's ninth-place claim depend on the order being the one
    computed here.
    """
    mean = mean_table.set_index("instrument")
    wide_r = per_quarter.pivot(index="instrument", columns="quarter", values="r")
    wide_p = per_quarter.pivot(index="instrument", columns="quarter", values="p")
    wide_n = per_quarter.pivot(index="instrument", columns="quarter", values="n")
    for frame in (wide_r, wide_p, wide_n):
        frame.columns = [f"Q{q}" for q in frame.columns]

    r = pd.concat([wide_r, mean["pearson_r"].rename("Mean")], axis=1)[COL_LABELS]
    p = pd.concat([wide_p, mean["pearson_p"].rename("Mean")], axis=1)[COL_LABELS]
    n = pd.concat([wide_n, mean["n"].rename("Mean")], axis=1)[COL_LABELS]

    # The sixteen column-based instruments, plus the derived row. `select` raises if
    # any registry instrument is absent, so a short grid cannot reach the figure.
    keep = list(si.select(r).index) + [si.STOPBANG_ROW]
    r, p, n = r.loc[keep], p.loc[keep], n.loc[keep]

    order = r["Mean"].abs().sort_values(ascending=False).index
    r, p, n = r.loc[order], p.loc[order], n.loc[order]

    rows = [
        {
            "rank": i + 1,
            "instrument": var,
            "label": si.label(var),
            "column": col,
            "r": r.at[var, col],
            "p": p.at[var, col],
            "n": int(n.at[var, col]),
            "significant": bool(p.at[var, col] <= ALPHA),
        }
        for i, var in enumerate(r.index)
        for col in COL_LABELS
    ]
    grid = pd.DataFrame(rows)
    if verbose:
        print(f"  grid: {len(order)} rows x {len(COL_LABELS)} columns, sorted by "
              f"|Mean r| (STOP-BANG ranks "
              f"{list(order).index(si.STOPBANG_ROW) + 1})")
    return grid


# ===================================================================
# Reading the grid back, and the numbers it implies
# ===================================================================

def pivot_grid(grid):
    """Tidy grid -> (r, p, n) frames in RANK order, columns COL_LABELS.

    The one place the saved order is honored. `pivot` alone sorts the index
    alphabetically, which would silently reorder Figure S3's rows; the figure step
    calls this rather than pivoting for itself.
    """
    order = (grid.sort_values("rank")["instrument"]
             .drop_duplicates().tolist())
    out = []
    for stat in ("r", "p", "n"):
        frame = grid.pivot(index="instrument", columns="column", values=stat)
        out.append(frame.loc[order, COL_LABELS])
    return tuple(out)


def check_grid(r, p, n):
    """Everything that must be true of the grid before anything is published from it."""
    expected = (si.N_ROWS, len(COL_LABELS))
    assert r.shape == expected, f"grid is {r.shape}, expected {expected}"
    assert si.STOPBANG_ROW in r.index, "the STOP-BANG row is missing"
    for var in si.PAIN_DIARY:
        assert var not in r.index, f"pain item {var} leaked into the figure"

    # `correlate` returns NaN rather than raising when an occasion is too sparse --
    # right for a quarterly cell, fatal for a row's headline number, which would
    # otherwise go blank without anything noticing.
    blank = [v for v in r.index if not np.isfinite(r.at[v, "Mean"])]
    assert not blank, f"Mean correlation is NaN for {blank}"

    absmean = r["Mean"].abs().values
    assert np.all(np.diff(absmean) <= 1e-12), (
        "rows are not ordered by descending |Mean r|; the saved `rank` no longer "
        "matches the values, so Figure S3 would be drawn out of order")
    assert (n[COL_LABELS] >= 0).values.all()


def compute_numbers(grid, per_quarter, stopbang, verbose=True):
    """Every quantity the documents quote, from the saved outputs alone.

    Both paths call this, so the default run and a --refit run cannot disagree about
    what the step reports.
    """
    r, p, n = pivot_grid(grid)
    check_grid(r, p, n)

    quarters = [c for c in COL_LABELS if c != "Mean"]
    q_r, q_p = r[quarters], p[quarters]

    sign_mean = np.sign(r["Mean"])
    reversal = q_r.apply(np.sign).ne(sign_mean, axis=0) & q_r.notna()
    significant = q_p <= ALPHA

    nums = {
        "n_analytic": int(stopbang["n_analytic"]),
        "n_quarters": len(quarters),
        "n_instruments": si.N_INSTRUMENTS,
        "n_figureS3_rows": si.N_ROWS,
        # The denominator behind every quarterly n: analytic participants with a
        # rating at that quarter. Stored so a change in the quarter filter, or in
        # step 00's export, shows up as a moved number instead of a silent one.
        "n_q13_observations": int(
            per_quarter.drop_duplicates("quarter")["n_q13"].sum()),
    }

    for var in r.index:
        nums[f"mean_r.{var}"] = float(r.at[var, "Mean"])
        nums[f"mean_p.{var}"] = float(p.at[var, "Mean"])
        nums[f"mean_n.{var}"] = int(n.at[var, "Mean"])

    # The two |r| bands the Results quote. The grouping is a registry decision, not an
    # inference made here: naps, sleep medication, alcohol and the PSQI duration item
    # are deliberately in NEITHER set.
    for tag, members in (("rating", si.RATING),
                         ("diary_recorded", si.DIARY_RECORDED)):
        vals = r.loc[[v for v in r.index if v in members], "Mean"].abs()
        assert len(vals) == len(members), (
            f"{tag}: registry lists {len(members)} members, grid has {len(vals)}")
        nums[f"abs_r_{tag}_min"] = float(vals.min())
        nums[f"abs_r_{tag}_max"] = float(vals.max())

    nums.update({
        "n_grid_cells": int(q_r.notna().values.sum()),
        "n_cells_significant": int(significant.values.sum()),
        # A reversal is a quarterly cell whose sign differs from the sign of its own
        # row's Mean. Pinned here because the documents assert these counts and no
        # script ever produced them.
        "n_sign_reversals_total": int(reversal.values.sum()),
        "n_sign_reversals_significant": int((reversal & significant).values.sum()),
        "n_rows_significant_all_quarters": int(significant.all(axis=1).sum()),
        "stopbang_rank_by_abs_mean_r": int(list(r.index).index(si.STOPBANG_ROW) + 1),
        "stopbang_n_quarters_same_sign": int((~reversal.loc[si.STOPBANG_ROW]).sum()),
    })

    for key, value in stopbang.items():
        nums[f"stopbang_{key}"] = value

    # One published number, one code path. The STOP-BANG correlation quoted in
    # Section S4 and reply R2.16 is the same cell the figure draws; if these ever
    # differ, a second Pearson has crept back in.
    summary_r, grid_r = nums["stopbang_r_q13"], nums[f"mean_r.{si.STOPBANG_ROW}"]
    assert np.isclose(summary_r, grid_r, rtol=0, atol=1e-12), (
        f"STOP-BANG r disagrees between the summary ({summary_r!r}) and the "
        f"grid ({grid_r!r}): there is a second Pearson in the step")
    assert int(nums["stopbang_n_q13"]) == int(nums[f"mean_n.{si.STOPBANG_ROW}"])

    if verbose:
        print(f"\n  {'':34s} {'Mean r':>8} {'p':>10} {'n':>5}")
        for var in r.index:
            print(f"  {si.label(var):34s} {r.at[var, 'Mean']:+8.3f} "
                  f"{p.at[var, 'Mean']:10.2e} {int(n.at[var, 'Mean']):5d}")
        print(f"\n  {nums['n_cells_significant']} of {nums['n_grid_cells']} quarterly "
              f"cells significant; {nums['n_sign_reversals_total']} sign reversal(s), "
              f"{nums['n_sign_reversals_significant']} of them significant")
        print(f"  {nums['n_rows_significant_all_quarters']} row(s) significant in all "
              f"{nums['n_quarters']} quarters")
    return nums


def _stored_numbers():
    """The previous run's numbers.json, unprefixed, or {} if there is none."""
    path = os.path.join(STEP_RESULTS_DIR, "numbers.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        block = json.load(fh)
    return {k.split(".", 1)[1]: v for k, v in block.items() if k.startswith("step06.")}


# ===================================================================
# Step entry point
# ===================================================================

def run_step06(verbose=True, refit=False):
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("STEP 06 — Correlates of the quarterly sleep measure (Figure S3, S4)")
        print("=" * 70)

    saved = [OUT_MEAN_CSV, OUT_QUARTER_CSV, OUT_STOPBANG_CSV, OUT_GRID_CSV]
    saved_exist = all(os.path.exists(f) for f in saved)
    if not refit and not saved_exist:
        if verbose:
            print("  Saved derivatives not found — forcing refit.")
        refit = True

    previous = _stored_numbers()

    if refit:
        # ------ FULL RECOMPUTE from data/ ------
        wide, dd, rep, ids = _load_inputs(verbose)

        # The gate. `summary()` regenerates the values Section S4 and reply R2.16
        # quote from a CSV whose producing script was lost; this asserts it still
        # does, on the frames already loaded here rather than on paths of its own.
        q13_person_mean = rep.groupby("ID")["value"].mean()
        sbg.test_reproduces_published(q13_person_mean=q13_person_mean, ids=ids,
                                      wide=wide)
        if verbose:
            print("  STOP-BANG reproduces every published number.")

        series, meta = _person_level_values(wide, dd, ids, verbose)
        mean_rows, quarter_rows = _sweep(series, meta, rep, verbose)

        all_means = pd.DataFrame(mean_rows)
        all_means = all_means.reindex(
            all_means["pearson_r"].abs().sort_values(ascending=False).index)
        per_quarter = pd.DataFrame(quarter_rows).sort_values(
            ["instrument", "quarter"], kind="stable")

        grid = _assemble_grid(all_means, per_quarter, verbose)
        stopbang = sbg.summary(q13_person_mean=q13_person_mean, ids=ids, wide=wide)

        # The person-mean table keeps the two nightly PAIN items -- they come off the
        # same diary form and were always computed -- but they are not sleep measures
        # and never reach the grid; the exclusion lives in the registry, so it is
        # auditable rather than silent. STOP-BANG is the other way round: it belongs
        # to the grid and to its own summary, not to the column-based table.
        person_mean = all_means[all_means["instrument"] != si.STOPBANG_ROW]

        person_mean.to_csv(OUT_MEAN_CSV, index=False)
        per_quarter.to_csv(OUT_QUARTER_CSV, index=False)
        pd.DataFrame([stopbang]).to_csv(OUT_STOPBANG_CSV, index=False)
        grid.to_csv(OUT_GRID_CSV, index=False)
        if verbose:
            for path in saved:
                print(f"  wrote {os.path.relpath(path, ROOT)}")
    else:
        # ------ DEFAULT: load the saved derivatives ------
        if verbose:
            print("  Loading saved derivatives (no Excel I/O).")
            print("  If upstream data or code changed, re-run with --refit.")
        # float_precision="round_trip" is not a nicety: pandas' default parser loses
        # the last digit, which would make numbers.json differ between this path and
        # --refit. Two runs of the same step must not report different numbers.
        person_mean = pd.read_csv(OUT_MEAN_CSV, float_precision="round_trip")
        per_quarter = pd.read_csv(OUT_QUARTER_CSV, float_precision="round_trip")
        stopbang = pd.read_csv(OUT_STOPBANG_CSV,
                               float_precision="round_trip").iloc[0].to_dict()
        grid = pd.read_csv(OUT_GRID_CSV, float_precision="round_trip")

    # ---- Figure S3 ---------------------------------------------------------
    # Drawn from the grid DATA, on both paths. Until this existed, step 06 computed
    # the grid and nothing rendered it: the only thing that could draw Figure S3 was
    # a sandbox script that was never migrated, so the figure in the supplement had
    # no pipeline source and could not be regenerated.
    r_grid, p_grid, n_grid = pivot_grid(grid)
    check_grid(r_grid, p_grid, n_grid)
    os.makedirs(SUPP_DIR, exist_ok=True)
    from heatmap import render as render_heatmap
    fig_path = render_heatmap(
        r_grid, p_grid, n_grid,
        columns=COL_LABELS,
        row_labels=[si.label(v) for v in r_grid.index],
        out_path=OUT_FIGURE_S3,
    )
    if verbose:
        print(f"  wrote {os.path.relpath(fig_path, ROOT)}")

    nums = compute_numbers(grid, per_quarter, stopbang, verbose)

    # The quarter filter was unified across a10b/a10c/a10e; this reports the drift a
    # unification is supposed to make visible. Not fatal -- upstream data legitimately
    # changes -- but never silent.
    before = previous.get("n_q13_observations")
    if before is not None and int(before) != int(nums["n_q13_observations"]):
        print(f"  WARNING: quarterly sleep ratings changed from {int(before)} to "
              f"{int(nums['n_q13_observations'])} since the last run.")

    from registry import write_numbers
    path = write_numbers(STEP_RESULTS_DIR, nums, prefix="step06")
    if verbose:
        print(f"\n  wrote {os.path.relpath(path, ROOT)} ({len(nums)} numbers)")
        print("=" * 70)
    return nums


def main():
    ap = argparse.ArgumentParser(
        description="Step 06 — correlates of the quarterly sleep measure "
                    "(Figure S3 grid, Section S4 STOP-BANG).")
    ap.add_argument("--refit", action="store_true",
                    help="Recompute from data/ instead of loading saved derivatives")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run_step06(verbose=not args.quiet, refit=args.refit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
