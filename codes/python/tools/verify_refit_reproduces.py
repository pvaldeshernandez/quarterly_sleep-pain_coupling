"""Did the re-run reproduce the results it was supposed to reproduce?

A pipeline change is only safe if the steps it did NOT mean to touch come back
identical. Every fit is seeded (`random_seed=42`) and the model graph is
deterministic, so a step whose inputs and code are unchanged must return the
same numbers to the last digit. Anything else is a silent change, and finding it
by noticing a shifted digit in a table months later is not a plan.

Compares every tabular artifact under a baseline tree against the live one:

    python tools/verify_refit_reproduces.py archive/preallsteps_20260808

Reports, per file: identical / differs (with the worst cell) / new / missing.
Exit code 1 if anything that existed in the baseline differs.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

#: below this, a difference is float-formatting noise from the CSV round trip,
#: not a changed result. Set deliberately tight: a real MCMC difference from a
#: changed graph is orders of magnitude larger than this.
TOL = 1e-9


def tabular(root):
    """{relative path: absolute path} for every CSV under `root`."""
    out = {}
    for dirpath, _, files in os.walk(root):
        # Skip anything archived. Superseded outputs are kept for reference and
        # comparing against them would report every one of them as a difference.
        if "archive" in dirpath.split(os.sep):
            continue
        for fn in files:
            if fn.endswith(".csv"):
                p = os.path.join(dirpath, fn)
                out[os.path.relpath(p, root)] = p
    return out


def compare(a_path, b_path):
    """(verdict, detail) for one pair of CSVs."""
    try:
        a = pd.read_csv(a_path, float_precision="round_trip")
        b = pd.read_csv(b_path, float_precision="round_trip")
    except Exception as exc:                                   # noqa: BLE001
        return "unreadable", f"{type(exc).__name__}: {exc}"

    if a.shape != b.shape:
        return "differs", f"shape {a.shape} -> {b.shape}"
    if list(a.columns) != list(b.columns):
        gained = [c for c in b.columns if c not in a.columns]
        lost = [c for c in a.columns if c not in b.columns]
        return "differs", f"columns +{gained} -{lost}"

    def numeric(s):
        # `is_numeric_dtype` is True for bool, and subtracting two boolean Series
        # raises rather than giving 0/1. A boolean column is compared as text.
        return (pd.api.types.is_numeric_dtype(s)
                and not pd.api.types.is_bool_dtype(s))

    worst, where = 0.0, ""
    for col in a.columns:
        x, y = a[col], b[col]
        if numeric(x) and numeric(y):
            d = (pd.to_numeric(x, errors="coerce")
                 - pd.to_numeric(y, errors="coerce")).abs()
            # NaN in the SAME cell on both sides is agreement, not a difference.
            mismatch_na = x.isna() != y.isna()
            if mismatch_na.any():
                return "differs", f"{col}: NaN pattern changed"
            if d.notna().any() and float(d.max()) > worst:
                worst, where = float(d.max()), f"{col} row {int(d.idxmax())}"
        else:
            if not x.astype(str).equals(y.astype(str)):
                bad = (x.astype(str) != y.astype(str))
                return "differs", f"{col}: text changed at row {int(bad.idxmax())}"

    if worst > TOL:
        return "differs", f"max |delta| {worst:.3e} at {where}"
    return "identical", f"max |delta| {worst:.1e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", help="directory holding the pre-change results/ "
                                     "and derivatives/ trees")
    ap.add_argument("--quiet", action="store_true",
                    help="list only the files that differ")
    a = ap.parse_args()

    base_root = a.baseline if os.path.isabs(a.baseline) \
        else os.path.join(ROOT, a.baseline)
    if not os.path.isdir(base_root):
        print(f"no such baseline: {base_root}")
        return 2

    n = {"identical": 0, "differs": 0, "new": 0, "missing": 0, "unreadable": 0}
    differing = []

    for tree in ("results", "derivatives"):
        b_dir, l_dir = os.path.join(base_root, tree), os.path.join(ROOT, tree)
        if not os.path.isdir(b_dir):
            continue
        baseline, live = tabular(b_dir), tabular(l_dir)
        print(f"\n{'=' * 74}\n{tree}/  —  {len(baseline)} baseline file(s)\n{'=' * 74}")

        for rel in sorted(baseline):
            if rel not in live:
                n["missing"] += 1
                print(f"  MISSING    {rel}")
                continue
            verdict, detail = compare(baseline[rel], live[rel])
            n[verdict] += 1
            if verdict == "identical":
                if not a.quiet:
                    print(f"  identical  {rel}")
            else:
                differing.append((tree, rel, detail))
                print(f"  DIFFERS    {rel}   {detail}")

        for rel in sorted(set(live) - set(baseline)):
            n["new"] += 1
            if not a.quiet:
                print(f"  new        {rel}")

    print(f"\n{'=' * 74}\nSUMMARY\n{'=' * 74}")
    for k in ("identical", "differs", "new", "missing", "unreadable"):
        print(f"  {k:11s} {n[k]:4d}")

    if differing:
        print("\nEVERY DIFFERENCE MUST BE EXPLAINED. A step whose code and inputs "
              "did not change\nmust reproduce exactly — the seed is fixed.")
        for tree, rel, detail in differing:
            print(f"  * {tree}/{rel} — {detail}")
    else:
        print("\nNothing that existed before has changed.")
    return 1 if differing or n["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
