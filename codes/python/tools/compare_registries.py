#!/usr/bin/env python3
"""Diff two runs of the pipeline, by NAME rather than by file.

    python tools/compare_registries.py OLD/reported_values.csv NEW/reported_values.csv
    python tools/compare_registries.py OLD NEW --dp 4

`tools/verify_refit_reproduces.py` compares CSVs cell by cell, which answers "is the output
byte-identical". This answers the different question the documents care about: **did any
NAMED quantity change, and by how much** -- because that is the list of sentences that may
need retyping.

TWO THINGS IT DOES THAT A NAIVE DIFF DOES NOT
---------------------------------------------
It keys on (name, step), not on name. `n_fit_records` is published by step 18 meaning "my
own 16 fits" and by step 24 meaning "all 53" -- collapsing them by name reported a change
of 16 -> 53 that never happened, which cost an hour of chasing.

It separates values that moved AT THE REPORTED PRECISION from values that moved further
down. A shift in the eighth decimal is the sampler being a sampler; a shift in the third is
a sentence to retype. Only the second kind is a worklist.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "reported_values.csv")
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["name"], r["step"])
            raw = (r["value"] or "").strip()
            if not raw:
                continue
            try:
                out[key] = float(raw)
            except ValueError:
                out[key] = raw
    return out, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--dp", type=int, default=3,
                    help="decimals at which a change counts as reportable (default 3)")
    args = ap.parse_args()

    old, old_p = load(args.old)
    new, new_p = load(args.new)
    print(f"old: {old_p}  ({len(old)} named values)")
    print(f"new: {new_p}  ({len(new)} named values)\n")

    gone = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    shared = sorted(set(old) & set(new))

    reportable, tiny, textual = [], [], []
    for k in shared:
        a, b = old[k], new[k]
        if isinstance(a, float) and isinstance(b, float):
            if a == b:
                continue
            if math.isnan(a) and math.isnan(b):
                continue
            if round(a, args.dp) == round(b, args.dp):
                tiny.append((k, a, b))
            else:
                reportable.append((k, a, b))
        elif a != b:
            textual.append((k, a, b))

    if reportable:
        print(f"--- {len(reportable)} changed at {args.dp} decimals (RETYPE THESE) ---")
        for (n, s), a, b in reportable:
            print(f"  {n:46s} {a:>14.6g} -> {b:<14.6g}  [{s}]")
    if textual:
        print(f"\n--- {len(textual)} text-valued changes ---")
        for (n, s), a, b in textual:
            print(f"  {n:46s} {str(a)[:22]:>22s} -> {str(b)[:22]:<22s}  [{s}]")
    if gone:
        print(f"\n--- {len(gone)} name(s) the new run does NOT publish ---")
        for n, s in gone:
            print(f"  {n}  [{s}]")
    if added:
        print(f"\n--- {len(added)} name(s) new in this run ---")
        for n, s in added:
            print(f"  {n}  [{s}]")

    print("\n" + "=" * 72)
    print(f"{len(reportable)} reportable change(s) | {len(tiny)} below {args.dp} dp "
          f"| {len(textual)} textual | {len(gone)} dropped | {len(added)} added")
    print(f"{len(shared) - len(reportable) - len(tiny) - len(textual)} identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
