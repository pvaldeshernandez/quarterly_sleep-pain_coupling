#!/usr/bin/env python3
"""Run the whole pipeline, in order, from a terminal.

    python run_pipeline.py                  # replot from saved derivatives (minutes)
    python run_pipeline.py --refit          # recompute everything (hours)
    python run_pipeline.py --from 07 --to 10
    python run_pipeline.py --list

Steps are discovered from the filenames rather than listed here, so inserting
`step09_posterior_predictive_check.py` makes it part of the pipeline with no edit to
this file. Numeric order IS execution order — that invariant is what the renumbering
to 26 steps exists to protect, and `--list` shows it.

Every step exposes `run_stepNN(verbose=..., refit=...)`; step 00 predates that
convention and exposes `main()`.

A failure does not stop the run by default. A late step that cannot load a missing
input is expected while the migration is incomplete, and stopping at the first one
would hide the state of everything after it. `--strict` stops instead.
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lib"))

STEP_RE = re.compile(r"^(step(\d{2})_[a-z0-9_]+)\.py$")


def discover():
    """[(number, module_name)] in execution order."""
    out = []
    for fn in sorted(os.listdir(HERE)):
        m = STEP_RE.match(fn)
        if m:
            out.append((m.group(2), m.group(1)))
    return out


def entry(mod, number):
    """The step's callable, and whether it accepts `refit`."""
    fn = getattr(mod, f"run_step{number}", None)
    if fn is not None:
        return fn, True
    fn = getattr(mod, "main", None)
    if fn is not None:
        return fn, False
    raise AttributeError(f"{mod.__name__}: no run_step{number}() and no main()")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refit", action="store_true",
                    help="recompute instead of loading saved derivatives")
    ap.add_argument("--from", dest="start", default="00", metavar="NN")
    ap.add_argument("--to", dest="stop", default="99", metavar="NN")
    ap.add_argument("--only", default=None, metavar="NN,NN")
    ap.add_argument("--list", action="store_true", help="print the order and exit")
    ap.add_argument("--strict", action="store_true", help="stop at the first failure")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    steps = discover()
    if args.only:
        # Accept "7" for "07": the step numbers are zero-padded on disk, and a
        # selection that silently matches nothing is worse than a typo.
        keep = {s.strip().zfill(2) for s in args.only.split(",") if s.strip()}
        unknown = keep - {s[0] for s in steps}
        if unknown:
            print(f"error: --only names no such step: {', '.join(sorted(unknown))}",
                  file=sys.stderr)
            return 2
        steps = [s for s in steps if s[0] in keep]
    else:
        steps = [s for s in steps
                 if args.start.zfill(2) <= s[0] <= args.stop.zfill(2)]

    if not steps:
        print("error: the selection matched no steps", file=sys.stderr)
        return 2

    if args.list:
        for num, name in steps:
            print(f"  {num}  {name}")
        print(f"\n{len(steps)} step(s)")
        return 0

    verbose = not args.quiet
    mode = "REFIT (recomputing)" if args.refit else "REPLOT (from saved derivatives)"
    print("=" * 72)
    print(f"PIPELINE — {mode}")
    print(f"{len(steps)} steps: {steps[0][0]} … {steps[-1][0]}")
    print("=" * 72)

    results, t_all = [], time.time()
    for num, name in steps:
        print(f"\n{'-' * 72}\n[{num}] {name}\n{'-' * 72}", flush=True)
        t0 = time.time()
        try:
            mod = importlib.import_module(name)
            fn, takes_refit = entry(mod, num)
            if takes_refit:
                fn(verbose=verbose, refit=args.refit)
            else:
                # A step whose only entry point is main() parses sys.argv itself, so
                # it would see the RUNNER's flags and argparse-exit the whole run.
                # Hand it the flags it understands and nothing else.
                argv = sys.argv
                sys.argv = [name] + (["--refit"] if args.refit else []) \
                    + ([] if verbose else ["--quiet"])
                try:
                    fn()
                finally:
                    sys.argv = argv
            dt = time.time() - t0
            results.append((num, name, "ok", dt, ""))
            print(f"[{num}] done in {dt:.1f}s", flush=True)
        # SystemExit is a BaseException, so a step that calls sys.exit() -- argparse
        # does, on any flag it does not recognize -- would otherwise kill the run
        # without ever reaching the summary.
        except (Exception, SystemExit) as exc:         # noqa: BLE001
            dt = time.time() - t0
            results.append((num, name, "FAIL", dt, f"{type(exc).__name__}: {exc}"))
            print(f"[{num}] FAILED after {dt:.1f}s: {type(exc).__name__}: {exc}",
                  flush=True)
            traceback.print_exc()
            if args.strict:
                break

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for num, name, status, dt, err in results:
        flag = "ok  " if status == "ok" else "FAIL"
        print(f"  {flag}  {num}  {name:42s} {dt:7.1f}s  {err[:60]}")
    n_ok = sum(1 for r in results if r[2] == "ok")

    # Collect the deliverables into the two document folders. Not a step: it computes
    # nothing, it only gathers what the steps produced under the names the documents use.
    if not args.only and n_ok == len(results):
        print(f"\n{'-' * 72}\n[--] collect deliverables\n{'-' * 72}", flush=True)
        try:
            import subprocess
            subprocess.run([sys.executable,
                            os.path.join(HERE, "tools", "collect_deliverables.py")],
                           check=False)
        except Exception as exc:                       # noqa: BLE001
            print(f"  collector failed: {type(exc).__name__}: {exc}")

    print(f"\n{n_ok}/{len(results)} step(s) succeeded in {time.time() - t_all:.1f}s")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
