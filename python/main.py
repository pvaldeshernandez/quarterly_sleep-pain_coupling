#!/usr/bin/env python3
"""
Main Pipeline: Quarterly Sleep-Pain Coupling Analysis
======================================================

Runs the complete analysis in manuscript order.

Steps
-----
  01  Prepare data         Factor scoring, within-person centering, lags
  02  Fit coupling model   Bayesian VARX(1) with contrast + age/sex
  03  Contrast moderation  JN analysis of contrast x coupling interaction
  04  NAcc moderation      Left NAcc BOLD moderation of SP coupling
  05  ACC moderation       Right dACC/MCC BOLD moderation of SP coupling
  06  Generate figures     All main + supplementary figures from saved results

Usage
-----
  python python/main.py                  # Run with real data
  python python/main.py --synthetic      # Run with synthetic data
  python python/main.py --step 3         # Start from step 3
  python python/main.py --step 6         # Regenerate figures only

Author: Pedro Valdes-Hernandez
"""

import argparse
import os
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

STEPS = [
    {
        "number": 1,
        "label": "Prepare data",
        "script": os.path.join(SCRIPT_DIR, "01_prepare_data.py"),
    },
    {
        "number": 2,
        "label": "Fit coupling model (base VARX(1))",
        "script": os.path.join(SCRIPT_DIR, "02_fit_coupling_model.py"),
    },
    {
        "number": 3,
        "label": "Contrast moderation (pain localization x coupling)",
        "script": os.path.join(SCRIPT_DIR, "03_contrast_moderation.py"),
    },
    {
        "number": 4,
        "label": "fMRI Sleep->Pain moderation (Krause ROIs + ACC)",
        "script": os.path.join(SCRIPT_DIR, "04_fmri_sp_moderation.py"),
    },
    {
        "number": 5,
        "label": "Arousal Pain->Sleep moderation (Lynch ROIs)",
        "script": os.path.join(SCRIPT_DIR, "05_arousal_ps_moderation.py"),
    },
    {
        "number": 6,
        "label": "Generate figures and tables",
        "script": os.path.join(SCRIPT_DIR, "06_generate_figures.py"),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds):
    """Format seconds as 'Xm Ys' or 'Xs'."""
    if seconds >= 60:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m {s:.1f}s"
    return f"{seconds:.1f}s"


def _run_step(step, synthetic, python_exe, data_dir=None, output_dir=None,
              figures_dir=None, interpolate=False):
    """Run one pipeline step as a subprocess, returning elapsed time.

    Parameters
    ----------
    step : dict
        Pipeline step definition (with "number", "label", "script" keys).
    synthetic : bool
        Forward --synthetic to the sub-script.
    python_exe : str
        Python interpreter to use.
    data_dir : str, optional
        Forward --data-dir to the sub-script.
    output_dir : str, optional
        Forward --output-dir to the sub-script.
    figures_dir : str, optional
        Forward --figures-dir to the sub-script (only step 6 uses it).
    interpolate : bool
        Forward --interpolate to step 1 (ignored by other steps).
    """
    script = step["script"]
    label = step["label"]
    number = step["number"]

    if not os.path.exists(script):
        print(f"  WARNING: {os.path.basename(script)} not found, skipping "
              f"step {number}")
        return 0.0

    cmd = [python_exe, script]
    if synthetic:
        cmd.append("--synthetic")

    # Step 1's output is the processed data CSV — it goes into
    # --output-dir if that's set, otherwise the default data/. For
    # step 1, the meaning of --data-dir is "where to find the raw
    # inputs" and --output-dir is "where to put the processed CSV".
    # For all other steps, --data-dir points at the processed CSV.
    if number == 1:
        # Step 1 reads raw inputs from data_dir and writes processed
        # output to output_dir. When running a sandbox, we typically
        # leave the raw data where it is and redirect the processed
        # output into the sandbox folder.
        if data_dir:
            cmd.extend(["--data-dir", data_dir])
        if output_dir:
            cmd.extend(["--output-dir", output_dir])
        if interpolate:
            cmd.append("--interpolate")
    else:
        # Steps 2-6: both flags point at the sandbox if used.
        # --data-dir is where to find the processed CSV (which is
        # produced by step 1 into output_dir).
        effective_data_dir = data_dir or output_dir
        if effective_data_dir:
            cmd.extend(["--data-dir", effective_data_dir])
        if output_dir:
            cmd.extend(["--output-dir", output_dir])
        if number == 6 and figures_dir:
            cmd.extend(["--figures-dir", figures_dir])

    print(f"\n{'=' * 60}")
    print(f"  Step {number}: {label}")
    print(f"  Script: {os.path.basename(script)}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  ERROR: Step {number} ({label}) failed with return code "
              f"{result.returncode}")
        sys.exit(result.returncode)

    print(f"\n  Step {number} completed in {_format_duration(elapsed)}")
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the full quarterly sleep-pain coupling pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python python/main.py                              "
            "# Full pipeline, real data, default paths\n"
            "  python python/main.py --synthetic                  "
            "# Full pipeline, synthetic data\n"
            "  python python/main.py --step 3                     "
            "# Start from step 3\n"
            "  python python/main.py --step 6                     "
            "# Regenerate figures only\n"
            "  python python/main.py --output-dir sandbox/run1    "
            "# Sandbox run with all outputs under sandbox/run1/\n"
            "  python python/main.py --synthetic "
            "--output-dir sandbox/synth    # Sandbox synthetic run\n"
        ),
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Run with synthetic data (no restricted data access needed).",
    )
    parser.add_argument(
        "--step", type=int, default=1, choices=range(1, len(STEPS) + 1),
        help="Start from this step number (default: 1).",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override input raw data directory (default: data/). "
             "Step 1 reads raw quarterly items and participants_wideformat "
             "from here. If --output-dir is set but --data-dir is not, "
             "the default data/ is still used for raw inputs.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory where all results, posterior draws, "
             "and figures are written. Useful for sandbox runs that must "
             "not touch the default results/ and figures/ folders. "
             "Step 1 writes processed_data_contrast.csv here; steps 2-5 "
             "read from here and write their own outputs here; step 6 "
             "writes figures to {output-dir}/figures/.",
    )
    parser.add_argument(
        "--figures-dir", default=None,
        help="Override figure output directory (default: {output-dir}/figures "
             "if --output-dir is set, else figures/).",
    )
    parser.add_argument(
        "--no-interpolate", action="store_true",
        help="Disable linear interpolation of single interior gaps in step 1. "
             "By default (to match the published sample of 229/1818), step 1 "
             "runs with --interpolate, which fills single missing quarters "
             "between observed ones and allows 13 additional subjects with "
             "short but interpolable segments to contribute to the model.",
    )
    args = parser.parse_args()

    python_exe = sys.executable
    mode_label = "SYNTHETIC" if args.synthetic else "REAL"

    # Resolve absolute paths for clarity in the log
    def _resolve_abs(path):
        if path is None:
            return None
        return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)

    abs_output_dir = _resolve_abs(args.output_dir)
    abs_data_dir = _resolve_abs(args.data_dir)
    abs_figures_dir = _resolve_abs(args.figures_dir)

    if abs_output_dir:
        os.makedirs(abs_output_dir, exist_ok=True)

    print()
    print("=" * 60)
    print("  Quarterly Sleep-Pain Coupling Pipeline")
    print(f"  Mode: {mode_label} DATA")
    print(f"  Starting from step: {args.step}")
    print(f"  Python: {python_exe}")
    print(f"  Repo:   {REPO_ROOT}")
    if abs_data_dir:
        print(f"  Data:   {abs_data_dir}")
    if abs_output_dir:
        print(f"  Output: {abs_output_dir}")
    if abs_figures_dir:
        print(f"  Figs:   {abs_figures_dir}")
    print("=" * 60)

    step_times = {}
    t_total_start = time.time()

    # By default we interpolate single gaps to match the published sample
    # of 229 subjects / 1818 observations. Users who want the stricter
    # no-interpolation sample (216/1571) can pass --no-interpolate.
    interpolate = not args.no_interpolate and not args.synthetic

    for step in STEPS:
        if step["number"] < args.step:
            continue
        elapsed = _run_step(
            step, args.synthetic, python_exe,
            data_dir=abs_data_dir,
            output_dir=abs_output_dir,
            figures_dir=abs_figures_dir,
            interpolate=interpolate,
        )
        step_times[step["number"]] = (step["label"], elapsed)

    t_total = time.time() - t_total_start

    # Summary
    print()
    print("=" * 60)
    print("  Pipeline Complete")
    print("=" * 60)
    for num, (label, elapsed) in sorted(step_times.items()):
        status = _format_duration(elapsed)
        print(f"  Step {num}: {label:30s} {status:>10s}")
    print(f"  {'':30s}  {'----------':>10s}")
    print(f"  {'Total':30s}  {_format_duration(t_total):>10s}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
