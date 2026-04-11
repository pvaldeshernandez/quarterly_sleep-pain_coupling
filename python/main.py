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


def _run_step(step, synthetic, python_exe):
    """Run one pipeline step as a subprocess, returning elapsed time."""
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

    print(f"\n{'=' * 60}")
    print(f"  Step {number}: {label}")
    print(f"  Script: {os.path.basename(script)}")
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
            "  python python/main.py                  "
            "# Full pipeline, real data\n"
            "  python python/main.py --synthetic      "
            "# Full pipeline, synthetic data\n"
            "  python python/main.py --step 3         "
            "# Start from step 3\n"
            "  python python/main.py --step 6         "
            "# Regenerate figures only\n"
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
    args = parser.parse_args()

    python_exe = sys.executable
    mode_label = "SYNTHETIC" if args.synthetic else "REAL"

    print()
    print("=" * 60)
    print("  Quarterly Sleep-Pain Coupling Pipeline")
    print(f"  Mode: {mode_label} DATA")
    print(f"  Starting from step: {args.step}")
    print(f"  Python: {python_exe}")
    print(f"  Repo:   {REPO_ROOT}")
    print("=" * 60)

    step_times = {}
    t_total_start = time.time()

    for step in STEPS:
        if step["number"] < args.step:
            continue
        elapsed = _run_step(step, args.synthetic, python_exe)
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
