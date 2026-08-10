#!/usr/bin/env python3
"""Regenerate all manuscript results — a thin shim over ``run_pipeline.py``.

This script used to hold its own hand-written list of steps to call. That list went
stale: it named 8 of the 24 steps, so most registries, tables and figures were left
untouched while it printed "All results regenerated", and it crashed on its first step
because it put the package directory on ``sys.path`` but not ``lib/``.

Both failure modes came from having a second, hand-maintained ordering. There is now one
ordering -- ``run_pipeline.discover()``, which reads it off the filenames -- and this
name is kept only so existing habits and older READMEs keep working.

    python generate_all_results.py            # replot from saved derivatives
    python generate_all_results.py --refit    # recompute everything (hours)

Prefer ``python run_pipeline.py``, which additionally offers --from/--to/--only/--list.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lib"))

import run_pipeline  # noqa: E402


if __name__ == "__main__":
    sys.exit(run_pipeline.main())
