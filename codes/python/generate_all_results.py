#!/usr/bin/env python3
"""
Regenerate all manuscript results from saved derivatives — no model refitting.

Calls each step's run function in default (replot) mode, which loads
saved posterior draws and intermediate outputs to regenerate all figures,
tables, and text numbers. No MCMC is re-run.

To re-run a specific step's computation from scratch, run that step
individually with the --refit flag:

    python step4_fit_coupling_model.py --refit

Usage:
    python generate_all_results.py
    python generate_all_results.py --quiet

Author: Pedro Valdes-Hernandez (with Claude Sonnet 4.6)
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate all manuscript results from saved derivatives."
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    if verbose:
        print("=" * 70)
        print("GENERATE ALL RESULTS (replot mode — no model refitting)")
        print("=" * 70)
        print()

    # Step 2: Factor validation (Figures S1, S2 + text numbers)
    if verbose:
        print("\n--- Step 2: Factor validation ---")
    import step2_contrast_validation as s2
    s2.run_step2(verbose=verbose)

    # Step 3: VARX data prep (Figure 1 + Table 3 + text numbers)
    if verbose:
        print("\n--- Step 3: Data preparation ---")
    import step3_prepare_varx_data as s3
    s3.run_step3(verbose=verbose)

    # Step 4: Coupling model (Figures 2-3 + Table 4 + text numbers)
    if verbose:
        print("\n--- Step 4: Coupling model figures ---")
    import step4_fit_coupling_model as s4
    s4.run_step4(verbose=verbose, refit=False)

    # Step 5: Contrast moderation JN (Figure 4 + Figure S3 + text numbers)
    if verbose:
        print("\n--- Step 5: Contrast moderation JN ---")
    import step5_contrast_moderation as s5
    s5.run_step5(verbose=verbose, refit=False)

    # Step 7: SP ROI extraction + Figure S4
    if verbose:
        print("\n--- Step 7: SP ROI maps (Figure S4) ---")
    import step7_extract_sp_rois as s7
    s7.generate_figure_s4(verbose=verbose)

    # Step 9: SP moderation JN (Figures 5, 6, S5 + text numbers)
    if verbose:
        print("\n--- Step 9: SP moderation JN ---")
    import step9_sp_moderation_jn as s9
    s9.run_step9(verbose=verbose, refit=False)

    # Step 10: PS ROI extraction + Figure S6
    if verbose:
        print("\n--- Step 10: PS arousal ROI maps (Figure S6) ---")
    import step10_extract_ps_rois as s10
    s10.generate_figure_s6(verbose=verbose)

    # Step 12: PS moderation JN (Figures S7, S8 + text numbers)
    if verbose:
        print("\n--- Step 12: PS moderation JN ---")
    import step12_ps_moderation_jn as s12
    s12.run_step12(verbose=verbose, refit=False)

    if verbose:
        print("\n" + "=" * 70)
        print("All results regenerated from saved derivatives.")
        print("Figures, tables, and text numbers saved to results/.")
        print("=" * 70)


if __name__ == "__main__":
    main()
