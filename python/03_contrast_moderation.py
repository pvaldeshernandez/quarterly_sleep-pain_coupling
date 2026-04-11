#!/usr/bin/env python3
"""
03 — Contrast Moderation Analysis and Johnson-Neyman Boundaries
================================================================

This script extracts the pain localization contrast moderation results from
the VARX(1) model fit (script 02) and performs Johnson-Neyman (JN) analysis
to identify the ranges of the contrast variable where the conditional
coupling effects are credibly different from zero.

What the Pain Localization Contrast Captures
---------------------------------------------
The contrast factor (Factor 2 from the 2-factor PAF model) separates:

  Positive values: Knee-dominant pain pattern (high knee, low body pain)
  Negative values: Body-dominant pain pattern (high body, low knee pain)
  Zero:            Balanced pattern (proportional knee and body pain)

This is orthogonal to overall pain severity (Factor 1), so the contrast
captures WHERE pain is concentrated, independently of HOW MUCH pain the
person reports. Within-person fluctuations in the contrast (K^w) capture
time-varying shifts in pain localization pattern.

Contrast Moderation Parameters
-------------------------------
Two parameters capture the role of pain localization in coupling:

  1. delta_s (b3): Direct effect of contrast on sleep.
     Does the pain localization pattern at t-1 independently predict
     sleep quality at t, beyond its effect through the coupling pathway?

  2. omega_ps (b4): Interaction term (contrast x pain -> sleep).
     Does pain localization moderate the pain-to-sleep coupling?
     When omega_ps < 0: body-dominant pain (K^w < 0) STRENGTHENS the
     negative pain-to-sleep coupling (pain disrupts sleep more when
     it is diffuse/widespread rather than localized to the knee).

The parallel parameters for the sleep-to-pain direction are:
  delta_p (a3): Direct contrast effect on pain
  omega_sp (a4): Interaction (contrast x sleep -> pain)

Johnson-Neyman Analysis
------------------------
The Johnson-Neyman technique (Johnson & Neyman, 1936) identifies "regions
of significance" — the ranges of a continuous moderator where the conditional
effect is statistically significant (or, in the Bayesian version, where the
95% credible interval excludes zero).

For the pain-to-sleep direction, the conditional coupling is:
  lambda_ps(K) = b1 + b4 * K^w

where b1 is the coupling at the sample mean of the contrast (K^w = 0) and
b4 is the slope of the coupling as a function of the contrast.

The Bayesian JN approach (implemented in coupling_model.compute_jn_curve)
evaluates the full posterior of lambda_ps(K) at 500 grid points across the
observed range of K. At each point, significance is determined by whether
the 95% CrI excludes zero. The JN boundary is found by linear interpolation
between adjacent grid points where significance status changes.

This is simpler and more flexible than the frequentist JN formula (which
requires solving a quadratic equation and assumes a known t-distribution),
and naturally handles non-normal posteriors.

Output Files
------------
  results/contrast_moderation_results.csv — Parameter estimates for delta
                                             and omega in both directions
  results/contrast_jn_boundary.txt        — JN boundary values and simple
                                             slopes at representative points

References
----------
Johnson, P. O., & Neyman, J. (1936). Tests of certain linear hypotheses and
  their application to some educational problems. Statistical Research
  Memoirs, 1, 57-93.

Bauer, D. J., & Curran, P. J. (2005). Probing interactions in fixed and
  multilevel regression: Inferential and graphical techniques. Multivariate
  Behavioral Research, 40, 373-400.

Author: Pedro Valdes-Hernandez
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
LIB_DIR = os.path.join(SCRIPT_DIR, "lib")
sys.path.insert(0, LIB_DIR)

DATA_DIR = os.path.join(REPO_DIR, "data")
RESULTS_DIR = os.path.join(REPO_DIR, "results")


# ===================================================================
# Simple Slope Computation
# ===================================================================

def compute_simple_slopes(intercept_draws, slope_draws, x_positions):
    """Compute simple slopes (conditional effects) from posterior draws.

    At a given moderator value x, the conditional coupling is:
      lambda(x) = intercept + slope * x

    The simple slope is the posterior distribution of lambda(x), from which
    we extract the mean, 95% CrI, and the probability that the effect is
    negative (directional evidence).

    Parameters
    ----------
    intercept_draws : ndarray, shape (n_draws,)
        Posterior draws of the coupling intercept (e.g., b1 or a2).
    slope_draws : ndarray, shape (n_draws,)
        Posterior draws of the interaction slope (e.g., b4 or a4).
    x_positions : list of (label, x_value)
        Named moderator values to evaluate (e.g., [("-2 SD", -0.5)]).

    Returns
    -------
    slopes : dict
        {label: {"beta": float, "ci_lo": float, "ci_hi": float,
                 "prob_neg": float}}
    """
    slopes = {}
    for label, x_val in x_positions:
        # Full posterior of the conditional effect at this x-value
        draws = intercept_draws + slope_draws * x_val
        beta = draws.mean()
        cl = np.percentile(draws, 2.5)
        ch = np.percentile(draws, 97.5)
        p_neg = (draws < 0).mean()
        slopes[label] = {
            "beta": beta,
            "ci_lo": cl,
            "ci_hi": ch,
            "prob_neg": p_neg,
        }
    return slopes


# ===================================================================
# Main Analysis
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract contrast moderation results and run JN analysis."
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data paths"
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="(Accepted for interface consistency with other steps but "
             "unused — step 3 reads only from --output-dir.)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory containing step-02 posterior draws and where "
             "step-03 results will be written. Default: results/ for "
             "real data, results/synthetic/ for synthetic."
    )
    args = parser.parse_args()

    # Resolve output directory (step 3 reads the step-2 output and writes
    # its own results alongside, so the directory must match)
    if args.output_dir:
        results_dir = args.output_dir
    elif args.synthetic:
        results_dir = os.path.join(RESULTS_DIR, "synthetic")
    else:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("CONTRAST MODERATION AND JOHNSON-NEYMAN ANALYSIS")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load saved posterior draws from script 02
    # ------------------------------------------------------------------
    posterior_path = os.path.join(results_dir, "contrast_posterior_draws.npz")
    if not os.path.exists(posterior_path):
        print(
            f"\n  ERROR: Posterior draws not found at {posterior_path}\n"
            f"  Run 02_fit_coupling_model.py first to generate them."
        )
        sys.exit(1)

    print(f"\n  Loading posterior draws from {posterior_path}")
    d = np.load(posterior_path)

    # Extract the draws we need for contrast moderation analysis.
    # These are the full posterior draws (8000 = 4 chains x 2000 draws):
    #   a2: lambda_sp (population SP coupling intercept)
    #   a4: omega_sp  (contrast x sleep interaction in pain equation)
    #   b1: lambda_ps (population PS coupling intercept)
    #   b4: omega_ps  (contrast x pain interaction in sleep equation)
    a2_draws = d["a2_draws"]    # SP coupling intercept
    a4_draws = d["a4_draws"]    # SP contrast moderation slope
    b1_draws = d["b1_draws"]    # PS coupling intercept
    b4_draws = d["b4_draws"]    # PS contrast moderation slope

    # Contrast values from observations (for grid range and rug plot)
    contrast_vals = d["contrast_vals"]

    # SD of the contrast (for defining -2 SD, 0, +2 SD evaluation points)
    c_sd = np.std(contrast_vals)

    print(f"  Posterior draws: {len(a2_draws)} per parameter")
    print(f"  Contrast SD: {c_sd:.4f}")

    # ------------------------------------------------------------------
    # Import JN computation from the shared library module
    # ------------------------------------------------------------------
    from coupling_model import compute_jn_curve

    # ------------------------------------------------------------------
    # Compute JN curves for both coupling directions
    # ------------------------------------------------------------------
    # The conditional coupling at a given contrast value K is:
    #   SP direction: lambda_sp(K) = a2 + a4 * K
    #   PS direction: lambda_ps(K) = b1 + b4 * K
    # We use clip_pct=(0, 100) to span the full observed contrast range.

    print("\n  Computing JN curves...")

    # Sleep-to-Pain direction (SP): lambda_sp as a function of contrast
    jn_sp = compute_jn_curve(
        a2_draws, a4_draws, contrast_vals, clip_pct=(0, 100)
    )

    # Pain-to-Sleep direction (PS): lambda_ps as a function of contrast
    jn_ps = compute_jn_curve(
        b1_draws, b4_draws, contrast_vals, clip_pct=(0, 100)
    )

    print(f"  SP JN boundaries: {jn_sp['jn_boundaries']}")
    print(f"  PS JN boundaries: {jn_ps['jn_boundaries']}")

    # ------------------------------------------------------------------
    # Compute simple slopes at representative contrast values
    # ------------------------------------------------------------------
    # Evaluate the conditional coupling at -2 SD, 0, and +2 SD of the
    # within-person contrast. These correspond to:
    #   -2 SD: strongly body-dominant pain pattern
    #       0: balanced (sample-average) pain pattern
    #   +2 SD: strongly knee-dominant pain pattern
    x_positions = [
        ("-2 SD", -2 * c_sd),
        ("0", 0.0),
        ("+2 SD", 2 * c_sd),
    ]

    slopes_sp = compute_simple_slopes(a2_draws, a4_draws, x_positions)
    slopes_ps = compute_simple_slopes(b1_draws, b4_draws, x_positions)

    print("\n  Simple Slopes (SP direction, sleep -> pain):")
    for label, vals in slopes_sp.items():
        sig = "*" if vals["prob_neg"] > 0.975 or vals["prob_neg"] < 0.025 else ""
        print(
            f"    {label:>5s}: {vals['beta']:.4f} "
            f"[{vals['ci_lo']:.4f}, {vals['ci_hi']:.4f}]{sig}"
        )

    print("\n  Simple Slopes (PS direction, pain -> sleep):")
    for label, vals in slopes_ps.items():
        sig = "*" if vals["prob_neg"] > 0.975 or vals["prob_neg"] < 0.025 else ""
        print(
            f"    {label:>5s}: {vals['beta']:.4f} "
            f"[{vals['ci_lo']:.4f}, {vals['ci_hi']:.4f}]{sig}"
        )

    # ------------------------------------------------------------------
    # Save contrast moderation results as CSV
    # ------------------------------------------------------------------
    results_rows = []

    # SP direction contrast moderation
    results_rows.append({
        "direction": "SP",
        "parameter": "omega_sp (a4)",
        "description": "Contrast moderation of sleep-to-pain coupling",
        "mean": a4_draws.mean(),
        "sd": a4_draws.std(),
        "ci_lo": np.percentile(a4_draws, 2.5),
        "ci_hi": np.percentile(a4_draws, 97.5),
        "prob_neg": (a4_draws < 0).mean(),
        "p_twotail": 2 * min((a4_draws < 0).mean(), (a4_draws > 0).mean()),
    })

    # PS direction contrast moderation (the key finding for manuscript)
    results_rows.append({
        "direction": "PS",
        "parameter": "omega_ps (b4)",
        "description": "Contrast moderation of pain-to-sleep coupling",
        "mean": b4_draws.mean(),
        "sd": b4_draws.std(),
        "ci_lo": np.percentile(b4_draws, 2.5),
        "ci_hi": np.percentile(b4_draws, 97.5),
        "prob_neg": (b4_draws < 0).mean(),
        "p_twotail": 2 * min((b4_draws < 0).mean(), (b4_draws > 0).mean()),
    })

    # Also save the direct contrast effects (delta_p, delta_s)
    # These come from a3 and b3 in the posterior draws.
    # Since a3 and b3 are not saved in the npz, we compute from the
    # coupling_results.csv if available, or note them as N/A.
    coupling_csv = os.path.join(results_dir, "coupling_results.csv")
    if os.path.exists(coupling_csv):
        coupling_df = pd.read_csv(coupling_csv)
        for code_var, direction, description in [
            ("a3", "SP", "Direct contrast effect on pain (delta_p)"),
            ("b3", "PS", "Direct contrast effect on sleep (delta_s)"),
        ]:
            if f"{code_var}_mean" in coupling_df.columns:
                row = coupling_df.iloc[0]
                results_rows.append({
                    "direction": direction,
                    "parameter": f"{code_var}",
                    "description": description,
                    "mean": row[f"{code_var}_mean"],
                    "sd": row[f"{code_var}_sd"],
                    "ci_lo": row[f"{code_var}_ci_lo"],
                    "ci_hi": row[f"{code_var}_ci_hi"],
                    "prob_neg": row[f"{code_var}_prob_neg"],
                    "p_twotail": 2 * min(
                        row[f"{code_var}_prob_neg"],
                        1 - row[f"{code_var}_prob_neg"],
                    ),
                })

    # Simple slopes at representative points
    for direction, slopes_dict, intercept_label in [
        ("SP", slopes_sp, "a2"),
        ("PS", slopes_ps, "b1"),
    ]:
        for level_label, vals in slopes_dict.items():
            results_rows.append({
                "direction": direction,
                "parameter": f"simple_slope_{level_label}",
                "description": (
                    f"Conditional coupling at contrast = {level_label}"
                ),
                "mean": vals["beta"],
                "sd": np.nan,  # not computed for simple slopes
                "ci_lo": vals["ci_lo"],
                "ci_hi": vals["ci_hi"],
                "prob_neg": vals["prob_neg"],
                "p_twotail": 2 * min(vals["prob_neg"], 1 - vals["prob_neg"]),
            })

    results_csv = os.path.join(results_dir, "contrast_moderation_results.csv")
    pd.DataFrame(results_rows).to_csv(results_csv, index=False)
    print(f"\n  Saved: {results_csv}")

    # ------------------------------------------------------------------
    # Save JN boundary information as text
    # ------------------------------------------------------------------
    jn_path = os.path.join(results_dir, "contrast_jn_boundary.txt")
    with open(jn_path, "w") as f:
        f.write("Johnson-Neyman Analysis: Contrast Moderation of Coupling\n")
        f.write("=" * 60 + "\n\n")
        f.write("Reference: Johnson & Neyman (1936); Bauer & Curran (2005)\n\n")
        f.write("Approach: Bayesian grid-based JN. At each of 500 grid points\n")
        f.write("across the observed contrast range, the full posterior of the\n")
        f.write("conditional coupling is computed. Significance is determined\n")
        f.write("by whether the 95% CrI excludes zero.\n\n")

        f.write(f"Contrast SD (within-person): {c_sd:.4f}\n\n")

        # --- SP direction ---
        f.write("Sleep-to-Pain (SP) Direction\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"  Coupling equation: lambda_sp(K) = "
            f"{jn_sp['intercept_mean']:.4f} + "
            f"({jn_sp['slope_mean']:.4f}) * K\n"
        )
        if jn_sp["jn_boundaries"]:
            for i, b in enumerate(jn_sp["jn_boundaries"]):
                f.write(f"  JN boundary {i + 1}: K = {b:.4f}\n")
        else:
            if np.all(jn_sp["sig"]):
                f.write("  No JN boundary: coupling credible across full range\n")
            else:
                f.write(
                    "  No JN boundary: coupling non-credible across full range\n"
                )
        # Report percentage in credible region
        if np.any(jn_sp["sig"]):
            x_grid = jn_sp["x_grid"]
            obs = jn_sp["obs_vals"]
            sig_mask = jn_sp["sig"]
            sig_lo = x_grid[sig_mask][0]
            sig_hi = x_grid[sig_mask][-1]
            pct = ((obs >= sig_lo) & (obs <= sig_hi)).mean() * 100
            f.write(f"  {pct:.1f}% of observations in credible region\n")

        f.write("\n  Simple slopes:\n")
        for label, vals in slopes_sp.items():
            sig = "*" if vals["prob_neg"] > 0.975 or vals["prob_neg"] < 0.025 else ""
            f.write(
                f"    K = {label:>5s}: lambda_sp = {vals['beta']:.4f} "
                f"[{vals['ci_lo']:.4f}, {vals['ci_hi']:.4f}]{sig}\n"
            )

        # --- PS direction ---
        f.write("\n\nPain-to-Sleep (PS) Direction\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"  Coupling equation: lambda_ps(K) = "
            f"{jn_ps['intercept_mean']:.4f} + "
            f"({jn_ps['slope_mean']:.4f}) * K\n"
        )
        if jn_ps["jn_boundaries"]:
            for i, b in enumerate(jn_ps["jn_boundaries"]):
                f.write(f"  JN boundary {i + 1}: K = {b:.4f}\n")
        else:
            if np.all(jn_ps["sig"]):
                f.write("  No JN boundary: coupling credible across full range\n")
            else:
                f.write(
                    "  No JN boundary: coupling non-credible across full range\n"
                )
        if np.any(jn_ps["sig"]):
            x_grid = jn_ps["x_grid"]
            obs = jn_ps["obs_vals"]
            sig_mask = jn_ps["sig"]
            sig_lo = x_grid[sig_mask][0]
            sig_hi = x_grid[sig_mask][-1]
            pct = ((obs >= sig_lo) & (obs <= sig_hi)).mean() * 100
            f.write(f"  {pct:.1f}% of observations in credible region\n")

        f.write("\n  Simple slopes:\n")
        for label, vals in slopes_ps.items():
            sig = "*" if vals["prob_neg"] > 0.975 or vals["prob_neg"] < 0.025 else ""
            f.write(
                f"    K = {label:>5s}: lambda_ps = {vals['beta']:.4f} "
                f"[{vals['ci_lo']:.4f}, {vals['ci_hi']:.4f}]{sig}\n"
            )

        # --- Interpretation guide ---
        f.write("\n\nInterpretation\n")
        f.write("-" * 60 + "\n")
        f.write("  Positive contrast (K > 0): knee-dominant pain\n")
        f.write("  Negative contrast (K < 0): body-dominant (diffuse) pain\n")
        f.write("  * = 95% CrI excludes zero (credible effect)\n")
        f.write("\n")
        f.write("  omega_ps < 0 means: body-dominant pain strengthens the\n")
        f.write("  negative pain-to-sleep coupling (more pain disrupts sleep\n")
        f.write("  more when pain is diffuse rather than knee-localized).\n")

    print(f"  Saved: {jn_path}")

    print("\n" + "=" * 70)
    print("CONTRAST MODERATION ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
