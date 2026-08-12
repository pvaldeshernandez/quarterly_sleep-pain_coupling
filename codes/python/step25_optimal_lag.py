"""
Step 25 — The optimal measurement interval implied by the fitted coupling matrix.

Section S15 of the supplement derives, from Dormann & Griffin (2015), the lag at which
a cross-lagged effect is maximally observable, and reports what fraction of that peak a
quarterly interval recovers. Every number in that derivation was arithmetic done by hand
on Table 4 and typed into the document: it had no pipeline source, so the checker could
not verify it and a refit could not update it.

This step is that arithmetic, done once, from `step08_table4_coupling.csv`.

It fits nothing. It reads the primary fit's four transition-matrix coefficients and
publishes every quantity Section S15, Section S15 and manuscript paragraph 174 quote.

WHAT IS DERIVED
---------------
The bivariate transition matrix of the VARX(1) is

    A = [[phi_p,   lambda_sp],
         [lambda_ps, phi_s  ]]

with eigenvalues

    lambda_{1,2} = ((phi_p + phi_s) +/- sqrt((phi_p - phi_s)^2 + 4*lambda_sp*lambda_ps)) / 2

Eq. S2, the bidirectional optimum, needs BOTH eigenvalues positive — it takes their
logarithms. Eq. S3, the unidirectional surrogate, needs only the two autoregressions
positive:

    omega_opt = -ln( ln(phi_p) / ln(phi_s) ) / ( ln(phi_p) - ln(phi_s) )

and the amplification at lag omega, normalized so that f(1) = 1, is

    f(omega) = (phi_p^omega - phi_s^omega) / (phi_p - phi_s)

Whether either equation APPLIES is itself a published claim, so it is computed and
published rather than assumed: `eq_s2_applicable` and `eq_s3_computable` are numbers in
the registry, not assertions in prose. Under the committed interpolation the minor
eigenvalue is negative and phi_s is positive, so S14 uses the unidirectional surrogate —
exactly what the document says. If a future refit made phi_s negative, this step would
publish `eq_s3_computable = 0` and omega_opt as NaN instead of silently producing the
logarithm of a negative number.

Outputs
-------
    results/step25_optimal_lag/step25_optimal_lag.csv    every derived quantity, tidy
    results/step25_optimal_lag/numbers.json              the registry keys

Run:  python step25_optimal_lag.py [--refit]

`--refit` is accepted for interface consistency and does nothing: there is no fit here
and the arithmetic is instantaneous, so the step always recomputes.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)
ROOT = os.path.dirname(os.path.dirname(HERE))

RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step25_optimal_lag")

IN_TABLE4 = os.path.join(RESULTS_DIR, "step08_coupling_model",
                         "step08_table4_coupling.csv")
OUT_CSV = os.path.join(STEP_RESULTS_DIR, "step25_optimal_lag.csv")

#: days per quarter, for reporting the optimal lag in the unit the Discussion uses
DAYS_PER_QUARTER = 91

#: script-internal name -> manuscript symbol, for the four transition-matrix entries
COEFFS = {"a1": "phi_p", "b2": "phi_s", "a2": "lambda_sp", "b1": "lambda_ps"}


def read_transition_matrix(path=IN_TABLE4):
    """The four coefficients of A, from the primary fit's published Table 4.

    Read from the table the manuscript prints rather than from the posterior, so this
    step derives from exactly the numbers a reader can see. Raises if a coefficient is
    missing: a partially-read transition matrix would give a plausible wrong answer.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist. Section S15 derives from the primary fit; "
            f"run step 07 first.")
    t = pd.read_csv(path, float_precision="round_trip").set_index("Parameter")
    missing = [k for k in COEFFS if k not in t.index]
    if missing:
        raise KeyError(f"Table 4 has no row(s) {missing}; cannot build A")
    return {sym: float(t.loc[key, "Estimate"]) for key, sym in COEFFS.items()}


def eigenvalues(phi_p, phi_s, lambda_sp, lambda_ps):
    """(lambda_1, lambda_2, discriminant) of the bivariate transition matrix."""
    disc = (phi_p - phi_s) ** 2 + 4 * lambda_sp * lambda_ps
    if disc < 0:
        # Complex eigenvalues: a genuinely oscillatory system in a different sense
        # than S14's "oscillatory regime", which is a NEGATIVE real minor eigenvalue.
        return np.nan, np.nan, float(disc)
    root = np.sqrt(disc)
    return ((phi_p + phi_s) + root) / 2, ((phi_p + phi_s) - root) / 2, float(disc)


def amplification(omega, phi_p, phi_s):
    """f(omega), the cross-lagged amplification at lag omega, normalized so f(1) = 1."""
    return (phi_p ** omega - phi_s ** omega) / (phi_p - phi_s)


def unidirectional_optimum(phi_p, phi_s):
    """Eq. S3's omega_opt, in quarters; NaN when the logarithms do not exist.

    Requires both autoregressions strictly positive AND different from each other:
    ln(phi) is undefined at or below zero, and the denominator vanishes when they are
    equal. Both are checked, because both produce a number rather than an error if
    left alone.
    """
    if not (phi_p > 0 and phi_s > 0):
        return float("nan")
    lp, ls = np.log(phi_p), np.log(phi_s)
    if np.isclose(lp, ls):
        return float("nan")
    return float(-np.log(lp / ls) / (lp - ls))


def derive(coeffs, verbose=True):
    """Every quantity Section S15 reports. Returns (tidy DataFrame, numbers dict)."""
    phi_p, phi_s = coeffs["phi_p"], coeffs["phi_s"]
    lambda_sp, lambda_ps = coeffs["lambda_sp"], coeffs["lambda_ps"]

    l1, l2, disc = eigenvalues(phi_p, phi_s, lambda_sp, lambda_ps)
    cross_product = lambda_sp * lambda_ps
    auto_product = phi_p * phi_s

    eq_s2_applicable = bool(np.isfinite(l1) and np.isfinite(l2) and l1 > 0 and l2 > 0)
    omega_opt = unidirectional_optimum(phi_p, phi_s)
    eq_s3_computable = bool(np.isfinite(omega_opt))

    f_quarterly = amplification(1.0, phi_p, phi_s)
    f_peak = amplification(omega_opt, phi_p, phi_s) if eq_s3_computable else float("nan")
    pct_of_peak = 100 * f_quarterly / f_peak if eq_s3_computable else float("nan")

    nums = {
        "phi_p": phi_p, "phi_s": phi_s,
        "lambda_sp": lambda_sp, "lambda_ps": lambda_ps,
        "cross_lagged_product": float(cross_product),
        "autoregressive_product": float(auto_product),
        "discriminant": disc,
        "eigenvalue_1": float(l1), "eigenvalue_2": float(l2),
        # Published as numbers, not asserted in prose: whether each equation applies
        # is itself a claim the supplement makes.
        "eq_s2_applicable": int(eq_s2_applicable),
        "eq_s3_computable": int(eq_s3_computable),
        # The two intermediates Eq. S3 is PRINTED with in the supplement. They are
        # published because the document shows them: a number a reader can see must
        # have a source, or the checker cannot tell a typo from a result.
        "ln_ratio": float(np.log(phi_p) / np.log(phi_s)) if eq_s3_computable
                    else float("nan"),
        "ln_difference": float(np.log(phi_p) - np.log(phi_s)) if eq_s3_computable
                         else float("nan"),
        "omega_opt_quarters": omega_opt,
        "omega_opt_days": omega_opt * DAYS_PER_QUARTER if eq_s3_computable
                          else float("nan"),
        "days_per_quarter": DAYS_PER_QUARTER,
        "amplification_at_one_quarter": float(f_quarterly),
        "amplification_at_optimum": float(f_peak),
        "pct_of_peak_captured_quarterly": float(pct_of_peak),
        # Section S15 states that the dominant eigenvalue "exceeds phi_p by 23%". That
        # percentage was computed by hand and had no name, so nothing could tell whether
        # it had followed the refit. Both the ratio and the excess are published because
        # the document phrases it as an excess.
        "eigenvalue_1_over_phi_p": float(l1 / phi_p) if phi_p else float("nan"),
        "eigenvalue_1_excess_over_phi_p_pct": float((l1 / phi_p - 1) * 100)
                                              if phi_p else float("nan"),
    }

    rows = [{"quantity": k, "value": v} for k, v in nums.items()]
    table = pd.DataFrame(rows)

    if verbose:
        print(f"  transition matrix   phi_p={phi_p:+.4f}  phi_s={phi_s:+.4f}  "
              f"lambda_sp={lambda_sp:+.4f}  lambda_ps={lambda_ps:+.4f}")
        print(f"  oscillatory regime  lambda_sp*lambda_ps = {cross_product:.5f} "
              f"{'>' if cross_product > auto_product else '<='} "
              f"phi_p*phi_s = {auto_product:.6f}")
        print(f"  eigenvalues         lambda_1 = {l1:+.4f}   lambda_2 = {l2:+.4f}")
        print(f"  Eq. S2 (needs both eigenvalues > 0): "
              f"{'APPLICABLE' if eq_s2_applicable else 'inapplicable'}")
        if eq_s3_computable:
            print(f"  Eq. S3 omega_opt    {omega_opt:.4f} quarters = "
                  f"{omega_opt * DAYS_PER_QUARTER:.1f} days")
            print(f"  amplification       f(1) = {f_quarterly:.2f}   "
                  f"f({omega_opt:.3f}) = {f_peak:.2f}")
            print(f"  quarterly measurement captures {pct_of_peak:.0f}% of the peak")
        else:
            print("  Eq. S3 omega_opt    UNDEFINED — an autoregression is not "
                  "positive, so ln(phi) does not exist. Section S15 has no "
                  "optimal-lag estimate under this fit.")
    return table, nums


def run_step25(verbose=True, refit=False):
    """Derive Section S15's optimal-lag quantities from the primary fit."""
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)
    if verbose:
        print("=" * 70)
        print("STEP 25 — Optimal measurement interval (Section S15)")
        print("=" * 70)

    coeffs = read_transition_matrix()
    table, nums = derive(coeffs, verbose=verbose)

    table.to_csv(OUT_CSV, index=False, float_format="%.17g")

    from registry import write_numbers
    path = write_numbers(STEP_RESULTS_DIR, nums, prefix="step25")

    if verbose:
        print(f"  Saved: {os.path.relpath(OUT_CSV, ROOT)}")
        print(f"  Saved numbers ({len(nums)} keys): {os.path.relpath(path, ROOT)}")
        print("=" * 70)
        print("STEP 25 COMPLETE")
        print("=" * 70)
    return table, nums


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refit", action="store_true",
                    help="accepted for interface consistency; this step never fits")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run_step25(verbose=not args.quiet, refit=args.refit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
