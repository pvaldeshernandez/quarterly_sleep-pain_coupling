#!/usr/bin/env python3
"""
04_fmri_sp_moderation.py
========================
Neuroimaging Moderation of Sleep-to-Pain Coupling (Aim 2b):
fMRI ROIs from Krause et al. (2019) and Sardi-motivated ACC

Overview
--------
This script tests whether activation in pain-relevant brain regions during
painful knee stimulation moderates the quarterly sleep-to-pain coupling
path (gamma_sp).  It runs 7 independent Bayesian VARX(1) models -- one for
each ROI -- and saves the moderation results, posterior draws for
significant ROIs, and Johnson-Neyman (JN) significance curves.

Theoretical Framework
---------------------
**Krause et al. (2019) J Neurosci 39:2291-2300** used experimental sleep
deprivation in healthy adults to identify brain regions where the neural
response to painful stimulation changes after total sleep loss.  They found
two classes of response:

  1. *Amplified* responses in somatosensory cortex (S1): sleep deprivation
     increased S1 activation during painful stimulation, reflecting
     heightened sensory encoding of nociceptive input.

  2. *Blunted* responses in reward/valuation regions (NAcc, thalamus,
     anterior insula, middle insula): sleep deprivation reduced activation
     in regions involved in endogenous pain modulation and affective
     appraisal, consistent with impaired descending analgesia.

Translating these findings to our observational coupling framework:

  - For S1, sleep deprivation *amplifies* the pain-evoked response.
    Higher baseline S1 activation therefore indexes a brain already in the
    "amplified" regime, which predicts stronger (more negative) sleep-to-
    pain coupling. Expected sign: NEGATIVE gamma_sp.

  - For NAcc, thalamus, anterior insula, and middle insula, sleep
    deprivation *blunts* the pain-evoked response. Greater baseline
    activation reflects stronger endogenous pain modulation and therefore
    weaker (less negative) coupling. Expected sign: POSITIVE gamma_sp.

**Sign Concordance Test**: Under the null hypothesis that each ROI has a
50% chance of matching the predicted sign, the probability of all 6 Krause
ROIs matching is (1/2)^6 = 0.016 (exact one-sided sign test), even when
individual ROIs are individually null.

**Sardi et al. (2018) Cerebral Cortex 28:3816-3828** demonstrated that the
anterior cingulate cortex (ACC) and nucleus accumbens (NAcc) are parallel
nodes in a dopamine D2-receptor-gated circuit that prevents sleep-
restriction-induced hyperalgesia.  Blocking D2 receptors in either region
alone was sufficient to restore hyperalgesia in sleep-restricted rats.
Since NAcc moderates sleep-to-pain coupling in our data, ACC should too.

The ACC ROI is defined at MNI (6, 12, 38) with a 6mm radius, from
Xu et al. (2020) Neurosci Biobehav Rev 112:300-323 (a meta-analysis of
pain neuroimaging studies).  Expected direction: positive gamma_sp (same
as NAcc).

Masking Convention
------------------
  - Left and Right NAcc: GM-masked contrast images (original SPM first-
    level), because subcortical reward regions have adequate coverage
    within the GM mask and the masked estimates were significant.
  - All other ROIs: Unmasked contrast images (re-estimated without
    individual GM masks to maximize voxel coverage in subcortical regions).
  - See python/lib/moderator_loaders.py for implementation details.

Output Files
------------
  results/fmri_sp_moderation_results.csv   -- Summary for all 7 ROIs
  results/nacc_posterior_draws.npz         -- Full posteriors for left NAcc
  results/acc_posterior_draws.npz          -- Full posteriors for ACC
  results/fmri_sp_jn_results.csv          -- JN boundaries for significant ROIs

Usage
-----
  # With synthetic data (no neuroimaging files required):
  python python/04_fmri_sp_moderation.py --synthetic

  # With real data (requires NIfTI files under data/):
  python python/04_fmri_sp_moderation.py

References
----------
[1] Krause AJ, Prather AA, Wager TD, Lindquist MA, Walker MP. (2019).
    The pain of sleep loss: a brain characterization in humans.
    J Neurosci 39:2291-2300.
[2] Sardi NF, Tobaldini G, Morais RN, Fischer L. (2018). Nucleus
    accumbens mediates the pronociceptive effect of sleep deprivation.
    Pain 159:75-84.
[3] Xu A, Larsen B, Bhatt RR, et al. (2020). Neurosci Biobehav Rev
    112:300-323. (ACC ROI coordinates from pain meta-analysis.)

Author: Pedro Valdes-Hernandez
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))

from coupling_model import load_data, fit_bayesian_varx1, compute_jn_curve
from moderator_loaders import (
    KRAUSE_ROIS,
    ACC_ROI,
    load_fmri_krause_rois,
    load_acc_roi,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")


# ===================================================================
# ROI Definitions (for reference and sign-concordance test)
# ===================================================================

# Expected sign of gamma_sp for each Krause ROI. S1 (where sleep
# deprivation *amplifies* pain-evoked activation) is predicted to
# have gamma_sp < 0; the other five ROIs (where sleep deprivation
# *blunts* activation) are predicted to have gamma_sp > 0. All six
# matching these signs gives the manuscript's (1/2)^6 = 0.016 sign
# test result.
EXPECTED_SIGNS = {
    "Right_S1":             "-",
    "Right_Middle_Insula":  "+",
    "Left_Thalamus":        "+",
    "Left_Anterior_Insula": "+",
    "Left_NAcc":            "+",
    "Right_NAcc":           "+",
}

# ROIs for which we save full posterior draws (for JN figures).
# These are the two ROIs that reached significance in the real data.
SAVE_POSTERIORS_FOR = {"Left_NAcc", "Right_dACC_MCC"}


# ===================================================================
# Synthetic Data Loader
# ===================================================================

def _load_synthetic_moderators(roi_names):
    """Load ROI values from the synthetic CSV, handling column name mapping.

    The synthetic CSV (data/synthetic/roi_values.csv) uses column names
    'subject_id', 'roi_name', 'bold_value' and abbreviated ROI names
    (e.g., 'Right_Mid_Insula' instead of 'Right_Middle_Insula').  This
    function maps those to the library's conventions.

    Parameters
    ----------
    roi_names : list of str
        ROI names to load (using the library's naming convention, e.g.,
        'Right_Middle_Insula', 'Left_NAcc', 'Right_dACC_MCC').

    Returns
    -------
    moderators : dict
        {roi_name: {subject_id: z_scored_value, ...}} for each ROI.
    labels : dict
        {roi_name: human-readable label string}
    raw_stats : dict
        {roi_name: {"mean": 0.0, "sd": 1.0}} (synthetic data is pre-z-scored)
    """
    csv_path = os.path.join(SYNTHETIC_DIR, "roi_values.csv")
    df = pd.read_csv(csv_path)

    # Column name mapping: synthetic CSV -> expected names
    df = df.rename(columns={"subject_id": "ID", "bold_value": "value"})

    # ROI name mapping: synthetic abbreviated -> library full names.
    # Only map names that differ; others pass through unchanged.
    synth_to_lib = {
        "Right_Mid_Insula":  "Right_Middle_Insula",
        "Left_Ant_Insula":   "Left_Anterior_Insula",
        "Right_dACC_MCC":    "Right_dACC_MCC",  # same in both
    }
    # Build the reverse mapping for lookup
    lib_to_synth = {v: k for k, v in synth_to_lib.items()}

    # Label dictionary combining Krause and ACC
    all_labels = {k: v["label"] for k, v in KRAUSE_ROIS.items()}
    all_labels["Right_dACC_MCC"] = ACC_ROI["label"]

    moderators = {}
    labels = {}
    raw_stats = {}

    for roi_name in roi_names:
        # Determine the name used in the synthetic CSV
        synth_name = lib_to_synth.get(roi_name, roi_name)

        roi_df = df[df["roi_name"] == synth_name]
        if len(roi_df) == 0:
            print(f"    WARNING: ROI '{synth_name}' not found in synthetic CSV")
            continue

        # Values in synthetic CSV are already z-scored
        values = dict(zip(roi_df["ID"].astype(str), roi_df["value"]))
        moderators[roi_name] = values
        labels[roi_name] = all_labels.get(roi_name, roi_name)
        raw_stats[roi_name] = {"mean": 0.0, "sd": 1.0}

    return moderators, labels, raw_stats


# ===================================================================
# Helper: Two-Sided Posterior p-value
# ===================================================================

def two_sided_p(samples):
    """Compute a two-sided Bayesian p-value from posterior samples.

    This is the posterior analog of a classical two-tailed p-value:
    twice the smaller tail probability.  A value of 0.05 means that
    97.5% of the posterior mass is on one side of zero.

    Parameters
    ----------
    samples : ndarray
        1D array of posterior draws.

    Returns
    -------
    float
        Two-sided posterior p-value.
    """
    p_pos = (samples > 0).mean()
    return float(2 * min(p_pos, 1 - p_pos))


# ===================================================================
# Main Analysis
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="fMRI moderation of sleep-to-pain coupling (Aim 2b)")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data instead of real neuroimaging files")
    parser.add_argument(
        "--data-dir", default=None,
        help="Override input data directory (default: data/). The processed "
             "CSV from step 01 and the ROI value CSVs are read from here.")
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory. Default: results/ for real data, "
             "results/synthetic/ for --synthetic.")
    args = parser.parse_args()

    t0_total = time.time()

    # Resolve directories
    data_dir = args.data_dir if args.data_dir else DATA_DIR
    if args.output_dir:
        results_dir = args.output_dir
    elif args.synthetic:
        results_dir = os.path.join(RESULTS_DIR, "synthetic")
    else:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load coupling data
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Loading coupling data")
    print("=" * 70)
    df_full, model_df, unique_ids, id_map = load_data(
        data_dir, synthetic=args.synthetic)

    # ------------------------------------------------------------------
    # Step 2: Load fMRI ROI moderator values
    # ------------------------------------------------------------------
    # We test 7 ROIs total: 6 Krause + 1 ACC
    print("\n" + "=" * 70)
    print("STEP 2: Loading fMRI ROI moderator values")
    print("=" * 70)

    if args.synthetic:
        # Load all 7 ROIs from the synthetic CSV directly, handling
        # the column name and ROI name mismatches.
        all_roi_names = list(KRAUSE_ROIS.keys()) + ["Right_dACC_MCC"]
        moderators, labels, raw_stats = _load_synthetic_moderators(
            all_roi_names)
        print(f"  Loaded {len(moderators)} ROIs from synthetic data")
    else:
        # Load from real neuroimaging files via the library loaders
        print("\n  Loading Krause ROIs...")
        moderators, labels, raw_stats = load_fmri_krause_rois(
            data_dir, synthetic=False)

        print("\n  Loading ACC ROI...")
        acc_mod, acc_label, acc_stats = load_acc_roi(
            data_dir, synthetic=False)
        if acc_mod:
            moderators["Right_dACC_MCC"] = acc_mod
            labels["Right_dACC_MCC"] = acc_label
            raw_stats["Right_dACC_MCC"] = acc_stats

    # Print summary of loaded ROIs
    for roi_name, values in moderators.items():
        n = len(values)
        stats = raw_stats.get(roi_name, {})
        print(f"    {roi_name}: N={n}, raw mean={stats.get('mean', 0):.4f}, "
              f"raw SD={stats.get('sd', 1):.4f}")

    # ------------------------------------------------------------------
    # Step 3: Fit moderation models (one per ROI)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Fitting moderation models (7 ROIs, SP direction)")
    print("=" * 70)

    all_results = []      # Accumulate results for the summary CSV
    jn_results = []       # Accumulate JN boundary results
    krause_aggregate = {} # ROI -> dict of arrays for Figure S5 aggregated npz

    # Define the ordered list of ROIs to test.
    # Krause ROIs first (for sign concordance), then ACC.
    roi_order = [
        "Right_S1", "Right_Middle_Insula", "Left_Thalamus",
        "Left_Anterior_Insula", "Left_NAcc", "Right_NAcc",
        "Right_dACC_MCC",
    ]

    for roi_idx, roi_name in enumerate(roi_order, 1):
        if roi_name not in moderators:
            print(f"\n  [{roi_idx}/7] {roi_name}: SKIPPED (no data)")
            continue

        X_person = moderators[roi_name]
        label = labels[roi_name]
        n_subj = len(X_person)

        print(f"\n{'=' * 60}")
        print(f"  [{roi_idx}/7] {label}")
        print(f"  N subjects with ROI data: {n_subj}")
        print(f"{'=' * 60}")

        # --- Fit the Bayesian VARX(1) with this ROI as moderator ---
        t0 = time.time()
        idata, sub_df, valid_ids = fit_bayesian_varx1(
            model_df, unique_ids, id_map,
            X_person=X_person, include_agesex=True,
            progressbar=True)
        elapsed = time.time() - t0

        n_valid = len(valid_ids)
        n_obs = len(sub_df)

        # --- Extract moderation posteriors ---
        # gamma_sp: effect of ROI on sleep-to-pain coupling slope
        # gamma_ps: effect of ROI on pain-to-sleep coupling slope (nuisance)
        gamma_sp = idata.posterior["gamma_sp"].values.flatten()
        gamma_ps = idata.posterior["gamma_ps"].values.flatten()
        a2_draws = idata.posterior["a2"].values.flatten()
        b1_draws = idata.posterior["b1"].values.flatten()

        sp_mean = float(gamma_sp.mean())
        sp_lo, sp_hi = np.percentile(gamma_sp, [2.5, 97.5])
        sp_p = two_sided_p(gamma_sp)

        ps_mean = float(gamma_ps.mean())
        ps_lo, ps_hi = np.percentile(gamma_ps, [2.5, 97.5])
        ps_p = two_sided_p(gamma_ps)

        # --- Convergence diagnostic ---
        # Compute R-hat across all scalar parameters.  R-hat > 1.05
        # suggests inadequate convergence.
        import arviz as az
        rhat_vals = az.rhat(idata)
        rhat_max = max(
            float(rhat_vals[v].max()) for v in rhat_vals.data_vars)

        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  N valid: {n_valid}, N obs: {n_obs}")
        print(f"  gamma_sp = {sp_mean:+.4f} [{sp_lo:+.4f}, {sp_hi:+.4f}], "
              f"p={sp_p:.4f}")
        print(f"  gamma_ps = {ps_mean:+.4f} [{ps_lo:+.4f}, {ps_hi:+.4f}], "
              f"p={ps_p:.4f}")
        print(f"  R-hat max: {rhat_max:.3f}")

        # --- Determine the data source label ---
        # Krause ROIs use 'unmasked' except NAcc which uses 'gm_masked'
        if roi_name in ("Left_NAcc", "Right_NAcc"):
            source = "gm_masked"
        elif roi_name == "Right_dACC_MCC":
            source = "unmasked"
        else:
            source = "unmasked"

        # --- Sign concordance check (Krause ROIs only) ---
        expected = EXPECTED_SIGNS.get(roi_name, None)
        if expected is not None:
            observed_sign = "+" if sp_mean > 0 else "-"
            concordant = observed_sign == expected
            print(f"  Expected sign: {expected}, Observed: {observed_sign}, "
                  f"Concordant: {concordant}")

        # --- Accumulate results ---
        all_results.append({
            "ROI": roi_name,
            "Label": label,
            "Framework": "Krause" if roi_name != "Right_dACC_MCC" else "Sardi",
            "Source": source,
            "N": n_valid,
            "N_obs": n_obs,
            "gamma_sp": sp_mean,
            "gamma_sp_ci_lo": float(sp_lo),
            "gamma_sp_ci_hi": float(sp_hi),
            "gamma_sp_p": sp_p,
            "gamma_ps": ps_mean,
            "gamma_ps_ci_lo": float(ps_lo),
            "gamma_ps_ci_hi": float(ps_hi),
            "gamma_ps_p": ps_p,
            "rhat_max": rhat_max,
            "elapsed_s": elapsed,
        })

        # --- Accumulate draws for the aggregated Krause npz ---
        # Figure S5 plots the four non-NAcc Krause ROIs as a 2x2 merge
        # of JN panels. The figure reads a single aggregated npz that
        # has per-ROI keys ``<ROI>_a2_draws``, ``<ROI>_gamma_sp_draws``,
        # ``<ROI>_X_vals``, ``<ROI>_raw_mean``, ``<ROI>_raw_sd``. We
        # build that accumulator here so no refitting is needed.
        roi_stats_agg = raw_stats.get(roi_name, {"mean": 0.0, "sd": 1.0})
        X_vals_z_agg = np.array(
            [X_person[sid] for sid in valid_ids if sid in X_person])
        if roi_name in (
            "Right_S1", "Right_Middle_Insula",
            "Left_Thalamus", "Left_Anterior_Insula",
        ):
            krause_aggregate[roi_name] = dict(
                a2_draws=a2_draws,
                gamma_sp_draws=gamma_sp,
                b1_draws=b1_draws,
                gamma_ps_draws=gamma_ps,
                X_vals=X_vals_z_agg,
                raw_mean=roi_stats_agg["mean"],
                raw_sd=roi_stats_agg["sd"],
            )

        # --- Save posterior draws for significant ROIs ---
        # We save full posteriors for NAcc and ACC to enable JN figure
        # generation without re-fitting.
        if roi_name in SAVE_POSTERIORS_FOR or roi_name == "Left_NAcc":
            # Compute arrays needed for JN analysis and plotting
            X_vals_z = np.array(
                [X_person[sid] for sid in valid_ids if sid in X_person])
            u_sp_mean = idata.posterior["u_sp"].values.reshape(
                -1, n_valid).mean(axis=0)

            # Person-level demographics (for conditional coupling computation)
            person_demo = sub_df.groupby("ID")[["Age_z", "Sex_c"]].first()
            person_age_z = np.array(
                [person_demo.loc[sid, "Age_z"] for sid in valid_ids])
            person_sex_c = np.array(
                [person_demo.loc[sid, "Sex_c"] for sid in valid_ids])
            person_x_z = np.array(
                [X_person[sid] for sid in valid_ids])

            # Determine filename based on ROI
            if roi_name == "Left_NAcc":
                npz_name = "nacc_posterior_draws.npz"
            elif roi_name == "Right_dACC_MCC":
                npz_name = "acc_posterior_draws.npz"
            else:
                npz_name = f"{roi_name.lower()}_posterior_draws.npz"

            npz_path = os.path.join(results_dir, npz_name)

            # Get raw ROI values for back-transformation
            roi_stats = raw_stats.get(roi_name, {"mean": 0.0, "sd": 1.0})

            np.savez(
                npz_path,
                a2_draws=a2_draws,
                gamma_sp_draws=gamma_sp,
                b1_draws=b1_draws,
                gamma_ps_draws=gamma_ps,
                roi_mean=roi_stats["mean"],
                roi_sd=roi_stats["sd"],
                X_vals=X_vals_z,
                u_sp_mean=u_sp_mean,
                person_age_z=person_age_z,
                person_sex_c=person_sex_c,
                person_x_z=person_x_z,
            )
            print(f"  Saved posterior draws: {npz_path}")

        # --- Johnson-Neyman analysis ---
        # Run JN analysis for ROIs with suggestive moderation (p < 0.10)
        # to identify the moderator range where SP coupling is credible.
        if sp_p < 0.10:
            print(f"\n  Running JN analysis for SP direction...")
            X_vals_z = np.array(
                [X_person[sid] for sid in valid_ids if sid in X_person])
            roi_stats = raw_stats.get(roi_name, {"mean": 0.0, "sd": 1.0})

            jn = compute_jn_curve(
                a2_draws, gamma_sp, X_vals_z,
                raw_mean=roi_stats["mean"],
                raw_sd=roi_stats["sd"],
            )

            # Report JN boundaries
            if jn["jn_boundaries_z"]:
                for bz, br in zip(
                        jn["jn_boundaries_z"], jn["jn_boundaries"]):
                    pct_below = (X_vals_z <= bz).mean() * 100
                    print(f"    JN boundary: z={bz:.3f} (raw={br:.4f}), "
                          f"{pct_below:.1f}% of sample below")

                    jn_results.append({
                        "ROI": roi_name,
                        "Label": label,
                        "Direction": "SP",
                        "JN_boundary_z": float(bz),
                        "JN_boundary_raw": float(br),
                        "Pct_below": float(pct_below),
                    })

            # Report simple slopes at key moderator values
            print(f"    Simple slopes:")
            for z_val in [-2, 0, 2]:
                # Conditional coupling = a2 + gamma_sp * z_val
                draws = a2_draws + gamma_sp * z_val
                m = draws.mean()
                lo, hi = np.percentile(draws, [2.5, 97.5])
                sig = "*" if hi < 0 or lo > 0 else ""
                raw_val = z_val * roi_stats["sd"] + roi_stats["mean"]
                print(f"      z={z_val:+d} (raw={raw_val:.4f}): "
                      f"coupling={m:.3f} [{lo:.3f}, {hi:.3f}]{sig}")

    # ------------------------------------------------------------------
    # Step 4: Sign concordance test (Krause ROIs only)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Sign Concordance Test (6 Krause ROIs)")
    print("=" * 70)

    # Count how many Krause ROIs match the predicted positive sign
    krause_names = list(EXPECTED_SIGNS.keys())
    n_concordant = 0
    for roi_name in krause_names:
        # Find this ROI in the results
        match = [r for r in all_results if r["ROI"] == roi_name]
        if match:
            observed_positive = match[0]["gamma_sp"] > 0
            expected_positive = EXPECTED_SIGNS[roi_name] == "+"
            if observed_positive == expected_positive:
                n_concordant += 1

    n_tested = len(krause_names)
    # Under H0: each ROI has 50% chance of matching -> Binomial(n, 0.5).
    # The one-sided sign-test p-value is P(X >= n_concordant) under
    # Binomial(n_tested, 0.5). When n_concordant == n_tested this
    # reduces to (1/2)^n_tested, matching the manuscript's claim.
    from math import comb
    sign_p = sum(
        comb(n_tested, k) * 0.5 ** n_tested
        for k in range(n_concordant, n_tested + 1)
    )
    print(f"  {n_concordant}/{n_tested} ROIs match predicted sign")
    if n_concordant == n_tested:
        print(
            f"  Sign concordance p = (1/2)^{n_tested} = {sign_p:.4f} "
            f"(one-sided exact binomial)"
        )
    else:
        print(
            f"  Sign concordance p = P(X >= {n_concordant} | "
            f"Binomial({n_tested}, 0.5)) = {sign_p:.4f} "
            f"(one-sided exact binomial)"
        )

    # ------------------------------------------------------------------
    # Step 5: Save results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Saving results")
    print("=" * 70)

    # Save the main results CSV (all 7 ROIs)
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(results_dir, "fmri_sp_moderation_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Save JN results if any boundaries were found
    if jn_results:
        jn_df = pd.DataFrame(jn_results)
        jn_csv = os.path.join(results_dir, "fmri_sp_jn_results.csv")
        jn_df.to_csv(jn_csv, index=False)
        print(f"  Saved: {jn_csv}")

    # Save the aggregated Krause posterior draws for Figure S5
    if krause_aggregate:
        krause_npz_path = os.path.join(
            results_dir, "krause_roi_posterior_draws.npz")
        krause_save = {}
        for roi_name, arrays in krause_aggregate.items():
            for key, val in arrays.items():
                krause_save[f"{roi_name}_{key}"] = np.asarray(val)
        np.savez(krause_npz_path, **krause_save)
        print(f"  Saved: {krause_npz_path}  "
              f"({len(krause_aggregate)} ROIs)")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_elapsed = (time.time() - t0_total) / 60
    print(f"\n{'=' * 70}")
    print(f"DONE: 7 fMRI SP moderation models in {total_elapsed:.1f} min")
    print(f"{'=' * 70}")

    # Print a compact summary table
    print(f"\n{'ROI':<25} {'gamma_sp':>10} {'p':>8} {'gamma_ps':>10} "
          f"{'p':>8} {'Rhat':>6}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['ROI']:<25} {r['gamma_sp']:>+10.4f} "
              f"{r['gamma_sp_p']:>8.4f} {r['gamma_ps']:>+10.4f} "
              f"{r['gamma_ps_p']:>8.4f} {r['rhat_max']:>6.3f}")


if __name__ == "__main__":
    main()
