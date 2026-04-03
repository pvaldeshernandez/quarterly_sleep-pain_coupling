#!/usr/bin/env python3
"""
05_arousal_ps_moderation.py
===========================
Neuroimaging Moderation of Pain-to-Sleep Coupling (Aim 2b):
Arousal-Pathway ROIs from Lynch et al. (2025)

Overview
--------
This script tests whether activity (fMRI BOLD) or structure (gray matter
volume) in subcortical arousal-relay nuclei moderates the quarterly
pain-to-sleep coupling path (gamma_ps).  It runs 10 independent Bayesian
VARX(1) models -- 5 ROIs x 2 modalities -- and saves the moderation
results and Johnson-Neyman (JN) significance curves.

Theoretical Framework
---------------------
**Lynch et al. (2025) Advanced Science** identified a nociceptive arousal
relay pathway originating from calcitonin gene-related peptide (CGRP)
neurons in the lateral parabrachial nucleus (PBN^elCGRP).  These neurons
project to four forebrain targets that drive wakefulness and arousal:

  PBN^elCGRP  -->  SI-BF (substantia innominata / basal forebrain, Ch4)
                   CeA  (central nucleus of the amygdala)
                   BNST (bed nucleus of the stria terminalis)
                   LH   (lateral hypothalamus, orexin/hypocretin neurons)

This pathway provides a direct neural mechanism by which nociceptive
input can fragment sleep via ascending arousal.  Importantly, it operates
independently of cortical pain processing -- these are subcortical
structures involved in basic arousal and threat detection, not pain
perception per se.

**Ito et al. (2023)** further demonstrated that chronic pain engages a
dynorphin/kappa-opioid receptor (KOR) wakefulness pathway: persistent
nociception activates dynorphin release in the pontine reticular
formation, which paradoxically promotes wakefulness through KOR signaling,
contributing to pain-related insomnia.

Expected Direction of Moderation
---------------------------------
We test whether individual differences in arousal-relay capacity moderate
the pain-to-sleep coupling path (gamma_ps).  The expected direction is
NEGATIVE gamma_ps:

  - Greater BOLD activation in arousal-relay regions during painful
    stimulation reflects a stronger nociceptive arousal response.
    Individuals with higher activation should show stronger pain-to-sleep
    coupling (more negative lambda_ps, meaning pain disrupts sleep more).

  - Greater gray matter volume in these regions reflects greater
    structural capacity for the arousal relay.  Individuals with larger
    arousal nuclei should show stronger pain-to-sleep coupling.

In both cases, negative gamma_ps means: higher moderator value (more
arousal capacity) -> more negative coupling (pain disrupts sleep more).

Why Pain-to-Sleep Direction
----------------------------
These arousal circuits concern how nociceptive signals disrupt sleep.
The direction of causal flow is: pain -> arousal activation -> sleep
fragmentation.  This maps to the pain-to-sleep (PS) coupling path in
our VARX(1) model, not the sleep-to-pain (SP) path.

Atlas Sources
-------------
Each ROI is defined using a published probabilistic or label atlas,
resampled to the target image resolution:

  PBN:      Singh et al. (2020), EagleVAC brainstem atlas.
            Labels 19/20 (left/right lateral parabrachial).

  SI-BF:    Zaborszky et al. (2008), Ch4 basal forebrain probabilistic
            atlas in MNI152 space.

  CeA:      Pauli et al. (2018), CIT168 crowd-sourced subcortical
            probabilistic atlas, bilateral central amygdala.

  BNST:     Theiss et al. (2017), 3T probabilistic atlas of the BNST.

  LH:       Neudorfer et al. (2020), hypothalamic atlas.
            Labels 25/26 (right/left lateral hypothalamus).

Masking Convention
------------------
  - fMRI: Most ROIs use unmasked contrast images.  LH uses GM-masked
    contrasts (adequate GM coverage at 3mm fMRI resolution).
  - VBM: Smoothed modulated gray matter images (smwc1 from CAT12),
    no masking needed (these are tissue probability maps).
  - PBN: At fMRI resolution (3mm), the PBN spans only ~3 voxels.  It was
    originally excluded from fMRI analyses due to 0 voxels within the GM
    mask; after re-estimation without masks, it became testable.  At VBM
    resolution (~1.5mm), PBN has adequate coverage.

Output Files
------------
  results/arousal_fmri_moderation_results.csv  -- fMRI BOLD, 5 ROIs
  results/arousal_vbm_moderation_results.csv   -- VBM GM volume, 5 ROIs
  results/arousal_jn_results.csv               -- JN boundaries (if any)

Usage
-----
  # With synthetic data (no neuroimaging files required):
  python python/05_arousal_ps_moderation.py --synthetic

  # With real data (requires NIfTI files under data/):
  python python/05_arousal_ps_moderation.py

References
----------
[1] Lynch JG, Berezowska S, Bhatt S, et al. (2025). A parabrachial to
    forebrain nociceptive relay promotes wakefulness. Advanced Science.
[2] Ito H, Navratilova E, Bhatt DK, Roe JD, Bhatt S, Dickenson AH,
    Porreca F. (2023). Chronic pain recruits hypothalamic dynorphin/kappa
    opioid receptor signalling to promote wakefulness and vigilance.
    Brain 146:4374-4392.
[3] Singh K, Indovina I, Augustinack J, et al. (2020). Probabilistic
    atlas of the lateral parabrachial nucleus, medial parabrachial
    nucleus, Koelliker-Fuse nucleus and ventral respiratory column in
    living humans from 7T MRI. NeuroImage 221:117187.
[4] Zaborszky L, Hoemke L, Mohlberg H, Schleicher A, Amunts K, Zilles K.
    (2008). Stereotaxic probabilistic maps of the magnocellular cell
    groups in human basal forebrain. NeuroImage 42:1127-1141.
[5] Pauli WM, Nili AN, Tyszka JM. (2018). A high-resolution probabilistic
    in vivo atlas of human subcortical brain nuclei. Scientific Data
    5:180063.
[6] Theiss JD, Ridgewell C, McHugo M, Heckers S, Blackford JU. (2017).
    Manual segmentation of the human bed nucleus of the stria terminalis
    using 3T MRI. NeuroImage 146:288-292.
[7] Neudorfer C, Germann J, Elias GJB, Gramer R, Boutet A, Lozano AM.
    (2020). A high-resolution in vivo magnetic resonance imaging atlas of
    the human hypothalamic region. Scientific Data 7:305.

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
    ATLAS_AROUSAL_ROIS,
    load_fmri_atlas_arousal,
    load_vbm_atlas_arousal,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")


# ===================================================================
# Expected Signs (for sign concordance analysis)
# ===================================================================

# Expected sign of gamma_ps for each arousal ROI.
# Based on the Lynch et al. (2025) framework: greater arousal-relay
# capacity should amplify pain-to-sleep disruption (more negative
# coupling), predicting negative gamma_ps.
EXPECTED_SIGNS_PS = {
    "PBN":       "-",
    "SI_BF_Ch4": "-",
    "CeA":       "-",
    "BNST":      "-",
    "LH":        "-",
}


# ===================================================================
# Synthetic Data Loaders
# ===================================================================

def _load_synthetic_fmri_arousal():
    """Load fMRI arousal ROI values from synthetic CSV.

    The synthetic CSV (data/synthetic/roi_values.csv) contains all ROIs
    in one file, using column names 'subject_id', 'roi_name', 'bold_value'.

    Returns
    -------
    moderators : dict
        {roi_name: {subject_id: z_scored_value, ...}}
    labels : dict
        {roi_name: human-readable label}
    raw_stats : dict
        {roi_name: {"mean": 0.0, "sd": 1.0}}
    """
    csv_path = os.path.join(SYNTHETIC_DIR, "roi_values.csv")
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"subject_id": "ID", "bold_value": "value"})

    moderators = {}
    labels = {}
    raw_stats = {}

    for roi_name, roi_cfg in ATLAS_AROUSAL_ROIS.items():
        roi_df = df[df["roi_name"] == roi_name]
        if len(roi_df) == 0:
            continue

        values = dict(zip(roi_df["ID"].astype(str), roi_df["value"]))
        moderators[roi_name] = values
        labels[roi_name] = roi_cfg["label"] + " (atlas BOLD)"
        raw_stats[roi_name] = {"mean": 0.0, "sd": 1.0}

    return moderators, labels, raw_stats


def _load_synthetic_vbm_arousal():
    """Load VBM arousal ROI volumes from synthetic CSV.

    The synthetic CSV (data/synthetic/vbm_volumes.csv) contains gray
    matter volumes with columns 'subject_id', 'roi_name', 'gm_volume'.

    Returns
    -------
    moderators : dict
        {roi_name: {subject_id: z_scored_value, ...}}
    labels : dict
        {roi_name: human-readable label}
    raw_stats : dict
        {roi_name: {"mean": 0.0, "sd": 1.0}}
    """
    csv_path = os.path.join(SYNTHETIC_DIR, "vbm_volumes.csv")
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"subject_id": "ID", "gm_volume": "value"})

    moderators = {}
    labels = {}
    raw_stats = {}

    for roi_name, roi_cfg in ATLAS_AROUSAL_ROIS.items():
        roi_df = df[df["roi_name"] == roi_name]
        if len(roi_df) == 0:
            continue

        values = dict(zip(roi_df["ID"].astype(str), roi_df["value"]))
        moderators[roi_name] = values
        labels[roi_name] = roi_cfg["label"] + " (atlas GM volume)"
        raw_stats[roi_name] = {"mean": 0.0, "sd": 1.0}

    return moderators, labels, raw_stats


# ===================================================================
# Helper: Two-Sided Posterior p-value
# ===================================================================

def two_sided_p(samples):
    """Compute a two-sided Bayesian p-value from posterior samples.

    This is twice the smaller tail probability: p = 2 * min(P(x>0), P(x<0)).
    A value of 0.05 means 97.5% of the posterior mass is on one side of zero.

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
# Run One Modality
# ===================================================================

def run_modality(modality_name, moderators, labels, raw_stats,
                 model_df, unique_ids, id_map):
    """Run moderation models for all ROIs in one imaging modality.

    Fits the VARX(1) model with each ROI as a moderator, extracting
    gamma_ps (the pain-to-sleep moderation coefficient) and computing
    JN curves for any ROIs with suggestive moderation (p < 0.10).

    Parameters
    ----------
    modality_name : str
        'fMRI BOLD' or 'VBM GM volume' -- used for logging.
    moderators : dict
        {roi_name: {subject_id: z_scored_value, ...}}
    labels : dict
        {roi_name: human-readable label}
    raw_stats : dict
        {roi_name: {"mean": float, "sd": float}}
    model_df : DataFrame
        Analysis-ready coupling data.
    unique_ids : list
        Subject IDs in the coupling dataset.
    id_map : dict
        {subject_ID: integer_index}

    Returns
    -------
    results : list of dict
        One dict per ROI with moderation results.
    jn_results : list of dict
        JN boundary information for ROIs with p < 0.10.
    """
    results = []
    jn_results = []

    # Ordered ROI list (same order as ATLAS_AROUSAL_ROIS definition)
    roi_order = list(ATLAS_AROUSAL_ROIS.keys())
    n_rois = len([r for r in roi_order if r in moderators])

    roi_count = 0
    for roi_name in roi_order:
        if roi_name not in moderators:
            continue
        roi_count += 1

        X_person = moderators[roi_name]
        label = labels[roi_name]
        n_subj = len(X_person)

        print(f"\n{'=' * 60}")
        print(f"  [{roi_count}/{n_rois}] {label}")
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
        # gamma_ps is the parameter of interest (pain-to-sleep direction)
        # gamma_sp is also extracted (as nuisance) for completeness
        gamma_sp = idata.posterior["gamma_sp"].values.flatten()
        gamma_ps = idata.posterior["gamma_ps"].values.flatten()
        b1_draws = idata.posterior["b1"].values.flatten()

        sp_mean = float(gamma_sp.mean())
        sp_lo, sp_hi = np.percentile(gamma_sp, [2.5, 97.5])
        sp_p = two_sided_p(gamma_sp)

        ps_mean = float(gamma_ps.mean())
        ps_lo, ps_hi = np.percentile(gamma_ps, [2.5, 97.5])
        ps_p = two_sided_p(gamma_ps)

        # --- Convergence diagnostic ---
        import arviz as az
        rhat_vals = az.rhat(idata)
        rhat_max = max(
            float(rhat_vals[v].max()) for v in rhat_vals.data_vars)

        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  N valid: {n_valid}, N obs: {n_obs}")
        print(f"  gamma_ps = {ps_mean:+.4f} [{ps_lo:+.4f}, {ps_hi:+.4f}], "
              f"p={ps_p:.4f}  <-- theory-relevant direction")
        print(f"  gamma_sp = {sp_mean:+.4f} [{sp_lo:+.4f}, {sp_hi:+.4f}], "
              f"p={sp_p:.4f}  (nuisance)")
        print(f"  R-hat max: {rhat_max:.3f}")

        # --- Sign concordance check ---
        expected = EXPECTED_SIGNS_PS.get(roi_name, None)
        if expected is not None:
            observed_sign = "-" if ps_mean < 0 else "+"
            concordant = observed_sign == expected
            print(f"  Expected sign (PS): {expected}, "
                  f"Observed: {observed_sign}, Concordant: {concordant}")

        # --- Accumulate results ---
        results.append({
            "ROI": roi_name,
            "Label": label,
            "Modality": modality_name,
            "N": n_valid,
            "N_obs": n_obs,
            "gamma_ps": ps_mean,
            "gamma_ps_ci_lo": float(ps_lo),
            "gamma_ps_ci_hi": float(ps_hi),
            "gamma_ps_p": ps_p,
            "gamma_sp": sp_mean,
            "gamma_sp_ci_lo": float(sp_lo),
            "gamma_sp_ci_hi": float(sp_hi),
            "gamma_sp_p": sp_p,
            "rhat_max": rhat_max,
            "elapsed_s": elapsed,
        })

        # --- Johnson-Neyman analysis ---
        # Run for ROIs with suggestive PS moderation (p < 0.10)
        if ps_p < 0.10:
            print(f"\n  Running JN analysis for PS direction...")
            X_vals_z = np.array(
                [X_person[sid] for sid in valid_ids if sid in X_person])
            roi_stats = raw_stats.get(roi_name, {"mean": 0.0, "sd": 1.0})

            jn = compute_jn_curve(
                b1_draws, gamma_ps, X_vals_z,
                raw_mean=roi_stats["mean"],
                raw_sd=roi_stats["sd"],
            )

            if jn["jn_boundaries_z"]:
                for bz, br in zip(
                        jn["jn_boundaries_z"], jn["jn_boundaries"]):
                    pct_below = (X_vals_z <= bz).mean() * 100
                    print(f"    JN boundary: z={bz:.3f} (raw={br:.4f}), "
                          f"{pct_below:.1f}% of sample below")

                    jn_results.append({
                        "ROI": roi_name,
                        "Label": label,
                        "Modality": modality_name,
                        "Direction": "PS",
                        "JN_boundary_z": float(bz),
                        "JN_boundary_raw": float(br),
                        "Pct_below": float(pct_below),
                    })

            # Report simple slopes at key moderator values
            print(f"    Simple slopes:")
            for z_val in [-2, 0, 2]:
                draws = b1_draws + gamma_ps * z_val
                m = draws.mean()
                lo, hi = np.percentile(draws, [2.5, 97.5])
                sig = "*" if hi < 0 or lo > 0 else ""
                raw_val = z_val * roi_stats["sd"] + roi_stats["mean"]
                print(f"      z={z_val:+d} (raw={raw_val:.4f}): "
                      f"coupling={m:.3f} [{lo:.3f}, {hi:.3f}]{sig}")

    return results, jn_results


# ===================================================================
# Main Analysis
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Arousal-pathway moderation of pain-to-sleep "
                    "coupling (Aim 2b)")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data instead of real neuroimaging files")
    args = parser.parse_args()

    t0_total = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load coupling data
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Loading coupling data")
    print("=" * 70)
    df_full, model_df, unique_ids, id_map = load_data(
        DATA_DIR, synthetic=args.synthetic)

    # ------------------------------------------------------------------
    # Step 2: Load fMRI arousal ROI values
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2A: Loading fMRI BOLD arousal ROI values")
    print("=" * 70)

    if args.synthetic:
        fmri_mods, fmri_labels, fmri_stats = _load_synthetic_fmri_arousal()
    else:
        fmri_mods, fmri_labels, fmri_stats = load_fmri_atlas_arousal(
            DATA_DIR, synthetic=False)

    print(f"  Loaded {len(fmri_mods)} fMRI arousal ROIs")
    for roi_name, values in fmri_mods.items():
        print(f"    {roi_name}: N={len(values)}")

    # ------------------------------------------------------------------
    # Step 3: Load VBM arousal ROI values
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2B: Loading VBM GM volume arousal ROI values")
    print("=" * 70)

    if args.synthetic:
        vbm_mods, vbm_labels, vbm_stats = _load_synthetic_vbm_arousal()
    else:
        vbm_mods, vbm_labels, vbm_stats = load_vbm_atlas_arousal(
            DATA_DIR, synthetic=False)

    print(f"  Loaded {len(vbm_mods)} VBM arousal ROIs")
    for roi_name, values in vbm_mods.items():
        print(f"    {roi_name}: N={len(values)}")

    # ------------------------------------------------------------------
    # Step 4: Run fMRI moderation models
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Running fMRI BOLD arousal PS moderation models")
    print("=" * 70)

    fmri_results, fmri_jn = run_modality(
        "fMRI BOLD", fmri_mods, fmri_labels, fmri_stats,
        model_df, unique_ids, id_map)

    # ------------------------------------------------------------------
    # Step 5: Run VBM moderation models
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Running VBM GM volume arousal PS moderation models")
    print("=" * 70)

    vbm_results, vbm_jn = run_modality(
        "VBM GM volume", vbm_mods, vbm_labels, vbm_stats,
        model_df, unique_ids, id_map)

    # ------------------------------------------------------------------
    # Step 6: Sign concordance tests
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Sign Concordance Tests")
    print("=" * 70)

    for modality_label, res_list in [
            ("fMRI BOLD", fmri_results), ("VBM GM volume", vbm_results)]:
        n_concordant = 0
        n_tested = 0
        for r in res_list:
            roi_name = r["ROI"]
            expected = EXPECTED_SIGNS_PS.get(roi_name, None)
            if expected is not None:
                n_tested += 1
                observed_negative = r["gamma_ps"] < 0
                if expected == "-" and observed_negative:
                    n_concordant += 1
                elif expected == "+" and not observed_negative:
                    n_concordant += 1

        if n_tested > 0:
            sign_p = 0.5 ** n_tested
            print(f"  {modality_label}: {n_concordant}/{n_tested} ROIs "
                  f"match predicted sign (negative)")
            print(f"    Sign concordance p = (1/2)^{n_tested} = "
                  f"{sign_p:.4f}")

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: Saving results")
    print("=" * 70)

    # Save fMRI results
    if fmri_results:
        fmri_df = pd.DataFrame(fmri_results)
        fmri_csv = os.path.join(
            RESULTS_DIR, "arousal_fmri_moderation_results.csv")
        fmri_df.to_csv(fmri_csv, index=False)
        print(f"  Saved: {fmri_csv}")

    # Save VBM results
    if vbm_results:
        vbm_df = pd.DataFrame(vbm_results)
        vbm_csv = os.path.join(
            RESULTS_DIR, "arousal_vbm_moderation_results.csv")
        vbm_df.to_csv(vbm_csv, index=False)
        print(f"  Saved: {vbm_csv}")

    # Save JN results (combined across modalities)
    all_jn = fmri_jn + vbm_jn
    if all_jn:
        jn_df = pd.DataFrame(all_jn)
        jn_csv = os.path.join(RESULTS_DIR, "arousal_jn_results.csv")
        jn_df.to_csv(jn_csv, index=False)
        print(f"  Saved: {jn_csv}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_elapsed = (time.time() - t0_total) / 60
    n_fmri = len(fmri_results)
    n_vbm = len(vbm_results)
    print(f"\n{'=' * 70}")
    print(f"DONE: {n_fmri} fMRI + {n_vbm} VBM arousal PS moderation models "
          f"in {total_elapsed:.1f} min")
    print(f"{'=' * 70}")

    # Print a compact summary table for fMRI
    if fmri_results:
        print(f"\nfMRI BOLD results:")
        print(f"{'ROI':<15} {'gamma_ps':>10} {'p':>8} {'gamma_sp':>10} "
              f"{'p':>8} {'Rhat':>6}")
        print("-" * 65)
        for r in fmri_results:
            print(f"{r['ROI']:<15} {r['gamma_ps']:>+10.4f} "
                  f"{r['gamma_ps_p']:>8.4f} {r['gamma_sp']:>+10.4f} "
                  f"{r['gamma_sp_p']:>8.4f} {r['rhat_max']:>6.3f}")

    # Print a compact summary table for VBM
    if vbm_results:
        print(f"\nVBM GM volume results:")
        print(f"{'ROI':<15} {'gamma_ps':>10} {'p':>8} {'gamma_sp':>10} "
              f"{'p':>8} {'Rhat':>6}")
        print("-" * 65)
        for r in vbm_results:
            print(f"{r['ROI']:<15} {r['gamma_ps']:>+10.4f} "
                  f"{r['gamma_ps_p']:>8.4f} {r['gamma_sp']:>+10.4f} "
                  f"{r['gamma_sp_p']:>8.4f} {r['rhat_max']:>6.3f}")


if __name__ == "__main__":
    main()
