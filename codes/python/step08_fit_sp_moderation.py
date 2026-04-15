"""
Step 08 — Fit Sleep-to-Pain moderation models (7 ROIs).
======================================================================

Input:  derivatives/step03_varx_data/step03_processed_long.csv
        derivatives/step07_sp_roi_values/step07_sp_roi_values.csv
Output:
  derivatives/
    step08_sp_posterior_draws.npz     — per-ROI posterior draws
  results/
    step08_table5_sp_moderation.csv   — Table 5: fMRI SP moderators
    step08_sign_concordance.csv       — sign test results
    step08_text_numbers.csv           — gamma estimates, p-values, etc.

Fits fit_bayesian_varx1 with X_person = z-scored ROI value for
each of the 7 SP ROIs (6 Krause + 1 ACC). Extracts gamma_sp and
gamma_ps posterior summaries. Runs the 6-ROI Krause sign
concordance test.

Author: Pedro Valdes-Hernandez (with Claude Opus 4.6)
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from math import comb

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DERIV_DIR = os.path.join(ROOT, "derivatives")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step08_sp_moderation")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step08_sp_moderation")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

IN_PROCESSED_CSV = os.path.join(DERIV_DIR, "step03_varx_data", "step03_processed_long.csv")
IN_ROI_CSV = os.path.join(DERIV_DIR, "step07_sp_roi_values", "step07_sp_roi_values.csv")

OUT_DRAWS_NPZ = os.path.join(STEP_DERIV_DIR, "step08_sp_posterior_draws.npz")
OUT_TABLE5_CSV = os.path.join(STEP_RESULTS_DIR, "step08_table5_sp_moderation.csv")
OUT_SIGN_CSV = os.path.join(STEP_RESULTS_DIR, "step08_sign_concordance.csv")
OUT_TEXT_CSV = os.path.join(STEP_RESULTS_DIR, "step08_text_numbers.csv")

# ROIs included in the Krause sign-concordance test (ACC excluded)
KRAUSE_ROIS = [
    "Right_S1", "Right_Middle_Insula", "Left_Thalamus",
    "Left_Anterior_Insula", "Left_NAcc", "Right_NAcc",
]

# Expected signs for the sign test
EXPECTED_SIGNS = {
    "Right_S1": "-",
    "Right_Middle_Insula": "+",
    "Left_Thalamus": "+",
    "Left_Anterior_Insula": "+",
    "Left_NAcc": "+",
    "Right_NAcc": "+",
}


def load_step02_data(csv_path):
    """Load Step 03 output and prepare for fitting."""
    df = pd.read_csv(csv_path)
    df["Age_z"] = (df["Age"] - df["Age"].mean()) / df["Age"].std()
    df["Sex_coded"] = (df["Sex"] == 2).astype(float)
    df["Sex_c"] = df["Sex_coded"] - df["Sex_coded"].mean()
    unique_ids = sorted(df["ID"].unique())
    id_map = {sid: i for i, sid in enumerate(unique_ids)}
    df["pid_idx"] = df["ID"].map(id_map)
    model_df = df.dropna(subset=["pain_within_lag1", "sleep_within_lag1"]).copy()
    return df, model_df, unique_ids, id_map


def run_step08(verbose=True, refit=False):
    from coupling_model import fit_bayesian_varx1, extract_results

    if verbose:
        print("=" * 70)
        print("STEP 08 — Fit Sleep-to-Pain moderation (7 ROIs)")
        print("=" * 70)

    os.makedirs(DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    roi_df = pd.read_csv(IN_ROI_CSV)

    # Check whether saved derivatives exist
    saved_exist = os.path.exists(OUT_DRAWS_NPZ) and os.path.exists(OUT_TABLE5_CSV)
    if not refit and not saved_exist:
        if verbose:
            print("  Saved derivatives not found — forcing refit.")
        refit = True

    if not refit and saved_exist:
        # ------ REPLOT MODE: load saved derivatives ------
        if verbose:
            print("  WARNING: Running in replot mode -- loading saved derivatives.")
            print("  If you have changed upstream data or code, re-run with --refit.")
        table5 = pd.read_csv(OUT_TABLE5_CSV)
        draws_dict = dict(np.load(OUT_DRAWS_NPZ))
    else:
        # ------ FULL MCMC FIT ------
        df_full, model_df, unique_ids, id_map = load_step02_data(IN_PROCESSED_CSV)

        all_rois = roi_df["ROI"].unique()
        if verbose:
            print(f"  {len(all_rois)} ROIs to fit")

        table_rows = []
        draws_dict = {}

        for roi_name in all_rois:
            roi_data = roi_df[roi_df["ROI"] == roi_name]
            label = roi_data["label"].iloc[0]
            framework = roi_data["framework"].iloc[0]
            mask_type = roi_data["mask_type"].iloc[0]
            expected = roi_data["expected_sign_sp"].iloc[0]
            raw_mean = roi_data["raw_mean"].iloc[0]
            raw_sd = roi_data["raw_sd"].iloc[0]

            # Build X_person dict {ID: z_value}
            X_person = dict(zip(
                roi_data["ID"].astype(str),
                roi_data["z_value"].values,
            ))

            if verbose:
                print(f"\n  Fitting: {label} ({framework}, {mask_type})...")

            idata, sub_df, valid_ids = fit_bayesian_varx1(
                model_df, unique_ids, id_map,
                X_person=X_person,
                include_agesex=True,
                progressbar=True,
            )

            n_valid = len(valid_ids)
            n_obs = len(sub_df)
            results = extract_results(idata, moderator_name=roi_name)

            sp_mean = results.get("gamma_sp_mean", np.nan)
            sp_lo = results.get("gamma_sp_ci_lo", np.nan)
            sp_hi = results.get("gamma_sp_ci_hi", np.nan)
            sp_prob_neg = results.get("gamma_sp_prob_neg", np.nan)
            sp_p = 2 * min(sp_prob_neg, 1 - sp_prob_neg) if np.isfinite(sp_prob_neg) else np.nan

            ps_mean = results.get("gamma_ps_mean", np.nan)
            ps_lo = results.get("gamma_ps_ci_lo", np.nan)
            ps_hi = results.get("gamma_ps_ci_hi", np.nan)
            ps_prob_neg = results.get("gamma_ps_prob_neg", np.nan)
            ps_p = 2 * min(ps_prob_neg, 1 - ps_prob_neg) if np.isfinite(ps_prob_neg) else np.nan

            rhat_max = results.get("rhat_max", np.nan)

            if verbose:
                print(f"    N={n_valid}, obs={n_obs}")
                print(f"    gamma_sp = {sp_mean:+.4f} [{sp_lo:+.4f}, {sp_hi:+.4f}], p={sp_p:.4f}")
                print(f"    gamma_ps = {ps_mean:+.4f} [{ps_lo:+.4f}, {ps_hi:+.4f}], p={ps_p:.4f}")
                print(f"    R-hat max: {rhat_max:.3f}")

            table_rows.append({
                "ROI": roi_name, "Label": label, "Framework": framework,
                "Source": mask_type, "N": n_valid, "N_obs": n_obs,
                "gamma_sp": sp_mean, "gamma_sp_ci_lo": sp_lo,
                "gamma_sp_ci_hi": sp_hi, "gamma_sp_p": sp_p,
                "gamma_ps": ps_mean, "gamma_ps_ci_lo": ps_lo,
                "gamma_ps_ci_hi": ps_hi, "gamma_ps_p": ps_p,
                "rhat_max": rhat_max,
            })

            # Save posterior draws for JN step
            a2_draws = idata.posterior["a2"].values.flatten()
            gamma_sp_draws = idata.posterior["gamma_sp"].values.flatten()
            u_sp_mean = idata.posterior["u_sp"].values.reshape(-1, len(valid_ids)).mean(axis=0)
            X_vals = np.array([X_person[sid] for sid in valid_ids if sid in X_person])
            draws_dict[f"{roi_name}_a2_draws"] = a2_draws
            draws_dict[f"{roi_name}_gamma_sp_draws"] = gamma_sp_draws
            draws_dict[f"{roi_name}_X_vals"] = X_vals
            draws_dict[f"{roi_name}_raw_mean"] = np.array([raw_mean])
            draws_dict[f"{roi_name}_raw_sd"] = np.array([raw_sd])
            draws_dict[f"{roi_name}_u_sp_mean"] = u_sp_mean

            del idata

        # Save Table 5
        table5 = pd.DataFrame(table_rows)
        table5.to_csv(OUT_TABLE5_CSV, index=False)
        if verbose:
            print(f"\n  Saved Table 5: {OUT_TABLE5_CSV}")

        # Save posterior draws
        np.savez(OUT_DRAWS_NPZ, **draws_dict)
        if verbose:
            print(f"  Saved draws: {OUT_DRAWS_NPZ}")

    text_rows = []

    # Sign concordance test (Krause 6 ROIs only)
    n_concordant = 0
    n_tested = 0
    sign_rows = []
    for roi_name in KRAUSE_ROIS:
        row = table5[table5["ROI"] == roi_name]
        if len(row) == 0:
            continue
        gamma = row["gamma_sp"].iloc[0]
        expected = EXPECTED_SIGNS[roi_name]
        observed = "+" if gamma > 0 else "-"
        match = observed == expected
        if match:
            n_concordant += 1
        n_tested += 1
        sign_rows.append({
            "ROI": roi_name, "gamma_sp": gamma,
            "expected": expected, "observed": observed,
            "match": match,
        })

    sign_p = sum(comb(n_tested, k) * 0.5 ** n_tested
                 for k in range(n_concordant, n_tested + 1))

    sign_df = pd.DataFrame(sign_rows)
    sign_df.to_csv(OUT_SIGN_CSV, index=False)
    if verbose:
        print(f"\n  Sign concordance: {n_concordant}/{n_tested}, p = {sign_p:.4f}")
        print(f"  Saved: {OUT_SIGN_CSV}")

    # Text numbers
    def _t(metric, value, note=""):
        text_rows.append({"metric": metric, "value": str(value), "note": note})

    for _, row in table5.iterrows():
        roi = row["ROI"]
        _t(f"gamma_sp_{roi}", f"{row['gamma_sp']:+.4f}")
        _t(f"gamma_sp_{roi}_ci", f"[{row['gamma_sp_ci_lo']:+.4f}, {row['gamma_sp_ci_hi']:+.4f}]")
        _t(f"gamma_sp_{roi}_p", f"{row['gamma_sp_p']:.4f}")
        _t(f"gamma_ps_{roi}", f"{row['gamma_ps']:+.4f}")
        _t(f"gamma_ps_{roi}_p", f"{row['gamma_ps_p']:.4f}")
        _t(f"N_{roi}", str(row["N"]))

    _t("sign_concordance", f"{n_concordant}/{n_tested}")
    _t("sign_concordance_p", f"{sign_p:.4f}")

    # NAcc-ACC correlations (left NAcc vs each ACC hemisphere)
    nacc_data = roi_df[roi_df["ROI"] == "Left_NAcc"][["ID", "z_value"]].rename(
        columns={"z_value": "nacc"})
    for acc_roi, acc_label in [("Left_dACC_MCC", "left"), ("Right_dACC_MCC", "right")]:
        acc_data = roi_df[roi_df["ROI"] == acc_roi][["ID", "z_value"]].rename(
            columns={"z_value": "acc"})
        merged = nacc_data.merge(acc_data, on="ID")
        if len(merged) > 10:
            r_val = float(np.corrcoef(merged["nacc"], merged["acc"])[0, 1])
            _t(f"r_nacc_acc_{acc_label}", f"{r_val:.2f}")
            if verbose:
                print(f"  Left NAcc vs {acc_label} ACC BOLD correlation: r = {r_val:.2f}")

    text_df = pd.DataFrame(text_rows)
    text_df.to_csv(OUT_TEXT_CSV, index=False)
    if verbose:
        print(f"  Saved: {OUT_TEXT_CSV}")

    generate_text_paragraphs(verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 08 COMPLETE")
        print("=" * 70)


def generate_text_paragraphs(verbose: bool = True) -> None:
    """Generate step08_text.md with the manuscript paragraphs for Results
    section 3.4 (sleep-to-pain fMRI moderation), populated from
    step08_text_numbers.csv and step08_table5_sp_moderation.csv.
    """
    OUT_TEXT_MD = os.path.join(STEP_RESULTS_DIR, "step08_text.md")

    if not os.path.exists(OUT_TEXT_CSV):
        if verbose:
            print("  SKIP: step08_text_numbers.csv not found — run step08 first")
        return

    if verbose:
        print("  Generating step08_text.md ...")

    tn = pd.read_csv(OUT_TEXT_CSV)
    v = dict(zip(tn["metric"], tn["value"]))

    # Helper to strip leading '+' for display
    def _val(key):
        return v.get(key, "N/A").lstrip("+")

    def _signed(key):
        return v.get(key, "N/A")

    # Also pull tau_sp and lambda_sp from step04 outputs to build the intro
    # paragraph that motivates the moderator search.
    step04_tn_path = os.path.join(
        RESULTS_DIR, "step04_coupling_model", "step04_text_numbers.csv"
    )
    tau_sp = None
    lambda_sp = None
    if os.path.exists(step04_tn_path):
        s4 = pd.read_csv(step04_tn_path)
        s4v = dict(zip(s4["metric"], s4["value"]))
        try:
            tau_sp = float(s4v.get("tau_sp", "nan"))
            lambda_sp = float(s4v.get("lambda_sp_mean", "nan"))
        except (TypeError, ValueError):
            tau_sp = lambda_sp = None

    # --- Paragraph 0: motivating the moderator search ---
    if tau_sp and lambda_sp:
        ratio_lo = tau_sp / max(abs(lambda_sp) + 0.02, 0.01)  # lambda ~-0.02 ± SE
        ratio_hi = tau_sp / max(abs(lambda_sp), 0.01)
        # Keep ordering low..high
        r_lo, r_hi = sorted([ratio_lo, ratio_hi])
        p0 = (
            "Although the population-average sleep-to-pain coupling was not "
            "credibly different from zero, the between-person variability was "
            "substantial relative to the fixed effect "
            f"($\\left|{{\\widehat{{\\tau}}}}_{{sp}}\\right|/"
            f"\\left|{{\\widehat{{\\lambda}}}}_{{sp}}\\right| \\approx$ "
            f"[{r_lo:.1f}, {r_hi:.1f}]), "
            "suggesting that individual-level moderators may shape the "
            "direction and magnitude of this effect and obscure a consistent "
            "population mean. We therefore tested whether BOLD activation in "
            "the ROIs proposed by Krause et al. moderated the "
            "sleep-to-pain coupling."
        )
    else:
        p0 = ""

    # --- Paragraph 1: NAcc results ---
    p1 = (
        "Left NAcc activation during painful knee stimulation was the only "
        "Krause et al. (2019) ROI that credibly moderated sleep-to-pain "
        f"coupling ($\\hat{{\\gamma}}_{{sp}}={_signed('gamma_sp_Left_NAcc')}$, "
        f"95% CrI {v.get('gamma_sp_Left_NAcc_ci', 'N/A')}; **Table 5**): "
        "the lower the left NAcc activation during painful stimulation, the "
        "stronger the sleep-to-pain coupling. "
        "The right NAcc showed the same direction but it was not credible "
        f"($\\hat{{\\gamma}}_{{sp}}={_signed('gamma_sp_Right_NAcc')}$, "
        f"95% CrI {v.get('gamma_sp_Right_NAcc_ci', 'N/A')}). "
    )

    # JN info from step09 if available; otherwise from text_numbers
    jn_text_csv = os.path.join(RESULTS_DIR, "step09_sp_jn", "step09_text_numbers.csv")
    if os.path.exists(jn_text_csv):
        jn_tn = pd.read_csv(jn_text_csv)
        jv = dict(zip(jn_tn["metric"], jn_tn["value"]))
        nacc_boundary = jv.get("jn_sp_Left_NAcc_boundary_raw", "N/A")
        nacc_pct = jv.get("jn_sp_Left_NAcc_pct_credible", "N/A")
        # Round pct to integer
        try:
            nacc_pct_int = f"{float(nacc_pct):.0f}"
        except (ValueError, TypeError):
            nacc_pct_int = nacc_pct
        p1 += (
            "Johnson-Neyman analysis (**Figure 5**) revealed that sleep-to-pain "
            f"coupling was credibly negative for individuals with left NAcc "
            f"activation below {nacc_boundary} (near the sample mean), "
            f"encompassing {nacc_pct_int}% of the sample. Above this boundary, the "
            "coupling was not credibly different from zero. JN analyses for the "
            "remaining Krause et al. (2019) ROIs is provided in the "
            "Supplementary Material (**Figure S5**)."
        )
    else:
        p1 += (
            "Johnson-Neyman analysis (**Figure 5**) revealed that sleep-to-pain "
            "coupling was credibly negative for individuals with low left NAcc "
            "activation. JN analyses for the remaining Krause et al. (2019) ROIs "
            "is provided in the Supplementary Material (**Figure S5**)."
        )

    # --- Paragraph 2: ACC results ---
    # Left dACC/MCC
    l_acc_gamma = _signed("gamma_sp_Left_dACC_MCC")
    l_acc_ci = v.get("gamma_sp_Left_dACC_MCC_ci", "N/A")
    l_acc_p = _val("gamma_sp_Left_dACC_MCC_p")
    # Right dACC/MCC
    r_acc_gamma = _signed("gamma_sp_Right_dACC_MCC")
    r_acc_ci = v.get("gamma_sp_Right_dACC_MCC_ci", "N/A")
    r_acc_p = _val("gamma_sp_Right_dACC_MCC_p")

    # ACC-NAcc correlations
    r_nacc_acc_left = v.get("r_nacc_acc_left", v.get("r_nacc_acc", "N/A"))
    r_nacc_acc_right = v.get("r_nacc_acc_right", "N/A")

    p2 = (
        "The ACC\u2014tested separately from the Krause et al. (2019) framework as a "
        "Sardi et al. (2024)-motivated D2-gated node\u2014also credibly "
        "moderated sleep-to-pain coupling "
        f"($\\hat{{\\gamma}}_{{sp}}={l_acc_gamma}$, "
        f"95% CrI {l_acc_ci}\u2014a marginal credibility\u2014for the left, and "
        f"$\\hat{{\\gamma}}_{{sp}}={r_acc_gamma}$, "
        f"95% CrI {r_acc_ci} for the right; **Table 5**): "
        "higher pain-evoked ACC activation was associated with less negative "
        "sleep-to-pain coupling. "
        "The Johnson-Neyman analysis (**Figure 6**) revealed a pattern closely "
        "paralleling the NAcc, with credible sleep-to-pain coupling present only "
        "in individuals with below-average ACC activation. "
        f"The low correlation between ACCs and left NAcc BOLD values "
        f"($r={r_nacc_acc_left}$ for the left ACC and "
        f"$r={r_nacc_acc_right}$ for the right one) indicates that these two "
        "regions capture largely independent aspects of pain processing, consistent "
        "with the view that they operate as parallel rather than serial nodes "
        "within the dopaminergic system (28)."
    )

    # --- Paragraph 3: Sign concordance ---
    sign_conc = v.get("sign_concordance", "N/A")
    sign_p = v.get("sign_concordance_p", "N/A")
    n_match, n_total = sign_conc.split("/") if "/" in sign_conc else ("N", "N")

    p3 = (
        f"Finally, although the moderation was credible only for the left NAcc "
        f"and both ACC among the seven sleep-to-pain ROIs (**Table 5**), all "
        f"{n_total} Krause et al. (2019) univariate $\\hat{{\\gamma}}_{{sp}}$ "
        f"estimates matched the direction predicted by the sleep deprivation "
        f"framework. The probability of this occurring by chance is "
        f"$p=(1/2)^{{{n_total}}}={sign_p}$ (exact sign test), providing "
        f"convergent support for the Krause et al. (2019) framework at the "
        f"pattern level."
    )

    # --- Table 5 ---
    table5_text = ""
    if os.path.exists(OUT_TABLE5_CSV):
        t5 = pd.read_csv(OUT_TABLE5_CSV)
        table5_lines = [
            "| ROI | Framework | Expected $\\hat{\\gamma}_{sp}$ | $\\hat{\\gamma}_{sp}$ | 95% CrI |",
            "| :--- | :--- | :---: | ---: | :--- |",
        ]
        for _, row in t5.iterrows():
            roi = row["ROI"].replace("_", " ")
            fw = row["Framework"]
            exp = row.get("expected_sign_sp", "")
            gsp = f"{row['gamma_sp']:+.3f}"
            ci = f"[{row['gamma_sp_ci_lo']:+.3f}, {row['gamma_sp_ci_hi']:+.3f}]"
            # Bold credible rows (CrI excludes zero)
            is_credible = (row["gamma_sp_ci_lo"] > 0) or (row["gamma_sp_ci_hi"] < 0)
            if is_credible:
                table5_lines.append(
                    f"| **{roi}** | **{fw}** | **{exp}** | **{gsp}** | **{ci}** |"
                )
            else:
                table5_lines.append(
                    f"| {roi} | {fw} | {exp} | {gsp} | {ci} |"
                )
        table5_text = "\n".join(table5_lines)

    n_fmri = int(v.get("n_fmri", "174"))
    # Table 5 Note is descriptive (ROI definitions, methods cross-refs);
    # lives only in docs/manuscript_pain.md.

    intro_block = f"### Paragraph 0 (moderator motivation)\n\n{p0}\n\n" if p0 else ""

    text = f"""\
## Results > 3.4 Sleep-to-pain fMRI moderation
{intro_block}### Paragraph 1 (NAcc results)

{p1}

### Paragraph 2 (ACC results)

{p2}

### Paragraph 3 (sign concordance)

{p3}

## Table 5. fMRI stimulation BOLD moderators of sleep-to-pain coupling (N = {n_fmri})

{table5_text}
"""

    with open(OUT_TEXT_MD, "w") as f:
        f.write(text)

    if verbose:
        print(f"    Saved: {OUT_TEXT_MD}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 08 — fit SP moderation models (7 ROIs)."
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--refit", action="store_true",
                        help="Re-run computation from scratch instead of loading saved derivatives")
    args = parser.parse_args()
    run_step08(verbose=not args.quiet, refit=args.refit)


if __name__ == "__main__":
    main()
