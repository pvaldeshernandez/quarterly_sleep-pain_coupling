"""
Step 03 — Fit the Bayesian VARX(1) coupling model + LOO-CV.
======================================================================

Input:  derivatives/step02_processed_long.csv
Output:
  derivatives/
    step03_posterior_draws.npz     — raw posterior arrays for downstream steps
    step03_person_coupling.csv     — per-person lambda_sp / lambda_ps
  results/
    step03_table4_coupling.csv     — Table 4: population parameters
    step03_loo_comparison.csv      — LOO-CV pairwise comparisons
    step03_text_numbers.csv        — every number stated in the text

This step fits the bivariate VARX(1) coupling model to the
within-person deviations produced by Step 02. It then fits three
additional nested models (no_PS, no_SP, null) for the LOO-CV
model comparison reported in Results §3.1.

The shared model code (``fit_bayesian_varx1``, ``extract_results``,
``compute_loo_comparison``) lives in ``codes/lib/coupling_model.py``.

Author: Pedro Valdes-Hernandez (with Claude Opus 4.6)
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =====================================================================
# Paths
# =====================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repo root
DERIV_DIR = os.path.join(ROOT, "derivatives")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step04_coupling_model")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step04_coupling_model")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

IN_PROCESSED_CSV = os.path.join(DERIV_DIR, "step03_varx_data", "step03_processed_long.csv")

# Derivatives
OUT_DRAWS_NPZ = os.path.join(STEP_DERIV_DIR, "step04_posterior_draws.npz")
OUT_PERSON_CSV = os.path.join(STEP_DERIV_DIR, "step04_person_coupling.csv")

# Results
OUT_TABLE4_CSV = os.path.join(STEP_RESULTS_DIR, "step04_table4_coupling.csv")
OUT_LOO_CSV = os.path.join(STEP_RESULTS_DIR, "step04_loo_comparison.csv")
OUT_TEXT_CSV = os.path.join(STEP_RESULTS_DIR, "step04_text_numbers.csv")
OUT_FIG2 = os.path.join(STEP_RESULTS_DIR, "step04_figure2_ps_coupling.png")
OUT_FIG3 = os.path.join(STEP_RESULTS_DIR, "step04_figure3_sp_coupling.png")


# =====================================================================
# Data loading
# =====================================================================

def load_data(csv_path: str):
    """Load Step 02 output and prepare for fit_bayesian_varx1.

    Returns (df_full, model_df, unique_ids, id_map) matching the
    interface that coupling_model.py expects.
    """
    df = pd.read_csv(csv_path)

    # Age z-score, Sex center
    df["Age_z"] = (df["Age"] - df["Age"].mean()) / df["Age"].std()
    df["Sex_coded"] = (df["Sex"] == 2).astype(float)
    sex_mean = df["Sex_coded"].mean()
    df["Sex_c"] = df["Sex_coded"] - sex_mean

    unique_ids = sorted(df["ID"].unique())
    id_map = {sid: i for i, sid in enumerate(unique_ids)}
    df["pid_idx"] = df["ID"].map(id_map)

    model_df = df.dropna(
        subset=["pain_within_lag1", "sleep_within_lag1"]
    ).copy()

    print(f"  Loaded: {len(unique_ids)} subjects, {len(model_df)} observations")
    print(f"  Age: mean={df['Age'].mean():.1f}, SD={df['Age'].std():.1f}")
    print(f"  Sex: {int((df['Sex']==2).sum())} female, "
          f"{int((df['Sex']==1).sum())} male")
    print(f"  Sex centering: mean(Sex_coded)={sex_mean:.4f}")

    return df, model_df, unique_ids, id_map


# =====================================================================
# Main pipeline
# =====================================================================

def _generate_coupling_figure(person_df, pop_mean, pop_ci_lo, pop_ci_hi,
                              col_mean, col_ci_lo, col_ci_hi, col_prob,
                              direction_label, out_path, prob_neg=None):
    """Draw a 2-panel coupling figure (boxstrip + forest) and save."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patheffects as patheffects
    from matplotlib.lines import Line2D

    df = person_df.sort_values(col_mean).reset_index(drop=True)
    n = len(df)
    vals = df[col_mean].values
    lo = df[col_ci_lo].values
    hi = df[col_ci_hi].values

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.12)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ---- Panel A: Boxstrip ----
    ax_a.boxplot(vals, vert=False, widths=0.5,
                 patch_artist=True,
                 boxprops=dict(facecolor="#E3F2FD", edgecolor="#1565C0",
                               linewidth=1.5),
                 medianprops=dict(color="#D32F2F", linewidth=2.5),
                 whiskerprops=dict(color="#1565C0", linewidth=1.2),
                 capprops=dict(color="#1565C0", linewidth=1.2),
                 flierprops=dict(marker="", markersize=0))

    rng = np.random.default_rng(42)
    jitter = rng.uniform(0.7, 1.3, n)
    dot_colors = ["#42A5F5" if x < 0 else "#EF5350" for x in vals]
    edge_colors = ["#0D47A1" if x < 0 else "#B71C1C" for x in vals]
    sc = ax_a.scatter(vals, jitter, c=dot_colors, alpha=0.7, s=80, zorder=3,
                      marker="o", edgecolors=edge_colors, linewidths=0.8)
    sc.set_path_effects([patheffects.withSimplePatchShadow(
        offset=(0.5, -0.5), shadow_rgbFace="#555555", alpha=0.25)])

    ax_a.axvline(0, color="black", linewidth=1, linestyle="-", zorder=2, alpha=0.5)
    mean_val = float(np.mean(vals))
    ax_a.plot(mean_val, 1.0, marker="D", color="#757575", markersize=12,
              zorder=5, markeredgecolor="white", markeredgewidth=1.2)

    # Unicode lambda: \u03bb, arrow: \u2192
    arrow = "\u2192"
    lam = "\u03bb"
    dir_parts = direction_label.split("->")
    if len(dir_parts) == 2:
        title_str = f"A.  {dir_parts[0].strip()} {arrow} {dir_parts[1].strip()} coupling"
    else:
        title_str = f"A.  {direction_label}"
    ax_a.set_title(title_str, fontsize=20, fontweight="bold", loc="left", pad=10)
    ax_a.set_xlabel(f"{lam} (posterior mean)", fontsize=18)
    ax_a.set_yticks([])
    ax_a.tick_params(axis="x", labelsize=16)

    # Population annotation
    if prob_neg is not None:
        pop_text = (f"Population {lam} = {pop_mean:.3f}\n"
                    f"[{pop_ci_lo:.3f}, {pop_ci_hi:.3f}]\n"
                    f"P({lam} < 0) = {prob_neg:.3f}")
    else:
        pop_text = (f"Population {lam} = {pop_mean:.3f}\n"
                    f"[{pop_ci_lo:.3f}, {pop_ci_hi:.3f}]")
    ax_a.text(0.02, 0.98, pop_text, transform=ax_a.transAxes,
              fontsize=14, va="top", ha="left", color="black", fontweight="bold")

    # Stats annotation
    med = float(np.median(vals))
    q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
    stats_text = (f"N = {n}\n"
                  f"mean = {mean_val:.3f}\n"
                  f"median = {med:.3f}\n"
                  f"IQR = [{q25:.3f}, {q75:.3f}]")
    ax_a.text(0.98, 0.98, stats_text, transform=ax_a.transAxes,
              fontsize=13, va="top", ha="right", color="#333333",
              fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor="#CCCCCC", alpha=0.9))

    legend_handles = [
        Line2D([0], [0], color="#D32F2F", linewidth=2.5, label="Median"),
        Line2D([0], [0], marker="D", color="#757575", markersize=10,
               markeredgecolor="white", markeredgewidth=0.8, linestyle="",
               label="Mean"),
        Line2D([0], [0], marker="o", color="#42A5F5", markersize=8,
               markeredgecolor="#0D47A1", markeredgewidth=0.8, linestyle="",
               label=f"Individual ({lam} < 0)"),
        Line2D([0], [0], marker="o", color="#EF5350", markersize=8,
               markeredgecolor="#B71C1C", markeredgewidth=0.8, linestyle="",
               label=f"Individual ({lam} > 0)"),
    ]
    ax_a.legend(handles=legend_handles, loc="lower right", fontsize=13,
                framealpha=0.9, edgecolor="#CCCCCC")

    # ---- Panel B: Forest ----
    for k in range(n):
        color = "#1565C0" if vals[k] < 0 else "#D32F2F"
        excludes_zero = (lo[k] > 0) or (hi[k] < 0)
        lw = 1.8 if excludes_zero else 0.7
        al = 0.7 if excludes_zero else 0.4
        ax_b.plot([lo[k], hi[k]], [k, k], color=color, linewidth=lw, alpha=al,
                  zorder=2)
        ax_b.plot(vals[k], k, marker="|", color=color,
                  markersize=3.5, alpha=0.8, zorder=3, markeredgewidth=0.9)

    ax_b.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5, zorder=1)
    ax_b.axvline(mean_val, color="#757575", linewidth=2, linestyle="--",
                 alpha=0.7, zorder=4)

    if len(dir_parts) == 2:
        title_str_b = f"B.  {dir_parts[0].strip()} {arrow} {dir_parts[1].strip()} coupling"
    else:
        title_str_b = f"B.  {direction_label}"
    ax_b.set_title(title_str_b, fontsize=20, fontweight="bold", loc="left", pad=10)
    ax_b.set_xlabel(f"{lam} (posterior mean \u00b1 95% CrI)", fontsize=18)
    ax_b.set_yticks([])
    ax_b.set_ylim(-1, n + 1)
    ax_b.tick_params(axis="x", labelsize=16)

    # Align panel heights
    fig.canvas.draw()
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    top = max(pos_a.y1, pos_b.y1)
    bottom = min(pos_a.y0, pos_b.y0)
    ax_a.set_position([pos_a.x0, bottom, pos_a.width, top - bottom])
    ax_b.set_position([pos_b.x0, bottom, pos_b.width, top - bottom])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_step04(verbose: bool = True, refit: bool = False):
    """Fit the coupling model and LOO-CV, produce all Step 03 outputs."""
    from coupling_model import (
        fit_bayesian_varx1,
        extract_results,
    )
    import arviz as az

    if verbose:
        print("=" * 70)
        print("STEP 03 — Fit Bayesian VARX(1) coupling model + LOO-CV")
        print("=" * 70)
        print(f"  Input: {IN_PROCESSED_CSV}")

    os.makedirs(DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    # Check whether saved derivatives exist
    saved_exist = (os.path.exists(OUT_DRAWS_NPZ) and os.path.exists(OUT_PERSON_CSV)
                   and os.path.exists(OUT_TABLE4_CSV) and os.path.exists(OUT_LOO_CSV)
                   and os.path.exists(OUT_TEXT_CSV))
    if not refit and not saved_exist:
        if verbose:
            print("  Saved derivatives not found — forcing refit.")
        refit = True

    if not refit and saved_exist:
        # ==============================================================
        # REPLOT MODE: load saved derivatives, regenerate figures only
        # ==============================================================
        if verbose:
            print("  WARNING: Running in replot mode -- loading saved derivatives.")
            print("  If you have changed upstream data or code, re-run with --refit.")

        person_df = pd.read_csv(OUT_PERSON_CSV)
        table4 = pd.read_csv(OUT_TABLE4_CSV)
        loo_df = pd.read_csv(OUT_LOO_CSV)
        text_df = pd.read_csv(OUT_TEXT_CSV)

        # Reconstruct results dict from Table 4 for figure generation
        results = {}
        for _, row in table4.iterrows():
            key = row["Parameter"]
            results[f"{key}_mean"] = row["Estimate"]
            results[f"{key}_sd"] = row["SD"]
            results[f"{key}_ci_lo"] = row["CrI_lo"]
            results[f"{key}_ci_hi"] = row["CrI_hi"]
            results[f"{key}_prob_neg"] = row["P_neg"]
        # Also read rhat_max from text numbers
        rhat_row = text_df[text_df["metric"] == "rhat_max"]
        if len(rhat_row) > 0:
            results["rhat_max"] = float(rhat_row["value"].iloc[0])

        if verbose:
            print(f"  Loaded Table 4: {OUT_TABLE4_CSV}")
            print(f"  Loaded person coupling: {OUT_PERSON_CSV}")
            print(f"  Loaded LOO: {OUT_LOO_CSV}")
    else:
        # ==============================================================
        # FULL MCMC FIT
        # ==============================================================
        df_full, model_df, unique_ids, id_map = load_data(IN_PROCESSED_CSV)
        n_persons = len(unique_ids)
        n_obs = len(model_df)

        # ==============================================================
        # 1. Fit the full VARX(1)
        # ==============================================================
        if verbose:
            print(f"\n  Fitting full model (include_agesex=True)...")
            print(f"    {n_persons} subjects, {n_obs} observations")
            print(f"    MCMC: 4 chains x 2000 draws, 2000 tuning")

        idata, sub_df, valid_ids = fit_bayesian_varx1(
            model_df, unique_ids, id_map,
            include_agesex=True,
            progressbar=True,
        )

        # ==============================================================
        # 2. Extract population parameters -> Table 4
        # ==============================================================
        results = extract_results(idata)
        results["n_persons"] = n_persons
        results["n_obs"] = n_obs

        table4_rows = []
        for key, desc in [
            ("a0", "Pain intercept (mu_p)"),
            ("a1", "Pain autoregression (phi_p)"),
            ("a2", "Sleep->Pain coupling (lambda_sp)"),
            ("a3", "Localization->Pain direct (delta_p)"),
            ("a4", "Sleep x Localization->Pain (omega_sp)"),
            ("b0", "Sleep intercept (mu_s)"),
            ("b1", "Pain->Sleep coupling (lambda_ps)"),
            ("b2", "Sleep autoregression (phi_s)"),
            ("b3", "Localization->Sleep direct (delta_s)"),
            ("b4", "Pain x Localization->Sleep (omega_ps)"),
            ("tau_sp", "SD: Sleep->Pain random slope"),
            ("tau_ps", "SD: Pain->Sleep random slope"),
            ("sigma_pain", "Innovation SD (pain)"),
            ("sigma_sleep", "Innovation SD (sleep)"),
            ("rho_innov", "Innovation correlation (rho)"),
        ]:
            table4_rows.append({
                "Parameter": key,
                "Description": desc,
                "Estimate": results.get(f"{key}_mean"),
                "SD": results.get(f"{key}_sd"),
                "CrI_lo": results.get(f"{key}_ci_lo"),
                "CrI_hi": results.get(f"{key}_ci_hi"),
                "P_neg": results.get(f"{key}_prob_neg"),
            })

        table4 = pd.DataFrame(table4_rows)
        table4.to_csv(OUT_TABLE4_CSV, index=False)
        if verbose:
            print(f"\n  Saved Table 4: {OUT_TABLE4_CSV}")

        # ==============================================================
        # 3. Per-person coupling slopes
        # ==============================================================
        a2_flat = idata.posterior["a2"].values.reshape(-1)
        b1_flat = idata.posterior["b1"].values.reshape(-1)
        u_sp_flat = idata.posterior["u_sp"].values.reshape(-1, n_persons)
        u_ps_flat = idata.posterior["u_ps"].values.reshape(-1, n_persons)

        lambda_sp_i = a2_flat[:, None] + u_sp_flat
        lambda_ps_i = b1_flat[:, None] + u_ps_flat

        person_rows = []
        for i, pid in enumerate(unique_ids):
            sp = lambda_sp_i[:, i]
            ps = lambda_ps_i[:, i]
            person_rows.append({
                "ID": pid,
                "beta_sp_mean": float(np.mean(sp)),
                "beta_sp_ci_lo": float(np.percentile(sp, 2.5)),
                "beta_sp_ci_hi": float(np.percentile(sp, 97.5)),
                "beta_sp_prob_neg": float((sp < 0).mean()),
                "beta_ps_mean": float(np.mean(ps)),
                "beta_ps_ci_lo": float(np.percentile(ps, 2.5)),
                "beta_ps_ci_hi": float(np.percentile(ps, 97.5)),
                "beta_ps_prob_neg": float((ps < 0).mean()),
            })
        person_df = pd.DataFrame(person_rows)
        person_df.to_csv(OUT_PERSON_CSV, index=False)
        if verbose:
            print(f"  Saved per-person coupling: {OUT_PERSON_CSV}")

        # ==============================================================
        # 4. Save posterior draws for downstream steps
        # ==============================================================
        a4_draws = idata.posterior["a4"].values.flatten()
        b4_draws = idata.posterior["b4"].values.flatten()
        u_sp_mean = idata.posterior["u_sp"].values.mean(axis=(0, 1))
        u_ps_mean = idata.posterior["u_ps"].values.mean(axis=(0, 1))
        obs_pid_idx = model_df["pid_idx"].values.astype(int)
        obs_contrast = model_df["contrast_within_lag1"].values.astype(float)
        contrast_vals = model_df["contrast_within_lag1"].dropna().values.astype(float)

        np.savez(
            OUT_DRAWS_NPZ,
            a2_draws=a2_flat, a4_draws=a4_draws,
            b1_draws=b1_flat, b4_draws=b4_draws,
            u_sp_mean=u_sp_mean, u_ps_mean=u_ps_mean,
            obs_pid_idx=obs_pid_idx, obs_contrast=obs_contrast,
            contrast_vals=contrast_vals,
        )
        if verbose:
            print(f"  Saved posterior draws: {OUT_DRAWS_NPZ}")

        # ==============================================================
        # 5. LOO-CV model comparison
        # ==============================================================
        if verbose:
            print("\n" + "=" * 70)
            print("LOO-CV MODEL COMPARISON")
            print("=" * 70)
            print("  Fitting 4 nested models...")

        # Free the main-fit idata before LOO
        del idata
        gc.collect()

        # The LOO comparison function in coupling_model.py expects a
        # data_dir with a processed CSV. We point it at our derivatives
        # folder where step02_processed_long.csv lives, but the function
        # looks for processed_data_contrast.csv or processed_data.csv.
        # Easiest: symlink or pass the data directly. Instead, we
        # inline the LOO logic here using the same primitives.

        idata_kwargs = {"log_likelihood": True}

        model_configs = [
            ("full",  True,  True,  "full model (both SP and PS paths)"),
            ("no_PS", True,  False, "no_PS model (SP only)"),
            ("no_SP", False, True,  "no_SP model (PS only)"),
            ("null",  False, False, "null model (no coupling)"),
        ]

        loo_dict = {}
        for name, inc_sp, inc_ps, desc in model_configs:
            if verbose:
                print(f"\n  Fitting {desc}...")
            idata_loo, _, _ = fit_bayesian_varx1(
                model_df, unique_ids, id_map,
                include_agesex=True,
                include_sp=inc_sp, include_ps=inc_ps,
                idata_kwargs=idata_kwargs,
                cores=1,
                progressbar=True,
            )
            if verbose:
                print(f"  Computing LOO for {name}...")
            ll = idata_loo.log_likelihood
            if "y_joint" not in ll.data_vars:
                import xarray as xr
                joint_vals = ll["y_pain"].values + ll["y_sleep"].values
                dims = ll["y_pain"].dims
                coords = {d: ll[d] for d in dims if d in ll.coords}
                idata_loo.log_likelihood = idata_loo.log_likelihood.drop_vars(
                    ["y_pain", "y_sleep"]
                )
                idata_loo.log_likelihood["y_joint"] = xr.DataArray(
                    joint_vals, dims=dims, coords=coords
                )
                del joint_vals
                gc.collect()
            loo_dict[name] = az.loo(idata_loo, pointwise=True, var_name="y_joint")
            del idata_loo, ll
            gc.collect()

        # Pairwise comparisons
        pairs = [
            ("full", "no_PS"),
            ("no_SP", "null"),
            ("full", "no_SP"),
            ("no_PS", "null"),
        ]
        loo_rows = []
        for a, b in pairs:
            d = loo_dict[a].loo_i.values - loo_dict[b].loo_i.values
            n = len(d)
            delta = float(d.sum())
            se = float(np.sqrt(n * d.var(ddof=1)))
            ratio = delta / se if se > 0 else float("nan")
            loo_rows.append({
                "model_a": a, "model_b": b,
                "delta_elpd": delta, "se": se, "delta_over_se": ratio,
            })
            if verbose:
                print(f"  {a:>6} vs {b:<6}: ΔELPD = {delta:+.2f}, "
                      f"SE = {se:.2f}, Δ/SE = {ratio:+.2f}")

        # Pareto k-hat diagnostics (full model)
        khat = loo_dict["full"].pareto_k.values
        k_max = float(np.nanmax(khat))
        k_bad = int((khat > 0.7).sum())
        n_loo_obs = len(khat)
        if verbose:
            print(f"  Pareto k-hat (full): max={k_max:.2f}, "
                  f"{k_bad}/{n_loo_obs} > 0.7")

        loo_df = pd.DataFrame(loo_rows)
        loo_df.to_csv(OUT_LOO_CSV, index=False)
        if verbose:
            print(f"\n  Saved LOO: {OUT_LOO_CSV}")

    # ==================================================================
    # 6. Text numbers (always regenerated)
    # ==================================================================
    text_rows = []

    def _t(metric, value, note=""):
        text_rows.append({"metric": metric, "value": str(value), "note": note})

    # Key coupling parameters
    _t("lambda_sp_mean", f"{results['a2_mean']:.4f}")
    _t("lambda_sp_ci", f"[{results['a2_ci_lo']:.4f}, {results['a2_ci_hi']:.4f}]")
    _t("lambda_sp_p_neg", f"{results['a2_prob_neg']:.3f}")
    _t("lambda_ps_mean", f"{results['b1_mean']:.4f}")
    _t("lambda_ps_ci", f"[{results['b1_ci_lo']:.4f}, {results['b1_ci_hi']:.4f}]")
    _t("lambda_ps_p_neg", f"{results['b1_prob_neg']:.3f}")
    _t("rho_innov_mean", f"{results['rho_innov_mean']:.4f}")
    _t("rho_innov_ci", f"[{results['rho_innov_ci_lo']:.4f}, {results['rho_innov_ci_hi']:.4f}]")
    _t("rho_innov_p_neg", f"{results['rho_innov_prob_neg']:.3f}")
    _t("tau_sp", f"{results['tau_sp_mean']:.4f}")
    _t("tau_sp_ci", f"[{results['tau_sp_ci_lo']:.4f}, {results['tau_sp_ci_hi']:.4f}]")
    _t("tau_ps", f"{results['tau_ps_mean']:.4f}")
    _t("tau_ps_ci", f"[{results['tau_ps_ci_lo']:.4f}, {results['tau_ps_ci_hi']:.4f}]")
    _t("rhat_max", f"{results['rhat_max']:.3f}")

    # Person-level range statistics
    _t("person_ps_range",
       f"[{person_df['beta_ps_mean'].min():.3f}, {person_df['beta_ps_mean'].max():.3f}]")
    _t("person_ps_sd", f"{person_df['beta_ps_mean'].std():.3f}")
    n_ps_cred = int((person_df['beta_ps_prob_neg'] > 0.95).sum())
    _t("person_ps_n_credible_neg", str(n_ps_cred))
    _t("person_sp_range",
       f"[{person_df['beta_sp_mean'].min():.3f}, {person_df['beta_sp_mean'].max():.3f}]")
    _t("person_sp_sd", f"{person_df['beta_sp_mean'].std():.3f}")
    n_sp_cred = int((person_df['beta_sp_prob_neg'] > 0.95).sum())
    _t("person_sp_n_credible_neg", str(n_sp_cred))

    # LOO numbers
    for _, row in loo_df.iterrows():
        _t(f"loo_{row['model_a']}_vs_{row['model_b']}_delta",
           f"{row['delta_elpd']:+.2f}")
        _t(f"loo_{row['model_a']}_vs_{row['model_b']}_se",
           f"{row['se']:.2f}")
        _t(f"loo_{row['model_a']}_vs_{row['model_b']}_ratio",
           f"{row['delta_over_se']:+.2f}")
    # Pareto k-hat from text numbers if in replot mode
    khat_row = text_df[text_df["metric"] == "pareto_khat_max"] if not refit and saved_exist else None
    if khat_row is not None and len(khat_row) > 0:
        _t("pareto_khat_max", khat_row["value"].iloc[0])
        above_row = text_df[text_df["metric"] == "pareto_khat_above_07"]
        _t("pareto_khat_above_07", above_row["value"].iloc[0] if len(above_row) > 0 else "")
        nobs_row = text_df[text_df["metric"] == "pareto_khat_n_obs"]
        _t("pareto_khat_n_obs", nobs_row["value"].iloc[0] if len(nobs_row) > 0 else "")
    else:
        _t("pareto_khat_max", f"{k_max:.2f}")
        _t("pareto_khat_above_07", str(k_bad))
        _t("pareto_khat_n_obs", str(n_loo_obs))

    text_df = pd.DataFrame(text_rows)
    text_df.to_csv(OUT_TEXT_CSV, index=False)
    if verbose:
        print(f"  Saved text numbers: {OUT_TEXT_CSV}")

    # ==================================================================
    # 7. Figures 2 & 3 — person-level coupling (always regenerated)
    # ==================================================================
    _generate_coupling_figure(
        person_df,
        pop_mean=results["b1_mean"],
        pop_ci_lo=results["b1_ci_lo"],
        pop_ci_hi=results["b1_ci_hi"],
        col_mean="beta_ps_mean", col_ci_lo="beta_ps_ci_lo",
        col_ci_hi="beta_ps_ci_hi", col_prob="beta_ps_prob_neg",
        direction_label="Pain -> Sleep",
        out_path=OUT_FIG2,
        prob_neg=results["b1_prob_neg"],
    )
    if verbose:
        print(f"  Saved Figure 2: {OUT_FIG2}")

    _generate_coupling_figure(
        person_df,
        pop_mean=results["a2_mean"],
        pop_ci_lo=results["a2_ci_lo"],
        pop_ci_hi=results["a2_ci_hi"],
        col_mean="beta_sp_mean", col_ci_lo="beta_sp_ci_lo",
        col_ci_hi="beta_sp_ci_hi", col_prob="beta_sp_prob_neg",
        direction_label="Sleep -> Pain",
        out_path=OUT_FIG3,
        prob_neg=results["a2_prob_neg"],
    )
    if verbose:
        print(f"  Saved Figure 3: {OUT_FIG3}")

    # Print key results
    if verbose:
        print(f"\n  Key results:")
        print(f"    lambda_sp = {results['a2_mean']:+.4f} "
              f"[{results['a2_ci_lo']:+.4f}, {results['a2_ci_hi']:+.4f}] "
              f"P(<0)={results['a2_prob_neg']:.3f}")
        print(f"    lambda_ps = {results['b1_mean']:+.4f} "
              f"[{results['b1_ci_lo']:+.4f}, {results['b1_ci_hi']:+.4f}] "
              f"P(<0)={results['b1_prob_neg']:.3f}")
        print(f"    rho = {results['rho_innov_mean']:+.4f}")
        print(f"    R-hat max: {results['rhat_max']:.3f}")
        print("\n" + "=" * 70)
        print("STEP 03 COMPLETE")
        print("=" * 70)

    generate_text_paragraphs(verbose)


def generate_text_paragraphs(verbose: bool = True) -> None:
    """Generate step04_text.md with the manuscript paragraphs for Results
    section 3.2 (population coupling estimates), populated from saved CSVs.

    Reads all computed numbers from text_numbers.csv, table4 CSV, LOO CSV,
    and person_coupling CSV and writes fully formatted markdown paragraphs
    into the results directory.
    """
    OUT_TEXT_MD = os.path.join(STEP_RESULTS_DIR, "step04_text.md")

    for required in (OUT_TEXT_CSV, OUT_TABLE4_CSV, OUT_LOO_CSV, OUT_PERSON_CSV):
        if not os.path.exists(required):
            if verbose:
                print(f"  SKIP: {os.path.basename(required)} not found — run step04 first")
            return

    if verbose:
        print("  Generating step04_text.md ...")

    text_df = pd.read_csv(OUT_TEXT_CSV)
    v = dict(zip(text_df["metric"], text_df["value"]))
    table4 = pd.read_csv(OUT_TABLE4_CSV)
    loo_df = pd.read_csv(OUT_LOO_CSV, keep_default_na=False)
    person_df = pd.read_csv(OUT_PERSON_CSV)

    # ----- Parse values from text_numbers.csv -----
    lps_mean = float(v["lambda_ps_mean"])
    lps_ci = v["lambda_ps_ci"]
    lps_pneg = float(v["lambda_ps_p_neg"])
    tau_ps = float(v["tau_ps"])
    ps_range = v["person_ps_range"]
    ps_sd = float(v["person_ps_sd"])
    n_ps_cred = int(v["person_ps_n_credible_neg"])

    lsp_mean = float(v["lambda_sp_mean"])
    lsp_ci = v["lambda_sp_ci"]
    lsp_pneg = float(v["lambda_sp_p_neg"])
    tau_sp = float(v["tau_sp"])
    sp_range = v["person_sp_range"]
    sp_sd = float(v["person_sp_sd"])
    n_sp_cred = int(v["person_sp_n_credible_neg"])

    rho_mean = float(v["rho_innov_mean"])
    rho_ci = v["rho_innov_ci"]
    rho_pneg = float(v["rho_innov_p_neg"])

    n_persons = len(person_df)

    # ----- LOO values -----
    def _loo(model_a, model_b):
        row = loo_df[(loo_df["model_a"] == model_a) & (loo_df["model_b"] == model_b)]
        delta = float(row["delta_elpd"].iloc[0])
        se = float(row["se"].iloc[0])
        ratio = float(row["delta_over_se"].iloc[0])
        return delta, se, ratio

    loo_full_noPS_d, loo_full_noPS_se, loo_full_noPS_r = _loo("full", "no_PS")
    loo_noSP_null_d, loo_noSP_null_se, loo_noSP_null_r = _loo("no_SP", "null")
    loo_full_noSP_d, loo_full_noSP_se, loo_full_noSP_r = _loo("full", "no_SP")
    loo_noPS_null_d, loo_noPS_null_se, loo_noPS_null_r = _loo("no_PS", "null")

    khat_max = float(v["pareto_khat_max"])
    khat_above = int(v["pareto_khat_above_07"])
    khat_nobs = int(v["pareto_khat_n_obs"])
    khat_pct = khat_above / khat_nobs * 100

    # ----- Paragraph 1: Pain-to-sleep coupling -----
    para1 = (
        f"**Table 4** presents the population parameters from the Bayesian "
        f"VARX(1) model. The pain-to-sleep pathway was the dominant coupling "
        f"direction. The population mean "
        f"($\\hat{{\\lambda}}_{{ps}}={lps_mean:.3f}$, "
        f"$P(\\hat{{\\lambda}}_{{ps}}<0)={lps_pneg:.3f}$, "
        f"95% CrI {lps_ci}) "
        f"indicated that a one-unit within-person increase in general pain "
        f"predicted a {abs(lps_mean):.3f}-unit decrease in next-quarter sleep "
        f"quality. The random effect SD was substantial "
        f"($\\hat{{\\tau}}_{{ps}}={tau_ps:.3f}$), reflecting meaningful "
        f"between-person heterogeneity: person-level posterior means ranged "
        f"from {ps_range} (SD = {ps_sd:.3f}), and {n_ps_cred} of "
        f"{n_persons} participants ({n_ps_cred/n_persons*100:.1f}%) had "
        f"$P(\\hat{{\\lambda}}_{{ps,i}}<0)>0.95$ (**Figure 2**)."
    )

    # ----- Paragraph 2: Sleep-to-pain coupling -----
    para2 = (
        f"The sleep-to-pain pathway did not reach credibility at the "
        f"population level "
        f"($\\hat{{\\lambda}}_{{sp}}={lsp_mean:.3f}$, "
        f"$P(\\hat{{\\lambda}}_{{sp}}<0)={lsp_pneg:.3f}$, "
        f"95% CrI {lsp_ci}). "
        f"The random effect SD ($\\hat{{\\tau}}_{{sp}}={tau_sp:.3f}$) was "
        f"smaller but non-negligible, with person-level posteriors ranging "
        f"from {sp_range} (SD = {sp_sd:.3f}). "
        f"{n_sp_cred} participants ({n_sp_cred/n_persons*100:.1f}%) had "
        f"$P(\\hat{{\\lambda}}_{{sp,i}}<0)>0.95$ (**Figure 3**)."
    )

    # ----- Paragraph 3: Innovation correlation -----
    para3 = (
        f"Same-quarter innovations in pain and sleep quality were negatively "
        f"correlated ($\\hat{{\\rho}}={rho_mean:.3f}$, $P(<0)={rho_pneg:.3f}$, "
        f"95% CrI {rho_ci}), indicating that within a given quarter, "
        f"unexplained increases in pain co-occurred with unexplained decreases "
        f"in sleep quality, or vice versa. Because this is a residual "
        f"correlation after controlling for all lagged effects, it may reflect "
        f"within-quarter bidirectional processes, shared contemporaneous "
        f"perturbations (e.g., a flare event affecting both pain and sleep "
        f"within the same assessment window), or measurement timing effects."
    )

    # ----- Paragraph 4: Model comparison -----
    para4 = (
        f"The model comparison (see Methods: Model comparison) confirmed the "
        f"dominance of the pain-to-sleep pathway. Including pain-to-sleep "
        f"coupling improved predictive accuracy substantially "
        f"($\\Delta LOO_{{elpd}}$ = {loo_full_noPS_d:+.1f}, "
        f"SE = {loo_full_noPS_se:.1f}, "
        f"$\\Delta$/SE = {loo_full_noPS_r:.2f} for full vs. no-PS; "
        f"$\\Delta LOO_{{elpd}}$ = {loo_noSP_null_d:+.1f}, "
        f"SE = {loo_noSP_null_se:.1f}, "
        f"$\\Delta$/SE = {loo_noSP_null_r:.2f} for no-SP vs. null), "
        f"whereas including sleep-to-pain coupling yielded negligible "
        f"improvement "
        f"($\\Delta LOO_{{elpd}}$ = {loo_full_noSP_d:+.1f}, "
        f"SE = {loo_full_noSP_se:.1f}, "
        f"$\\Delta$/SE = {loo_full_noSP_r:.2f} for full vs. no-SP; "
        f"$\\Delta LOO_{{elpd}}$ = {loo_noPS_null_d:+.1f}, "
        f"SE = {loo_noPS_null_se:.1f}, "
        f"$\\Delta$/SE = {loo_noPS_null_r:.2f} for no-PS vs. null). "
        f"Pareto $\\hat{{k}}$ diagnostics confirmed the reliability of the "
        f"LOO-CV approximation: only {khat_above} of {khat_nobs} observations "
        f"({khat_pct:.1f}%) exceeded the 0.7 threshold (maximum "
        f"$\\hat{{k}}$ = {khat_max:.2f}). By the conventional threshold of "
        f"$|\\Delta/SE|>2$, pain-to-sleep coupling substantially improved "
        f"prediction while sleep-to-pain coupling did not."
    )

    text = f"""\
## Results > 3.2 Population coupling estimates
### Paragraph 1 (pain-to-sleep coupling)

{para1}

### Paragraph 2 (sleep-to-pain coupling)

{para2}

### Paragraph 3 (innovation correlation)

{para3}

### Paragraph 4 (model comparison)

{para4}
"""

    with open(OUT_TEXT_MD, "w") as f:
        f.write(text)

    if verbose:
        print(f"    Saved: {OUT_TEXT_MD}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 03 — fit the Bayesian VARX(1) coupling model + LOO-CV."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--refit", action="store_true",
        help="Re-run computation from scratch instead of loading saved derivatives",
    )
    args = parser.parse_args()
    run_step04(verbose=not args.quiet, refit=args.refit)


if __name__ == "__main__":
    main()
