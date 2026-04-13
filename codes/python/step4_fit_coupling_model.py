"""
Step 3 — Fit the Bayesian VARX(1) coupling model + LOO-CV.
======================================================================

Input:  derivatives/step2_processed_long.csv
Output:
  derivatives/
    step3_posterior_draws.npz     — raw posterior arrays for downstream steps
    step3_person_coupling.csv     — per-person lambda_sp / lambda_ps
  results/
    step3_table4_coupling.csv     — Table 4: population parameters
    step3_loo_comparison.csv      — LOO-CV pairwise comparisons
    step3_text_numbers.csv        — every number stated in the text

This step fits the bivariate VARX(1) coupling model to the
within-person deviations produced by Step 2. It then fits three
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
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step4")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step4")
os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

LIB_DIR = os.path.join(HERE, "lib")
sys.path.insert(0, LIB_DIR)

IN_PROCESSED_CSV = os.path.join(DERIV_DIR, "step3", "step3_processed_long.csv")

# Derivatives
OUT_DRAWS_NPZ = os.path.join(STEP_DERIV_DIR, "step4_posterior_draws.npz")
OUT_PERSON_CSV = os.path.join(STEP_DERIV_DIR, "step4_person_coupling.csv")

# Results
OUT_TABLE4_CSV = os.path.join(STEP_RESULTS_DIR, "step4_table4_coupling.csv")
OUT_LOO_CSV = os.path.join(STEP_RESULTS_DIR, "step4_loo_comparison.csv")
OUT_TEXT_CSV = os.path.join(STEP_RESULTS_DIR, "step4_text_numbers.csv")
OUT_FIG2 = os.path.join(STEP_RESULTS_DIR, "step4_figure2_ps_coupling.png")
OUT_FIG3 = os.path.join(STEP_RESULTS_DIR, "step4_figure3_sp_coupling.png")


# =====================================================================
# Data loading
# =====================================================================

def load_data(csv_path: str):
    """Load Step 2 output and prepare for fit_bayesian_varx1.

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


def run_step4(verbose: bool = True):
    """Fit the coupling model and LOO-CV, produce all Step 3 outputs."""
    from coupling_model import (
        fit_bayesian_varx1,
        extract_results,
    )
    import arviz as az

    if verbose:
        print("=" * 70)
        print("STEP 3 — Fit Bayesian VARX(1) coupling model + LOO-CV")
        print("=" * 70)
        print(f"  Input: {IN_PROCESSED_CSV}")

    os.makedirs(DERIV_DIR, exist_ok=True)
    os.makedirs(STEP_DERIV_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)

    df_full, model_df, unique_ids, id_map = load_data(IN_PROCESSED_CSV)
    n_persons = len(unique_ids)
    n_obs = len(model_df)

    # ==================================================================
    # 1. Fit the full VARX(1)
    # ==================================================================
    if verbose:
        print(f"\n  Fitting full model (include_agesex=True)...")
        print(f"    {n_persons} subjects, {n_obs} observations")
        print(f"    MCMC: 4 chains x 2000 draws, 2000 tuning")

    idata, sub_df, valid_ids = fit_bayesian_varx1(
        model_df, unique_ids, id_map,
        include_agesex=True,
        progressbar=True,
    )

    # ==================================================================
    # 2. Extract population parameters -> Table 4
    # ==================================================================
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

    # ==================================================================
    # 3. Per-person coupling slopes
    # ==================================================================
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

    # ==================================================================
    # 4. Save posterior draws for downstream steps
    # ==================================================================
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

    # ==================================================================
    # 5. LOO-CV model comparison
    # ==================================================================
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
    # folder where step2_processed_long.csv lives, but the function
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
    # 6. Text numbers
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
    _t("pareto_khat_max", f"{k_max:.2f}")
    _t("pareto_khat_above_07", str(k_bad))
    _t("pareto_khat_n_obs", str(n_loo_obs))

    text_df = pd.DataFrame(text_rows)
    text_df.to_csv(OUT_TEXT_CSV, index=False)
    if verbose:
        print(f"  Saved text numbers: {OUT_TEXT_CSV}")

    # ==================================================================
    # 7. Figures 2 & 3 — person-level coupling
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
        print("STEP 3 COMPLETE")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 3 — fit the Bayesian VARX(1) coupling model + LOO-CV."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()
    run_step4(verbose=not args.quiet)


if __name__ == "__main__":
    main()
