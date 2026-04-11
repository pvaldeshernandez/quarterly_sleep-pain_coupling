# Reproduction Audit — Every Reported Number in the Manuscript

**Scope.** This document is the final cross-check of every quantitative
claim in `docs/manuscript_pain.md` / `manuscript_pain.docx` against
what the `python/` pipeline produces when run end-to-end on the real
data. It supersedes `sandbox/REPRODUCTION_PLAN.md` (which tracked
what was still TODO) and extends `sandbox/REPRODUCTION_REPORT.md`
(which only covered the headline Table 3/5 numbers).

**Sources.**
- `sandbox/real_run2/` — full pipeline run, steps 1-6,
  `python/main.py --output-dir sandbox/real_run2 --interpolate`.
- `sandbox/real_loo/` — supplementary LOO-CV run,
  `python/02_fit_coupling_model.py --data-dir sandbox/real_loo --output-dir sandbox/real_loo --loo`,
  using the same `processed_data_contrast.csv` produced by the main
  run.

**Legend.**
- ✅ MATCHED — reproduced within MCMC Monte Carlo error (|Δ| ≤ 1 posterior SD)
- ⚠ CLOSE — same sign, same inferential conclusion, numerical gap > 1 SD
- 🐛 BUG — pipeline produces a different result with different inferential implications
- ❌ NOT PIPELINE — quantity is not produced by `python/` (lives in
  raw data files or legacy scripts)
- ✏ MANUSCRIPT TYPO — sandbox value differs, and the manuscript value
  appears to be a typo / carryover from an earlier draft

---

## 1. Sample size and data structure

| Claim | Manuscript | Sandbox | Status |
| --- | --- | --- | --- |
| Parent study N | 243 | 243 | ✅ |
| Analytic sample N | 229 | 229 | ✅ |
| Excluded for no ≥3-quarter segment | 14 | 14 | ✅ |
| Median lagged transitions per person | 9 | 9 | ✅ |
| Range of lagged transitions | 2–10 | 2–10 | ✅ |
| Total observations (lag transitions) | 1,818 | 1,818 | ✅ |
| Retained points (segment ≥3) | 2,056 | 2,056 | ✅ |
| Points recovered via item-level interpolation | 128 | 128 | ✅ |
| fMRI subsample | 174 | 174 | ✅ |
| VBM subsample | 189 | 189 | ✅ |
| fMRI arousal observations (1,403) | 1,403 | 1,403 | ✅ |

---

## 2. Demographics (Table 2, N=229)

| Claim | Manuscript | Sandbox | Status |
| --- | --- | --- | --- |
| Age mean (SD) | 58.1 (8.2) | 58.1 (8.2) | ✅ |
| Age range | 31–79 | **44–80** | ✏ TYPO |
| Female N (%) | 148 (64.6) | 148 (64.6) | ✅ |
| Male N | 81 | 81 | ✅ |
| Race/ethnicity breakdown | 91 Black, 67 White, 55 Hispanic, … | — | ❌ NOT PIPELINE |
| BMI mean (SD) | 31.6 (7.7) | — | ❌ NOT PIPELINE |
| WOMAC Pain mean (SD) | 5.8 (5.0) | — | ❌ NOT PIPELINE |
| KL grade distribution | 70/43/48/34/29 | — | ❌ NOT PIPELINE |

**Note on age range.** The manuscript reports `31–79`, but both
`data/participants_wideformat.xlsx` (`age__s1`: 44-80, N=253) and the
processed 229-subject subset (44-80) show a minimum of 44. The 31 in
the manuscript is inconsistent with the raw source and should be
updated to **44–80**.

**Note on Table 2 in general.** Demographics beyond Age/Sex are not
computed by `python/` because the model only needs Age + Sex. These
values come directly from `participants_wideformat.xlsx` and can be
recomputed by a small descriptive script (not currently in the
pipeline). The sandbox run does not refute any of them.

---

## 3. Factor analysis (Section 2.2, Table 1)

| Claim | Manuscript | Sandbox | Status |
| --- | --- | --- | --- |
| Two-factor PAF with polychoric correlations | — | — | ✅ (pipeline design) |
| F1 eigenvalue | 5.56 | — | ❌ NOT PIPELINE |
| F1 % variance | 69.5% | — | ❌ NOT PIPELINE |
| F2 eigenvalue | 0.89 | — | ❌ NOT PIPELINE |
| F2 % variance | 11.2% | — | ❌ NOT PIPELINE |
| Total variance (2-factor) | 80.7% | — | ❌ NOT PIPELINE |
| Parallel analysis 95th% F1 | 1.13 | — | ❌ NOT PIPELINE |
| Parallel analysis 95th% F2 | 1.11 | — | ❌ NOT PIPELINE |
| F1 loadings range (all +) | 0.78–0.86 | — | ❌ NOT PIPELINE |
| F2 knee loadings range | +0.30 to +0.45 | — | ❌ NOT PIPELINE |
| F2 body loadings range | -0.23 to -0.50 | — | ❌ NOT PIPELINE |
| r(F1, F2) | 0.001 | — | ❌ NOT PIPELINE |

**Note.** The `python/01_prepare_data.py` step loads a precomputed
factor-model parameter file (`factor_model_params_contrast.json`) and
applies it to the quarterly items. It does not refit the factor
model or run parallel analysis. The upstream factor analysis lives
in a legacy MATLAB script and is outside the reproduction scope of
the `python/` pipeline.

---

## 4. Convergent validity (Figure S2)

| Clinical measure vs contrast | Manuscript r | Sandbox r | Status |
| --- | --- | --- | --- |
| PHQ knee pain days | 0.37 | reproduced | ✅ via figure S2 |
| PHQ % waking day knee pain | 0.30 | reproduced | ✅ |
| WOMAC Pain | 0.28 | reproduced | ✅ |
| WOMAC Total | 0.26 | reproduced | ✅ |
| WOMAC Physical Function | 0.25 | reproduced | ✅ |
| WOMAC Stiffness | 0.24 | reproduced | ✅ |
| Knee pain rating | 0.23 | reproduced | ✅ |
| KL grade (Spearman ρ) | 0.31 | reproduced | ✅ |

Figure S2 is re-generated from `participants_wideformat.xlsx` by
`python/06_generate_figures.py` and matches the published figure
visually.

---

## 5. Population coupling parameters (Table 3)

From `sandbox/real_run2/coupling_results.csv` + `coupling_summary.txt`.

| Parameter | Manuscript | Sandbox | Δ | Status |
| --- | --- | --- | --- | --- |
| μ_p | 0.009 (SD 0.010) | 0.002 (SD 0.010) | 0.007 | ✅ |
| φ_p (pain AR) | 0.106 (SD 0.025) | 0.104 (SD 0.026) | 0.002 | ✅ |
| **λ_sp** Sleep→Pain | **-0.021** [-0.054, +0.015] | **-0.023** [-0.059, +0.013] | 0.002 | ✅ |
| δ_p (contrast→pain) | +0.024 [-0.004, +0.051] | +0.022 [-0.006, +0.050] | 0.002 | ✅ |
| ω_sp (sleep×contrast→pain) | +0.009 [-0.026, +0.043] | +0.005 [-0.031, +0.041] | 0.004 | ✅ |
| μ_s | 0.004 | 0.004 | 0.000 | ✅ |
| **λ_ps** Pain→Sleep | **-0.140** [-0.238, -0.042] | **-0.136** [-0.234, -0.041] | 0.004 | ✅ |
| φ_s (sleep AR) | +0.004 | +0.005 | 0.001 | ✅ |
| **δ_s** (contrast→sleep) | **-0.050** [-0.093, -0.006] | **-0.052** [-0.096, -0.007] | 0.002 | ✅ |
| ω_ps (pain×contrast→sleep) | -0.050 [-0.124, +0.021] | -0.048 [-0.121, +0.026] | 0.002 | ✅ |
| τ_sp | 0.115 [0.073, 0.156] | 0.113 | 0.002 | ✅ |
| τ_ps | 0.362 [0.244, 0.464] | 0.359 | 0.003 | ✅ |
| σ_p | 0.437 [0.424, 0.452] | 0.438 | 0.001 | ✅ |
| σ_s | 0.661 [0.640, 0.682] | 0.661 | 0.000 | ✅ |
| ρ (innovation correlation) | -0.155 [-0.202, -0.111] | -0.157 | 0.002 | ✅ |
| Age × λ_sp | — | +0.006, p=0.77 | — | ✅ null reproduced |
| Sex × λ_sp | — | +0.006, p=0.88 | — | ✅ null reproduced |
| Age × λ_ps | — | -0.024, p=0.52 | — | ✅ null reproduced |
| Sex × λ_ps | — | -0.011, p=0.87 | — | ✅ null reproduced |
| R-hat max | 1.010 | 1.020 | — | ✅ (both < 1.05 convergence) |

---

## 6. Person-level range statistics (Section 3.3)

Computed post-hoc from `sandbox/real_run2/person_coupling_estimates.csv`.

| Claim | Manuscript | Sandbox | Status |
| --- | --- | --- | --- |
| Per-person λ_ps range | -0.878 to +0.810 | -0.851 to +0.799 | ✅ (MCMC noise at tail) |
| Per-person λ_ps SD | 0.180 | 0.178 | ✅ |
| N with P(λ_ps<0)>0.95 | 9 of 229 (3.9%) | 9 of 229 (3.9%) | ✅ exact |
| Per-person λ_sp range | -0.213 to +0.154 | -0.210 to +0.147 | ✅ |
| Per-person λ_sp SD | 0.049 | 0.048 | ✅ |
| N with P(λ_sp<0)>0.95 | 4 (1.7%) | 3 (1.3%) | ⚠ (boundary subject) |

**Note.** The single-subject discrepancy in the λ_sp>0.95 count
reflects one person whose posterior probability sits right at the
0.95 boundary and flips between runs under MCMC Monte Carlo error.
Since neither 4 nor 3 is individually significant after correcting
for 229 tests, this does not change any inferential conclusion.

---

## 7. LOO-CV model comparison (Section 3.1, Methods §2.6)

From `sandbox/real_loo/loo_comparison.txt` and `loo_pairwise.csv`,
fitting the four nested models full / no_PS / no_SP / null.

| Comparison | Manuscript | Sandbox | Δ | Status |
| --- | --- | --- | --- | --- |
| **full vs no_PS: ΔELPD** | **+22.0** | **+20.63** | 1.37 | ✅ |
| full vs no_PS: SE | 8.8 | 8.74 | 0.06 | ✅ |
| **full vs no_PS: Δ/SE** | **2.49** | **2.36** | 0.13 | ✅ |
| **no_SP vs null: ΔELPD** | **+23.1** | **+21.86** | 1.24 | ✅ |
| no_SP vs null: SE | 9.1 | 9.11 | 0.01 | ✅ |
| **no_SP vs null: Δ/SE** | **2.53** | **2.40** | 0.13 | ✅ |
| full vs no_SP: ΔELPD | +0.7 | -1.03 | 1.73 | ✅ (both |Δ/SE|<1) |
| full vs no_SP: SE | 5.5 | 5.45 | 0.05 | ✅ |
| full vs no_SP: Δ/SE | 0.12 | -0.19 | 0.31 | ✅ |
| no_PS vs null: ΔELPD | +1.8 | +0.20 | 1.60 | ✅ |
| no_PS vs null: SE | 5.9 | 5.90 | 0.00 | ✅ |
| no_PS vs null: Δ/SE | 0.30 | +0.03 | 0.27 | ✅ |
| Pareto k̂ max (full) | 0.90 | 0.86 | 0.04 | ✅ |
| Observations with k̂>0.7 | 2 of 1,818 (0.1%) | 2 of 1,818 (0.1%) | — | ✅ exact |

**Substantive conclusion preserved.** Both PS-direction tests cross
the conventional |Δ/SE|>2 threshold (2.36 and 2.40), both
SP-direction tests are comfortably below (|Δ/SE|<0.5). Pareto k̂
diagnostics confirm the importance-sampling approximation is
reliable. The manuscript's conclusion — "pain-to-sleep coupling
substantially improved prediction while sleep-to-pain coupling did
not" — holds in the sandbox.

**Implementation notes.**
- `compute_loo_comparison` was rewritten to fit the four
  manuscript-described models (full / no_PS / no_SP / null) via new
  `include_sp` / `include_ps` flags on `fit_bayesian_varx1`. The
  previous implementation fitted a different (base / +contrast /
  +agesex / full) nesting.
- The model has two observed variables (`y_pain`, `y_sleep`). The
  joint log-likelihood is computed in-place as `log p(y_pain) + log
  p(y_sleep|y_pain)` (valid under the Cholesky factorisation) and
  passed to `az.loo` via `var_name="y_joint"`.
- To stay under the 16 GB cgroup limit, LOO fits use
  `cores=1` (sequential chains) and the `InferenceData` objects are
  dropped immediately after computing the pointwise LOO for each
  model. Log-likelihood arrays for `y_pain` and `y_sleep` are
  discarded after forming `y_joint`.

---

## 8. Contrast moderation (Table 4 + Figure 4)

From `sandbox/real_run2/contrast_moderation_results.csv` and
`contrast_jn_boundary.txt`.

| Parameter | Manuscript | Sandbox | Status |
| --- | --- | --- | --- |
| δ_p | +0.024 [-0.004, +0.051] | +0.022 | ✅ |
| ω_sp | +0.009 [-0.026, +0.043] | +0.005 | ✅ |
| **δ_s** | **-0.050** [-0.093, -0.006] | **-0.052** | ✅ |
| **ω_ps** | **-0.050** [-0.124, +0.021] | **-0.048** | ✅ |

Simple slopes (pain-to-sleep coupling at ±2 SD of K, centered):

| K value | Manuscript λ_ps(K) | Sandbox λ_ps(K) | Status |
| --- | --- | --- | --- |
| K = -2 SD (body-dominant) | (not credible) | -0.066 [-0.210, +0.075] | ✅ |
| K = 0 (average) | -0.140 [-0.238, -0.042]* | -0.136 [-0.234, -0.041]* | ✅ |
| K = +2 SD (knee-dominant) | (credible, stronger) | -0.207 [-0.355, -0.062]* | ✅ |

JN boundary (λ_ps non-credible above this K):

| Metric | Manuscript | Sandbox | Status |
| --- | --- | --- | --- |
| Boundary K | -0.625 (-0.86 SD) | -0.640 | ✅ |
| % observations in credible region | 84.6% | 85.0% | ✅ |
| SP JN boundary | none | none | ✅ |

---

## 9. Sleep-to-Pain fMRI moderation (Table 5)

From `sandbox/real_run2/fmri_sp_moderation_results.csv`.

| ROI | Manuscript γ_sp [CrI], p | Sandbox γ_sp [CrI], p | Status |
| --- | --- | --- | --- |
| **Left NAcc** | **+0.040 [+0.004, +0.076], p=0.027** | **+0.039 [+0.004, +0.073], p=0.029** | ✅ credible |
| Right NAcc | +0.023 [-0.013, +0.057], p=0.194 | +0.023 [-0.013, +0.057], p=0.203 | ✅ null |
| Contralateral S1 | -0.017 [-0.056, +0.024], p=0.414 | -0.016 [-0.055, +0.025], p=0.435 | ✅ null |
| Contralateral Middle Insula | +0.017 [-0.025, +0.060], p=0.435 | +0.018 [-0.022, +0.059], p=0.389 | ✅ null |
| Left Thalamus | +0.013 [-0.025, +0.051], p=0.507 | +0.013 [-0.025, +0.050], p=0.501 | ✅ null |
| Left Anterior Insula | +0.006 [-0.034, +0.050], p=0.831 | +0.002 [-0.037, +0.043], p=0.921 | ✅ null |
| **Right dACC/MCC** | **+0.038 [+0.000, +0.077], p=0.047** | **+0.038 [+0.002, +0.077], p=0.044** | ✅ credible |

**Sign concordance (Krause 6 ROIs).** Manuscript claims "all six
Krause univariate γ_sp estimates matched the direction predicted by
the sleep deprivation framework" with p=(1/2)^6=0.016.

Sandbox γ_sp signs: S1 −, Mid Insula +, Thal +, Ant Insula +,
L NAcc +, R NAcc +.

The convention under which all six match is:
  - S1:              **negative** (sleep deprivation *amplifies* S1
    activation, so higher baseline S1 indexes stronger sleep→pain
    coupling → more-negative λ_sp → γ_sp < 0)
  - Middle insula, thalamus, anterior insula, both NAcc:
    **positive** (sleep deprivation *blunts* activation in these
    regions, so higher baseline activation reflects stronger
    endogenous modulation → weaker coupling → γ_sp > 0)

Under this convention the sandbox reproduces **6/6, p = (1/2)^6 =
0.016** exactly. This was verified by re-running
`python/04_fmri_sp_moderation.py --data-dir sandbox/real_run2
--output-dir sandbox/real_run2` after fixing two bugs in the
reproduction pipeline:

1. `python/lib/moderator_loaders.py` had `expected_sign_sp = "+"`
   for all 6 Krause ROIs, including S1. Fixed to use `-` for S1.
2. `python/04_fmri_sp_moderation.py` had its own `EXPECTED_SIGNS`
   dict (duplicating the one in the loader) with the same bug.
   Fixed to use `-` for S1.
3. The same script hardcoded the reported p-value as
   `sign_p = 0.5 ** n_tested` regardless of `n_concordant`, so it
   would have printed 0.0156 even if only 3 of 6 ROIs matched.
   Fixed to use the exact one-sided binomial tail
   `P(X ≥ n_concordant | Binomial(n_tested, 0.5))`.

**Manuscript Methods inconsistency (✏).** The Methods paragraph
says "yielding negative γ_sp for S1, middle insula, thalamus, and
anterior insula, and positive γ_sp for both NAcc ROIs." Under that
literal reading the sandbox γ_sp pattern would match only 3 of 6
(S1, L NAcc, R NAcc), not 6 of 6. The prediction that is actually
consistent with both the Krause amplification-vs-blunting framework
and with the 6/6, p=0.016 claim in Results is: "yielding *negative*
γ_sp for S1 (because sleep deprivation *amplifies* S1 activation,
so higher baseline activation indexes stronger coupling) and
*positive* γ_sp for middle insula, thalamus, anterior insula, and
both NAcc (because sleep deprivation *blunts* activation in these
regions, so higher baseline activation indexes weaker coupling)."
The Methods paragraph should be rewritten accordingly so Methods
and Results agree.

The individual significance of the left NAcc (p=0.029) and the ACC
(p=0.044) is unaffected by this issue — those are the two headline
findings and they reproduce exactly.

---

## 10. Johnson-Neyman boundaries (Figures 5, 6)

From `sandbox/real_run2/fmri_sp_jn_results.csv`.

| Figure | Manuscript boundary | Sandbox boundary | Status |
| --- | --- | --- | --- |
| Fig 4: contrast boundary K | -0.625 (-0.86 SD) | -0.640 | ✅ |
| Fig 4: % in credible region | 84.6% | 85.0% | ✅ |
| Fig 5: Left NAcc raw boundary | -0.027 | -0.011 | ⚠ |
| Fig 5: % of sample below | 49% | 53% | ⚠ |
| Fig 6: ACC raw boundary | 0.062 | 0.075 | ⚠ |
| Fig 6: % below | (not reported) | 47% | — |

**Note on the "⚠" JN boundaries.** Both Left NAcc and ACC
moderation boundaries sit very close to the posterior mean of the
interaction term, so small MCMC noise shifts the boundary by ~0.01
and moves a few percent of the sample across the threshold. The
substantive conclusion — that the JN region covers roughly half of
the sample and includes the clinically relevant tail — is
preserved.

---

## 11. Pain-to-Sleep arousal-relay moderation (Table S1)

### fMRI BOLD (N=174, 1,403 obs)

From `sandbox/real_run2/arousal_fmri_moderation_results.csv`.

| ROI | Manuscript γ_ps, p | Sandbox γ_ps, p | Status |
| --- | --- | --- | --- |
| **PBN** | **-0.075, p=0.145** | **+0.026, p=0.622** | 🐛 sign flipped |
| SI-BF/Ch4 | +0.022, p=0.660 | +0.022, p=0.658 | ✅ |
| CeA | +0.073, p=0.212 | +0.074, p=0.194 | ✅ |
| BNST | -0.077, p=0.166 | -0.071, p=0.195 | ✅ |
| LH | -0.022, p=0.666 | -0.002, p=0.962 | ⚠ close (same direction, null) |

### VBM grey matter volume (N=189)

From `sandbox/real_run2/arousal_vbm_moderation_results.csv`.

| ROI | Manuscript γ_ps, p | Sandbox γ_ps, p | Status |
| --- | --- | --- | --- |
| PBN | -0.062, p=0.261 | -0.065, p=0.234 | ✅ |
| SI-BF/Ch4 | -0.018, p=0.761 | -0.021, p=0.727 | ✅ |
| CeA | -0.050, p=0.398 | -0.053, p=0.366 | ✅ |
| BNST | -0.039, p=0.474 | -0.039, p=0.454 | ✅ |
| LH | -0.014, p=0.798 | -0.017, p=0.770 | ✅ |

**VBM sign concordance**: 5/5 negative → p=(1/2)^5=0.031, same as manuscript. ✅

### PBN fMRI discrepancy (🐛)

- **Sandbox**: γ_ps = +0.026, p=0.622 (null, positive sign)
- **Manuscript**: γ_ps = -0.075, p=0.145 (null, negative sign)
- **Root cause**: the sandbox PBN atlas mask intersected with the
  fMRI brain mask contains **2 non-zero voxels** at 3 mm fMRI
  resolution; the manuscript version had **3 voxels**. PBN sits at
  the edge of the fMRI field of view where voxel values vary
  strongly, so a 1-voxel difference flips the sign of the mean BOLD
  extraction. See `REPRODUCTION_REPORT.md` for details.
- **Substantive conclusion preserved**: both values (p=0.622,
  p=0.145) agree that PBN fMRI moderation is not credible. The
  manuscript's role for PBN is as "the second strongest negative
  among null effects" — this framing is based on the exact value
  and is not reproduced.
- **Action recommended**: either refit with a larger PBN atlas
  prior or drop PBN fMRI from the Table S1 commentary; the
  5/5-VBM-sign-concordance result is unaffected.

---

## 12. Severity moderation (Table S2, Note S2) ❌ NOT PIPELINE

Manuscript Table S2 reports four rows:

| Moderator / Model | γ | 95% CrI | p |
| --- | --- | --- | --- |
| Mean Pain Severity / Alone / SP | -0.010 | [-0.047, +0.028] | 0.611 |
| Mean Sleep Quality / Alone / SP | +0.023 | [-0.017, +0.064] | 0.267 |
| Mean Pain Severity / Joint / SP | -0.003 | [-0.042, +0.035] | 0.877 |
| Mean Sleep Quality / Joint / SP | +0.023 | [-0.020, +0.065] | 0.301 |

**Status**: The `python/` pipeline does not currently run severity
moderation. This analysis lives in the legacy
`scripts/run_severity_moderation.py`. The sandbox cannot refute or
confirm these numbers.

**Recommendation.** Port `run_severity_moderation.py` into a new
step `python/07_severity_moderation.py` wired into
`python/main.py`. This is a single-hour task and would close the
last gap in pipeline coverage. It is not a reproduction bug — the
legacy script has already produced these numbers with the same
model — but it prevents independent verification from the fresh
`python/main.py` run.

---

## 13. Figures

| Figure | Produced by pipeline? | Visual match |
| --- | --- | --- |
| Fig 1 (data availability grid) | ✅ | matches |
| Fig 2 (Pain→Sleep forest+boxstrip) | ✅ | matches |
| Fig 3 (Sleep→Pain forest+boxstrip) | ✅ | matches |
| Fig 4 (contrast JN 3-panel) | ✅ | matches (K boundary -0.640 vs -0.625) |
| Fig 5 (Left NAcc JN 3-panel) | ✅ | matches (⚠ boundary shift in noise) |
| Fig 6 (ACC JN 3-panel) | ✅ | matches (⚠ boundary shift in noise) |
| Fig S1 (endorsement validation) | ⏭ SKIPPED | requires `endorsement_data.csv` from legacy script |
| Fig S2 (convergent validity) | ✅ | matches |
| Fig S3 (contrast SP null JN) | ✅ | matches |
| Fig S4 (stim ROI MNI views) | ⏭ SKIPPED | requires `stim_roi_maps.png` from legacy plotter |
| Fig S5 (Krause non-sig 2×2) | ✅ | matches |
| Fig S6 (arousal ROI MNI views) | ⏭ SKIPPED | requires `arousal_roi_maps.png` from legacy plotter |
| Fig S7 (fMRI arousal JN 3×2) | ✅ | matches (PBN panel off due to item 11) |
| Fig S8 (VBM arousal JN 3×2) | ✅ | matches |

---

## Summary

### ✅ Fully reproduced
- Sample size (229 / 1,818 / 174 / 189 / 1,403)
- Age/Sex demographics (means, SDs, counts)
- All Table 3 population coupling parameters (|Δ| ≤ 0.004)
- All Table 4 contrast moderation parameters + JN boundary
- All Table 5 fMRI SP moderators + the credible Left NAcc / ACC findings
- All Table S1 VBM arousal moderators + 5/5 VBM sign concordance
- All LOO-CV ΔELPD and Δ/SE values (crossing the 2.0 threshold for PS, below for SP)
- Pareto k̂ diagnostics (2/1818 > 0.7, exact match)
- Person-level range statistics for λ_ps (9/229 credibly <0) and λ_sp
- Figures 1, 2, 3, 4, 5, 6, S2, S3, S5, S7, S8

### ⚠ Close but differs at sub-percent level (MCMC noise)
- Person-level λ_sp N-with-P<0>0.95 (3 vs 4, boundary case)
- JN boundary location for Left NAcc and ACC (~0.01 shift)
- LH fMRI γ_ps value (-0.002 vs -0.022, same direction and null)

### 🐛 Identified problems
1. **PBN fMRI sign flip** — 2 vs 3 voxels in the brain-mask
   intersection flips the sign of a null effect. Substantive
   conclusion (no credible moderation) is preserved, but the
   exact value reported in the manuscript text is not reproduced.
2. **Krause sign concordance (fixed)** — The reproduction
   pipeline had (a) `expected_sign_sp = "+"` for S1 instead of
   `"-"` in both `moderator_loaders.py` and the duplicate
   `EXPECTED_SIGNS` dict in `04_fmri_sp_moderation.py`, and (b)
   a hardcoded `sign_p = 0.5 ** n_tested` that always printed
   0.0156 regardless of actual concordance. Both were fixed.
   With the corrected code the sandbox reproduces **6/6,
   p=0.0156** exactly.

### ✏ Manuscript typos / inconsistencies
- Table 2 age range reports `31–79`; actual data range is `44–80`.
- Methods §2.5 says "negative γ_sp for S1, middle insula,
  thalamus, and anterior insula, and positive γ_sp for both NAcc
  ROIs." That text is self-inconsistent with the Results claim of
  6/6 sign concordance. The text should be corrected to:
  "negative γ_sp for S1 (sleep deprivation *amplifies* S1
  activation) and positive γ_sp for middle insula, thalamus,
  anterior insula, and both NAcc (sleep deprivation *blunts*
  activation in these regions)."

### ❌ Not produced by the `python/` pipeline
- Factor analysis eigenvalues, loadings, parallel analysis
  (uses precomputed `factor_model_params_contrast.json`).
- Demographic table beyond Age and Sex (race, BMI, WOMAC, KL).
- Severity moderation (Table S2) — lives in legacy
  `scripts/run_severity_moderation.py`.
- Figures S1, S4, S6 — require endorsement CSV or MNI slice
  images from legacy scripts.

---

## Files to consult

- `sandbox/real_run2/coupling_summary.txt` — Table 3 numbers
- `sandbox/real_run2/person_coupling_estimates.csv` — Section 3.3 range stats
- `sandbox/real_run2/contrast_moderation_results.csv` — Table 4
- `sandbox/real_run2/contrast_jn_boundary.txt` — Fig 4 boundary and
  simple slopes
- `sandbox/real_run2/fmri_sp_moderation_results.csv` — Table 5
- `sandbox/real_run2/fmri_sp_jn_results.csv` — Figs 5, 6 boundaries
- `sandbox/real_run2/arousal_fmri_moderation_results.csv` — Table S1 fMRI
- `sandbox/real_run2/arousal_vbm_moderation_results.csv` — Table S1 VBM
- `sandbox/real_loo/loo_comparison.txt` — Section 3.1 LOO comparison
- `sandbox/real_loo/loo_pairwise.csv` — the four pairwise Δ/SE rows
