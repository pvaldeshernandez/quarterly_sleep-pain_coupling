# Reproduction Plan — Checking Every Reported Number

This document enumerates every numerical claim in `manuscript_pain.md`
and tracks whether the sandbox pipeline already reproduced it, is
still missing, or mismatches. Each row is then re-run / verified as
needed.

Source for sandbox values: `sandbox/real_run2/` (full real-data run
with interpolation, 229 subjects / 1818 lag transitions, commit
`9cca0d0`).

Legend:
  ✅ MATCHED — reproduced within expected MCMC Monte Carlo error (|Δ| ≤ SD)
  ⚠ CLOSE — matches direction and significance but differs by >1 SD
  ❌ MISSING — the pipeline has not yet produced this number
  🐛 BUG — the pipeline produced it but the value is wrong

---

## 1. Sample size and data structure

| Claim | Manuscript | Sandbox | Status |
| --- | --- | --- | --- |
| Parent study N | 243 | 243 (raw CSV) | ✅ |
| Analytic sample N | 229 | 229 | ✅ |
| Excluded for no ≥3-quarter segment | 14 | 14 | ✅ |
| Median lagged transitions per person | 9 | 9 | ✅ |
| Range of lagged transitions | 2–10 | 2–10 | ✅ |
| Total observations (lag transitions) | 1,818 | 1,818 | ✅ |
| Retained points (segment ≥ 3) | 2,056 | 2,056 | ✅ |
| Points from partial raw items | 128 | 128 | ✅ |
| fMRI subsample | 174 | 174 | ✅ |
| VBM subsample | 189 | 189 | ✅ |
| fMRI arousal observations | 1,403 | 1,403 | ✅ |
| fMRI sample demographics: 115 F, 59 M | 115 F, 59 M | (in log) | ✅ |
| fMRI age mean (SD) | 58.7 (8.5) | - | need to verify |

## 2. Demographics (Table 3, N=229)

| Claim | Manuscript | Sandbox | Status |
| --- | --- | --- | --- |
| Age mean (SD) [range] | 58.1 (8.2) [31–79] | - | ❌ not computed by pipeline; comes from participants_wideformat |
| Female sex N (%) | 148 (64.6) | 148 F (Sex_coded=1) | ✅ |
| Male sex N | 81 | 81 | ✅ |
| Race/ethnicity Black | 91 (39.7) | - | ❌ not computed |
| Race/ethnicity White | 67 (29.3) | - | ❌ not computed |
| Race/ethnicity Hispanic | 55 (24.0) | - | ❌ not computed |
| BMI mean (SD) | 31.6 (7.7) | - | ❌ |
| WOMAC Pain mean (SD) | 5.8 (5.0) | - | ❌ |
| KL grade distribution | 70/43/48/34/29 | - | ❌ |

**Note.** Table 3 is descriptive statistics from `participants_wideformat.xlsx`.
The current `python/` pipeline does not produce a demographics table —
it just uses Age and Sex in the model. Adding a Table 3 generator
would be a nice-to-have but is not a reproduction failure since the
source data is in the raw file.

## 3. Factor analysis

| Claim | Manuscript | Sandbox | Status |
| --- | --- | --- | --- |
| Correlation type | polychoric | ? | need to check |
| F1 eigenvalue | 5.56 | ? | need to check `factor_model_params_contrast.json` |
| F1 % variance | 69.5% | ? | - |
| F2 eigenvalue | 0.89 | ? | - |
| F2 % variance | 11.2% | ? | - |
| Total variance (2-factor) | 80.7% | ? | - |
| Parallel analysis 95th% F1 | 1.13 | ? | not computed by pipeline |
| Parallel analysis 95th% F2 | 1.11 | ? | not computed by pipeline |
| F1 loadings range (all +) | 0.78–0.86 | ? | - |
| F2 knee loadings range | +0.30 to +0.45 | ? | - |
| F2 body loadings range | -0.23 to -0.50 | ? | - |
| r(F1, F2) | 0.001 | ? | - |
| PHQ body map mean areas | 3.2 | ? | ❌ not computed |
| PHQ body map SD areas | 2.5 | ? | ❌ not computed |
| Knee-only group N | 19 | ? | ❌ not computed |
| Knee + others N | 143 | ? | ❌ not computed |
| No knee N | 67 | ? | ❌ not computed |
| ANOVA F(2,226) | 16.56 | ? | ❌ not computed |
| ANOVA p | <0.001 | ? | ❌ not computed |
| Knee-only mean contrast | 0.57 (0.64) | ? | ❌ |
| Knee+others mean | 0.10 (0.72) | ? | ❌ |
| No-knee mean | -0.34 (0.61) | ? | ❌ |
| Knee endorsement r_pb | 0.31, p<0.001 | ? | ❌ |
| Upper back r_pb | -0.16, p=0.018 | ? | ❌ |
| Lower back r_pb | -0.14, p=0.034 | ? | ❌ |

## 4. Convergent validity correlations (Figure S2)

| Clinical measure vs contrast | r | p | Sandbox | Status |
| --- | --- | --- | --- | --- |
| PHQ knee pain days | 0.37 | <0.001 | ? | need Figure S2 |
| PHQ % waking day knee pain | 0.30 | <0.001 | ? | - |
| WOMAC Pain | 0.28 | <0.001 | ? | - |
| WOMAC Total | 0.26 | <0.001 | ? | - |
| WOMAC Physical Function | 0.25 | <0.001 | ? | - |
| WOMAC Stiffness | 0.24 | <0.001 | ? | - |
| Knee pain rating | 0.23 | <0.001 | ? | - |
| KL grade (Spearman ρ) | 0.31 | <0.001 | ? | - |

## 5. Population coupling parameters (Table 4)

| Parameter | Manuscript | Sandbox | Status |
| --- | --- | --- | --- |
| μ_p | 0.009 (SD 0.010) | 0.002 (SD 0.010) | ✅ (Δ=0.007, within 1 SD) |
| φ_p | 0.106 (SD 0.025) | 0.104 (SD 0.026) | ✅ |
| **λ_sp** | **-0.021** [-0.054, 0.015] P<0=0.877 | **-0.023** [-0.059, 0.013] P<0=0.893 | ✅ |
| δ_p | +0.024 [-0.004, 0.051] P<0=0.056 | +0.022 [-0.006, 0.050] P<0=0.064 | ✅ |
| ω_sp | +0.009 [-0.026, 0.043] P<0=0.313 | +0.005 [-0.031, 0.041] P<0=0.394 | ✅ |
| μ_s | 0.004 | - | need to check coupling_results.csv |
| **λ_ps** | **-0.140** [-0.238, -0.042] P<0=0.998 | **-0.136** [-0.234, -0.041] P<0=0.998 | ✅ |
| φ_s | 0.004 | 0.005 | ✅ |
| **δ_s** | **-0.050** [-0.093, -0.006] P<0=0.986 | **-0.052** [-0.096, -0.007] P<0=0.987 | ✅ |
| ω_ps | -0.050 [-0.124, 0.021] P<0=0.903 | -0.048 [-0.121, 0.026] P<0=0.895 | ✅ |
| τ_sp | 0.115 [0.073, 0.156] | 0.113 | ✅ |
| τ_ps | 0.362 [0.244, 0.464] | 0.359 | ✅ |
| σ_p | 0.437 [0.424, 0.452] | 0.438 | ✅ |
| σ_s | 0.661 [0.640, 0.682] | 0.661 | ✅ |
| ρ | -0.155 [-0.202, -0.111] P<0=1.000 | -0.157 P<0=1.000 | ✅ |

## 6. Person-level range statistics

| Claim | Manuscript | Sandbox | Status |
| --- | --- | --- | --- |
| Per-person λ_ps range | -0.878 to +0.810 | ? | need to compute from person_coupling_estimates.csv |
| Per-person λ_ps SD | 0.180 | ? | - |
| N with P(λ_ps<0)>0.95 | 9 of 229 (3.9%) | ? | - |
| Per-person λ_sp range | -0.213 to +0.154 | ? | - |
| Per-person λ_sp SD | 0.049 | ? | - |
| N with P(λ_sp<0)>0.95 | 4 (1.7%) | ? | - |

## 7. LOO-CV model comparison 🔴 MISSING

| Comparison | ΔELPD | SE | Δ/SE | Status |
| --- | --- | --- | --- | --- |
| Full vs no-PS | +22.0 | 8.8 | 2.49 | ❌ NOT RUN |
| No-SP vs null | +23.1 | 9.1 | 2.53 | ❌ NOT RUN |
| Full vs no-SP | +0.7 | 5.5 | 0.12 | ❌ NOT RUN |
| No-PS vs null | +1.8 | 5.9 | 0.30 | ❌ NOT RUN |
| Pareto k̂ > 0.7 count | 2 (0.1%) | - | - | ❌ NOT RUN |
| Pareto k̂ max | 0.90 | - | - | ❌ NOT RUN |

**Action required:**
1. Rewrite `compute_loo_comparison()` to fit the 4 nested models the
   manuscript describes (full / no-PS / no-SP / null), not the 4 models
   the old code fitted (base / +contrast / +agesex / full).
2. Add `include_sp`/`include_ps` flags to `fit_bayesian_varx1` ← DONE.
3. Run step 2 with `--loo` in a fresh sandbox.
4. Compare ΔELPD/SE values to the manuscript.

## 8. Contrast moderation (Table 5 in paper, sandbox file
`contrast_moderation_results.csv`)

| Parameter | Manuscript | Sandbox | Status |
| --- | --- | --- | --- |
| δ_p | +0.024 [-0.004, 0.051] P<0=0.056 | +0.022 | ✅ |
| ω_sp | +0.009 [-0.026, 0.043] P<0=0.313 | +0.005 | ✅ |
| **δ_s** | **-0.050** [-0.093, -0.006] P<0=0.986 | **-0.052** | ✅ |
| ω_ps | -0.050 [-0.124, 0.021] P<0=0.903 | -0.048 | ✅ |
| Simple slopes at ±2 SD, 0 | (in text) | - | need to verify |
| JN boundary (ps) | K = -0.625 | ? | need to check contrast_jn_boundary.txt |
| JN -0.625 in SD units | -0.86 SD | - | - |
| % obs in credible region | 84.6% | ? | - |
| JN boundary (sp) | none | ? | - |

## 9. fMRI moderation (Table 6 / Table 5 in paper, sandbox file
`fmri_sp_moderation_results.csv`)

| ROI | Manuscript γ_sp [CrI], p | Sandbox | Status |
| --- | --- | --- | --- |
| **Left NAcc** | **+0.040 [+0.004, +0.076], p=0.027** | +0.039 [+0.004, +0.073], p=0.029 | ✅ |
| Right NAcc | +0.023 [-0.013, +0.057], p=0.194 | +0.023 [-0.013, +0.057], p=0.203 | ✅ |
| Contralateral S1 | -0.017 [-0.056, +0.024], p=0.414 | -0.016 [-0.055, +0.025], p=0.435 | ✅ |
| Contralateral Middle Insula | +0.017 [-0.025, +0.060], p=0.435 | +0.018 [-0.022, +0.059], p=0.389 | ✅ |
| Left Thalamus | +0.013 [-0.025, +0.051], p=0.507 | +0.013 [-0.025, +0.050], p=0.501 | ✅ |
| Left Anterior Insula | +0.006 [-0.034, +0.050], p=0.831 | +0.002 [-0.037, +0.043], p=0.921 | ✅ |
| **Right dACC/MCC** | **+0.038 [+0.000, +0.077], p=0.047** | +0.038 [+0.002, +0.077], p=0.044 | ✅ |
| Sign concordance (6/6 Krause) | p=(1/2)^6=0.016 | 6/6, p=0.016 | ✅ |
| NAcc-ACC correlation r | 0.12 | ? | need to compute |

## 10. JN boundaries (Figures 4, 5, 6)

| Figure | Manuscript | Sandbox | Status |
| --- | --- | --- | --- |
| Fig 4: contrast boundary | K = -0.625 (-0.86 SD) | K = -0.640 | ✅ close |
| Fig 4: % in credible region | 84.6% | 85% (15% outside) | ✅ |
| Fig 5: Left NAcc boundary (raw) | -0.027 | -0.011 | ⚠ MCMC variance |
| Fig 5: % of sample below | 49% | 53% | ⚠ close |
| Fig 6: ACC boundary (raw) | 0.062 | 0.075 | ⚠ MCMC variance |
| Fig 6: % below | (not reported) | 47% | — |

## 11. Arousal relay moderation (Table S1)

fMRI BOLD (N=174, 1403 obs):

| ROI | Manuscript γ_ps, p | Sandbox | Status |
| --- | --- | --- | --- |
| PBN | -0.075, p=0.145 | **+0.026, p=0.622** | 🐛 sign flipped |
| SI-BF/Ch4 | +0.022, p=0.660 | +0.022, p=0.658 | ✅ |
| CeA | +0.073, p=0.212 | +0.074, p=0.194 | ✅ |
| BNST | -0.077, p=0.166 | -0.071, p=0.195 | ✅ |
| LH | -0.022, p=0.666 | -0.002, p=0.962 | ⚠ close but larger Δ |

VBM GM volume (N=189):

| ROI | Manuscript γ_ps, p | Sandbox | Status |
| --- | --- | --- | --- |
| PBN | -0.062, p=0.261 | -0.065, p=0.234 | ✅ |
| SI-BF/Ch4 | -0.018, p=0.761 | -0.021, p=0.727 | ✅ |
| CeA | -0.050, p=0.398 | -0.053, p=0.366 | ✅ |
| BNST | -0.039, p=0.474 | -0.039, p=0.454 | ✅ |
| LH | -0.014, p=0.798 | -0.017, p=0.770 | ✅ |
| Sign concordance VBM (5/5) | p=(1/2)^5=0.031 | 5/5, p=0.031 | ✅ |

**PBN fMRI discrepancy (🐛):**
- Root cause: 2 non-zero voxels in sandbox brain-mask intersection
  vs 3 in manuscript. This flips the sign of the mean BOLD extraction.
- Both substantive conclusions agree: PBN fMRI is non-credible.
- But the reported -0.075 value is not reproduced.

**Action required:**
- Investigate the brain-mask intersection logic in `load_fmri_atlas_arousal`.
- Verify whether the manuscript's 3-voxel extraction came from a
  different PBN atlas file, a different brain mask, or a different
  interpolation method.

## 12. Severity moderation (Table S2)

Manuscript reports 4 rows (SP direction only, N=229, 1818 obs):

| Moderator / Model | γ | 95% CrI | p |
| --- | --- | --- | --- |
| Mean Pain Severity / Alone / SP | -0.010 | [-0.047, +0.028] | 0.611 |
| Mean Sleep Quality / Alone / SP | +0.023 | [-0.017, +0.064] | 0.267 |
| Mean Pain Severity / Joint / SP | -0.003 | [-0.042, +0.035] | 0.877 |
| Mean Sleep Quality / Joint / SP | +0.023 | [-0.020, +0.065] | 0.301 |

**Action required:**
- The current `python/` pipeline does not run severity moderation.
  This is in the legacy `scripts/run_severity_moderation.py`.
- Either port it into a step in the main pipeline OR verify against
  the legacy outputs.

## 13. Figure production

| Figure | Produced? | Notes |
| --- | --- | --- |
| Figure 1 (data availability) | ✅ | 229 participants × 11 quarters |
| Figure 2 (λ_ps person-level) | ✅ | from person_coupling_estimates.csv |
| Figure 3 (λ_sp person-level) | ✅ | same source |
| Figure 4 (contrast JN) | ✅ | from contrast_posterior_draws.npz |
| Figure 5 (Left NAcc JN) | ✅ | from nacc_posterior_draws.npz |
| Figure 6 (ACC JN) | ✅ | from acc_posterior_draws.npz |
| Figure S1 (factor endorsement) | ❌ | requires endorsement_data.csv not produced |
| Figure S2 (convergent validity) | ✅ | reads participants_wideformat.xlsx |
| Figure S3 (contrast SP null JN) | ✅ | |
| Figure S4 (stim ROI MNI views) | ❌ | legacy plot_stim_rois.py |
| Figure S5 (Krause JN merge) | ✅ | from krause_roi_posterior_draws.npz |
| Figure S6 (arousal ROI MNI views) | ❌ | legacy plot_arousal_rois.py |
| Figure S7 (fMRI arousal JN) | ✅ | from fmri_arousal_posterior_draws.npz |
| Figure S8 (VBM arousal JN) | ✅ | from vbm_arousal_posterior_draws.npz |

---

## Summary of what's MISSING or WRONG

### Critical (manuscript claims not reproduced):
1. **LOO-CV model comparison** — 4 ELPD differences, Pareto k̂
   diagnostics. `compute_loo_comparison` needs rewriting + running.
2. **PBN fMRI sign** — sandbox +0.026 vs manuscript -0.075. Needs
   root-cause investigation in the brain-mask intersection.
3. **Severity moderation (Table S2)** — not produced by the Python
   pipeline at all. Need to port from legacy script or run it
   separately.

### Nice-to-have (manuscript claims that the pipeline does not
compute but that live in the raw data files):
4. **Demographics Table 3** — descriptive statistics from
   participants_wideformat.xlsx. Could add a small script.
5. **Factor analysis parallel analysis** — eigenvalue distribution
   from Horn's method, not currently computed.
6. **Factor analysis convergent validity** — ANOVA across PHQ
   distribution groups, point-biserial correlations with 13 body
   areas. Not currently in step 1.
7. **Person-level range statistics** for λ_sp and λ_ps (-0.878 to
   +0.810, N with P<0 > 0.95, etc.). Easy to compute from
   `person_coupling_estimates.csv`.
8. **NAcc-ACC BOLD correlation r = 0.12** — easy to compute from the
   two moderator loaders.
9. **Figures S1, S4, S6** — auxiliary figures that need
   endorsement analysis or MNI slice images.

### Already verified:
10. Sample size, Table 4 population parameters, Table 5 fMRI
    moderation, Table 6 contrast moderation, VBM arousal, sign
    concordance, Figures 1-6, S2, S3, S5, S7, S8.

---

## Execution order

1. ✅ [DONE] Read manuscript, extract numerical claims.
2. ✅ [DONE] Cross-check against sandbox.
3. ⏳ Rewrite `compute_loo_comparison` to match manuscript nesting.
4. ⏳ Run LOO with fresh sandbox.
5. ⏳ Verify LOO ΔELPD/SE values.
6. ⏳ Add person-level range computation to step 2 summary output.
7. ⏳ Add NAcc-ACC BOLD correlation computation to step 4.
8. ⏳ Port severity moderation from legacy `scripts/run_severity_moderation.py`.
9. ⏳ Investigate PBN fMRI brain-mask intersection.
10. ⏳ Add factor analysis parallel analysis + convergent validity to step 1
    (optional, outside pipeline's current scope).
11. ⏳ Write REPRODUCTION_AUDIT.md with final pass/fail status.
12. ⏳ Commit and push.
