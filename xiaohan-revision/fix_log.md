# Fix log — response to Xiaohan's reproducibility review

Source of issues: `xiaohan-revision/issues.html` (delivered 2026-04-18).
Two-part structure:

1. **Manuscript edits still needed** — `docs/manuscript_pain.md` edits
   that have not been applied yet.
2. **Code changes already made** — committed and pushed to
   `origin/main`, end-to-end refit passes.

Status legend: `[ ]` open · `[~]` in progress · `[x]` done · `[-]` wontfix

---

# PART 1 — Manuscript edits still needed

All of these are edits to `docs/manuscript_pain.md` (and possibly
`docs/supplementary_materials.md`). None require a further code
refit; the code-side of each issue is either resolved or N/A.

## M1. Remove "varimax rotation" claim (D1)
- Status: `[ ]`
- Location: `docs/manuscript_pain.md` §Methods, factor analysis
  (lines ~311–313).
- Current text: "the eight items (q2–q5, q7–q10) were submitted to
  exploratory factor analysis (EFA) with polychoric correlations and
  varimax rotation."
- Edit: delete "and varimax rotation"; replace with a description of
  what the code actually does — iterative principal-axis factoring on
  the polychoric correlation matrix, with a sign-flip step to orient
  F1 (severity) and F2 (knee-vs-body contrast). Justify briefly that
  no rotation is applied because unrotated PAF already delivers the
  target decomposition: F1 is the first common-variance axis and F2
  is by construction orthogonal to it. Any orthogonal rotation
  (e.g., varimax) would mix severity into both axes.

## M2. Update fMRI cohort N and side counts (D2b)
- Status: `[x]` in `manuscript_pain.md`; Pedro will port to the docx.
- Location: `docs/manuscript_pain.md` line 107.
- Previous text: "A subset of 188 participants ... right in 125,
  left in 96."
- Applied edit: **N = 182; right in 98, left in 84.** The 6 baseline-
  only subjects (1011, 1014, 2095-2, 2102-2, 836-24, 836-395) are
  excluded because they have no quarterly data. Side from s1 with s2
  fallback (rescues 836-163).
- Downstream N = 174 (SP moderation) and 189 (VBM) are already
  correct elsewhere in the manuscript.

## M3. Rename S1 and Middle Insula to "Contralateral" (D2)
- Status: `[-]` — not needed. Manuscript already labels the two ROIs
  as "Contralateral S1" and "Contralateral Middle Insula" in Table 5
  (md lines 328–329) and in the Methods narrative. The previous
  mismatch was purely code-side (dict keys `Right_S1` /
  `Right_Middle_Insula`), and has been resolved by renaming the code
  keys to `Contra_S1` / `Contra_Middle_Insula` in step 07/08/09.

## M4. Revise ACC narrative: right-lateralized → bilateral (D2 + D5)
- Status: `[ ]`
- Locations: Results §3.5 and Discussion (§ discussing ACC
  moderation); currently frames ACC as a right-lateralized result.
- New result after the γ_sp-only respec: **bilateral dACC/MCC
  moderation of SP coupling**, both hemispheres credible:
  - Left dACC/MCC:  γ_sp = +0.044 [+0.003, +0.083]  (credible)
  - Right dACC/MCC: γ_sp = +0.039 [+0.001, +0.077]  (credible)
- Edit:
  - Replace right-only language with bilateral language.
  - Update or drop the Symonds 2006 citation if it was the basis for
    the right-lateralized framing (its theoretical content still
    stands; the bilateral result is more consistent with Sardi et al.
    (2024), which did not claim lateralization).

## M5. Retract "10 models (hemisphere-split)" sensitivity claim (D3)
- Status: `[x]` in `manuscript_pain.md`; Pedro will port to the docx.
- Location: `docs/manuscript_pain.md` line 348 (Results §3.5).
- Previous text: "Moderation was not credible for any ROI in either
  modality (Table S1)—a sensitivity analysis testing each hemisphere
  separately (10 models) confirmed that bilateral averaging did not
  mask lateralized effects."
- Applied edit: deleted the hemisphere-sensitivity clause. Kept the
  not-credible statement.
- Step 11 text generator (`codes/python/step11_fit_ps_moderation.py`)
  updated to emit the shorter paragraph; next --refit will produce
  text that matches the manuscript.
- Supplementary materials do not contain this claim; no supp edit
  needed.

## M6. Update Table 5 numbers after refit
- Status: `[ ]`
- Location: Table 5 in the manuscript (also results/step08_sp_moderation/…).
- All γ_sp values shifted slightly after the D2 contralateralization
  and D5 γ_sp-only respec. New values:

  | ROI | γ_sp | CrI | Credible |
  |---|---|---|---|
  | Contralateral S1 | −0.015 | [−0.055, +0.025] | no |
  | Contralateral Middle Insula | +0.018 | [−0.023, +0.059] | no |
  | Left Thalamus | +0.013 | [−0.024, +0.049] | no |
  | Left Anterior Insula | +0.002 | [−0.038, +0.043] | no |
  | **Left NAcc** | **+0.040** | **[+0.004, +0.075]** | **yes** |
  | Right NAcc | +0.023 | [−0.013, +0.059] | no |
  | **Left dACC/MCC** | **+0.044** | **[+0.003, +0.083]** | **yes (new)** |
  | **Right dACC/MCC** | **+0.039** | **[+0.001, +0.077]** | **yes** |

- Sign concordance (Krause 6 ROIs): **6/6, p = 0.0156** (unchanged).
- γ_ps columns: Krause/ACC ROIs are now tested only on SP, so γ_ps
  is not estimated. Either drop the γ_ps columns from Table 5 or
  mark "not estimated."

## M7. Update Table S2 (severity) — drop joint-model PS γ rows (D7)
- Status: `[ ]`
- Location: Table S2, Supplementary Materials; Discussion
  (Limitations) "six moderation parameters" sentence.
- New Table S2 rows (6 total):

  | Moderator | Model | Dir | γ | CrI |
  |---|---|---|---|---|
  | Mean Pain Severity | Alone | SP | −0.010 | [−0.049, +0.028] |
  | Mean Pain Severity | Alone | PS | −0.050 | [−0.163, +0.063] |
  | Mean Sleep Quality | Alone | SP | +0.023 | [−0.018, +0.064] |
  | Mean Sleep Quality | Alone | PS | +0.050 | [−0.050, +0.148] |
  | Mean Pain Severity | Joint | SP | −0.003 | [−0.043, +0.036] |
  | Mean Sleep Quality | Joint | SP | +0.021 | [−0.023, +0.065] |

  All 6 non-credible; limitation claim holds.
- Joint model now fits 2 γ's (γ_sp_pain, γ_sp_sleep) only — consistent
  with the "six moderation parameters" phrasing.

## M8. Check/update Aim 1 numbers (Table 4)
- Status: `[ ]`
- Minor shifts after D4 person-level centering:

  | Param | Old | New |
  |---|---|---|
  | λ̂_sp | −0.019 [−0.055, +0.017] | −0.022 [−0.057, +0.014] |
  | λ̂_ps | −0.141 [−0.234, −0.042] | −0.136 [−0.235, −0.038] |
  | τ̂_sp | 0.115 | 0.114 |
  | τ̂_ps | 0.352 | 0.361 |
  | ρ̂_innov | (unreported?) | −0.157 [−0.203, −0.111] |

- All within MCMC noise; conclusions unchanged. Update the table
  with the new point estimates and CrIs.

## M9. Update person-dot overlays to include Age/Sex (D9)
- Status: `[ ]` (cosmetic, figure edit only)
- Location: Figures 2, 3, 5.
- Per-person fitted coupling values in the overlays should include
  the age/sex adjustment terms from the written model:
  `λ_sp,i = λ_sp + γ_sp_age·Age_z + γ_sp_sex·Sex_c + γ·X + u_sp`.
- Current figure code omits the γ_age·Age_z + γ_sex·Sex_c terms.
- Fix this in the figure-generation code (step04, step09, step12),
  or clarify in each caption that person dots display the
  age/sex-zero-centered fitted slope.

## M10. Widen JN grid to 0–100% (D8)
- Status: `[ ]` (cosmetic)
- Location: Figures 5, 6, S7, S8.
- Step 05 (contrast JN) uses `clip_pct=(0, 100)` and matches the
  manuscript claim ("spans the observed moderator range"). Steps 09
  and 12 (neuroimaging JN) use `clip_pct=(1, 99)`, trimming outer 1%.
- Fix in `step09_sp_moderation_jn.py` and `step12_ps_moderation_jn.py`
  (one-line change to the `compute_jn_curve` call) and regenerate
  figures.

---

# PART 2 — Code changes made

All changes committed to `origin/main` and pushed:
- `984617d` — D2/D4/D5/D6/D7 fixes
- `89d0860` — refactor helpers + refit driver

End-to-end refit via `xiaohan-revision/refit_all.sh`: all 9 impacted
steps (04, 05, 07, 08, 09, 10, 11, 12, 13) exit 0. Total ~36 min.

## C1. Reproduction dependencies (R1, R2)
- Status: `[ ]` (not yet in `environment.yml`)
- Xiaohan reported that a fresh `conda env create -f environment.yml`
  fails at step 02 (`statsmodels`) and at steps 06/07/10 (`nilearn`).
- Pending: add both packages to `environment.yml`.

## C2. Contralateral S1 and Middle Insula (D2) — `[x]`
- `step07_extract_sp_rois.py`: ROIs renamed to `Contra_S1` and
  `Contra_Middle_Insula`, both carry `mni_mirror` and
  `contralateralize=True`. New `load_stim_side_map()` reads
  `img_test_site__s1` with `img_test_site__s2` fallback from
  `participants_wideformat.xlsx`, restricted to IDs in
  `step00_extracted_long.csv` (N = 182).
- Per-subject ROI sphere placed on the hemisphere contralateral to
  the stimulated knee. **The ROI x-coordinate is mirrored; the con
  image is not flipped.**
- Downstream renames in `step08` (KRAUSE_ROIS, EXPECTED_SIGNS) and
  `step09` (S5_ROIS).

## C3. Hypothesis-driven moderator direction (D5) — `[x]`
- `lib/coupling_model.py:fit_bayesian_varx1`: new parameter
  `moderator_direction ∈ {"both", "sp", "ps", "none"}`. Default
  `"both"` preserves legacy behavior for other callers.
  `include_sp`/`include_ps` still control whether each coupling
  direction exists at all (reserved for uncoupled-model fits).
  Validation raises if `moderator_direction` asks for a disabled
  direction.
- `step08_fit_sp_moderation.py` passes `moderator_direction="sp"`:
  X attached only to λ_sp for Krause/ACC ROIs (the hypothesis).
- `step11_fit_ps_moderation.py` passes `moderator_direction="ps"`:
  X attached only to λ_ps for Lynch ROIs (the hypothesis).
- `extract_results()` updated to handle either γ being absent from
  the posterior (e.g., γ_ps not estimated when
  `moderator_direction="sp"`).

## C4. Person-level Age/Sex centering (D4) — `[x]`
- Four loader functions (`step04.load_data`,
  `step08.load_step02_data`, `step11.load_step02_data`,
  `step13.load_data`) now compute Age_z and Sex_c on
  `df.groupby("ID").first()` (person level) instead of the
  observation-weighted long df.
- Refactored into a single shared helper
  `lib/coupling_model.py:load_varx_frame` (see C8) so all four steps
  call one implementation.

## C5. LH GM-mask honored (D6) — `[x]`
- `step10_extract_ps_rois.py:extract_fmri_arousal` iterates per ROI
  and routes through `FMRI_MASKED_DIR` when the ROI's
  `fmri_mask == "gm_masked"` flag is set, else `FMRI_UNMASKED_DIR`.
- Previously the unmasked dir was used unconditionally and the flag
  only affected the Figure S6 caption.
- Impact: LH extraction now reads GM-masked con values (cohort N
  unchanged; only the numeric values differ).

## C6. Joint severity model spec (D7) — `[x]`
- `step13_severity_moderation.py:_fit_joint_model`: removed γ_ps_pain
  and γ_ps_sleep from the joint model. Joint λ_ps now receives only
  the intercept, age/sex γ's, and the random effect. The joint model
  now contributes 2 γ's (both on λ_sp), matching the manuscript's
  "six moderation parameters" phrasing (2 individual-pain + 2
  individual-sleep + 2 joint).

## C7. Joint severity model speedup — `[x]` (bonus fix)
- `step13` joint model was using `pm.MvNormal(chol=L)` for the
  bivariate innovations likelihood. This is O(n³) per posterior
  evaluation and was taking 40+ min to sample. Replaced with the
  sequential-conditioning Cholesky trick (pain marginal + sleep
  conditioned on pain residual) already used by the library; the
  joint fit now finishes in ~90 s with identical posterior up to
  MCMC noise.

## C8. Shared library helpers — `[x]` (refactor)
Added to `lib/coupling_model.py`:

- **`add_bivariate_innovations_likelihood(mu_p, mu_s, y_p, y_s)`** —
  declares `sigma_pain`, `sigma_sleep`, `rho_raw`, `rho_innov` and
  attaches the bivariate innovations via sequential conditioning.
  Used by both `fit_bayesian_varx1` and `step13._fit_joint_model`
  so the likelihood and its priors live in one place.
- **`load_varx_frame(csv_path, verbose=False)`** — reads the step03
  CSV, computes person-level Age_z and Sex_c, merges onto the long
  frame, builds `pid_idx`, drops lag NaNs. Replaces four duplicated
  loaders (step04, 08, 11, 13).
- **`two_tail_p(prob_neg)`** — NaN-safe
  `2 * min(P(<0), 1 - P(<0))`. Replaces five inline copies
  across step08, step11, step13, and `extract_results`.
- **`sign_concordance_p(n_concordant, n_tested)`** — upper-tail
  binomial. Replaces two inline copies in step08 and step11.

## C9. Refit driver — `[x]`
- `xiaohan-revision/refit_all.sh`: portable bash driver. Runs steps
  04, 05, 07, 08, 09, 10, 11, 12, 13 sequentially with `--refit`.
  Resolves the repo root from the script's own location (no
  hardcoded paths), uses an existing `PYTENSOR_FLAGS` if set.

---

# Deferred issues (MATLAB, out of scope)

Noted for completeness; not actionable for this revision:

- **D10** `fmri_t0 = 1` vs. middle-slice reference
  (`create_first_level_job.m`).
- **D11** A.5–A.11 normalized-series handoff — which prefix
  (`sd*`, `sd2*`, `sw*`) feeds first-level is not fully traceable
  from the released MATLAB. Document the CONN→SPM handoff in
  `REPRODUCE_FOR_XIAOHAN.md` when time permits.
