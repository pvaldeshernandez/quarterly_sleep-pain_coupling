# Fix log — response to Xiaohan's reproducibility review

Source of issues: `xiaohan-revision/issues.html` (delivered 2026-04-18).
Review covered 14-step pipeline run on HiPerGator; reproduction matched
manuscript to |Δ| ≤ 0.005 (MCMC tolerance) after two dependency fixes.

Status legend: `[ ]` open · `[~]` in progress · `[x]` done · `[-]` wontfix

---

## Reproduction issues (env)

- [ ] **R1. `statsmodels` missing from `environment.yml`**
  - Action: add `statsmodels` to `environment.yml`.
  - Files: `UPLOAD2/environment.yml`

- [ ] **R2. `nilearn` missing from `environment.yml`**
  - Action: add `nilearn` to `environment.yml`.
  - Files: `UPLOAD2/environment.yml`

---

## Manuscript ↔ code divergences

### D1. Varimax rotation claim (step 01)
- Status: `[~]` — manuscript-only edit; **no code change**
- Xiaohan: manuscript lines 311–313 say "polychoric correlations and
  varimax rotation"; code performs iterative PAF + sign flips only, no
  rotation matrix.
- Decision: **fix manuscript, keep code as-is.** Rationale:
  - Target constructs are F1 = severity and F2 = knee-vs-body contrast
    orthogonal to severity.
  - Unrotated PAF gives F1 = first common-variance axis (≈ severity) and
    F2 ⟂ F1 (pure contrast orthogonal to severity) — exactly the target.
  - Varimax maximizes per-factor variance of squared loadings → rotates
    toward item clusters → F̃1 ≈ knee, F̃2 ≈ body, destroying the
    severity/contrast decomposition.
  - Varimax would be the **wrong** rotation for the stated goal; the
    code is correct and no code change is needed.
- Action: edit `docs/manuscript_pain.md` §Methods factor analysis only.
  Remove "varimax rotation" from the description. State what was done:
  iterative principal-axis factoring on the polychoric correlation
  matrix, with sign-flip orientation, no rotation applied. Justify:
  unrotated PAF directly delivers the target (severity, contrast)
  decomposition because F1 is the first common-variance axis and F2 is
  by construction orthogonal to it; any orthogonal rotation would mix
  severity into both axes and defeat the purpose.
- Files: `docs/manuscript_pain.md` (lines ~311–313) **only**

### D2. Contralateral S1 / mid-insula (step 07)
- Status: `[x]` — code fixed; step 07 and step 08 re-run
- Confirmed: `step07_extract_sp_rois.py` extracts both `Right_S1` and
  `Right_Middle_Insula` from fixed +x coordinates for every subject.
  `FIG_S4_ROIS["Contra_*"]` carries an `mni_mirror` key but it is used
  only for Figure S4 rendering (to draw both spheres on the schematic);
  the extractor does not consult it. No per-subject flip exists.
- Manuscript claim (lines 121–125): "adapted hemisphere assignments for
  somatotopically organized regions to match the stimulated side, which
  varied across our sample" — S1 and Middle Insula extracted from the
  hemisphere contralateral to the stimulated knee, by mirroring the x
  coordinate (not flipping the con image).
- Variable in data dictionary: `img_test_site` (1 = Right knee,
  2 = Left knee), present in the wide-format file as
  `img_test_site__s1` (baseline) and `img_test_site__s2` (session 2).
  Pipeline uses only s1. Contralateral mapping:
    - `test_site = 1` (R knee stim) → contra = LEFT hemisphere  (x = -36 / -32)
    - `test_site = 2` (L knee stim) → contra = RIGHT hemisphere (x = +36 / +32)
- **Cohort decision (Pedro, 2026-04-18)**: The analytic fMRI cohort for
  this issue is "scanned ∩ present in `step00_extracted_long.csv`",
  **N = 182**. The 6 scanned IDs that step00 drops as baseline-only
  (1011, 1014, 2095-2, 2102-2, 836-24, 836-395) can't contribute to SP
  moderation (they have no quarterly data), so it does not make sense
  to work with them. For side assignment, use s1 with s2 fallback;
  that rescues 836-163 (s1 NaN, s2 = L-stim).
- **Sandbox results** (xiaohan-revision/sandbox_contra/, N = 182):
  - Side distribution: **98 R-stim, 84 L-stim** (0 missing).
  - S1:  released-right mean = −0.211; contralateral mean = −0.140;
    r(released, contra) = 0.899; paired t = +4.43, p ≈ 0;
    R-stim subgroup mean Δ = +0.131.
  - Middle Insula: released-right mean = +0.077; contralateral mean
    = +0.122; r(released, contra) = 0.965; paired t = +3.76, p = 0.0002;
    R-stim subgroup mean Δ = +0.084.
  - Interpretation: released-right values are a good proxy for
    contralateral values among the 84 L-stim subjects (trivially
    identical) but systematically biased among the 98 R-stim subjects,
    where "released-right" = **ipsilateral**. The SP model therefore
    used ipsilateral values for 54% of the cohort (98/182).
- Resolution (2026-04-18): option (a) — fixed code. Changes:
  - `step07_extract_sp_rois.py`: renamed the two ROIs to `Contra_S1`
    and `Contra_Middle_Insula`; added `contralateralize=True` and
    `mni_mirror`; added `load_stim_side_map()` reading s1 with s2
    fallback from `participants_wideformat.xlsx`, restricted to IDs
    in `step00_extracted_long.csv`; per-subject ROI sphere placed on
    the hemisphere contralateral to the stimulated knee (x-coordinate
    mirrored, con image NOT flipped).
  - Propagated rename to `step08_fit_sp_moderation.py` (KRAUSE_ROIS,
    EXPECTED_SIGNS) and `step09_sp_moderation_jn.py` (S5_ROIS).
- Re-run impact (Table 5 comparison, γ_sp after **both** contralateral
  extraction and the D5 γ_sp-only respec):
  - Contra S1:  −0.017 → −0.015 [−0.056, +0.027]  (not credible → not credible)
  - Contra Mid Insula: +0.018 → +0.018 [−0.022, +0.059] (not credible)
  - Left Thalamus: +0.012 → +0.013 (not credible)
  - Left Anterior Insula: +0.002 → +0.003 (not credible)
  - Left NAcc:   +0.040 → +0.041 [+0.005, +0.076] (credible → credible)
  - Right NAcc:  +0.014 → +0.024 (not credible)
  - **Left dACC/MCC:** +0.038 → **+0.043 [+0.005, +0.081]** (boundary → credible)
  - **Right dACC/MCC:** +0.039 → **+0.039 [+0.001, +0.077]** (boundary → credible)
  - Krause sign concordance: **6/6, p = 0.0156** (unchanged)
- Interpretation: the contralateral flip moved raw S1/MidIns values
  meaningfully (mean +0.07 and +0.05) but barely moved γ_sp. The D5
  respec is what tightened ACC to clearly credible bilaterally. Main
  NAcc conclusion unchanged.
- Files touched: `codes/python/step07_extract_sp_rois.py`,
  `codes/python/step08_fit_sp_moderation.py`,
  `codes/python/step09_sp_moderation_jn.py`.
- Downstream to re-run: step 09 (done), Figure 6 / Figure 5 /
  Figure S5, Table 5 CSV — already regenerated by refit.
- Manuscript edits now needed:
  - Table 5 labels: Right_S1 → Contralateral S1; Right_Middle_Insula
    → Contralateral Middle Insula.
  - ACC narrative: currently frames ACC as a right-lateralized
    result. New result: **bilateral dACC/MCC moderation of SP**.
    Update Results §3.5 and Discussion accordingly (Sardi et al.
    motivation did not claim lateralization, so the bilateral finding
    is more consistent with theory).

### D2b. fMRI cohort N and side proportions in manuscript (line 107)
- Status: `[ ]` — text fix
- Manuscript (line 107): "A subset of 188 participants completed
  task-based fMRI … the participant's most painful knee (right in
  125 participants, left in 96)."
- **Both numbers need updating:**
  - **N = 188** is the count of subjects with a `con_0001.nii` file
    after step 06. But 6 of those (1011, 1014, 2095-2, 2102-2, 836-24,
    836-395) are baseline-only and are dropped by step 00, so they
    never enter any subsequent analysis. The analytic fMRI cohort is
    **N = 182** (scanned ∩ step00 long frame).
  - **125/96** are whole-study counts (N = 221 with s1 side info,
    combining scanned and unscanned subjects). They are not the
    fMRI-cohort counts. For N = 182 with s1→s2 fallback on side:
    **98 right-knee stimulated, 84 left-knee stimulated** (all
    assigned; s2 fallback rescues 836-163).
- Action: edit line 107 to report N = 182 with 98 right / 84 left.
  If the enrollment count of 188 is worth keeping as context, phrase
  it as "188 scanned, of whom 182 contributed to the coupling analysis
  after excluding baseline-only participants." Downstream N = 174 (SP
  moderation) is the further subset with sufficient quarterly data to
  fit the coupling model; that number is already correct in the
  manuscript and does not need changing.
- Files: `docs/manuscript_pain.md` (line 107). Check Table/Figure
  captions for any other reference to N = 188 or 125/96.

### D3. "10 models" hemisphere sensitivity analysis (step 11)
- Status: `[~]` — decision made, text edit pending
- Xiaohan: auto-generated paragraph claims "sensitivity analysis testing
  each hemisphere separately (10 models)"; step10 defines 5 bilateral
  ROIs only; the 10 fits in step11 are 5 fMRI + 5 VBM (modality-split,
  not hemisphere-split).
- Decision (2026-04-18): option (b) — retract the claim. The Lynch
  ROIs were designed as bilateral atlases because the theoretical
  framework (Lynch et al. 2025) specifies bilateral projections, and
  the manuscript already justifies bilateral extraction (line 151:
  "these structures belong to the spino-parabrachio-amygdaloid and
  medial pain pathways, which receive predominantly bilateral
  projections"). The hemisphere-sensitivity sentence appears to have
  drifted in from an earlier draft. Also, the PBN is only 3 voxels at
  3 mm fMRI resolution — splitting it is unviable.
- Action:
  - `codes/python/step11_fit_ps_moderation.py`: remove the
    "sensitivity analysis testing each hemisphere separately (10
    models)" clause from the text-generator paragraph (lines ~354-364)
    and rephrase as "5 fMRI + 5 VBM = 10 separate models" (modality
    split), or simply drop the clause if the information is redundant.
  - `docs/manuscript_pain.md` Results §3.5 (lines ~1129-1132): strip
    the hemisphere-sensitivity sentence.
- No code-model re-run needed.

### D4. Age/Sex centering on long df vs. person-level (steps 04/08/11/13)
- Status: `[~]` — code fixed; re-run pending
- Code changes (permanent, all four steps):
  - `step04_fit_coupling_model.py:load_data`,
  - `step08_fit_sp_moderation.py:load_step02_data`,
  - `step11_fit_ps_moderation.py:load_step02_data`,
  - `step13_severity_moderation.py:load_data`
  — each now computes Age z and Sex_c on
  `df.groupby("ID").first()` (one row per subject) and merges those
  columns back onto the long df.
- Observed vs person-level mean (reference):
  - Age: long = 58.10 (SD 8.13); person = 58.06 (SD 8.16).
  - Sex: long = 0.670; person = 0.646 (manuscript value). Shift 0.024.
- Expected numeric impact: negligible on γ's (centering is a
  reparameterization of the intercept); person-level fitted values
  shift by γ_sex × 0.024.
- Re-run pending (steps 04, 08, 11, 13).

### D5. γ attached to both directions simultaneously (step 08/11)
- Status: `[~]` — SP side fixed & re-run; PS side (step 11) code
  updated, re-run pending
- Decision (2026-04-18): ROIs are hypothesis-driven and can only
  enter the coupling slope the theory motivates. An ROI cannot
  legitimately moderate the opposite direction, so the correct spec
  is γ_sp-only for Krause/ACC ROIs and γ_ps-only for Lynch ROIs.
  The previous joint-γ fit was misspecified.
- Code changes (permanent):
  - `lib/coupling_model.py`: new parameter
    `moderator_direction ∈ {"both","sp","ps","none"}`, default
    `"both"` (preserves legacy behavior for other callers). When
    `"sp"`, only γ_sp is declared and attached to λ_sp,i; γ_ps is
    absent. When `"ps"`, mirror image. `include_sp`/`include_ps`
    still control whether each coupling direction exists at all
    (so we can fit uncoupled models in the future). Validation
    raises if `moderator_direction` asks for a direction that is
    not included. `extract_results()` updated to handle either γ
    being absent from the posterior.
  - `step08_fit_sp_moderation.py`: passes
    `moderator_direction="sp"`. γ_ps columns in Table 5 become NaN
    (not estimated for Krause/ACC ROIs).
  - `step11_fit_ps_moderation.py`: passes
    `moderator_direction="ps"`. γ_sp columns become NaN.
- Re-run status:
  - SP (step 08, step 09): DONE — see D2 for numeric impact.
  - PS (step 11, step 12): CODE UPDATED, refit pending.
- Follow-up action: refit step 11 + step 12 after we finish
  discussing D2/D2b; update the PS ROI figures and Table in
  manuscript accordingly.

### D6. LH GM-masking flag ignored (step 10)
- Status: `[~]` — code fixed; re-run pending (will happen with D5 refit)
- Code change (permanent): `step10_extract_ps_rois.py:extract_fmri_arousal`
  now iterates per ROI and picks `FMRI_MASKED_DIR` when the ROI's
  `fmri_mask == "gm_masked"` flag is set, else `FMRI_UNMASKED_DIR`.
  Previously the unmasked dir was used unconditionally and the flag
  only affected the Figure S6 caption.
- Impact: LH extraction now reads GM-masked con values, matching the
  caption and the manuscript (line 157: "mask applied only to NAcc
  and LH"). Changes LH raw values only; does not affect the other
  four Lynch ROIs.
- Re-run: step 10 (extraction), step 11 (PS moderation — bundled
  with D5 refit).

### D7. Step 13 joint model writes 2 of 4 γ's (step 13)
- Status: `[~]` — code fixed; re-run pending
- Decision: drop γ_ps_pain and γ_ps_sleep from the joint model
  specification, matching the manuscript's "six moderation parameters"
  claim (individual-pain SP + PS, individual-sleep SP + PS, joint
  model pain SP + sleep SP = 6).
- Code change (permanent, `step13_severity_moderation.py:_fit_joint_model`):
  removed declarations of `gamma_ps_pain` and `gamma_ps_sleep`; removed
  their attachment to `lambda_ps`. Joint-model λ_ps now receives only
  the intercept, Age/Sex γ's, and the random effect u_ps,i.
- Re-run: step 13.

### D8. JN grid clipped to 1st–99th percentile (steps 09/12)
- Status: `[ ]` (cosmetic)
- Xiaohan: manuscript says grid "spans the observed moderator range";
  neuroimaging JN code uses `clip_pct=(1, 99)`, while the contrast JN
  uses `(0, 100)` and matches text.
- Action: change `clip_pct=(1, 99)` to `(0, 100)` in steps 09 and 12 for
  consistency, re-run figures S7/S8.

### D9. Person-dot overlays omit age/sex terms (steps 04/09/12)
- Status: `[ ]` (cosmetic)
- Xiaohan: figure person-dots compute `a2 + γ·X + u_sp` (and analogs)
  without the age/sex nuisance terms that are in the written slope.
- Action: add `γ_age·Age_z + γ_sex·Sex_c` to the person-level fitted
  values in figure overlays for Figures 2/3/5, or clarify caption.

### D10. SPM `fmri_t0 = 1` vs. middle-slice reference (MATLAB)
- Status: `[ ]` (MATLAB)
- Xiaohan: `create_first_level_job.m` sets `fmri_t0 = 1` with a comment
  saying "correction to the first slice", but `slice_timing.m` uses
  `refslice = round(nslices/2)` (middle slice).
- Action: change to `fmri_t0 = round(fmri_t/2) = 8` and re-estimate
  first-level, or document the inconsistency and its (likely small)
  effect. Needs magnitude check.

### D11. A.5–A.11 normalized-series handoff (MATLAB)
- Status: `[ ]` (MATLAB, traceability)
- Xiaohan: `deform_all.m` emits `d*`/`sd*` (indirect) and `d2*`/`sd2*`
  (direct); `create_first_level_job.m` picks `^sw.*\.nii$`; working
  directory hardcoded to `CONN2SPM_dartel_BIDS_indirect`. No in-tree
  step bridges the prefixes.
- Action: document the CONN→SPM handoff explicitly in
  `REPRODUCE_FOR_XIAOHAN.md` and/or ship the bridging script.

---

## Notes

- Reproduction spot-check (2026-04-16) passed: coupling-value |Δ| ≤ 0.005,
  6/6 Krause concordance, 5/5 VBM concordance. None of D4/D5 appear to
  change results enough to flip sign/significance at the main-finding
  level.
- D1, D4, D5 warrant re-running models to quantify the numeric impact
  on reported γ's and λ's.
