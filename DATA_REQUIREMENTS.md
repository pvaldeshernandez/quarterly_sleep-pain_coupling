# Data Requirements for Full Reproduction

Everything the `python/main.py` pipeline consumes, from raw inputs
through to the final figures. **No intermediate results are cached
or read from disk outside of what this pipeline writes itself.** Each
run of `python python/main.py` recomputes every model from scratch.

The pipeline is fast (~12 min end-to-end on 4 cores) because the
model is small (~20 population parameters + 2 × N random effects),
not because anything is precomputed. Only the PyTensor JIT **compile**
cache is reused across runs (`~/.pytensor/`, ~118 MB), which saves
about 10 s per fit but does **not** cache MCMC draws.

---

## Quick summary

| Resource | Size | Required for |
| --- | --- | --- |
| `data/quarterly_data_long.csv` | 245 KB | Step 1 (factor analysis) |
| `data/participants_wideformat.xlsx` | 4.2 MB | Step 1 (demographics, Figure S2 scatter) |
| `data/fmri_contrasts/` (SPM GM-masked con images) | ~41 GB | Step 4 (NAcc only) |
| `data/spm_nomask/` (SPM unmasked con images) | 197 MB | Step 4 (all other Krause + ACC), Step 5 (fMRI arousal) |
| `data/vbm/` (CAT12 smwc1 GM images) | 1.7 GB | Step 5 (VBM arousal) |
| `data/atlases/` (probabilistic subcortical atlases) | 331 MB | Step 5 (atlas-weighted ROI extraction) |
| **Total** | **~43 GB** | (most of the bulk is fmri_contrasts) |

On HiPerGator the bulk directories already exist in their native
locations. The `data/` folder in the repo contains symlinks to them:

```
data/fmri_contrasts -> /orange/cruzalmeida/pvaldeshernandez/Data/UPLOAD2/
                       CONN2SPM_dartel_BIDS_indirect/
                       GLM-CONN2SPM-conn_stim_ses-01_dartel_BIDS_indirect-simplest
data/vbm            -> /orange/cruzalmeida/jperazagoicolea/UPLOAD2/
                       dset/derivatives/LocalizedBrainAge/ses-01/gm_data
data/atlases        -> <repo>/atlases/
```

A fresh clone on a different machine needs to either recreate these
symlinks or copy the contents into place under `data/`.

---

## 1. Questionnaire data (steps 1, 6)

### 1a. `data/quarterly_data_long.csv`

**Content.** One row per person per quarter, with the 25 quarterly
items (knee pain, body pain, sleep quality, treatment, and other
questions) plus demographics (age, race, sex) and the DTI-ALPS eigenmaps
at baseline.

**Required columns (these are the only ones the pipeline uses):**

| Column | Type | Description |
| --- | --- | --- |
| `ID` | string/int | Participant identifier (e.g., `1001`, `836-500`) |
| `quarter` | int | Quarterly visit index, 0 (baseline) through 11 |
| `age` | float | Age in years at the visit |
| `gender` | int | 1 = male, 2 = female (recoded by step 1 to 0/1) |
| `q1_knee_pain` | 0/1 | Gateway: any knee pain this quarter? |
| `q2_knee_pain` | 0–10 | Worst knee pain in last 7 days |
| `q3_knee_pain` | 0–10 | Average knee pain in last 7 days |
| `q4_knee_pain` | 0–10 | Current knee pain |
| `q5_knee_pain` | 0–10 | (optional, loaded but may be empty) |
| `q6_body_pain` | 0/1 | Gateway: any body pain this quarter? |
| `q7_body_pain` | 0–10 | Worst body pain |
| `q8_body_pain` | 0–10 | Average body pain |
| `q9_body_pain` | 0–10 | Current body pain |
| `q10_body_pain` | 0–10 | (optional) |
| `q13_sleep` | 0–10 | Sleep quality this week (higher = better) |

Step 1 applies **gateway imputation**: if `q1_knee_pain == 0` the
intensity items q2/q3/q4 are set to 0 where missing (the respondent
skipped them because they had no knee pain to rate). Same for
`q6_body_pain` → q7/q8/q9.

**Read by:** `python/01_prepare_data.py:load_real_data()`

### 1b. `data/participants_wideformat.xlsx`

**Content.** One row per participant with baseline clinical and
demographic variables. The pipeline only uses a small subset, but
Figure S2 (convergent validity) reads many more if available.

**Required columns (for the pipeline):**

| Column | Description |
| --- | --- |
| `ID` | Participant identifier (must match the quarterly file) |
| `age` | Baseline age |
| `gender` | Baseline sex (1 = M, 2 = F) |

**Optional columns (Figure S2 will render them as scatter panels):**

| Column | Description |
| --- | --- |
| `phq_knee_pain_days__s1` | PHQ knee pain days per week |
| `phq_percent_pain__s1` | PHQ % waking day in knee pain |
| `womac_pain__s1` | WOMAC Pain subscale |
| `total_womac__s1` | WOMAC Total |
| `womac_phys_function__s1` | WOMAC Physical Function |
| `womac_stiffness__s1` | WOMAC Stiffness |
| `qst_knee_pain_rating__s1` | Knee pain rating (QST) |

**Read by:** `python/01_prepare_data.py` (for demographics),
`python/06_generate_figures.py:generate_figure_s2()` (for the scatter panels).

---

## 2. SPM first-level contrast images (step 4, step 5)

### 2a. `data/spm_nomask/<subj_id>/con_0001.nii` (unmasked contrasts)

**Content.** First-level SPM contrast images for the
pain-stimulation-vs-baseline contrast, re-estimated without a
subject-specific grey-matter mask. The re-estimation was needed
because subcortical ROIs (thalamus, NAcc) would otherwise lose
voxels where individual GM segmentation excluded them, producing
NaNs and biased means.

**Format.** One NIfTI file per subject (`con_0001.nii`) inside a
directory named after the subject ID. Directory listing looks like:

```
data/spm_nomask/
    1001/con_0001.nii
    1002/con_0001.nii
    ...
    2113/con_0001.nii
    836-50/con_0001.nii
    836-500/con_0001.nii
    ...
```

**Subject count.** 188 subjects on HiPerGator. Not every participant
has fMRI data.

**Space + resolution.** MNI152, 3 mm isotropic.

**Used for:** Contralateral S1, Contralateral Middle Insula, Left
Thalamus, Left Anterior Insula, Right dACC/MCC (all Krause + ACC
except NAcc), and the 5 Lynch atlas-arousal ROIs for fMRI BOLD
extraction in step 5.

**Read by:** `python/lib/moderator_loaders.py` →
`load_fmri_krause_rois()`, `load_acc_roi()`, `load_fmri_atlas_arousal()`.

**HiPerGator location.**
```
/orange/cruzalmeida/pvaldeshernandez/Sleep-Pain_Coupling/UPLOAD2/data/spm_nomask/
```

### 2b. `data/fmri_contrasts/<subj_id>/con_0001.nii` (GM-masked)

**Content.** The **original** SPM first-level contrast images with
individual GM masks applied (via SPM's `spmT_*` → `con_*` masking
default). Used only for the Left and Right NAcc extraction, where
the manuscript chose the GM-masked version because it gave a cleaner
signal and a credible moderation (`γ_sp = +0.040, p = 0.027`).

**Format and structure.** Same as `spm_nomask/`: one directory per
subject, with `con_0001.nii` inside.

**Subject count.** Same 188 subjects (one-to-one with `spm_nomask/`).

**HiPerGator location.**
```
/orange/cruzalmeida/pvaldeshernandez/Data/UPLOAD2/
    CONN2SPM_dartel_BIDS_indirect/
    GLM-CONN2SPM-conn_stim_ses-01_dartel_BIDS_indirect-simplest
```

Symlinked into `data/fmri_contrasts`. This directory is **large
(~41 GB)** because it also contains per-subject SPM workflow artefacts
(`SPM.mat`, `ResMS.nii`, `Res_*.nii`) that the pipeline does not read
but that live alongside the contrast images.

**Read by:** `python/lib/moderator_loaders.py:load_fmri_krause_rois()`
(NAcc only).

---

## 3. VBM smwc1 images (step 5)

### 3. `data/vbm/smwc1<subj_id>_ses-01_T1w.nii`

**Content.** Smoothed, modulated segmented GM images from CAT12
(the `smwc1` prefix) used for voxel-based morphometry. Modulation
preserves total tissue volume, so integrating a probability-weighted
atlas mask against the smwc1 image gives a valid "total GM volume"
measurement for that ROI per subject.

**Format.** One NIfTI file per subject in a flat directory. Filename
convention `smwc1<ID>_ses-01_T1w.nii` (no subject subfolders):

```
data/vbm/
    smwc11001_ses-01_T1w.nii
    smwc11002_ses-01_T1w.nii
    ...
    smwc1836x500_ses-01_T1w.nii
    ...
```

The pipeline parses the subject ID from the filename:
```python
subj_id = fname.replace("smwc1", "").replace("_ses-01_T1w.nii", "")
# '836x500' -> '836-500' (the x becomes a dash)
```

**Subject count.** 189 subjects on HiPerGator (one more than the
fMRI subset because a few subjects have VBM but not fMRI).

**Space + resolution.** MNI152, 1.5 mm isotropic. Higher resolution
than the fMRI data means all 5 arousal ROIs — including PBN — are
extractable.

**Read by:**
`python/lib/moderator_loaders.py:load_vbm_atlas_arousal()`.

**HiPerGator location.**
```
/orange/cruzalmeida/jperazagoicolea/UPLOAD2/
    dset/derivatives/LocalizedBrainAge/ses-01/gm_data
```

Symlinked into `data/vbm`.

---

## 4. Atlas files (step 5)

### 4. `data/atlases/` (probabilistic subcortical atlases)

**Content.** Published probabilistic or label atlases used to define
the 5 Lynch arousal-relay ROIs. The pipeline resamples each atlas
to the fMRI or VBM resolution and extracts a probability-weighted
mean BOLD (fMRI) or total GM volume (VBM) per subject.

**Required files and their ROI mapping:**

| File | Used for | Source |
| --- | --- | --- |
| `atlases/atlas_b2_brainstem.nii.gz` | **PBN** (Parabrachial nucleus) | Singh et al. 2022 Hum Brain Mapp (Brainstem Navigator / EagleVAC) |
| `atlases/zaborszky_bf/Ch4_basal_forebrain_prob_MNI152.nii.gz` | **SI-BF/Ch4** (substantia innominata / basal forebrain) | Zaborszky et al. 2008 NeuroImage |
| `atlases/CIT168_CeA_prob_bilat_MNI152_1mm.nii.gz` | **CeA** (central amygdala) | Pauli et al. 2018 Sci Data (CIT168) |
| `atlases/Blackford_BNST_3T.nii.gz` | **BNST** (bed nucleus of the stria terminalis) | Theiss et al. 2017 NeuroImage (3T probabilistic) |
| `atlases/hypothalamus_neudorfer2020/atlas_labels_0.5mm.nii.gz` | **LH** (lateral hypothalamus) | Neudorfer et al. 2020 Sci Data |

All files are NIfTI. Most are probabilistic (values in [0, 1]);
Neudorfer's hypothalamus atlas is a **label** atlas (integer indices)
and the pipeline selects label values `{25, 26}` (lateral hypothalamic
area, left and right).

**Download instructions** (for a fresh install without HiPerGator access):
- **PBN (Brainstem Navigator / EagleVAC)**: https://www.nitrc.org/projects/brainstemnav/
- **Zaborszky BF Ch4**: https://identifiers.org/neurovault.collection:8102 (ch4 map)
- **CIT168 CeA**: https://osf.io/jkzwp/ (CIT168 reinforcement learning atlas)
- **Blackford BNST**: https://www.nitrc.org/projects/bnst_atlas
- **Neudorfer hypothalamus**: https://www.nature.com/articles/s41597-020-00644-6 (supplementary data)

All atlases are already in MNI152 space in the repo copy; third-party
downloads may be in native atlas space and would need `fsl-flirt` /
`antsApplyTransforms` to register into MNI152 first.

**Read by:**
`python/lib/moderator_loaders.py:load_fmri_atlas_arousal()` and
`load_vbm_atlas_arousal()`.

**HiPerGator location.**
```
/orange/cruzalmeida/pvaldeshernandez/Sleep-Pain_Coupling/UPLOAD2/atlases/
```

Symlinked into `data/atlases`.

---

## 5. Software environment

Installed with `conda env create -f environment.yml`. Key packages
and versions (what the sandbox run actually used):

| Package | Version | Purpose |
| --- | --- | --- |
| python | 3.13.x | interpreter |
| numpy | 1.26+ | arrays |
| scipy | 1.11+ | polychoric correlations, percentile |
| pandas | 2.0+ | DataFrames |
| pymc | 5.10+ | Bayesian VARX(1) model and NUTS sampler |
| pytensor | ≥ 2.18 | compute graph backend for PyMC |
| arviz | 0.17+ | posterior summaries, R-hat |
| nibabel | 5.1+ | NIfTI I/O |
| nilearn | 0.10+ | atlas resampling (`resample_to_img`) |
| matplotlib | 3.7+ | figures |
| openpyxl | 3.1+ | reading `participants_wideformat.xlsx` |

On HiPerGator:
```bash
module load conda/25.7.0
conda activate base
```
or
```bash
module load conda/25.3.1
```

Both interpreters have these packages installed.

---

## 6. Inputs NOT needed (internal to the pipeline)

These files appear in `data/` but are **produced by the pipeline
itself**, not inputs:

- `data/processed_data_contrast.csv` (produced by step 1)
- `data/factor_model_params_contrast.json` (produced by step 1)
- `data/factor_scores.csv` (legacy, not read by the python/ pipeline)

And all of `results/`, `figures/`, `sandbox/*` are outputs.

---

## 7. What's produced end-to-end

Running `python python/main.py --output-dir OUT` from scratch
produces the following in `OUT/`:

### Processed data (step 1)
- `processed_data_contrast.csv` (229 subjects, 1818 lag observations,
  with factor scores, within-between decomposition, lag-1 predictors,
  and interaction terms).
- `factor_model_params_contrast.json` (PAF loadings, item means/SDs,
  eigenvalues for the 2-factor pain model).

### Population-level coupling (step 2)
- `coupling_results.csv` (single-row summary of all 20+ population
  parameters with posterior mean, SD, 95% CrI, P(<0)).
- `coupling_summary.txt` (human-readable print of the above).
- `person_coupling_estimates.csv` (one row per subject with
  `beta_sp_mean`, `beta_sp_ci_lo/hi`, `beta_ps_mean`, `beta_ps_ci_lo/hi`
  — used by Figures 2 and 3).
- `contrast_posterior_draws.npz` (full posterior draws of a2, a4,
  b1, b4, random effects, per-observation contrast values — used
  by Figure 4 and Figure S3).

### Contrast moderation (step 3)
- `contrast_moderation_results.csv` (Table 4 row for the contrast ×
  coupling interaction, with simple slopes at −2 SD, 0, +2 SD).
- `contrast_jn_boundary.txt` (Johnson-Neyman analysis text summary).

### fMRI stimulation ROI moderation (step 4)
- `fmri_sp_moderation_results.csv` (Table 5; 7 rows for Contra_S1,
  Contra_Middle_Insula, Left_Thalamus, Left_Anterior_Insula,
  Left_NAcc, Right_NAcc, Right_dACC_MCC).
- `fmri_sp_jn_results.csv` (JN boundaries for the significant ROIs).
- `nacc_posterior_draws.npz` (Left NAcc draws for Figure 5).
- `acc_posterior_draws.npz` (ACC draws for Figure 6).
- `krause_roi_posterior_draws.npz` (4 non-NAcc Krause ROIs aggregated,
  for Figure S5).

### Arousal pathway moderation (step 5)
- `arousal_fmri_moderation_results.csv` (Table S1 fMRI rows — 5 ROIs).
- `arousal_vbm_moderation_results.csv` (Table S1 VBM rows — 5 ROIs).
- `arousal_jn_results.csv` (JN boundaries for any ROI with p < 0.10).
- `fmri_arousal_posterior_draws.npz` (5 ROIs aggregated, for Figure S7).
- `vbm_arousal_posterior_draws.npz` (5 ROIs aggregated, for Figure S8).

### Figures (step 6)

Main manuscript:
- `figures/figure1.png` — data availability grid (229 × 11 quarters).
- `figures/figure2.png` — Pain→Sleep person-specific coupling (forest + boxstrip).
- `figures/figure3.png` — Sleep→Pain person-specific coupling (forest + boxstrip).
- `figures/figure4.png` — Contrast moderation of Pain→Sleep coupling (JN).
- `figures/figure5.png` — Left NAcc moderation of Sleep→Pain coupling (JN).
- `figures/figure6.png` — ACC moderation of Sleep→Pain coupling (JN).

Supplementary:
- `figures/figure_s2.png` — Convergent validity of the contrast factor.
- `figures/figure_s3.png` — Contrast moderation of Sleep→Pain (null JN).
- `figures/figure_s5.png` — Krause non-significant ROI JN (2x2 merge).
- `figures/figure_s7.png` — Arousal fMRI BOLD JN panels (5 ROIs).
- `figures/figure_s8.png` — Arousal VBM GM volume JN panels (5 ROIs).

The remaining supplementary figures (S1 factor-validation endorsement
plot, S4 stim ROI MNI views, S6 arousal ROI MNI views) are generated
by **separate legacy scripts under `scripts/`** and are not part of
the `python/main.py` pipeline. They require:
- `endorsement_data.csv` (from `scripts/analyze_endorsement_options.py`).
- `stim_roi_maps.png` (pre-rendered ROI montage from
  `scripts/plot_stim_rois.py` or `rearrange_roi_maps.py`).
- `arousal_roi_maps.png` (pre-rendered montage from
  `scripts/plot_arousal_rois.py`).

If those files are placed inside `OUT/` (the output directory), step 6
will copy them into `OUT/figures/figure_s{1,4,6}.png`.

---

## 8. Minimal file checklist for a fresh install

To run `python python/main.py --output-dir sandbox/myrun` end-to-end
on a new machine, the `data/` folder must contain (or symlink to):

```
data/
├── quarterly_data_long.csv          (245 KB) REQUIRED, step 1
├── participants_wideformat.xlsx     (4.2 MB) REQUIRED, step 1 + fig S2
├── fmri_contrasts/                  (~41 GB) REQUIRED for NAcc (step 4)
│   └── <subj>/con_0001.nii            × 188 subjects
├── spm_nomask/                      (~200 MB) REQUIRED for other fMRI (steps 4, 5)
│   └── <subj>/con_0001.nii            × 188 subjects
├── vbm/                             (~1.7 GB) REQUIRED for VBM (step 5)
│   └── smwc1<subj>_ses-01_T1w.nii     × 189 subjects
└── atlases/                         (~331 MB) REQUIRED for arousal ROIs (step 5)
    ├── atlas_b2_brainstem.nii.gz           (PBN)
    ├── zaborszky_bf/Ch4_basal_forebrain_prob_MNI152.nii.gz  (SI-BF)
    ├── CIT168_CeA_prob_bilat_MNI152_1mm.nii.gz              (CeA)
    ├── Blackford_BNST_3T.nii.gz                              (BNST)
    └── hypothalamus_neudorfer2020/atlas_labels_0.5mm.nii.gz  (LH)
```

Everything else in `atlases/` (other CIT168 files, Juelich amygdala
variants, `holiatlas_v1.0/`, etc.) is exploratory and **not used**
by the current pipeline.

Total on disk: ~43 GB. The bulk (41 GB) is the `fmri_contrasts/`
directory, which contains the SPM first-level workflow artefacts
alongside the contrast images. If only the `con_0001.nii` files
were copied, the directory would be ~200 MB.

---

## 9. Synthetic data demo (no real data needed)

To test the pipeline without restricted data, use `--synthetic`:

```bash
python python/main.py --synthetic --output-dir sandbox/synth
```

This consumes only:
- `data/synthetic/quarterly_data_long.csv` (~35 KB, pre-simulated)
- `data/synthetic/participants_wideformat.csv` (~8 KB, demographics)
- `data/synthetic/roi_values.csv` (~12 KB, fMRI ROI values — one row per subject × ROI)
- `data/synthetic/vbm_volumes.csv` (~9 KB, VBM GM volumes)
- `data/synthetic/ground_truth.json` (known-true parameter values)

All of these are committed to the repo under `data/synthetic/` and
are produced by `simulate/generate_synthetic_data.py`.

The synthetic data preserves the real sample size (229/1818), the
observation structure (median 9 quarters per person), and the known
model parameters, so the pipeline's parameter recovery can be checked
against `ground_truth.json`.

---

## 10. Raw MRI data (NOT needed by this pipeline)

The raw T1w / BOLD DICOM / NIfTI images, motion-corrected functional
series, segmented tissue maps, DARTEL flow fields, and so on are
**not inputs to the `python/` pipeline**. They were inputs to the
**MATLAB preprocessing pipeline** (under `matlab/preprocessing/` and
`matlab/first_level/`), which produced the SPM contrast images that
the Python pipeline then consumes.

If you need to rerun the preprocessing from raw DICOM, see
`matlab/preprocessing/script_study.m` as the driver. That pipeline
requires SPM12, a DARTEL template-building step, individual T1w
segmentation, slice-timing correction, motion correction with unwarp,
normalization to MNI152, and a 5-block GLM — all together about a
week of wall time on HiPerGator for 229 subjects. See
`matlab/preprocessing/` and the main `README.md` for details.

This document **assumes** the MATLAB pipeline has already been run
and its outputs are staged in `data/spm_nomask/`, `data/fmri_contrasts/`,
and `data/vbm/`.
