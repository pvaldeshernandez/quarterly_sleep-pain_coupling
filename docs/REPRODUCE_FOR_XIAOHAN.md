# Reproducing the Results — Guide for Xiaohan

Hi Xiaohan, this walks you through reproducing every number and
figure in the paper from scratch on your own computer.

---

## 1. Setup

```bash
git clone git@github.com:pvaldeshernandez/quarterly_sleep-pain_coupling.git
cd quarterly_sleep-pain_coupling
conda env create -f environment.yml
conda activate sleep-pain-coupling
```

If SSH doesn't work, use HTTPS:
```bash
git clone https://github.com/pvaldeshernandez/quarterly_sleep-pain_coupling.git
```

Verify imports:
```bash
python -c "import pymc, arviz, nibabel, pandas; print('ok')"
```

---

## 2. Data

Pedro will send you the data files. Place them so the repo looks
like this:

```
data/
├── original/
│   ├── participants_wideformat.xlsx
│   └── UPLOAD2_Data_Dictionary.xlsx
├── fmri_contrasts/<subject_id>/con_0001.nii
├── spm_nomask/<subject_id>/con_0001.nii
├── vbm/smwc1<subject_id>_ses-01_T1w.nii
└── atlases/*.nii.gz
```

**Why both `fmri_contrasts/` and `spm_nomask/`?** NAcc and LH ROIs use
GM-masked contrasts (`fmri_contrasts/`). All other fMRI ROIs use
unmasked re-estimated contrasts (`spm_nomask/`), because subcortical
and brainstem ROIs lose coverage under per-subject GM masking.

---

## 3. Run the pipeline

From the repo root:

```bash
cd codes/python
python step00_extract_data.py
python step01_factor_analysis.py
python step02_contrast_validation.py
python step03_prepare_varx_data.py
python step04_fit_coupling_model.py
python step05_contrast_moderation.py
python step06_estimate_fmri_contrasts.py
python step07_extract_sp_rois.py
python step08_fit_sp_moderation.py
python step09_sp_moderation_jn.py
python step10_extract_ps_rois.py
python step11_fit_ps_moderation.py
python step12_ps_moderation_jn.py
python step13_severity_moderation.py
```

Total time: ~45 minutes on a 4-core machine with 16 GB RAM. Steps
4, 8, 11, and 13 are the slow ones (Bayesian model fitting via MCMC).

---

## 4. What each step does

| Step | What it does | What it produces |
|------|-------------|-----------------|
| 0 | Extract paper-relevant variables from the legacy wide-format xlsx, apply gateway imputation | `data/step00_extracted_long.csv` |
| 1 | Factor analysis (polychoric, 2-factor PAF, parallel analysis), Bartlett scoring, interpolation | Factor scores + model JSON |
| 2 | External validation of contrast factor: point-biserial correlations, convergent validity | Figures S1, S2 + text numbers |
| 3 | Segment filter (>=3 consecutive quarters), within-between decomposition, lag creation | VARX-ready data + Figure 1 + Table 3 |
| 4 | Fit the Bayesian VARX(1) coupling model + LOO-CV (4 nested models) | Tables 3-4, Figures 2-3, posterior draws |
| 5 | Johnson-Neyman analysis of pain localization moderation | Figure 4, Figure S3 |
| 6 | Re-estimate SPM contrast images without GM mask (OLS replication of SPM GLM) | Unmasked con images in `data/spm_nomask/` |
| 7 | Extract mean fMRI BOLD in 8 spherical ROIs (6 Krause + bilateral dACC/MCC) + brain maps | ROI values CSV, Figure S4 |
| 8 | Fit 8 Sleep-to-Pain moderation models, Krause sign concordance test | Table 5 + sign concordance |
| 9 | Johnson-Neyman analysis for SP moderation ROIs | Figures 5, 6, S5 |
| 10 | Extract arousal relay ROI values (5 atlas ROIs x 2 modalities: fMRI + VBM) + brain maps | ROI values CSVs, Figure S6 |
| 11 | Fit 10 Pain-to-Sleep moderation models, VBM sign concordance | Table S1 |
| 12 | Johnson-Neyman analysis for PS moderation ROIs | Figures S7, S8 |
| 13 | Person-mean severity moderation of coupling (3 models) | Table S2 |

---

## 5. Where to find outputs

- **`data/`** — only the Step 00 extraction. No other step writes here.
- **`derivatives/`** — intermediate files passed between steps (factor scores, processed data, posterior draws, ROI values). Each subfolder is named `stepN_description/`. You don't need to look at these unless debugging.
- **`results/`** — every table (CSV), figure (PNG), and text number (CSV) that appears in the manuscript:
  - Main-text figures and tables are in step-specific subfolders (e.g. `results/step04_coupling_model/`)
  - **All supplementary materials** (Figures S1-S8, Tables S1-S2) are in `results/supplementary_materials/`

---

## 6. Checking your results

Each step produces a `text_numbers.csv` companion that lists
every number stated in the manuscript text from that step. Compare
your values against the manuscript — they should match within MCMC
noise (|delta| <= 0.005 for coupling parameters, <= 0.02 for JN
boundaries).

If something looks off:
- **Sample size wrong?** Check that `data/original/` has the right
  xlsx files and that `data/fmri_contrasts/`, `data/spm_nomask/`,
  etc. are populated.
- **Factor scores different?** Step 01 uses polychoric correlations
  by default. Make sure you're running it without extra flags.
- **VBM N too low?** The VBM filenames use `x` instead of `-` in
  subject IDs (BIDS convention). Step 10 handles this mapping.
  If you see N < 189 for VBM, the mapping may need updating.

---

## 7. Replotting figures without refitting models

If you only need to change figure aesthetics (colors, fonts, axis
limits) without re-running MCMC:

```bash
python generate_all_results.py
```

This reads saved posterior draws from `derivatives/` and regenerates
all figures, tables, and text paragraphs. Takes ~30 seconds.

To re-run a specific step's computation from scratch (e.g., after
changing upstream data or code):

```bash
python step04_fit_coupling_model.py --refit
```

Every step supports the `--refit` flag. Without it, the step loads
saved derivatives and only regenerates results (figures, tables, text).
A warning is printed when running in this default mode.

---

## 8. What the MATLAB code does

You don't need to run the MATLAB pipeline — the neuroimaging
inputs (`fmri_contrasts/`, `spm_nomask/`, `vbm/`, `atlases/`) are
provided pre-computed. The MATLAB code is in `codes/matlab/` for
transparency; it documents how the raw DICOM files were
preprocessed into the NIfTI contrast images that the Python
pipeline reads.

---

## 9. Questions

Ping Pedro or open an issue on the repo.
