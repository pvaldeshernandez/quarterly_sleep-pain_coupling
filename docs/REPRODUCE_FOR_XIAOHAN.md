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

**Why both `fmri_contrasts/` and `spm_nomask/`?** NAcc ROIs use
GM-masked contrasts (`fmri_contrasts/`). All other fMRI ROIs use
unmasked re-estimated contrasts (`spm_nomask/`), because subcortical
and brainstem ROIs lose coverage under per-subject GM masking.

---

## 3. Run the pipeline

From the repo root:

```bash
cd python
python step0_extract_data.py
python step1_factor_analysis.py
python step2_prepare_varx_data.py
python step3_fit_coupling_model.py
python step4_contrast_moderation.py
python step5_extract_sp_rois.py
python step6_fit_sp_moderation.py
python step7_sp_moderation_jn.py
python step8_extract_ps_rois.py
python step9_fit_ps_moderation.py
python step10_ps_moderation_jn.py
python step11_remaining_figures.py
```

Total time: ~30 minutes on a 4-core machine with 16 GB RAM. Steps
3, 6, and 9 are the slow ones (Bayesian model fitting).

---

## 4. What each step does

| Step | What it does | What it produces |
|------|-------------|-----------------|
| 0 | Extract paper-relevant variables from the legacy wide-format xlsx, apply gateway imputation | `data/step0_extracted_long.csv` |
| 1 | Factor analysis (polychoric, 2-factor PAF, parallel analysis), Bartlett scoring, interpolation | Factor scores + model JSON |
| 2 | Segment filter (≥3 consecutive quarters), within-between decomposition, lag creation | VARX-ready data + Figure 1 + Table 3 |
| 3 | Fit the Bayesian VARX(1) coupling model + LOO-CV (4 nested models) | Table 4 + LOO comparison + posterior draws |
| 4 | Johnson-Neyman analysis of pain localization moderation | Figures 4, S3 |
| 5 | Extract mean fMRI BOLD in 7 spherical ROIs (6 Krause + ACC) | ROI values CSV |
| 6 | Fit 7 Sleep-to-Pain moderation models, sign concordance test | Table 5 + sign concordance |
| 7 | Johnson-Neyman analysis for SP moderation ROIs | Figures 5, 6, S5 |
| 8 | Extract arousal relay ROI values (5 atlas ROIs x 2 modalities: fMRI + VBM) | ROI values CSVs |
| 9 | Fit 10 Pain-to-Sleep moderation models, VBM sign concordance | Table S1 |
| 10 | Johnson-Neyman analysis for PS moderation ROIs | Figures S7, S8 |
| 11 | Person-level coupling plots, factor validation, convergent validity | Figures 2, 3, S1, S2 |

---

## 5. Where to find outputs

- **`data/`** — only the Step 0 extraction. No other step writes here.
- **`derivatives/`** — intermediate files passed between steps (factor scores, processed data, posterior draws, ROI values). You don't need to look at these unless debugging.
- **`results/`** — every table (CSV), figure (PNG), and text number (CSV) that appears in the manuscript. Each file is prefixed with its step number so you know where it came from.

---

## 6. Checking your results

Each results file has a `text_numbers.csv` companion that lists
every number stated in the manuscript text from that step. Compare
your values against the manuscript — they should match within MCMC
noise (|Δ| ≤ 0.005 for coupling parameters, ≤ 0.02 for JN
boundaries).

If something looks off:
- **Sample size wrong?** Check that `data/original/` has the right
  xlsx files and that `data/fmri_contrasts/`, `data/spm_nomask/`,
  etc. are populated.
- **Factor scores different?** Step 1 uses polychoric correlations
  by default. Make sure you're running it without extra flags.
- **VBM N too low?** The VBM filenames use `x` instead of `-` in
  subject IDs (BIDS convention). Step 8 handles this mapping.
  If you see N < 189 for VBM, the mapping may need updating.

---

## 7. What the MATLAB code does

You don't need to run the MATLAB pipeline — the neuroimaging
inputs (`fmri_contrasts/`, `spm_nomask/`, `vbm/`, `atlases/`) are
provided pre-computed. The MATLAB code is in the repo for
transparency; it documents how the raw DICOM files were
preprocessed into the NIfTI contrast images that the Python
pipeline reads.

---

## 8. Questions

Ping Pedro or open an issue on the repo.
