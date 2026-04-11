# Reproducing the Quarterly Sleep-Pain Coupling Results — Guide for Xiaohan

Hi Xiaohan — this file walks you through reproducing every number and
figure in the quarterly sleep-pain coupling paper from scratch, on
your own computer, using the data Pedro will send you separately.
By the end you will have:

- All tables (3, 4, 5) reproduced from posterior draws.
- All main figures (1-6) regenerated.
- All supplementary figures that the Python pipeline covers (S2,
  S3, S5, S7, S8).
- The LOO-CV model-comparison numbers reported in Results §3.1.
- The Krause 6/6 sign-concordance test from Discussion.

You do **not** need any cluster access. Everything runs on a single
laptop with ≥ 16 GB of RAM.

---

## 1. What Pedro will send you

Pedro will give you a folder containing:

```
data/
├── quarterly_data_long.csv              # quarterly pain + sleep items per subject
├── participants_wideformat.xlsx         # baseline variables (demographics + WOMAC + PHQ)
├── factor_model_params_contrast.json    # precomputed 2-factor PAF model
├── fmri_contrasts/                      # GM-masked first-level SPM contrasts
│   ├── <subject_id_1>/
│   │   └── con_0001.nii                 # one per subject
│   └── ...
├── spm_nomask/                          # unmasked re-estimated contrasts
│   ├── <subject_id_1>/
│   │   └── con_0001.nii                 # one per subject
│   └── ...
├── vbm/                                 # VBM gray matter images
│   └── smwc1<subject_id>_ses-01_T1w.nii
│       # one per subject with VBM data
└── atlases/                             # probabilistic arousal ROI atlases
    ├── atlas_b2_brainstem.nii.gz             # PBN
    ├── Ch4_basal_forebrain_prob_MNI152.nii.gz # SI-BF / Ch4
    ├── CIT168_CeA_prob_bilat_MNI152_1mm.nii.gz # CeA
    ├── Blackford_BNST_3T.nii.gz               # BNST
    └── atlas_labels_0.5mm.nii.gz              # LH
```

### Why each file is needed

| File / folder | What it feeds | Why both masked and unmasked fMRI |
| --- | --- | --- |
| `quarterly_data_long.csv` | Step 1 — within-person decomposition, lagged variables, contrast scores; also provides baseline Age and Sex for the coupling model | Long-format quarterly items |
| `participants_wideformat.xlsx` | Step 6 — **Figure S2 only** (convergent-validity scatter) | Baseline WOMAC / PHQ / QST columns. If you skip Figure S2 you don't need this file at all. |
| `factor_model_params_contrast.json` | Step 1 — precomputed 2-factor PAF loadings | Factor scores are computed here, not refit |
| `fmri_contrasts/` (GM-masked) | Step 4 — **Left NAcc, Right NAcc** Sleep→Pain moderation (Table 5) | NAcc ROIs were reported with the GM-masked contrasts in the paper |
| `spm_nomask/` (unmasked) | Step 4 — S1, Middle Insula, Thalamus, Anterior Insula, dACC/MCC Sleep→Pain moderation; Step 5 — all fMRI arousal ROIs Pain→Sleep moderation | Subcortical and brainstem ROIs lose coverage under per-subject GM masking, so these use the re-estimated unmasked contrasts |
| `vbm/` (smwc1 GM images) | Step 5 — VBM arousal ROI volumes (Table S1, VBM panel) | Pain→Sleep structural moderation |
| `atlases/` | Step 5 — probability-weighted extraction of the five Lynch arousal ROIs | PBN, SI-BF/Ch4, CeA, BNST, LH |

You do **not** need the `data/synthetic/` folder and you should not
run anything with `--synthetic`.

---

## 2. Install Python + conda

Any of the following conda distributions works:

- [Miniforge](https://github.com/conda-forge/miniforge/releases)
  (recommended — conda-forge channel by default)
- Anaconda or Miniconda

On macOS or Linux the install is a one-line shell script from the
Miniforge releases page. On Windows install the `.exe` and open the
"Miniforge Prompt" from the Start menu for all subsequent commands.

Verify:

```bash
conda --version   # should print e.g. "conda 24.x"
```

---

## 3. Clone the repo

```bash
git clone git@github.com:pvaldeshernandez/quarterly_sleep-pain_coupling.git
cd quarterly_sleep-pain_coupling
```

If you don't have SSH keys set up with GitHub yet, use HTTPS:

```bash
git clone https://github.com/pvaldeshernandez/quarterly_sleep-pain_coupling.git
```

---

## 4. Create the conda environment

From the repo root:

```bash
conda env create -f environment.yml
conda activate sleep-pain-coupling
```

This installs Python 3.11+, PyMC 5.27.1, ArviZ 0.23.4, NumPy,
SciPy, pandas, matplotlib, scikit-learn, PyTensor, nibabel, and
openpyxl. Creation typically takes 3-5 minutes.

Smoke test that the imports work:

```bash
python -c "import pymc, arviz, nibabel, pandas; print('ok')"
```

---

## 5. Drop the data in place

Unpack the data folder Pedro sent you so that the following files
exist relative to the repo root:

```
quarterly_sleep-pain_coupling/
├── data/
│   ├── quarterly_data_long.csv
│   ├── participants_wideformat.xlsx
│   ├── factor_model_params_contrast.json
│   ├── fmri_contrasts/<subject_id>/con_0001.nii
│   ├── spm_nomask/<subject_id>/con_0001.nii
│   ├── vbm/smwc1<subject_id>_ses-01_T1w.nii
│   └── atlases/*.nii.gz
```

You can also put the data anywhere and point the pipeline at it
with `--data-dir /some/other/path`. The pipeline first looks under
`--data-dir`, and if it can't find a neuroimaging subdirectory
there (e.g., `spm_nomask/`) it falls back to the repo default
`data/` — so a mixed setup also works if you prefer to keep the
big NIfTI folders on an external drive and only copy the CSVs into
the repo.

---

## 6. Run the full pipeline

From the repo root:

```bash
cd python
python main.py --output-dir ../results --figures-dir ../figures
```

What this does, in order:

1. **`01_prepare_data.py`** — reads the quarterly CSV, applies the
   precomputed 2-factor PAF, runs within-person decomposition,
   builds lagged + interaction variables. Writes
   `processed_data_contrast.csv` into the output dir. Also runs
   default-on interpolation of single-quarter missingness to
   retain the N=229 / 1,818 sample (use `--no-interpolate` if you
   want to skip that).
2. **`02_fit_coupling_model.py`** — fits the Bayesian VARX(1)
   coupling model with 4 chains × 2,000 draws. Writes
   `coupling_results.csv`, `coupling_summary.txt`,
   `person_coupling_estimates.csv`, `contrast_posterior_draws.npz`.
   Takes about 30 s on 4 cores.
3. **`03_contrast_moderation.py`** — Johnson-Neyman analysis of
   contrast moderation for Sleep→Pain and Pain→Sleep. Writes
   `contrast_jn_boundary.txt`, `contrast_moderation_results.csv`.
4. **`04_fmri_sp_moderation.py`** — reads `fmri_contrasts/` and
   `spm_nomask/`, fits the 6 Krause ROIs + the ACC ROI as
   Sleep→Pain moderators, runs Johnson-Neyman for each, runs the
   sign-concordance test. Writes
   `fmri_sp_moderation_results.csv`, `fmri_sp_jn_results.csv`,
   `nacc_posterior_draws.npz`, `acc_posterior_draws.npz`,
   `krause_roi_posterior_draws.npz`. Takes about 4 min.
5. **`05_arousal_ps_moderation.py`** — reads the five Lynch
   arousal atlases, fits both fMRI BOLD and VBM volume versions
   as Pain→Sleep moderators (10 models total). Writes
   `arousal_fmri_moderation_results.csv`,
   `arousal_vbm_moderation_results.csv`,
   `fmri_arousal_posterior_draws.npz`,
   `vbm_arousal_posterior_draws.npz`. Takes about 2 min.
6. **`06_generate_figures.py`** — reads every posterior draws npz
   and every results CSV and generates the main figures (1-6)
   and the supplementary figures that the pipeline covers.

Total wall time: **~10 minutes** on a 4-core laptop with 16 GB RAM.

If you want to run individual steps, each script also accepts
`--data-dir` and `--output-dir` the same way `main.py` does. For
example:

```bash
python 02_fit_coupling_model.py --output-dir ../results
```

### Optional: LOO-CV model comparison

The LOO-CV results in Results §3.1 are not run by default because
they take an extra ~5 min. To include them, pass `--loo` to step 2:

```bash
python 02_fit_coupling_model.py --output-dir ../results --loo
```

That fits four nested models (full / no-PS / no-SP / null) with
cores=1 (sequential chains) and writes
`loo_comparison.csv`, `loo_pairwise.csv`, `loo_comparison.txt`.

---

## 7. Check the numbers against the manuscript

`sandbox/REPRODUCTION_AUDIT.md` in the repo lists every numerical
claim in the paper (Tables 3, 4, 5, S1, S2; LOO ΔELPD values;
person-level range statistics; JN boundaries; sign-concordance
test) and cross-checks it against a sandbox run done on the same
pipeline. Every number should reproduce to within MCMC Monte Carlo
error (typically |Δ| ≤ 0.005 on the coupling parameters, and to
the second decimal on the LOO Δ/SE values).

If you see differences larger than that, check these first:

- **Sample size off by 13 or so?** Step 1's default-on
  interpolation is what keeps N=229 instead of dropping to 216.
  If you passed `--no-interpolate` you'll see the smaller sample.
- **PBN fMRI value different?** The PBN atlas mask intersected
  with the brain mask lands on the edge of the fMRI field of
  view. A 1-voxel difference in the intersection flips the sign.
  The substantive conclusion (PBN fMRI is non-credible) is
  preserved either way — see the PBN section of
  `sandbox/REPRODUCTION_AUDIT.md`.
- **Anything else different by > 0.005?** Open an issue or ping
  Pedro and me; the commit log has the fix history.

---

## 8. What the pipeline does *not* reproduce

These are in the paper but live outside the `python/` pipeline:

| Claim | Where it lives | Why |
| --- | --- | --- |
| Factor analysis eigenvalues, loadings, parallel analysis | Precomputed into `factor_model_params_contrast.json`; original fit is in a legacy MATLAB script | The Python pipeline applies the factor model, it does not refit |
| Table 2 demographics (race, BMI, WOMAC means, KL grade) | Descriptive statistics of `participants_wideformat.xlsx` | Just needs a small descriptive script; we didn't write one |
| Table S2 severity moderation | `scripts/run_severity_moderation.py` (legacy) | Not ported to `python/` yet |
| Figures S1, S4, S6 | Legacy plot scripts | Require auxiliary inputs (`endorsement_data.csv`, MNI slice renders) |

None of these are reproduction failures — they're just not wired
into the Python pipeline. If you need any of them, ping Pedro.

---

## 9. Code layout

```
python/
├── main.py                         # orchestrator
├── 01_prepare_data.py              # step 1
├── 02_fit_coupling_model.py        # step 2 (Aim 1 fit + optional LOO)
├── 03_contrast_moderation.py       # step 3 (Aim 2 contrast moderation)
├── 04_fmri_sp_moderation.py        # step 4 (Krause + ACC fMRI moderation)
├── 05_arousal_ps_moderation.py     # step 5 (Lynch arousal moderation)
├── 06_generate_figures.py          # step 6 (all figures)
└── lib/
    ├── coupling_model.py           # shared Bayesian model, JN, LOO
    └── moderator_loaders.py        # NIfTI I/O, atlas extraction, z-scoring
```

The model lives in `python/lib/coupling_model.py` —
`fit_bayesian_varx1` is the single PyMC model builder used by
every step. When you want to understand the math, start there.
The companion paper Methods §2.4-§2.5 describes the model in the
same notation the code uses internally.

---

## 10. Common pitfalls

- **`KeyError: q13_sleep_quality`** — the quarterly CSV is missing
  the sleep-quality column. Check the column names with
  `pd.read_csv(...).columns.tolist()` and make sure the file you
  received matches what step 1 expects.
- **`FileNotFoundError: ... con_0001.nii`** — the `fmri_contrasts/`
  or `spm_nomask/` directory layout isn't quite right. Each
  subject needs its own subdirectory named by `ID` containing
  `con_0001.nii` directly (no extra nesting).
- **OOM during LOO** — the LOO run is memory-bounded, not
  compute-bounded. Close Chrome and anything else heavy before
  running it. If you still OOM, let Pedro or me know; we can
  reduce the chain count for your laptop.
- **`libscipy_openblas64_` errors** — something was `pip install`ed
  into the conda env. Run `pip uninstall <package>` for the
  offending package and then `conda install <package>` instead.

---

## 11. Questions

Ping Pedro or open an issue on the repo. Happy analyzing, Xiaohan —
you'll find the code straightforward once you've run it once
through.
