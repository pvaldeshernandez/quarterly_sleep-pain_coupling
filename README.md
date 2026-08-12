# Quarterly Sleep-Pain Coupling in Knee Pain

Reproducible analysis code for:

> Valdes-Hernandez PA, Montesino-Goicolea S, Li X, Peraza JA, Weber E, Mickle AM, Staud R, Lai S, Sibille KT, Goodin BR, Fillingim RB, Cruz-Almeida Y. **Quarterly Sleep-Pain Coupling in Knee Pain: Pain-to-Sleep Dominance and NAcc-Gated Sleep-to-Pain Coupling.** *[Journal TBD]*, 2026.

---

## Quick Start

```bash
git clone git@github.com:pvaldeshernandez/quarterly_sleep-pain_coupling.git
cd quarterly_sleep-pain_coupling

conda env create -f environment.yml
conda activate sleep-pain-coupling
```

Place the data files into `data/original/` and the neuroimaging directories into `data/`
(see [Data Availability](#data-availability)). The atlases must be downloaded separately —
see [Atlases](#atlases). Then:

```bash
cd codes/python
python run_pipeline.py --refit     # recompute everything, in order
```

There is no list of steps to copy and paste. `run_pipeline.py` discovers them from the
filenames and runs them in numeric order, so **numeric order is execution order** — that is
the invariant the whole layout exists to protect.

```bash
python run_pipeline.py             # replot from saved derivatives (minutes)
python run_pipeline.py --refit     # recompute (hours; the 26 steps, in order)
python run_pipeline.py --list      # print the order and exit
python run_pipeline.py --from 07 --to 10
python run_pipeline.py --only 15,18
python run_pipeline.py --strict    # stop at the first failure instead of continuing
```

Without `--refit`, each step loads its saved derivative and redraws — so changing a figure's
styling costs seconds, not a refit. With `--refit`, it recomputes.

A full `--refit` of all 26 steps takes **three hours**, measured on a run from an empty
`derivatives/` and `results/`. Nearly all of it is sampling: step 08 takes 28 minutes for
its four models, step 18 forty-four for its sixteen, step 16 twenty-two for its eight.
**Step 13** re-estimates all 188 first-level GLMs in 8 minutes — it solves the OLS directly
from the SPM design matrices rather than re-running SPM, so it is not the bottleneck an
imaging step usually is.

Replotting everything from saved derivatives takes under three minutes.

---

## Reproducibility

Two properties are deliberate, and both were bought at the cost of a bug that took three
attempts to find. Do not undo either.

**The polychoric correlation is computed in closed form.** `lib/measurement.bvn_cdf` uses
Owen's T. `scipy.stats.multivariate_normal.cdf` integrates by randomized quasi-Monte-Carlo,
which made the factor loadings — and therefore every factor score and every number derived
from one — different on every run.

**Every step reads a CSV with `float_precision="round_trip"`.** The default pandas parser is
off by one unit in the last place. That is harmless in a report and not harmless upstream of
NUTS: a last-bit difference in a model covariate moves posterior means in the fourth decimal.

Together these make the pipeline bit-reproducible: two `--refit` runs from the same data
produce the same numbers.

---

## Repository Structure

```
quarterly_sleep-pain_coupling/
├── codes/
│   ├── python/
│   │   ├── run_pipeline.py              # the runner — discovers and orders the steps
│   │   ├── generate_all_results.py      # redraw everything from saved derivatives
│   │   ├── step00_extract_data.py .. step25_optimal_lag.py
│   │   ├── lib/                         # shared, concept-level functions
│   │   │   ├── coupling_model.py        # the Bayesian VARX(1) model — ONE implementation
│   │   │   ├── measurement.py           # polychoric correlation, Owen's T
│   │   │   ├── analytic_sample.py       # who is in the study, asked once
│   │   │   ├── nuisance.py              # residualization shared by steps 15 and 18
│   │   │   ├── descriptives.py, ppc.py, registry.py, heatmap.py
│   │   │   └── baseline_stability.py, sleep_instruments.py, stopbang.py
│   │   ├── tools/
│   │   │   ├── collect_deliverables.py  # step outputs -> document-facing names
│   │   │   ├── verify_refit_reproduces.py
│   │   │   ├── compare_interpolation_arms.py
│   │   │   └── make_functions_md.py     # regenerates FUNCTIONS.md from the code
│   │   └── FUNCTIONS.md                 # generated registry of all 297 callables
│   │
│   └── matlab/                          # MRI preprocessing (MATLAB/SPM12)
│       ├── preprocessing/               # DARTEL pipeline
│       ├── first_level/                 # first-level GLM
│       └── lib/                         # helpers
│
├── data/                                # inputs (not committed)
├── derivatives/                         # what a step computed — one folder per step
├── results/                             # what a step reported — one folder per step
│   ├── manuscript/                      # figure1.png .. figure6.png
│   ├── supplementary_materials/         # figureS*.png, tableS*.csv
│   └── reported_values.csv              # every named value, and the step that made it
│
├── environment.yml
├── README.md
└── LICENSE
```

**One step, one folder.** A step writes into `derivatives/stepNN_*/` and `results/stepNN_*/`
and nowhere else, under names that say what the output *shows* — not what number the
document gives it.

---

## How results reach the documents

A step cannot know that its figure is "Figure S6", because that changes: inserting one
supplement section renumbered six figures. So the numbering lives in exactly one place —
the `MANUSCRIPT` and `SUPPLEMENT` maps in `tools/collect_deliverables.py`.

```bash
python tools/collect_deliverables.py --dry-run   # show what would be copied
python tools/collect_deliverables.py             # copy, and rebuild reported_values.csv
```

It does two jobs:

- **Copies** each step's figures and tables into `results/manuscript/` and
  `results/supplementary_materials/` under their document-facing names. The copies are
  duplicates, not moves: the step folder stays the record of what that step produced.
- **Merges** every named value into `results/reported_values.csv` — name, value, producing
  step, source file. Values come from each step's `numbers.json` and from any CSV whose
  header contains a `metric` column. That last rule is by **schema, not filename**: steps
  name these files inconsistently, and globbing for `*text_numbers*` silently missed two,
  including the eigenvalues that open the Results.

`reported_values.csv` is what a checker needs to answer "does this sentence's number match
what the code produced". Renumbering the documents touches `collect_deliverables.py` and
nothing else.

The runner calls the collector automatically after a full run.

---

## Pipeline Overview

Step order follows the order the paper reports things, so every supplement section is
produced before it is cited. `--list` is the authority; this table is the commentary.

| Step | What it does |
|---|---|
| 00 | `extract_data` — pull paper-relevant variables out of the legacy wide-format xlsx |
| 01 | `factor_analysis` — 2-factor PAF on polychoric correlations, Horn's parallel analysis, Bartlett scoring |
| 02 | `measurement_checks` — invariance and congruence across samplings |
| 03 | `data_curation` — segment filter; who is in the analytic sample, and the figure and table that report it |
| 04 | `raw_descriptives` — person-quarter descriptives, ICCs, within/between decomposition |
| 05 | `contrast_validation` — external validation of the localization factor |
| 06 | `sleep_measure_correlates` — convergent validity and stability of the sleep measure |
| 07 | `prepare_varx_data` — within-between decomposition and lag construction |
| 08 | `fit_coupling_model` — the Bayesian VARX(1) fit, LOO-CV, the coupling figures |
| 09 | `posterior_predictive_check` |
| 10 | `timevarying_covariates` — sensitivity to time-varying confounders |
| 11 | `interpolation_sensitivity` |
| 12 | `contrast_moderation` — Johnson-Neyman for pain localization |
| 13 | `estimate_fmri_contrasts` — re-estimate the 188 first-level GLMs without the GM mask |
| 14 | `extract_sp_rois` — mean BOLD in the 8 sleep-to-pain ROIs, plus the ROI figure |
| 15 | `imaging_qc` — motion, scanner site, evoked pain; the descriptive half |
| 16 | `fit_sp_moderation` — the 8 sleep-to-pain moderation fits, sign concordance |
| 17 | `sp_moderation_jn` — Johnson-Neyman for the sleep-to-pain moderators |
| 18 | `nuisance_adjusted` — the 16 nuisance-adjusted refits (4 ROIs x 4 schemes) |
| 19 | `ps_specificity` — directional specificity controls |
| 20 | `extract_ps_rois` — arousal ROI values, fMRI and VBM |
| 21 | `fit_ps_moderation` — the 10 pain-to-sleep moderation fits, VBM sign concordance |
| 22 | `ps_moderation_jn` |
| 23 | `severity_moderation` — person-mean severity as a moderator |
| 24 | `diagnostics_summary` — R-hat, ESS and BFMI across all 52 fits |
| 25 | `optimal_lag` — the timescale at which coupling would peak |

**One order inversion is accepted, not a bug.** The model-diagnostics supplement section is
cited early but fed from both ends: the posterior predictive check at 09 and
`diagnostics_summary` at 24, which globs the diagnostics of all 52 fits and so cannot run
until every fit exists. Splitting the section to give it one producer would have cost 49
reference rewrites across three documents. Recorded so it is not rediscovered and "fixed".

---

## Atlases

Step 20 uses five published probabilistic atlases to define the pain-arousal relay ROIs
(PBN, SI-BF/Ch4, CeA, BNST, LH). These files are not distributed with this repository and
must be downloaded from the original sources, then placed under `data/atlases/` using the
exact filenames and subfolder structure below.

```
data/atlases/
├── atlas_b2_brainstem.nii.gz                                  (PBN)
├── atlas_b2_brainstem.txt                                     (optional metadata)
├── atlas_b2_brainstem_roi_numbers.txt                         (optional metadata)
├── Blackford_BNST_3T.nii.gz                                   (BNST)
├── CIT168_CeA_prob_bilat_MNI152_1mm.nii.gz                    (CeA; see note)
├── zaborszky_bf/
│   └── Ch4_basal_forebrain_prob_MNI152.nii.gz                 (SI-BF/Ch4)
└── hypothalamus_neudorfer2020/
    ├── atlas_labels_0.5mm.nii.gz                              (LH)
    └── Volumes_names-labels.csv
```

### Sources

| File(s) | Atlas | Source |
|---|---|---|
| `atlas_b2_brainstem.nii.gz` | Brainstem Navigator (Bianciardi lab) — labels 19 and 20 = left/right lateral parabrachial nucleus | <https://www.nitrc.org/projects/brainstemnavig/> |
| `Blackford_BNST_3T.nii.gz` | BNST probabilistic atlas (Theiss et al., *NeuroImage* 2017; Blackford lab) | <https://www.nitrc.org/projects/bnst_atlas/> |
| `CIT168_CeA_prob_bilat_MNI152_1mm.nii.gz` | Derived from the CIT168 in vivo subcortical atlas (Pauli et al., *Sci Data* 2018). The published atlas provides an extended-amygdala volume combining CeA + BNST; the CeA-specific probabilistic map used here was built from the crowd-sourced individual-observer AMY_CEN labelings in the CIT168 repository, averaged into a probability map and registered to MNI152 1 mm space (see Methods §"Pain-to-sleep ROIs"). | CIT168 atlas: <https://osf.io/jkzwp/> — the bilateral CeA-only file is not distributed by the original atlas; reconstruct it following the Methods, or request it from the corresponding author. |
| `zaborszky_bf/Ch4_basal_forebrain_prob_MNI152.nii.gz` | Probabilistic basal forebrain cytoarchitectonic atlas, Ch4 cell group (Zaborszky et al., *NeuroImage* 2008) | Distributed with SPM Anatomy Toolbox: <https://www.fz-juelich.de/en/inm/inm-7/resources/jubrain-anatomy-toolbox> (Ch4 map). |
| `hypothalamus_neudorfer2020/atlas_labels_0.5mm.nii.gz` + `Volumes_names-labels.csv` | Probabilistic hypothalamus atlas (Neudorfer et al., *Sci Data* 2020) — labels 25 and 26 = left/right lateral hypothalamus | <https://www.lead-dbs.org/helpsupport/knowledge-base/atlasesresources/cobralab-hypothalamic-subnuclei-atlas/> (or the publication's supplementary materials) |

### Notes

- File names and subfolder paths must match exactly. `step20_extract_ps_rois.py` references
  them by hard-coded path.
- All atlases are expected in MNI152 space. Step 20 resamples to the fMRI resolution (3 mm)
  for BOLD extraction and to VBM resolution (1.5 mm) for grey-matter volume extraction.
- The `.txt` files for the brainstem atlas are metadata only and not required.
- If the corresponding author can share a pre-built bundle of these atlases (subject to
  redistribution licenses), please request `data/atlases/` directly.

---

## MATLAB Pipeline

The MATLAB code preprocesses raw MRI and fits the first-level GLMs. It is **not needed for
the statistical analysis** — only for reproducing the neuroimaging preprocessing from raw
DICOM. Requires MATLAB R2018b or later and SPM12.

| Script | Description |
|---|---|
| `preprocessing/script_study.m` | Main driver — configure paths and run the whole preprocessing |
| `preprocessing/*.m` | DARTEL pipeline: slice timing, realign/unwarp, segmentation, template creation, normalization to MNI |
| `first_level/run_first_level.m` | First-level GLM, pain stimulation paradigm |
| `lib/*.m` | Helpers |

Two jobs that used to live here are now in Python, and the MATLAB versions have been
retired: the no-mask re-estimation is `step13_estimate_fmri_contrasts.py`, which solves the
OLS directly from the SPM design matrix, and ROI construction and extraction happen inside
steps 14 and 20.

---

## Data Availability

The data that support the findings of this study contain protected health information from
human subjects and cannot be shared publicly due to IRB and HIPAA restrictions. De-identified
data are available from the corresponding author upon reasonable request, subject to
institutional data use agreements.

| File | Description | Size |
|------|-------------|------|
| `data/original/participants_wideformat.xlsx` | Legacy wide-format dataset (all baseline + quarterly variables) | ~4 MB |
| `data/original/UPLOAD2_Data_Dictionary.xlsx` | Variable definitions and coding | ~220 KB |
| `data/fmri_contrasts/` | GM-masked first-level SPM contrast images (one `con_0001.nii` per subject) | ~41 GB |
| `data/spm_nomask/` | Unmasked re-estimated contrast images (written by step 13) | ~200 MB |
| `data/vbm/` | Smoothed modulated grey matter images (`smwc1*_ses-01_T1w.nii`) | ~1.7 GB |
| `data/atlases/` | Probabilistic arousal ROI atlases — see [Atlases](#atlases) | ~330 MB |

Step 00 writes `data/step00_extracted_long.csv`; it is the only step that writes into `data/`.

---

## Scope

This repository generates results. It contains no code that reads or writes a manuscript,
a supplement or a response document — those tools are kept outside it, because putting
results into documents is an interactive activity, not a pipeline stage.

---

## Citation

```
Valdes-Hernandez PA, Montesino-Goicolea S, Li X, Peraza JA, Weber E,
Mickle AM, Staud R, Lai S, Sibille KT, Goodin BR, Fillingim RB,
Cruz-Almeida Y. Quarterly Sleep-Pain Coupling in Knee Pain: Pain-to-Sleep
Dominance and NAcc-Gated Sleep-to-Pain Coupling. [Journal TBD], 2026.
```

---

## License

MIT License. See [LICENSE](LICENSE).

## Contact

Pedro A. Valdes-Hernandez — pvaldeshernandez@ufl.edu
