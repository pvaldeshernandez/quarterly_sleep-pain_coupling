# Atlas Sources for ROI Definition

This directory should contain the atlas files used by the `create_*_atlas_roi.m`
scripts in the parent directory. Due to licensing and file size, most atlases are
**not included** in this repository. Below are the download instructions and
citations for each.

---

## 1. Brainstem Navigator (PBN)

**Used by:** `create_pbn_atlas_roi.m`

**Citation:**
Singh K, Indovina I, Augustinack JC, et al. An optimized probabilistic atlas of
brainstem nuclei in the human brain using the EagleVAC framework. 2024 (in
preparation).

**License:** Freely available for academic use.

**Download:** <https://www.nitrc.org/projects/brainstemnavig/>

**File needed:** `BrainstemNavigator_labels_v0.9.nii.gz` (or latest version).
This is an integer-labeled volume at 0.5 mm resolution in MNI152 space.

**Labels used:** 19 (left lateral PBN), 20 (right lateral PBN).

**Included in repo:** No.

---

## 2. Zaborszky Basal Forebrain Atlas (SI/BF Ch4)

**Used by:** `create_sibf_atlas_roi.m`

**Citation:**
Zaborszky L, Hoemke L, Mohlberg H, Schleicher A, Amunts K, Zilles K.
Stereotaxic probabilistic maps of the magnocellular cell groups in human basal
forebrain. NeuroImage. 2008;42(3):1127-1141.
doi:[10.1016/j.neuroimage.2008.05.055](https://doi.org/10.1016/j.neuroimage.2008.05.055)

**License:** Available through the SPM Anatomy Toolbox (academic use).

**Download:** <https://www.fz-juelich.de/en/inm/inm-1/research/atlases/>
(also distributed with the SPM Anatomy Toolbox).

**File needed:** Ch4 probability map NIfTI (continuous values, 0 to 1 or 0 to
100%). The exact filename depends on the toolbox version.

**Included in repo:** No.

---

## 3. CIT168 Subcortical Atlas (CeA)

**Used by:** `create_cea_atlas_roi.m`

**Citation:**
Pauli WM, Nili AN, Tyszka JM. A high-resolution probabilistic in vivo atlas of
human subcortical brain nuclei. Scientific Data. 2018;5:180063.
doi:[10.1038/sdata.2018.63](https://doi.org/10.1038/sdata.2018.63)

**License:** CC-BY-4.0.

**Download:** <https://osf.io/jkzwp/>

**Files needed:**
- Individual-observer labeling volumes (`CIT168_*_label_*.nii.gz`)
- CIT168 T1w template image (`CIT168_T1w_700um.nii.gz` or similar)

**Labels used:** 4 (AMY\_CEN = central nucleus of amygdala).

**Note:** The CIT168 atlas is in its own template space. The creation script
registers it to MNI152 using FSL FLIRT + FNIRT, which requires a standard
MNI152 T1 1 mm template (e.g., from `$FSLDIR/data/standard/`).

**Included in repo:** No.

---

## 4. Theiss BNST Atlas (BNST)

**Used by:** `create_bnst_atlas_roi.m`

**Citation:**
Theiss JD, Ridgewell C, McHugo M, Heckers S, Blackford JU. Manual segmentation
of the human bed nucleus of the stria terminalis using 3T MRI. NeuroImage.
2017;146:288-292.
doi:[10.1016/j.neuroimage.2016.11.047](https://doi.org/10.1016/j.neuroimage.2016.11.047)

**License:** Available from the authors upon request.

**Download:** Contact the corresponding author (JU Blackford, Vanderbilt
University).

**File needed:** BNST probability map NIfTI in MNI152 space.

**Included in repo:** No.

---

## 5. Neudorfer Hypothalamic Atlas (LH)

**Used by:** `create_lh_atlas_roi.m`

**Citation:**
Neudorfer C, Germann J, Elias GJB, Gramer R, Boutet A, Lozano AM. A
high-resolution in vivo magnetic resonance imaging atlas of the human
hypothalamic region. Scientific Data. 2020;7:305.
doi:[10.1038/s41597-020-00644-6](https://doi.org/10.1038/s41597-020-00644-6)

**License:** CC-BY-4.0.

**Download:** <https://zenodo.org/record/3942115>

**File needed:** Hypothalamic atlas label NIfTI (integer labels in MNI152 space).

**Labels used:** 25 (left lateral hypothalamic area), 26 (right lateral
hypothalamic area).

**Included in repo:** No.

---

## Note on Spherical ROIs

The Krause et al. (2019) and Xu et al. (2020) spherical ROIs do **not** require
atlas files. They are constructed algorithmically from MNI coordinates and radii.
See `create_krause_spherical_rois.m` and `create_acc_spherical_roi.m`.
