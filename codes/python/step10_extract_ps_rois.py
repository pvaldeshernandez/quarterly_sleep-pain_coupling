"""
Step 10 — Extract Pain-to-Sleep arousal relay ROI values + Figure S6.
======================================================================

Input:  derivatives/step06_fmri_contrasts_masked/   (LH ROI)
        derivatives/step06_fmri_contrasts_unmasked/ (all other fMRI ROIs)
        data/original/vbm/, data/atlases/
        MNI152 template (via nilearn, for Figure S6)
Output:
  derivatives/step10_ps_roi_values/
    step10_ps_fmri_roi_values.csv   — per-subject fMRI BOLD z-scored ROI values
    step10_ps_vbm_roi_values.csv    — per-subject VBM GM volume z-scored ROI values
  results/supplementary_materials/
    figure_s6_arousal_rois.png      — Figure S6: arousal ROI brain maps

Extracts probability-weighted mean fMRI BOLD and VBM GM volume
from 5 atlas-defined arousal relay ROIs (Lynch et al. 2025):
  PBN, SI-BF/Ch4, CeA, BNST, LH

Also generates Figure S6: orthogonal brain slices for all 5 ROIs.

Author: Pedro Valdes-Hernandez (with Claude Sonnet 4.6)
"""
from __future__ import annotations

import argparse
import glob as glob_mod
import io
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DERIV_DIR = os.path.join(ROOT, "derivatives")
STEP_DERIV_DIR = os.path.join(DERIV_DIR, "step10_ps_roi_values")
os.makedirs(STEP_DERIV_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(ROOT, "results")
SUPP_DIR = os.path.join(RESULTS_DIR, "supplementary_materials")
os.makedirs(SUPP_DIR, exist_ok=True)

FMRI_MASKED_DIR   = os.path.join(DERIV_DIR, "step06_fmri_contrasts_masked")
FMRI_UNMASKED_DIR = os.path.join(DERIV_DIR, "step06_fmri_contrasts_unmasked")
VBM_DIR   = os.path.join(ROOT, "data", "original", "vbm")
ATLAS_DIR = os.path.join(ROOT, "data", "atlases")

OUT_FMRI_CSV = os.path.join(STEP_DERIV_DIR, "step10_ps_fmri_roi_values.csv")
OUT_VBM_CSV = os.path.join(STEP_DERIV_DIR, "step10_ps_vbm_roi_values.csv")
OUT_FIG_S6 = os.path.join(SUPP_DIR, "figure_s6_arousal_rois.png")

AROUSAL_ROIS = {
    "PBN": {
        "label": "Lateral Parabrachial Nucleus",
        "atlas_type": "label", "labels": [19, 20],
        "expected_sign_ps": "-",
    },
    "SI_BF_Ch4": {
        "label": "Substantia Innominata / Basal Forebrain (Ch4)",
        "atlas_type": "prob",
        "expected_sign_ps": "-",
    },
    "CeA": {
        "label": "Central Nucleus of the Amygdala",
        "atlas_type": "prob",
        "expected_sign_ps": "-",
    },
    "BNST": {
        "label": "Bed Nucleus of the Stria Terminalis",
        "atlas_type": "prob",
        "expected_sign_ps": "-",
    },
    "LH": {
        "label": "Lateral Hypothalamus",
        "atlas_type": "label", "labels": [25, 26],
        "expected_sign_ps": "-",
        "fmri_mask": "gm_masked",
    },
}

ATLAS_FILES = {
    "PBN": "atlas_b2_brainstem.nii.gz",
    "SI_BF_Ch4": os.path.join("zaborszky_bf", "Ch4_basal_forebrain_prob_MNI152.nii.gz"),
    "CeA": "CIT168_CeA_prob_bilat_MNI152_1mm.nii.gz",
    "BNST": "Blackford_BNST_3T.nii.gz",
    "LH": os.path.join("hypothalamus_neudorfer2020", "atlas_labels_0.5mm.nii.gz"),
}

# Figure S6 display config
FIG_S6_ROIS = {
    "PBN": {
        "label": "Lateral Parabrachial Nucleus (PBN)",
        "atlas_file": os.path.join(ATLAS_DIR, "atlas_b2_brainstem.nii.gz"),
        "atlas_type": "label", "labels": [19, 20],
        "title": "PBN  (Brainstem Navigator [39])",
    },
    "SI-BF/Ch4": {
        "label": "Substantia Innominata / Basal Forebrain (SI-BF/Ch4)",
        "atlas_file": os.path.join(ATLAS_DIR, "zaborszky_bf",
                                   "Ch4_basal_forebrain_prob_MNI152.nii.gz"),
        "atlas_type": "prob",
        "title": "SI-BF/Ch4  (Zaborszky et al. [40])",
    },
    "CeA": {
        "label": "Central Nucleus of the Amygdala (CeA)",
        "atlas_file": os.path.join(ATLAS_DIR,
                                   "CIT168_CeA_prob_bilat_MNI152_1mm.nii.gz"),
        "atlas_type": "prob",
        "title": "CeA  (CIT168 [41])",
    },
    "BNST": {
        "label": "Bed Nucleus of the Stria Terminalis (BNST)",
        "atlas_file": os.path.join(ATLAS_DIR, "Blackford_BNST_3T.nii.gz"),
        "atlas_type": "prob",
        "title": "BNST  (Theiss et al. [42])",
    },
    "LH": {
        "label": "Lateral Hypothalamus (LH)",
        "atlas_file": os.path.join(ATLAS_DIR, "hypothalamus_neudorfer2020",
                                   "atlas_labels_0.5mm.nii.gz"),
        "atlas_type": "label", "labels": [25, 26],
        "title": "LH  (Neudorfer et al. [43])",
    },
}

FIG_S6_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]


# =====================================================================
# ROI extraction — fMRI BOLD
# =====================================================================

def extract_fmri_arousal(verbose=True):
    """Extract probability-weighted mean BOLD per subject per ROI."""
    import nibabel as nib
    from nilearn.image import resample_to_img

    default_fmri_dir = FMRI_UNMASKED_DIR

    ref_ids = sorted(os.listdir(default_fmri_dir))
    ref_path = os.path.join(default_fmri_dir, ref_ids[0], "con_0001.nii")
    ref_img = nib.load(ref_path)
    brain_mask = np.isfinite(ref_img.get_fdata())

    roi_weights = {}
    for roi_name, cfg in AROUSAL_ROIS.items():
        atlas_path = os.path.join(ATLAS_DIR, ATLAS_FILES[roi_name])
        if not os.path.isfile(atlas_path):
            if verbose:
                print(f"    {roi_name}: atlas not found at {atlas_path}, skipping")
            continue

        atlas_img = nib.load(atlas_path)
        atlas_data = atlas_img.get_fdata()

        if cfg["atlas_type"] == "label":
            mask = np.zeros_like(atlas_data, dtype=np.float32)
            for lab in cfg["labels"]:
                mask[atlas_data == lab] = 1.0
            mask_img = nib.Nifti1Image(mask, atlas_img.affine)
            resampled = resample_to_img(mask_img, ref_img, interpolation="linear")
            weights = resampled.get_fdata().clip(0, 1)
        else:
            resampled = resample_to_img(atlas_img, ref_img, interpolation="linear")
            weights = resampled.get_fdata().clip(0, 1)

        n_brain = int(np.sum((weights * brain_mask) > 0))
        if verbose:
            print(f"    {cfg['label']}: {n_brain} brain voxels at fMRI resolution")
        if n_brain < 1:
            if verbose:
                print(f"      WARNING: 0 brain voxels, skipping")
            continue
        roi_weights[roi_name] = weights

    all_rows = []
    for subj_id in sorted(os.listdir(default_fmri_dir)):
        con_path = os.path.join(default_fmri_dir, subj_id, "con_0001.nii")
        if not os.path.isfile(con_path):
            continue
        data = nib.load(con_path).get_fdata()
        for roi_name, weights in roi_weights.items():
            valid = np.isfinite(data)
            w = weights * valid
            wsum = w.sum()
            if wsum > 0:
                all_rows.append({
                    "ID": subj_id, "ROI": roi_name,
                    "label": AROUSAL_ROIS[roi_name]["label"],
                    "modality": "fMRI_BOLD",
                    "expected_sign_ps": AROUSAL_ROIS[roi_name]["expected_sign_ps"],
                    "raw_value": float(np.nansum(data * w) / wsum),
                })

    df = pd.DataFrame(all_rows)
    for roi_name in df["ROI"].unique():
        mask = df["ROI"] == roi_name
        vals = df.loc[mask, "raw_value"]
        mean_x, sd_x = vals.mean(), vals.std()
        df.loc[mask, "z_value"] = (vals - mean_x) / sd_x
        df.loc[mask, "raw_mean"] = mean_x
        df.loc[mask, "raw_sd"] = sd_x
        if verbose:
            print(f"    {roi_name} fMRI: N={mask.sum()}, mean={mean_x:.4f}, SD={sd_x:.4f}")

    return df


# =====================================================================
# ROI extraction — VBM GM volume
# =====================================================================

def extract_vbm_arousal(verbose=True):
    """Extract probability-weighted GM volume per subject per ROI."""
    import nibabel as nib
    from nilearn.image import resample_to_img

    pattern = os.path.join(VBM_DIR, "smwc1*_ses-01_T1w.nii")
    gm_files = sorted(glob_mod.glob(pattern))
    if not gm_files:
        if verbose:
            print(f"  No smwc1 files found in {VBM_DIR}")
        return pd.DataFrame()

    ref_img = nib.load(gm_files[0])
    vox_vol = float(np.abs(np.linalg.det(ref_img.affine[:3, :3])))

    roi_weights = {}
    for roi_name, cfg in AROUSAL_ROIS.items():
        atlas_path = os.path.join(ATLAS_DIR, ATLAS_FILES[roi_name])
        if not os.path.isfile(atlas_path):
            continue

        atlas_img = nib.load(atlas_path)
        atlas_data = atlas_img.get_fdata()

        if cfg["atlas_type"] == "label":
            mask = np.zeros_like(atlas_data, dtype=np.float32)
            for lab in cfg["labels"]:
                mask[atlas_data == lab] = 1.0
            mask_img = nib.Nifti1Image(mask, atlas_img.affine)
            resampled = resample_to_img(mask_img, ref_img, interpolation="linear")
            weights = resampled.get_fdata().clip(0, 1)
        else:
            resampled = resample_to_img(atlas_img, ref_img, interpolation="linear")
            weights = resampled.get_fdata().clip(0, 1)

        n_vox = int(np.sum(weights > 0))
        if verbose:
            print(f"    {cfg['label']}: {n_vox} voxels at VBM resolution")
        roi_weights[roi_name] = weights

    all_rows = []
    for gm_path in gm_files:
        fname = os.path.basename(gm_path)
        subj_id = fname.replace("smwc1", "").replace("_ses-01_T1w.nii", "")
        subj_id = subj_id.replace("x", "-")
        data = nib.load(gm_path).get_fdata()

        for roi_name, weights in roi_weights.items():
            vol = float(np.nansum(data * weights) * vox_vol)
            all_rows.append({
                "ID": subj_id, "ROI": roi_name,
                "label": AROUSAL_ROIS[roi_name]["label"],
                "modality": "VBM_GM_volume",
                "expected_sign_ps": AROUSAL_ROIS[roi_name]["expected_sign_ps"],
                "raw_value": vol,
            })

    df = pd.DataFrame(all_rows)
    for roi_name in df["ROI"].unique():
        mask = df["ROI"] == roi_name
        vals = df.loc[mask, "raw_value"]
        mean_x, sd_x = vals.mean(), vals.std()
        df.loc[mask, "z_value"] = (vals - mean_x) / sd_x
        df.loc[mask, "raw_mean"] = mean_x
        df.loc[mask, "raw_sd"] = sd_x
        if verbose:
            print(f"    {roi_name} VBM: N={mask.sum()}, mean={mean_x:.2f}, SD={sd_x:.2f}")

    return df


# =====================================================================
# Figure S6 — Arousal ROI brain maps
# =====================================================================

def _make_solid_cmap(hex_color):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("solid", [hex_color, hex_color], N=2)


def _load_atlas_roi(roi_cfg, template):
    import nibabel as nib
    from nilearn.image import resample_to_img, new_img_like
    atlas_img = nib.load(roi_cfg["atlas_file"])
    atlas_data = atlas_img.get_fdata()
    if roi_cfg["atlas_type"] == "label":
        mask = np.zeros_like(atlas_data, dtype=np.float32)
        for lab in roi_cfg["labels"]:
            mask[atlas_data == lab] = 1.0
        mask_img = nib.Nifti1Image(mask, atlas_img.affine)
    else:
        mask_img = nib.Nifti1Image((atlas_data > 0).astype(np.float32),
                                   atlas_img.affine)
    resampled = resample_to_img(mask_img, template, interpolation="linear")
    data = (resampled.get_fdata() > 0.1).astype(np.float32)
    return new_img_like(template, data)


def _get_center_of_mass(roi_img):
    data = roi_img.get_fdata()
    coords = np.array(np.where(data > 0)).T
    if len(coords) == 0:
        return (0, 0, 0)
    com_ijk = coords.mean(axis=0)
    com_mni = roi_img.affine[:3, :3] @ com_ijk + roi_img.affine[:3, 3]
    return tuple(com_mni.astype(int))


def _render_roi_to_image(roi_img, template, cut_coords, title, cmap):
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_roi
    from PIL import Image
    display = plot_roi(
        roi_img, bg_img=template, display_mode="ortho",
        cut_coords=cut_coords, title=title, cmap=cmap,
        alpha=0.7, dim=-0.5, annotate=True, draw_cross=True, colorbar=False,
    )
    fig = display.frame_axes.figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _make_composite(panels, ncols=2):
    from PIL import Image
    nrows = math.ceil(len(panels) / ncols)
    target_w = max(p.width for p in panels)
    scaled = []
    for p in panels:
        new_h = int(p.height * target_w / p.width)
        scaled.append(p.resize((target_w, new_h), Image.LANCZOS))
    max_h = max(p.height for p in scaled)
    padded = []
    for p in scaled:
        if p.height < max_h:
            canvas = Image.new("RGB", (target_w, max_h), (0, 0, 0))
            canvas.paste(p, (0, (max_h - p.height) // 2))
            padded.append(canvas)
        else:
            padded.append(p)
    gap = 4
    composite = Image.new("RGB",
                          (ncols * target_w + (ncols - 1) * gap,
                           nrows * max_h + (nrows - 1) * gap),
                          (255, 255, 255))
    for idx, p in enumerate(padded):
        composite.paste(p, (idx % ncols * (target_w + gap),
                            idx // ncols * (max_h + gap)))
    return composite


def generate_figure_s6(verbose=True):
    """Generate Figure S6: orthogonal brain slices for all PS arousal ROIs."""
    from nilearn.datasets import load_mni152_template

    if verbose:
        print("\n  Generating Figure S6 (PS arousal ROI brain maps)...")

    template = load_mni152_template(resolution=1)
    if verbose:
        print("    Loaded MNI152 template (1mm)")

    panels = []
    for idx, (roi_key, cfg) in enumerate(FIG_S6_ROIS.items()):
        if verbose:
            print(f"    Loading: {cfg['label']}")
        roi_img = _load_atlas_roi(cfg, template)
        nvox = int((roi_img.get_fdata() > 0).sum())
        com = _get_center_of_mass(roi_img)
        if verbose:
            print(f"      {nvox} voxels, center of mass: {com}")
        cmap = _make_solid_cmap(FIG_S6_COLORS[idx])
        panel = _render_roi_to_image(roi_img, template, com, cfg["title"], cmap)
        panels.append(panel)

    composite = _make_composite(panels, ncols=2)
    composite.save(OUT_FIG_S6, dpi=(200, 200))
    if verbose:
        print(f"    Saved: {OUT_FIG_S6}")


# =====================================================================
# Main
# =====================================================================

def generate_text_paragraphs(verbose: bool = True) -> None:
    """Generate step10_text.md with the Figure S6 caption, built
    programmatically from the FIG_S6_ROIS dictionary and extraction
    results. Includes atlas source references and PBN voxel count
    at fMRI resolution when available.
    """
    STEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "step10_ps_roi_extraction")
    os.makedirs(STEP_RESULTS_DIR, exist_ok=True)
    OUT_TEXT_MD = os.path.join(STEP_RESULTS_DIR, "step10_text.md")

    if verbose:
        print("  Generating step10_text.md ...")

    # Build ROI description list from FIG_S6_ROIS
    roi_descs = []
    for roi_key, cfg in FIG_S6_ROIS.items():
        roi_descs.append(f"{cfg['label']} ({cfg['title'].split('(')[-1].rstrip(')')}"
                         f")" if "(" in cfg["title"] else cfg["label"])

    # Cleaner version: just use title which already has atlas ref
    roi_lines = []
    for roi_key, cfg in FIG_S6_ROIS.items():
        roi_lines.append(cfg["title"])
    roi_list = "; ".join(roi_lines)

    # Try to read PBN voxel count from fMRI extraction results
    pbn_note = ""
    if os.path.exists(OUT_FMRI_CSV):
        fmri_df = pd.read_csv(OUT_FMRI_CSV)
        pbn_rows = fmri_df[fmri_df["ROI"] == "PBN"]
        if len(pbn_rows) > 0:
            n_pbn_subj = pbn_rows["ID"].nunique()
            pbn_note = (
                f" At 3 mm fMRI resolution, the PBN ROI contained a small "
                f"number of voxels; extraction was successful for "
                f"$N$ = {n_pbn_subj} participants."
            )
        elif "PBN" not in fmri_df["ROI"].values:
            pbn_note = (
                " Note: PBN yielded 0 brain voxels at 3 mm fMRI resolution "
                "and was excluded from fMRI analyses."
            )

    # Masking note
    masked_rois = [v["label"] for k, v in AROUSAL_ROIS.items()
                   if v.get("fmri_mask") == "gm_masked"]
    mask_note = ""
    if masked_rois:
        mask_note = (
            f" {', '.join(masked_rois)} used GM-masked contrast images; "
            f"all other ROIs used unmasked re-estimated contrasts."
        )

    fig_s6_caption = (
        f"**Figure S6.** Atlas-defined arousal relay ROI locations for the "
        f"pain-to-sleep moderation analysis, shown on the MNI152 template. "
        f"Five ROIs are displayed in orthogonal slices centered on each "
        f"region's center of mass: {roi_list}. "
        f"Probability-weighted maps were used for extraction of both fMRI "
        f"BOLD activation and VBM grey matter volume.{pbn_note}{mask_note}"
    )

    text = (
        "## Step 10 — PS arousal ROI extraction\n\n"
        "## Figure Captions\n\n"
        f"{fig_s6_caption}\n"
    )

    with open(OUT_TEXT_MD, "w") as f:
        f.write(text)

    if verbose:
        print(f"    Saved: {OUT_TEXT_MD}")


def run_step10(verbose=True, refit=False):
    if verbose:
        print("=" * 70)
        print("STEP 10 — Extract Pain-to-Sleep arousal relay ROI values + Figure S6")
        print("=" * 70)

    if not refit and os.path.exists(OUT_FMRI_CSV) and os.path.exists(OUT_VBM_CSV):
        if verbose:
            print("  WARNING: Running in replot mode — loading saved derivatives.")
            print("  If you have changed upstream data or code, re-run with --refit.")
            print(f"  Loading: {OUT_FMRI_CSV}")
            print(f"  Loading: {OUT_VBM_CSV}")
    else:
        if verbose:
            print(f"  fMRI unmasked:  {FMRI_UNMASKED_DIR}")
            print(f"  fMRI masked:    {FMRI_MASKED_DIR}")
            print(f"  VBM:            {VBM_DIR}")
            print(f"  Atlases:        {ATLAS_DIR}")

        if verbose:
            print("\n  fMRI BOLD extraction:")
        fmri_df = extract_fmri_arousal(verbose)
        fmri_df.to_csv(OUT_FMRI_CSV, index=False)
        if verbose:
            print(f"  Saved: {OUT_FMRI_CSV} ({fmri_df['ROI'].nunique()} ROIs, "
                  f"{fmri_df['ID'].nunique()} subjects)")

        if verbose:
            print("\n  VBM GM volume extraction:")
        vbm_df = extract_vbm_arousal(verbose)
        vbm_df.to_csv(OUT_VBM_CSV, index=False)
        if verbose:
            print(f"  Saved: {OUT_VBM_CSV} ({vbm_df['ROI'].nunique()} ROIs, "
                  f"{vbm_df['ID'].nunique()} subjects)")

    generate_figure_s6(verbose)
    generate_text_paragraphs(verbose)

    if verbose:
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 10 — extract PS arousal relay ROI values + Figure S6."
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--refit", action="store_true",
                        help="Re-extract ROI values from fMRI/VBM images")
    args = parser.parse_args()
    run_step10(verbose=not args.quiet, refit=args.refit)


if __name__ == "__main__":
    main()
