"""Replace every figure in the submission with the one the pipeline just produced.

Two different mechanisms, because the two documents store figures differently:

  * the MANUSCRIPT embeds nothing — its six figures are separate `figureN.tiff` files
    submitted alongside it, so they are converted from the regenerated PNGs;
  * the SUPPLEMENT embeds all nine, each stored TWICE (a DrawingML copy under
    `mc:Choice` and a VML copy under `mc:Fallback`), so eighteen media parts move.

THE TRAP THIS SCRIPT EXISTS TO AVOID. The supplement was renumbered when Section S4 was
inserted, but the pipeline's filenames were not. `figure_s3_jn_localization_sp.png` is
Figure **S4** in the document. Mapping by number would put the Johnson-Neyman panel under
"Per-quarter stability of sleep quality correlations" and nothing would complain — the
caption and the image would simply disagree. Every pairing below was therefore verified
against the caption TEXT, and the script re-verifies it at run time.

Figure S3 comes from the sandbox (`a10d_render_heatmap.py`) because step 06 is not built
yet; it is the only figure not produced by a numbered step.

    python tools/update_figures.py --dry-run
    python tools/update_figures.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CODE))
sys.path.insert(0, os.path.join(CODE, "revision"))

RESULTS = os.path.join(ROOT, "results")
DERIV = os.path.join(ROOT, "derivatives")
SUB = os.path.join(ROOT, "docs/manuscript_JPAIN_resubmission")

#: manuscript figure -> the PNG the pipeline writes
MANUSCRIPT = {
    "figure1.tiff": f"{RESULTS}/step04_varx_data/step04_figure1.png",
    "figure2.tiff": f"{RESULTS}/step07_coupling_model/step07_figure2_ps_coupling.png",
    "figure3.tiff": f"{RESULTS}/step07_coupling_model/step07_figure3_sp_coupling.png",
    "figure4.tiff": f"{RESULTS}/step11_contrast_moderation/step11_figure4_jn_localization_ps.png",
    "figure5.tiff": f"{RESULTS}/step16_sp_jn/step16_figure5_jn_nacc.png",
    "figure6.tiff": f"{RESULTS}/step16_sp_jn/step16_figure6_jn_acc.png",
}

#: supplement figure label -> (source PNG, a distinctive phrase of its caption)
#: The phrase is the guard: it is checked against the live caption before the swap.
SUPPLEMENT = {
    "Figure S1": (f"{RESULTS}/supplementary_materials/figure_s1_endorsement.png",
                  "PHQ body map endorsements"),
    "Figure S2": (f"{RESULTS}/supplementary_materials/figure_s2_convergent.png",
                  "Convergent validity"),
    # Step 06 now renders this; it used to exist only as a sandbox output, so the
    # supplement's Figure S3 had no pipeline source. The two files are byte-identical,
    # so this is a provenance change, not a content one.
    "Figure S3": (f"{RESULTS}/supplementary_materials/figure_s3_sleep_stability_heatmap.png",
                  "Per-quarter stability"),
    "Figure S4": (f"{RESULTS}/supplementary_materials/figure_s3_jn_localization_sp.png",
                  "pain localization moderation"),
    "Figure S5": (f"{RESULTS}/supplementary_materials/figure_s4_stim_rois.png",
                  "Spherical regions of interest"),
    "Figure S6": (f"{RESULTS}/supplementary_materials/figure_s5_krause_jn.png",
                  "non-credible Krause ROI"),
    "Figure S7": (f"{RESULTS}/supplementary_materials/figure_s6_arousal_rois.png",
                  "Atlas-defined probabilistic"),
    "Figure S8": (f"{RESULTS}/supplementary_materials/figure_s7_fmri_arousal_jn.png",
                  "fMRI BOLD moderation"),
    "Figure S9": (f"{RESULTS}/supplementary_materials/figure_s8_vbm_arousal_jn.png",
                  "matter volume moderation"),
}


def update_manuscript_tiffs(dry):
    """Convert each regenerated PNG to the TIFF the journal receives."""
    from PIL import Image
    n = 0
    for tiff, png in MANUSCRIPT.items():
        if not os.path.exists(png):
            print(f"  {tiff:14s} SKIP — {os.path.relpath(png, ROOT)} not produced yet")
            continue
        dst = os.path.join(SUB, tiff)
        im = Image.open(png)
        if im.mode == "RGBA":                     # TIFF via LZW dislikes alpha
            bg = Image.new("RGB", im.size, "white")
            bg.paste(im, mask=im.split()[3])
            im = bg
        print(f"  {tiff:14s} <- {os.path.relpath(png, ROOT):58s} {im.size}")
        if not dry:
            im.save(dst, format="TIFF", compression="tiff_lzw", dpi=(300, 300))
        n += 1
    return n


def update_supplement(dry):
    """Swap both stored copies of each embedded figure, guarded by its caption."""
    from docx_edit import Doc
    d = Doc("supplementary_materials.docx")
    swapped = 0
    for i, p in enumerate(d.paragraphs):
        t = d.text_of(p).replace("\xa0", " ").strip()
        m = re.match(r"(Figure S\d+)\.", t)
        if not m:
            continue
        parts = d.image_parts_for(p)
        if not parts:
            continue
        label = m.group(1)
        if label not in SUPPLEMENT:
            print(f"  {label}: no source mapped — left alone")
            continue
        png, phrase = SUPPLEMENT[label]
        # The guard. Mapping by number alone would silently pair a caption with the
        # wrong panel, which is exactly what the renumbering made possible.
        assert phrase in t, (f"{label}: caption does not contain {phrase!r} — the "
                             f"figure mapping is stale, refusing to swap")
        if not os.path.exists(png):
            print(f"  {label:11s} SKIP — {os.path.relpath(png, ROOT)} not produced yet")
            continue
        assert len(parts) == 2, f"{label}: expected 2 stored copies, found {parts}"
        for part in parts:
            if not dry:
                d.replace_media(part, png)
        print(f"  {label:11s} <- {os.path.relpath(png, ROOT):58s} x{len(parts)} copies")
        swapped += 1
    if not dry and swapped:
        d.save()
    return swapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=== manuscript (separate TIFF files) ===")
    n_t = update_manuscript_tiffs(args.dry_run)
    print("\n=== supplement (embedded, two copies each) ===")
    n_s = update_supplement(args.dry_run)

    print(f"\n{n_t} manuscript figure(s), {n_s} supplement figure(s)")
    if args.dry_run:
        print("dry run -- nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
