#!/usr/bin/env python3
"""Copy each step's figures and tables into the two document folders.

    results/stepNN_*/           what a step produced, named for what it shows
        |
        v  this script
    results/manuscript/         figure1.png .. figure6.png
    results/supplementary_materials/   figureS1.png .. figureS9.png, tableS9_*.csv ..
    results/reported_values.csv        every named value, with the step that produced it

WHY THIS EXISTS. Steps write into their own folder -- one step, one folder -- and name
their outputs for what they SHOW, because a step cannot know what number the document
will give it. The document decides that, and it changes: inserting Section S4 last week
shifted six figure numbers, and splitting Section S9 yesterday shifted four more.

So the numbering lives in exactly one place: MANUSCRIPT and SUPPLEMENT below. Renumber
the documents and only this file moves. Everything downstream then works by NAME --
`figureS6.png` is Figure S6, with nothing to look up -- which is why
`docs/tools/update_figures.py` needs no map of its own.

THE THIRD OUTPUT is `reported_values.csv`: every value the pipeline names, merged from
each step's `numbers.json` and `step*_text_numbers.csv`. Figures and tables are not the
only deliverables -- most of what a reader checks is stated in PROSE, and until now those
values lived scattered across twelve registries and nine text-number files with nothing
gathering them. One file, three columns (name, value, step), is what a checker needs to
answer "does this sentence's number match what the code produced".

The copies are deliberate duplicates, not moves. A step's folder stays the record of what
that step produced; the document folders are a view of it.

    python tools/collect_deliverables.py --dry-run
    python tools/collect_deliverables.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CODE))
RESULTS = os.path.join(ROOT, "results")

OUT_MANUSCRIPT = os.path.join(RESULTS, "manuscript")
OUT_SUPPLEMENT = os.path.join(RESULTS, "supplementary_materials")

#: document name -> the step folder and file that produces it.
MANUSCRIPT = {
    "figure1.png": ("step03_curation", "step03_figure1.png"),
    "figure2.png": ("step08_coupling_model", "step08_figure2_ps_coupling.png"),
    "figure3.png": ("step08_coupling_model", "step08_figure3_sp_coupling.png"),
    "figure4.png": ("step12_contrast_moderation", "step12_figure4_jn_localization_ps.png"),
    "figure5.png": ("step17_sp_jn", "step17_figure5_jn_nacc.png"),
    "figure6.png": ("step17_sp_jn", "step17_figure6_jn_acc.png"),
}

SUPPLEMENT = {
    "figureS1.png": ("step05_contrast_validation", "figure_endorsement.png"),
    "figureS2.png": ("step05_contrast_validation", "figure_convergent.png"),
    "figureS3.png": ("step06_sleep_measure_correlates", "figure_sleep_stability_heatmap.png"),
    "figureS4.png": ("step12_contrast_moderation", "figure_jn_localization_sp.png"),
    "figureS5.png": ("step14_sp_roi_extraction", "figure_stim_rois.png"),
    "figureS6.png": ("step17_sp_jn", "figure_krause_jn.png"),
    "figureS7.png": ("step20_ps_roi_extraction", "figure_arousal_rois.png"),
    "figureS8.png": ("step22_ps_jn", "figure_fmri_arousal_jn.png"),
    "figureS9.png": ("step22_ps_jn", "figure_vbm_arousal_jn.png"),
    "tableS9_fmri_arousal.csv": ("step21_ps_moderation", "table_s1_fmri_arousal.csv"),
    "tableS9_vbm_arousal.csv": ("step21_ps_moderation", "table_s1_vbm_arousal.csv"),
    "tableS10_severity.csv": ("step23_severity_moderation", "table_s2_severity.csv"),
}


def collect(mapping, outdir, dry):
    os.makedirs(outdir, exist_ok=True)
    ok = missing = 0
    for doc_name, (step_dir, src_name) in sorted(mapping.items()):
        src = os.path.join(RESULTS, step_dir, src_name)
        if not os.path.exists(src):
            print(f"  MISSING  {doc_name:26s} <- {step_dir}/{src_name}")
            missing += 1
            continue
        print(f"  ok       {doc_name:26s} <- {step_dir}/{src_name}")
        if not dry:
            shutil.copy2(src, os.path.join(outdir, doc_name))
        ok += 1
    return ok, missing


def collect_values(dry):
    """Merge every named value into one file: name, value, producing step."""
    import csv
    import glob
    import json

    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*", "numbers.json"))):
        step = os.path.basename(os.path.dirname(path))
        for key, val in json.load(open(path)).items():
            # keys are already "stepNN.name"; the folder is the authority on which step
            rows.append({"name": key.split(".", 1)[-1], "value": val,
                         "step": step, "source": "numbers.json"})

    # Found by SCHEMA, not by filename. Steps name these inconsistently --
    # step01_factor_results.csv, step08_text_numbers.csv, text_numbers_severity.csv --
    # and globbing "*text_numbers*" silently missed two of them, including step01's
    # eigenvalues, which open the Results. Any CSV whose header is metric,value is a
    # named-value file whatever it is called.
    for root in (os.path.join(ROOT, "derivatives"), RESULTS):
        for path in sorted(glob.glob(os.path.join(root, "*", "*.csv"))):
            step = os.path.basename(os.path.dirname(path))
            try:
                with open(path, newline="") as fh:
                    rdr = csv.DictReader(fh)
                    if not rdr.fieldnames or "metric" not in rdr.fieldnames:
                        continue
                    for r in rdr:
                        if r.get("metric"):
                            rows.append({"name": r["metric"],
                                         "value": r.get("value", ""),
                                         "step": step,
                                         "source": os.path.basename(path)})
            except (OSError, UnicodeDecodeError):
                continue

    out = os.path.join(RESULTS, "reported_values.csv")
    dup = len(rows) - len({(r["name"], r["step"]) for r in rows})
    print(f"  {len(rows)} named value(s) from {len({r['step'] for r in rows})} step(s)"
          + (f"   ({dup} duplicate name+step)" if dup else ""))
    if not dry:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["name", "value", "step", "source"])
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {os.path.relpath(out, ROOT)}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=== manuscript ===")
    a_ok, a_missing = collect(MANUSCRIPT, OUT_MANUSCRIPT, args.dry_run)
    print("\n=== supplementary materials ===")
    b_ok, b_missing = collect(SUPPLEMENT, OUT_SUPPLEMENT, args.dry_run)

    print("\n=== reported values ===")
    n_vals = collect_values(args.dry_run)

    print(f"\n{a_ok + b_ok} file(s) collected, {a_missing + b_missing} missing, "
          f"{n_vals} named value(s)")
    if args.dry_run:
        print("dry run -- nothing copied")
    return 1 if (a_missing + b_missing) else 0


if __name__ == "__main__":
    sys.exit(main())
