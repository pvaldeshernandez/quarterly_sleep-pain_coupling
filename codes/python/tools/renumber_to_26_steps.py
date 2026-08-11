"""Renumber the pipeline from 24 steps to 26, in one sentinel-protected pass.

Same machinery as the 13->23 migration this file grew from: the map is a PERMUTATION,
so every replacement writes a sentinel first and the sentinels are stripped only after
every rule has run. Applying a permutation twice is a different permutation, not a
no-op, so the run is one-shot and guarded by its own stamp.

This pass only MOVES steps. The two splits (step04 -> curation + varx prep,
step15 -> qc + nuisance-adjusted) are done afterwards, by hand, into the numbers this
map leaves free: 03 and 18.

    python tools/renumber_to_26_steps.py --dry-run
    python tools/renumber_to_26_steps.py
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CODE))

#: old step number -> new step number. Identity entries are listed so the map is
#: readable as the whole plan rather than as a diff.
MAP = {
    "00": "00",   # extract_data
    "01": "01",   # factor_analysis
    "02": "02",   # measurement_checks
                  # 03 data_curation            SPLIT OUT of prepare_varx_data, after
    "05": "04",   # raw_descriptives
    "03": "05",   # contrast_validation
    "06": "06",   # sleep_measure_correlates
    "04": "07",   # prepare_varx_data (keeps the VARX half only)
    "07": "08",   # fit_coupling_model
    "08": "09",   # posterior_predictive_check
    "10": "10",   # timevarying_covariates
    "09": "11",   # interpolation_sensitivity
    "11": "12",   # contrast_moderation
    "12": "13",   # estimate_fmri_contrasts
    "13": "14",   # extract_sp_rois
    "15": "15",   # imaging_qc (keeps the QC half only)
    "14": "16",   # fit_sp_moderation
    "16": "17",   # sp_moderation_jn
                  # 18 nuisance_adjusted        SPLIT OUT of imaging_qc, after
    "17": "19",   # ps_specificity
    "18": "20",   # extract_ps_rois
    "19": "21",   # fit_ps_moderation
    "20": "22",   # ps_moderation_jn
    "21": "23",   # severity_moderation
    "22": "24",   # diagnostics_summary
    "23": "25",   # optimal_lag
}

#: directories whose contents carry `stepNN` tokens
TEXT_ROOTS = [os.path.join(CODE), os.path.join(CODE, "lib"),
              os.path.join(CODE, "tools"), os.path.join(CODE, "revision")]

#: directories holding stepNN-named folders to rename
DIR_ROOTS = [os.path.join(ROOT, "derivatives"), os.path.join(ROOT, "results")]

TOKEN = re.compile(r"step(\d{2})")

#: Built from parts so the literal never appears in this file's own source.
SENT_L, SENT_R = "<" + "<", ">" + ">"


def moved(old):
    """True when a number actually changes; identity entries are left alone."""
    return MAP.get(old) not in (None, old)


def rewrite_text(path, dry):
    """Sentinel-protected rewrite of every `stepNN` token in one file."""
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    out = TOKEN.sub(lambda m: f"step{SENT_L}{MAP[m.group(1)]}{SENT_R}"
                    if m.group(1) in MAP else m.group(0), src)
    out = out.replace(SENT_L, "").replace(SENT_R, "")
    if out == src:
        return 0
    n = sum(1 for m in TOKEN.finditer(src) if moved(m.group(1)))
    if not dry:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(out)
    return n


def git_mv(src, dst, dry):
    if dry:
        return
    try:
        subprocess.run(["git", "mv", src, dst], cwd=ROOT, check=True,
                       capture_output=True)
    except subprocess.CalledProcessError:
        os.rename(src, dst)          # untracked paths (derivatives/, results/)


#: written once the migration has run; its presence makes a second run refuse
STAMP = os.path.join(CODE, ".renumbered_to_26_steps")


def main():
    dry = "--dry-run" in sys.argv
    assert len(set(MAP.values())) == len(MAP), "map is not injective"

    # A permutation applied twice is a DIFFERENT permutation, not a no-op: run this
    # again and contrast_validation goes 03 -> 04 on top of 02 -> 03, silently landing
    # on prepare_varx_data's number. There is no way to detect that after the fact, so
    # the only safe design is to make the second run impossible.
    if os.path.exists(STAMP):
        print(f"Already renumbered ({os.path.relpath(STAMP, ROOT)} exists).")
        print("This migration is ONE-SHOT: re-applying the map would corrupt the tree.")
        print("Delete the stamp only if you have restored the pre-migration numbering.")
        return 1

    # ---- 1. rewrite tokens, before anything moves ---------------------------
    total = 0
    for root in TEXT_ROOTS:
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            # Never rewrite this file, nor the tool that recorded the PREVIOUS
            # migration: its map documents history and must keep saying 13 -> 23.
            if path in (os.path.abspath(__file__),
                        os.path.join(HERE, "renumber_pipeline.py")):
                continue                      # never rewrite this file's own source
            if os.path.isfile(path) and name.endswith(".py"):
                n = rewrite_text(path, dry)
                if n:
                    print(f"  {os.path.relpath(path, ROOT):48s} {n} token(s)")
                    total += n
    print(f"\n{total} token(s) rewritten")

    # ---- 2. rename the scripts ---------------------------------------------
    renamed = 0
    for name in sorted(os.listdir(CODE), reverse=True):
        m = re.match(r"step(\d{2})_(.+\.py)$", name)
        if not m or not moved(m.group(1)):
            continue
        dst = f"step{MAP[m.group(1)]}_{m.group(2)}"
        print(f"  {name}  ->  {dst}")
        git_mv(os.path.join(CODE, name), os.path.join(CODE, dst), dry)
        renamed += 1

    # ---- 3. rename derivative and results folders --------------------------
    # Descending by TARGET number, so a folder is never renamed onto a name that is
    # still occupied by a folder yet to move.
    for root in DIR_ROOTS:
        if not os.path.isdir(root):
            continue
        entries = [e for e in os.listdir(root) if re.match(r"step\d{2}_", e)]
        for e in sorted(entries, key=lambda e: -int(MAP.get(e[4:6], e[4:6]))):
            old = e[4:6]
            if not moved(old):
                continue
            dst = f"step{MAP[old]}_{e[7:]}"
            print(f"  {os.path.relpath(os.path.join(root, e), ROOT)}  ->  {dst}")
            git_mv(os.path.join(root, e), os.path.join(root, dst), dry)
            renamed += 1

    # ---- 4. rename stepNN-prefixed FILES inside those folders --------------
    # The code addresses its own outputs by name (`step07_posterior_draws.npz`), and
    # step 1 has just rewritten those literals. Without this the load-and-plot default
    # path breaks on every step until a full refit happens to overwrite everything.
    files = 0
    for root in DIR_ROOTS:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                m = re.match(r"step(\d{2})_(.+)$", fn)
                if not m or not moved(m.group(1)):
                    continue
                dst = f"step{MAP[m.group(1)]}_{m.group(2)}"
                git_mv(os.path.join(dirpath, fn), os.path.join(dirpath, dst), dry)
                files += 1
    print(f"{files} file(s) renamed inside derivatives/ and results/")

    print(f"{renamed} path(s) renamed")
    if dry:
        print("\ndry run -- nothing written")
        return 0
    with open(STAMP, "w") as fh:
        fh.write("codes/python renumbered 13 -> 23 steps; see "
                 "docs/plans/2026-08-08-pipeline-fork-design.md\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
