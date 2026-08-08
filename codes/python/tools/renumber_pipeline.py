"""Renumber the pipeline from 13 steps to 23, in one sentinel-protected pass.

Existing steps keep their names and take new numbers; nine new steps are inserted at the
positions the design fixes. Scripts, derivative folders, results folders and every
`stepNN` token in the tree move together.

WHY SENTINELS. The map is a PERMUTATION: 04 -> 07 and 07 -> 13 both appear, so a naive
sequential rewrite lets a value that has already been renamed be caught again by a later
rule, and applying the whole map twice produces a different permutation entirely. That is
not a hypothetical — it is exactly how the supplement's ten table titles were corrupted on
Aug 6 and had to be repaired on Aug 7. Every replacement therefore writes `<<NN>>` first and the sentinels are stripped only after every rule has run.

The sentinel is assembled at runtime rather than written as a literal, and this file
excludes itself from the sweep. Both are necessary: on the first run this script
processed its own source and its own sentinel-stripping line deleted the sentinels
from its own string literals, leaving `out.replace("", "")`. A tool that edits source
files must not be in its own input set.

Renames go through `git mv` so history follows the file.

    python tools/renumber_pipeline.py --dry-run
    python tools/renumber_pipeline.py
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
                  # 02 measurement_checks        NEW
    "02": "03",   # contrast_validation
    "03": "04",   # prepare_varx_data
                  # 05 raw_descriptives          NEW
                  # 06 sleep_measure_correlates  NEW
    "04": "07",   # fit_coupling_model
                  # 08 posterior_predictive_check NEW
                  # 09 interpolation_sensitivity  NEW
                  # 10 timevarying_covariates     NEW
    "05": "11",   # contrast_moderation
    "06": "12",   # estimate_fmri_contrasts
    "07": "13",   # extract_sp_rois
    "08": "14",   # fit_sp_moderation
                  # 15 imaging_qc                NEW
    "09": "16",   # sp_moderation_jn
                  # 17 ps_specificity            NEW
    "10": "18",   # extract_ps_rois
    "11": "19",   # fit_ps_moderation
    "12": "20",   # ps_moderation_jn
    "13": "21",   # severity_moderation
                  # 22 diagnostics_summary       NEW
}

#: directories whose contents carry `stepNN` tokens
TEXT_ROOTS = [os.path.join(CODE), os.path.join(CODE, "lib"), os.path.join(CODE, "tools")]

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
STAMP = os.path.join(CODE, ".renumbered_to_23_steps")


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
            if path == os.path.abspath(__file__):
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
