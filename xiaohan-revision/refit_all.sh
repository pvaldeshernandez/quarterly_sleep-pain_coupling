#!/bin/bash
# Full end-to-end refit after Xiaohan-review code fixes (D2, D4, D5, D6, D7).
# Runs every step that depends on the modified code: 04, 05, 07, 08, 09,
# 10, 11, 12, 13. Steps 00-03 are untouched by the fixes and do not need
# a refit.
#
# Usage: from the repo root, run `bash xiaohan-revision/refit_all.sh`.
set -e

# Resolve repo root as the parent of this script's directory.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$REPO_ROOT"

# PyTensor cache: reuse any configured location; otherwise cache inside repo.
: "${PYTENSOR_FLAGS:=base_compiledir=$REPO_ROOT/.pytensor_cache}"
export PYTENSOR_FLAGS

t0=$(date +%s)
for step in \
    step04_fit_coupling_model \
    step05_contrast_moderation \
    step07_extract_sp_rois \
    step08_fit_sp_moderation \
    step09_sp_moderation_jn \
    step10_extract_ps_rois \
    step11_fit_ps_moderation \
    step12_ps_moderation_jn \
    step13_severity_moderation; do
    echo "============================================================"
    echo "RUNNING $step  (elapsed $(( $(date +%s) - t0 ))s)"
    echo "============================================================"
    python codes/python/${step}.py --refit
done
echo "============================================================"
echo "DONE  (total $(( $(date +%s) - t0 ))s)"
echo "============================================================"
