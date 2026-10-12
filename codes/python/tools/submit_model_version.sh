#!/bin/bash
# Refit the whole paper under one model version on SLURM, as dependent jobs.
#
#   bash codes/python/tools/submit_model_version.sh latent
#
# Outputs go to derivatives_<model>/ and results_<model>/ (lib/model_version.py);
# logs to derivatives_<model>/logs/. Stages, each a run_pipeline.py --refit call:
#   A  00-08   data preparation and the primary fit with its LOO models
#   B  09-12   checks and sensitivity analyses on the primary fit   (after A)
#   C  13-23   imaging moderation, specificity and severity          (after A)
#   D  24-25   diagnostics, optimal lag, and the deliverables        (after B and C)
# B and C write to different step folders, so they run side by side. CPU only:
# NUTS on these models gains nothing from a GPU.
set -euo pipefail
MODEL=${1:?usage: submit_model_version.sh <model>}
CODE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$CODE/../.." && pwd)"
LOGS="$ROOT/derivatives_${MODEL}/logs"
mkdir -p "$LOGS"

submit() {  # name, dependency ("" for none), run_pipeline.py arguments...
  local name=$1 dep=$2; shift 2
  sbatch --parsable --job-name="${MODEL}_${name}" ${dep:+--dependency=afterany:$dep} \
    --account=cruzalmeida --partition=hpg-default \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=4-00:00:00 \
    --output="$LOGS/%x_%j.log" \
    --wrap="module load conda/25.7.0 && \
export PYTENSOR_FLAGS=base_compiledir=/orange/cruzalmeida/pvaldeshernandez/.pytensor_cache/${MODEL}_\$SLURM_JOB_ID && \
cd '$CODE' && python run_pipeline.py --refit --model $MODEL $*"
}

A=$(submit A_00-08 "" --from 00 --to 08)
B=$(submit B_09-12 "$A" --from 09 --to 12)
C=$(submit C_13-23 "$A" --from 13 --to 23)
D=$(submit D_24-25 "$B:$C" --from 24 --to 25)
echo "submitted: A=$A B=$B C=$C D=$D  logs: $LOGS"
