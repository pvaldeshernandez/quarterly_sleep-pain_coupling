#!/bin/bash
# One SLURM job per fit (handoff section 7). Run from the repository root.
cd "$(dirname "$0")/../.."
# third field: initial-point treatment for latent fits (conditional | stationary)
SPECS=${SPECS:-"real,latent,conditional sim1,observed,conditional sim1,latent,conditional sim2,observed,conditional sim2,latent,conditional sim3,observed,conditional sim3,latent,conditional real,latent,stationary sim1,latent,stationary sim2,latent,stationary sim3,latent,stationary"}
for spec in $SPECS; do
  IFS=, read -r d c i <<< "$spec"; set -- $d $c $i
  sbatch --job-name="lc_$1_$2_$3" --account=cruzalmeida --partition=hpg-default \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=12:00:00 \
    --output="extensions/latent_centering/logs/%x_%j.log" \
    --wrap="module load conda/25.7.0 && export PYTENSOR_FLAGS=base_compiledir=/orange/cruzalmeida/pvaldeshernandez/.pytensor_cache/lc_\$SLURM_JOB_ID && python codes/python/extensions/latent_centering.py fit --data $1 --centering $2 --initial $3"
done
