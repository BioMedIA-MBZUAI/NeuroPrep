#!/bin/bash
#SBATCH --job-name=np-catalog-validate
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu-03
#SBATCH --chdir=/nfs-stor/salma.hassan/NeuroPrep-agentic
#SBATCH --output=/nfs-stor/salma.hassan/NeuroPrep-agentic/logs/%x-%j.out
#SBATCH --error=/nfs-stor/salma.hassan/NeuroPrep-agentic/logs/%x-%j.err

set -euo pipefail

APP_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic
. "$APP_ROOT/scripts/ciai_env.sh"
CACHE_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-cache
RUNTIME_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-runtime/catalog-validation-$SLURM_JOB_ID
OUTPUT_ROOT=$APP_ROOT/output/catalog-validation-$SLURM_JOB_ID

export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/apps/local/anaconda3/bin
export TMPDIR=$RUNTIME_ROOT/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export XDG_CONFIG_HOME=$RUNTIME_ROOT/xdg-config
export XDG_DATA_HOME=$RUNTIME_ROOT/xdg-data
export XDG_STATE_HOME=$RUNTIME_ROOT/xdg-state
export UV_CACHE_DIR=$CACHE_ROOT/uv
export PIP_CACHE_DIR=$CACHE_ROOT/pip
export SINGULARITY_CACHEDIR=$CACHE_ROOT/singularity
export SINGULARITY_TMPDIR=$RUNTIME_ROOT/singularity-tmp

mkdir -p "$APP_ROOT/logs" "$OUTPUT_ROOT" "$TMPDIR" "$SINGULARITY_TMPDIR" \
  "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" \
  "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$SINGULARITY_CACHEDIR"

test -x /usr/bin/singularity
/usr/bin/singularity --version
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

"$APP_ROOT/scripts/install_cluster_tools.sh"
"$APP_ROOT/.venv/bin/ruff" check "$APP_ROOT"
"$APP_ROOT/.venv/bin/python" -m pytest -q "$APP_ROOT/tests"
"$APP_ROOT/.venv/bin/neuroprep-agentic" executor-coverage \
  --output "$OUTPUT_ROOT/executor-coverage.json"
"$APP_ROOT/.venv/bin/neuroprep-agentic" catalog-images --execute \
  --output "$OUTPUT_ROOT/catalog-images.json"

test "$(jq -r .ready "$OUTPUT_ROOT/executor-coverage.json")" = true
test "$(jq -r .ready "$OUTPUT_ROOT/catalog-images.json")" = true
test "$(jq -r .missing_count "$OUTPUT_ROOT/catalog-images.json")" = 0
test "$(jq -r .invalid_count "$OUTPUT_ROOT/catalog-images.json")" = 0
printf '%s\n' "ready" >"$OUTPUT_ROOT/READY"
