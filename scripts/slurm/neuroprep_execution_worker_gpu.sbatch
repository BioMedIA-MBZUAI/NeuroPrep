#!/bin/bash
#SBATCH --job-name=np-executor
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=gpu-03,gpu-05,gpu-50
#SBATCH --chdir=/nfs-stor/salma.hassan
#SBATCH --output=/nfs-stor/salma.hassan/NeuroPrep-agentic-runtime/%x-%j.out
#SBATCH --error=/nfs-stor/salma.hassan/NeuroPrep-agentic-runtime/%x-%j.err

set -euo pipefail

export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/apps/local/anaconda3/bin
APP_ROOT=${NEUROPREP_AGENTIC_PROJECT:-/nfs-stor/salma.hassan/NeuroPrep-agentic}
. "$APP_ROOT/scripts/ciai_env.sh"
CACHE_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-cache
RUNTIME_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-runtime/worker-$SLURM_JOB_ID
test -n "$NEUROPREP_STATE_PATH"

case "$APP_ROOT" in
  /nfs-stor/salma.hassan/*|/l/users/salma.hassan/*) ;;
  *) echo "NEUROPREP_AGENTIC_PROJECT must be on approved cluster storage" >&2; exit 2 ;;
esac
APP_ROOT=$(readlink -f "$APP_ROOT")
case "$APP_ROOT" in
  /nfs-stor/salma.hassan/*|/l/users/salma.hassan/*) ;;
  *) echo "Resolved NEUROPREP_AGENTIC_PROJECT must be approved" >&2; exit 2 ;;
esac
REPOSITORY_ROOT=$(git -C "$APP_ROOT" rev-parse --show-toplevel 2>/dev/null || true)
test -n "$REPOSITORY_ROOT"
test "$(readlink -f "$REPOSITORY_ROOT")" = "$APP_ROOT"
export NEUROPREP_AGENTIC_PROJECT=$APP_ROOT
cd "$APP_ROOT"

export TMPDIR=$RUNTIME_ROOT/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export XDG_CONFIG_HOME=$RUNTIME_ROOT/xdg-config
export XDG_DATA_HOME=$RUNTIME_ROOT/xdg-data
export XDG_STATE_HOME=$RUNTIME_ROOT/xdg-state
export UV_CACHE_DIR=$CACHE_ROOT/uv
export SINGULARITY_CACHEDIR=$CACHE_ROOT/singularity
export SINGULARITY_TMPDIR=$RUNTIME_ROOT/singularity-tmp
export CREWAI_TELEMETRY_OPT_OUT=true
export OTEL_SDK_DISABLED=true

mkdir -p \
  "$APP_ROOT/logs" "$TMPDIR" "$RUNTIME_ROOT/var-tmp" \
  "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" \
  "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

test -x /usr/bin/singularity
test -s "$NEUROPREP_STATE_PATH"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

WORKER_ID=slurm-$SLURM_JOB_ID-$HOSTNAME
exec "$APP_ROOT/.venv/bin/neuroprep-agentic" worker \
  --state "$NEUROPREP_STATE_PATH" \
  --worker-id "$WORKER_ID" \
  --resource-class cpu \
  --resource-class gpu \
  --resource-class high_memory \
  --lease-seconds 3600
