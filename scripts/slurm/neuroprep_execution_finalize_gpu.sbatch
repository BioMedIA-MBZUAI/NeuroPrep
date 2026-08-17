#!/bin/bash
#SBATCH --job-name=np-finalize
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu-03,gpu-05,gpu-50
#SBATCH --chdir=/nfs-stor/salma.hassan
#SBATCH --output=/nfs-stor/salma.hassan/NeuroPrep-agentic-runtime/%x-%j.out
#SBATCH --error=/nfs-stor/salma.hassan/NeuroPrep-agentic-runtime/%x-%j.err

set -euo pipefail

export PATH=/usr/bin:/bin:/usr/sbin:/sbin
APP_ROOT=${NEUROPREP_AGENTIC_PROJECT:-/nfs-stor/salma.hassan/NeuroPrep-agentic}
RUNTIME_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-runtime/finalize-$SLURM_JOB_ID
test -n "$NEUROPREP_EXECUTION_PLAN_PATH"
test -n "$NEUROPREP_RELEASE_ROOT"

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
export XDG_CACHE_HOME=$RUNTIME_ROOT/xdg-cache
export XDG_CONFIG_HOME=$RUNTIME_ROOT/xdg-config
export XDG_DATA_HOME=$RUNTIME_ROOT/xdg-data
export XDG_STATE_HOME=$RUNTIME_ROOT/xdg-state

mkdir -p "$APP_ROOT/logs" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" "$XDG_STATE_HOME"
test -s "$NEUROPREP_EXECUTION_PLAN_PATH"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

exec "$APP_ROOT/.venv/bin/neuroprep-agentic" finalize-run \
  "$NEUROPREP_EXECUTION_PLAN_PATH" \
  --output-root "$NEUROPREP_RELEASE_ROOT"
