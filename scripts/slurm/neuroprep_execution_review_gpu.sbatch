#!/bin/bash
#SBATCH --job-name=np-agent-review
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu-03
#SBATCH --chdir=/nfs-stor/salma.hassan/NeuroPrep-agentic
#SBATCH --output=/nfs-stor/salma.hassan/NeuroPrep-agentic/logs/%x-%j.out
#SBATCH --error=/nfs-stor/salma.hassan/NeuroPrep-agentic/logs/%x-%j.err

set -euo pipefail

APP_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic
CACHE_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-cache
RUNTIME_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-runtime/review-$SLURM_JOB_ID
OLLAMA_IMAGE=$CACHE_ROOT/containers/ollama-0.32.6.sif
test -n "$NEUROPREP_EXECUTION_PLAN_PATH"

export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/apps/local/anaconda3/bin
export TMPDIR=$RUNTIME_ROOT/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export XDG_CONFIG_HOME=$RUNTIME_ROOT/xdg-config
export XDG_DATA_HOME=$RUNTIME_ROOT/xdg-data
export XDG_STATE_HOME=$RUNTIME_ROOT/xdg-state
export CREWAI_STORAGE_DIR=$APP_ROOT/state/crewai
export CREWAI_TELEMETRY_OPT_OUT=true
export OTEL_SDK_DISABLED=true
OLLAMA_PORT=$((20000 + SLURM_JOB_ID % 20000))
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
export OLLAMA_BASE_URL=http://127.0.0.1:$OLLAMA_PORT
export SINGULARITYENV_OLLAMA_HOST=$OLLAMA_HOST
export OLLAMA_MODELS=/root/.ollama/models
export OLLAMA_NO_CLOUD=true
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_MANAGER_MODEL=ollama/qwen3:14b
export OLLAMA_WORKER_MODEL=ollama/qwen3:8b
export SINGULARITY_TMPDIR=$RUNTIME_ROOT/singularity-tmp
export SINGULARITY_CACHEDIR=$CACHE_ROOT/singularity

mkdir -p \
  "$APP_ROOT/logs" "$APP_ROOT/output" "$CREWAI_STORAGE_DIR" \
  "$CACHE_ROOT/ollama" "$CACHE_ROOT/singularity" \
  "$TMPDIR" "$RUNTIME_ROOT/var-tmp" "$SINGULARITY_TMPDIR" \
  "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME"

test -x /usr/bin/singularity
test -s "$OLLAMA_IMAGE"
test -s "$NEUROPREP_EXECUTION_PLAN_PATH"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
"$APP_ROOT/.venv/bin/neuroprep-agentic" prefetch-images \
  "$NEUROPREP_EXECUTION_PLAN_PATH" --execute
"$APP_ROOT/.venv/bin/neuroprep-agentic" prepare-assets \
  "$NEUROPREP_EXECUTION_PLAN_PATH" --execute
"$APP_ROOT/.venv/bin/neuroprep-agentic" preflight-run \
  "$NEUROPREP_EXECUTION_PLAN_PATH"
"$APP_ROOT/.venv/bin/neuroprep-agentic" run-init "$NEUROPREP_EXECUTION_PLAN_PATH"

OLLAMA_BIND=$CACHE_ROOT/ollama:/root/.ollama
TMP_BIND=$TMPDIR:/tmp
VAR_TMP_BIND=$RUNTIME_ROOT/var-tmp:/var/tmp
OLLAMA_LOG=$APP_ROOT/logs/ollama-review-$SLURM_JOB_ID.log

/usr/bin/singularity exec --nv \
  --bind "$OLLAMA_BIND" --bind "$TMP_BIND" --bind "$VAR_TMP_BIND" \
  "$OLLAMA_IMAGE" ollama serve >"$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT

curl --fail --silent --retry 120 --retry-delay 2 --retry-connrefused \
  --max-time 300 "$OLLAMA_BASE_URL/api/tags" >/dev/null

for model in qwen3:8b qwen3:14b; do
  if ! /usr/bin/singularity exec --nv --bind "$OLLAMA_BIND" \
    "$OLLAMA_IMAGE" ollama show "$model" >/dev/null 2>&1; then
    /usr/bin/singularity exec --nv --bind "$OLLAMA_BIND" \
      "$OLLAMA_IMAGE" ollama pull "$model"
  fi
done

"$APP_ROOT/.venv/bin/neuroprep-agentic" crew \
  --workflow execution_review \
  --request "Review exact execution readiness, resources, and recovery boundaries."
"$APP_ROOT/.venv/bin/neuroprep-agentic" validate-control-artifact \
  "$APP_ROOT/output/control-execution_review-$SLURM_JOB_ID.json" --workflow execution_review
