#!/bin/bash
#SBATCH --job-name=np-prompt-policy
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu-03
#SBATCH --chdir=/nfs-stor/salma.hassan/NeuroPrep-agentic
#SBATCH --output=/nfs-stor/salma.hassan/NeuroPrep-agentic/logs/%x-%j.out
#SBATCH --error=/nfs-stor/salma.hassan/NeuroPrep-agentic/logs/%x-%j.err

set -euo pipefail

APP_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic
CACHE_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-cache
RUNTIME_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-runtime/prompt-policy-$SLURM_JOB_ID
OLLAMA_IMAGE=$CACHE_ROOT/containers/ollama-0.32.6.sif
: "${NEUROPREP_REQUEST_FILE:?NEUROPREP_REQUEST_FILE is required}"
: "${NEUROPREP_POLICY_OUTPUT:?NEUROPREP_POLICY_OUTPUT is required}"

case "$NEUROPREP_REQUEST_FILE" in
  /nfs-stor/salma.hassan/*|/l/users/salma.hassan/*) ;;
  *) echo "Request file must be on approved cluster storage" >&2; exit 2 ;;
esac
case "$NEUROPREP_POLICY_OUTPUT" in
  /nfs-stor/salma.hassan/*|/l/users/salma.hassan/*) ;;
  *) echo "Policy output must be on approved cluster storage" >&2; exit 2 ;;
esac

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

mkdir -p "$APP_ROOT/logs" "$CACHE_ROOT/ollama" "$TMPDIR" \
  "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" \
  "$(dirname "$NEUROPREP_POLICY_OUTPUT")"
test -s "$NEUROPREP_REQUEST_FILE"
test -s "$OLLAMA_IMAGE"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

HARDWARE=$RUNTIME_ROOT/hardware.json
"$APP_ROOT/.venv/bin/neuroprep-agentic" hardware-profile --output "$HARDWARE"

/usr/bin/singularity exec --nv --bind "$CACHE_ROOT/ollama:/root/.ollama" \
  --bind "$TMPDIR:/tmp" "$OLLAMA_IMAGE" ollama serve \
  >"$APP_ROOT/logs/ollama-policy-$SLURM_JOB_ID.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT
curl --fail --silent --retry 120 --retry-delay 2 --retry-connrefused \
  --max-time 300 "$OLLAMA_BASE_URL/api/tags" >/dev/null

for model in qwen3:8b qwen3:14b; do
  if ! /usr/bin/singularity exec --nv --bind "$CACHE_ROOT/ollama:/root/.ollama" \
    "$OLLAMA_IMAGE" ollama show "$model" >/dev/null 2>&1; then
    /usr/bin/singularity exec --nv --bind "$CACHE_ROOT/ollama:/root/.ollama" \
      "$OLLAMA_IMAGE" ollama pull "$model"
  fi
done

"$APP_ROOT/.venv/bin/neuroprep-agentic" prompt-policy \
  --request-file "$NEUROPREP_REQUEST_FILE" \
  --hardware-profile "$HARDWARE" \
  --output "$NEUROPREP_POLICY_OUTPUT"
"$APP_ROOT/.venv/bin/neuroprep-agentic" backend-plan \
  --policy "$NEUROPREP_POLICY_OUTPUT" \
  --hardware-profile "$HARDWARE" \
  --output "$NEUROPREP_POLICY_OUTPUT.backends.json"
