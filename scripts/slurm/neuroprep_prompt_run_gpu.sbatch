#!/bin/bash
#SBATCH --job-name=np-prompt-run
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu-03,gpu-05,gpu-50
#SBATCH --chdir=/nfs-stor/salma.hassan/NeuroPrep-agentic
#SBATCH --output=/nfs-stor/salma.hassan/NeuroPrep-agentic/logs/%x-%j.out
#SBATCH --error=/nfs-stor/salma.hassan/NeuroPrep-agentic/logs/%x-%j.err

set -euo pipefail

APP_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic
CACHE_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-cache
RUNTIME_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-runtime/prompt-run-$SLURM_JOB_ID
OLLAMA_IMAGE=$CACHE_ROOT/containers/ollama-0.32.6.sif
: "${NEUROPREP_REQUEST_FILE:?NEUROPREP_REQUEST_FILE is required}"

approved_path() {
  case "$1" in
    /nfs-stor/salma.hassan/*|/l/users/salma.hassan/*) return 0 ;;
    *) return 1 ;;
  esac
}

if ! approved_path "$NEUROPREP_REQUEST_FILE"; then
  echo "Request file must be on approved cluster storage" >&2
  exit 2
fi
for variable in \
  NEUROPREP_SOURCE_ROOT NEUROPREP_OUTPUT_ROOT NEUROPREP_MANIFEST \
  NEUROPREP_IDENTITY_KEY NEUROPREP_BACKEND_EVIDENCE \
  NEUROPREP_BACKEND_EVIDENCE_KEY NEUROPREP_FREESURFER_LICENSE; do
  value=${!variable:-}
  if [ -n "$value" ] && [ "$variable" != "NEUROPREP_IDENTITY_KEY" ] && \
      ! approved_path "$value"; then
    echo "$variable must be on approved cluster storage" >&2
    exit 2
  fi
done

export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/apps/local/anaconda3/bin
export TMPDIR=$RUNTIME_ROOT/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export XDG_CONFIG_HOME=$RUNTIME_ROOT/xdg-config
export XDG_DATA_HOME=$RUNTIME_ROOT/xdg-data
export XDG_STATE_HOME=$RUNTIME_ROOT/xdg-state
export CREWAI_STORAGE_DIR=$RUNTIME_ROOT/crewai
export CREWAI_TELEMETRY_OPT_OUT=true
export OTEL_SDK_DISABLED=true
export SINGULARITY_TMPDIR=$RUNTIME_ROOT/singularity-tmp
export SINGULARITY_CACHEDIR=$CACHE_ROOT/singularity
OLLAMA_PORT=$((20000 + SLURM_JOB_ID % 20000))
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
export OLLAMA_BASE_URL=http://127.0.0.1:$OLLAMA_PORT
export SINGULARITYENV_OLLAMA_HOST=$OLLAMA_HOST
export OLLAMA_MODELS=/root/.ollama/models
export OLLAMA_NO_CLOUD=true
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_MANAGER_MODEL=${OLLAMA_MANAGER_MODEL:-ollama/qwen3:14b}
export OLLAMA_WORKER_MODEL=${OLLAMA_WORKER_MODEL:-ollama/qwen3:8b}

mkdir -p \
  "$APP_ROOT/logs" "$CACHE_ROOT/ollama" "$CACHE_ROOT/singularity" \
  "$TMPDIR" "$RUNTIME_ROOT/var-tmp" "$SINGULARITY_TMPDIR" \
  "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" \
  "$CREWAI_STORAGE_DIR"
test -x /usr/bin/singularity
test -x "$APP_ROOT/.venv/bin/neuroprep-agentic"
test -s "$NEUROPREP_REQUEST_FILE"
test -s "$OLLAMA_IMAGE"
test -n "${CUDA_VISIBLE_DEVICES:-}"
timeout 20 stat "$APP_ROOT" "$OLLAMA_IMAGE" >/dev/null
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
  --query-gpu=name,uuid,memory.total --format=csv,noheader
/usr/bin/singularity exec --nv "$OLLAMA_IMAGE" \
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
  --query-gpu=name,uuid,memory.total --format=csv,noheader

HARDWARE=$RUNTIME_ROOT/hardware.json
"$APP_ROOT/.venv/bin/neuroprep-agentic" hardware-profile --output "$HARDWARE"

OLLAMA_BIND=$CACHE_ROOT/ollama:/root/.ollama
TMP_BIND=$TMPDIR:/tmp
VAR_TMP_BIND=$RUNTIME_ROOT/var-tmp:/var/tmp
OLLAMA_LOG=$APP_ROOT/logs/ollama-prompt-run-$SLURM_JOB_ID.log
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

arguments=(
  prompt-run
  --request-file "$NEUROPREP_REQUEST_FILE"
  --hardware-profile "$HARDWARE"
  --gpu-workers "${NEUROPREP_GPU_WORKERS:-2}"
)
optional_value() {
  local environment_name=$1
  local option_name=$2
  local value=${!environment_name:-}
  if [ -n "$value" ]; then
    arguments+=("$option_name" "$value")
  fi
}
optional_flag() {
  local environment_name=$1
  local option_name=$2
  if [ "${!environment_name:-false}" = "true" ]; then
    arguments+=("$option_name")
  fi
}
optional_value NEUROPREP_SOURCE_ROOT --source-root
optional_value NEUROPREP_OUTPUT_ROOT --output-root
optional_value NEUROPREP_MANIFEST --manifest
optional_value NEUROPREP_DATASET_ID --dataset-id
optional_value NEUROPREP_ADAPTER_ID --adapter-id
optional_value NEUROPREP_IDENTITY_KEY --identity-key
optional_value NEUROPREP_BACKEND_EVIDENCE --backend-evidence
optional_value NEUROPREP_BACKEND_EVIDENCE_KEY --backend-evidence-key
optional_value NEUROPREP_FREESURFER_LICENSE --freesurfer-license
optional_flag NEUROPREP_EXECUTE_OPTIONAL --execute-optional
optional_flag NEUROPREP_ALLOW_QUARANTINE --allow-quarantine
optional_flag NEUROPREP_ALLOW_UNSUPPORTED --allow-unsupported
optional_flag NEUROPREP_INITIALIZE --initialize
optional_flag NEUROPREP_SUBMIT --submit

"$APP_ROOT/.venv/bin/neuroprep-agentic" "${arguments[@]}"
