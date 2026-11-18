#!/bin/bash
#SBATCH --job-name=neuroprep-agentic
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
. "$APP_ROOT/scripts/ciai_env.sh"
CACHE_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-cache
STATE_ROOT=${APP_ROOT}/state
RUNTIME_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-runtime/slurm-${SLURM_JOB_ID}
IMAGE=${CACHE_ROOT}/containers/ollama-0.32.6.sif
IMAGE_SOURCE=docker://ollama/ollama:0.32.6
UV=/apps/local/anaconda3/bin/uv
VENV=${APP_ROOT}/.venv
EXAMPLE_MANIFEST=${APP_ROOT}/examples/synthetic_multimodal_manifest.json

export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/apps/local/anaconda3/bin
export TMPDIR=${RUNTIME_ROOT}/tmp
export TMP=${TMPDIR}
export TEMP=${TMPDIR}
export XDG_CACHE_HOME=${CACHE_ROOT}/xdg
export XDG_CONFIG_HOME=${RUNTIME_ROOT}/xdg-config
export XDG_DATA_HOME=${RUNTIME_ROOT}/xdg-data
export XDG_STATE_HOME=${RUNTIME_ROOT}/xdg-state
export PIP_CACHE_DIR=${CACHE_ROOT}/pip
export UV_CACHE_DIR=${CACHE_ROOT}/uv
export UV_PYTHON_INSTALL_DIR=${CACHE_ROOT}/uv-python
export UV_LINK_MODE=copy
export SINGULARITY_CACHEDIR=${CACHE_ROOT}/singularity
export SINGULARITY_TMPDIR=${RUNTIME_ROOT}/singularity-tmp
export CREWAI_STORAGE_DIR=${STATE_ROOT}/crewai
export CREWAI_TELEMETRY_OPT_OUT=true
export OTEL_SDK_DISABLED=true
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_MODELS=/root/.ollama/models
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_NO_CLOUD=true
export SINGULARITYENV_OLLAMA_HOST=${OLLAMA_HOST}
export SINGULARITYENV_OLLAMA_MODELS=${OLLAMA_MODELS}
export SINGULARITYENV_OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE}
export SINGULARITYENV_OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS}
export SINGULARITYENV_OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL}
export SINGULARITYENV_OLLAMA_NO_CLOUD=${OLLAMA_NO_CLOUD}
export OLLAMA_MANAGER_MODEL=${OLLAMA_MANAGER_MODEL:-ollama/qwen3:14b}
export OLLAMA_WORKER_MODEL=${OLLAMA_WORKER_MODEL:-ollama/qwen3:8b}
export NEUROPREP_WORKFLOW=${NEUROPREP_WORKFLOW:-coverage}
export NEUROPREP_REQUEST=${NEUROPREP_REQUEST:-Review exact registered coverage and gaps.}

mkdir -p \
  "${APP_ROOT}/logs" \
  "${APP_ROOT}/output" \
  "${STATE_ROOT}/crewai" \
  "${CACHE_ROOT}/containers" \
  "${CACHE_ROOT}/ollama" \
  "${CACHE_ROOT}/pip" \
  "${CACHE_ROOT}/uv" \
  "${CACHE_ROOT}/uv-python" \
  "${CACHE_ROOT}/singularity" \
  "${CACHE_ROOT}/xdg" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_DATA_HOME}" \
  "${XDG_STATE_HOME}" \
  "${TMPDIR}" \
  "${RUNTIME_ROOT}/var-tmp" \
  "${SINGULARITY_TMPDIR}"

test -x /usr/bin/singularity
test -x "${UV}"
/usr/bin/singularity --version
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

if [[ ! -s "${IMAGE}" ]]; then
  /usr/bin/singularity pull "${IMAGE}" "${IMAGE_SOURCE}"
fi

cd "${APP_ROOT}"
"${UV}" sync --extra dev --frozen
"${VENV}/bin/python" -m pytest -q
"${VENV}/bin/ruff" check .
"${VENV}/bin/neuroprep-agentic" coverage \
  --output "${APP_ROOT}/output/coverage-${SLURM_JOB_ID}.json"
"${VENV}/bin/neuroprep-agentic" ingest-plan "${EXAMPLE_MANIFEST}" \
  --identity-key synthetic-validation-key-2026 \
  --output "${APP_ROOT}/output/synthetic-plan-${SLURM_JOB_ID}.json"

OLLAMA_BIND=${CACHE_ROOT}/ollama:/root/.ollama
TMP_BIND=${TMPDIR}:/tmp
VAR_TMP_BIND=${RUNTIME_ROOT}/var-tmp:/var/tmp
OLLAMA_LOG=${APP_ROOT}/logs/ollama-${SLURM_JOB_ID}.log

/usr/bin/singularity exec --nv \
  --bind "${OLLAMA_BIND}" \
  --bind "${TMP_BIND}" \
  --bind "${VAR_TMP_BIND}" \
  "${IMAGE}" ollama serve >"${OLLAMA_LOG}" 2>&1 &
OLLAMA_PID=$!
trap 'kill "${OLLAMA_PID}" 2>/dev/null || true' EXIT

if ! curl --fail --silent --retry 120 --retry-delay 2 --retry-connrefused \
  --max-time 240 http://127.0.0.1:11434/api/tags >/dev/null; then
  sed -n '1,200p' "${OLLAMA_LOG}"
  exit 1
fi

/usr/bin/singularity exec --nv --bind "${OLLAMA_BIND}" --bind "${TMP_BIND}" \
  --bind "${VAR_TMP_BIND}" "${IMAGE}" ollama pull qwen3:8b
/usr/bin/singularity exec --nv --bind "${OLLAMA_BIND}" --bind "${TMP_BIND}" \
  --bind "${VAR_TMP_BIND}" "${IMAGE}" ollama pull qwen3:14b

if [[ "${NEUROPREP_WORKFLOW}" == "ingest_plan" ]]; then
  export NEUROPREP_MANIFEST_PATH=${NEUROPREP_MANIFEST_PATH:-${EXAMPLE_MANIFEST}}
  : "${NEUROPREP_IDENTITY_KEY:?ingest_plan requires NEUROPREP_IDENTITY_KEY}"
fi

"${VENV}/bin/neuroprep-agentic" crew \
  --workflow "${NEUROPREP_WORKFLOW}" \
  --request "${NEUROPREP_REQUEST}"

"${VENV}/bin/neuroprep-agentic" validate-control-artifact \
  "${APP_ROOT}/output/control-${NEUROPREP_WORKFLOW}-${SLURM_JOB_ID}.json" \
  --workflow "${NEUROPREP_WORKFLOW}"
