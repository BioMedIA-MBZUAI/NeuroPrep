#!/usr/bin/env bash
# Start the user-local Ollama and OpenJev services for one Slurm allocation.
# The allocation owns the processes; canceling the job stops both services.
# Required environment: NEUROPREP_ROOT, OPENJEV_MODEL_DIR.
# Optional: OLLAMA_MODEL (default qwen3:14b), OPENJEV_PORT (default 3000).
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@30

set -Eeuo pipefail

NEUROPREP_ROOT="${NEUROPREP_ROOT:-$HOME/neuroprep-agentic-0.4.0}"
OPENJEV_MODEL_DIR="${OPENJEV_MODEL_DIR:-$HOME/openjev-fp8}"
OPENJEV_GGUF="${OPENJEV_GGUF:-$HOME/openjev-gguf/OpenJev-Q4_K_M.gguf}"
OPENJEV_SHIM="${OPENJEV_SHIM:-$HOME/openjev-gguf/shim.py}"
OPENJEV_BACKEND="${OPENJEV_BACKEND:-ollama}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:14b}"
OPENJEV_PORT="${OPENJEV_PORT:-3000}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
VENV="${NEUROPREP_VENV:-$HOME/neuroprep-venv}"
OPENJEV_VENV="${OPENJEV_VENV:-$HOME/openjev-venv}"
NEUROPREP_FSL_MRS_ROOT="${NEUROPREP_FSL_MRS_ROOT:-$HOME/fsl-mrs-root}"
NEUROPREP_ANTS_IMAGE="${NEUROPREP_ANTS_IMAGE:-$HOME/ants-2.6.5.sif}"

test -x "$HOME/.local/ollama"
test -x "$VENV/bin/python"
test -x "$OPENJEV_VENV/bin/python"
if [[ "$OPENJEV_BACKEND" == "ollama" ]]; then
  test -f "$OPENJEV_GGUF"
  test -f "$OPENJEV_SHIM"
else
  test -f "$OPENJEV_MODEL_DIR/config.json"
  test -f "$OPENJEV_MODEL_DIR/.download-complete" || {
    echo "OpenJev checkpoint is incomplete: $OPENJEV_MODEL_DIR" >&2
    exit 2
  }
fi

export PATH="$HOME/.local:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$OPENJEV_VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$OPENJEV_VENV/lib/python3.12/site-packages/nvidia/cublas/lib:$OPENJEV_VENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export OLLAMA_HOST
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
export NEUROPREP_FSL_MRS_ROOT NEUROPREP_ANTS_IMAGE
export NEUROPREP_OPENJEV_ENDPOINT="http://127.0.0.1:${OPENJEV_PORT}/v1/systemone"

cleanup() {
  jobs -pr | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ollama serve >"$HOME/neuroprep-ollama-${SLURM_JOB_ID}.log" 2>&1 &
ollama_pid=$!
for _ in $(seq 1 60); do
  curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null
ollama pull "$OLLAMA_MODEL" >/dev/null

if [[ "$OPENJEV_BACKEND" == "ollama" ]]; then
  ollama create qwen -f <(printf 'FROM %s\nPARAMETER num_ctx 4096\nPARAMETER temperature 0\n' "$OPENJEV_GGUF") >/dev/null
  VLLM=http://127.0.0.1:11434/v1 TOKENIZER="$OPENJEV_MODEL_DIR" \
    "$OPENJEV_VENV/bin/python" "$OPENJEV_SHIM" --host 127.0.0.1 --port "$OPENJEV_PORT" \
    >"$HOME/neuroprep-openjev-${SLURM_JOB_ID}.log" 2>&1 &
  openjev_pid=$!
else
  "$OPENJEV_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$OPENJEV_MODEL_DIR" --host 127.0.0.1 --port 8000 \
    --served-model-name qwen --enable-prefix-caching --max-model-len 16384 \
    --gpu-memory-utilization 0.90 --max-num-seqs 16 --trust-remote-code \
    --quantization fp8 >"$HOME/neuroprep-vllm-${SLURM_JOB_ID}.log" 2>&1 &
  vllm_pid=$!
  for _ in $(seq 1 180); do
    curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
    sleep 1
  done
  curl -fsS http://127.0.0.1:8000/v1/models >/dev/null
  VLLM=http://127.0.0.1:8000/v1 TOKENIZER="$OPENJEV_MODEL_DIR" \
    "$OPENJEV_VENV/bin/python" "$OPENJEV_SHIM" --host 127.0.0.1 --port "$OPENJEV_PORT" \
    >"$HOME/neuroprep-openjev-${SLURM_JOB_ID}.log" 2>&1 &
  openjev_pid=$!
fi
for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:${OPENJEV_PORT}/v1/version" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "http://127.0.0.1:${OPENJEV_PORT}/v1/version"
"$VENV/bin/neuroprep-agentic" runtime-check || true
wait "$openjev_pid" "$ollama_pid"
