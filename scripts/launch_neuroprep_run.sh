#!/bin/bash

set -euo pipefail

export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/apps/local/anaconda3/bin
APP_ROOT=${NEUROPREP_AGENTIC_PROJECT:-/nfs-stor/salma.hassan/NeuroPrep-agentic}
REVIEW_JOB=""
WORKER_JOBS=""
FINALIZE_JOB=""

report_incomplete_launch() {
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "launch_complete=false" >&2
    echo "review_job=${REVIEW_JOB:-not-submitted}" >&2
    echo "worker_jobs=${WORKER_JOBS:-not-submitted}" >&2
    echo "finalize_job=${FINALIZE_JOB:-not-submitted}" >&2
    echo "Preserve any reported job IDs; resume only the missing stages." >&2
  fi
}
trap report_incomplete_launch EXIT

if [ "$#" -lt 1 ]; then
  echo "Usage: launch_neuroprep_run.sh EXECUTION_PLAN.json [GPU_WORKERS]" >&2
  exit 2
fi
PLAN=$1
GPU_WORKERS=2
if [ "$#" -ge 2 ]; then
  GPU_WORKERS=$2
fi

case "$PLAN" in
  /nfs-stor/salma.hassan/*|/l/users/salma.hassan/*) ;;
  *) echo "Plan must be on approved cluster storage" >&2; exit 2 ;;
esac
case "$APP_ROOT" in
  /nfs-stor/salma.hassan/*|/l/users/salma.hassan/*) ;;
  *) echo "NEUROPREP_AGENTIC_PROJECT must be on approved cluster storage" >&2; exit 2 ;;
esac
APP_ROOT=$(readlink -f "$APP_ROOT")
case "$APP_ROOT" in
  /nfs-stor/salma.hassan/*|/l/users/salma.hassan/*) ;;
  *) echo "Resolved NEUROPREP_AGENTIC_PROJECT must be on approved cluster storage" >&2; exit 2 ;;
esac
REPOSITORY_ROOT=$(git -C "$APP_ROOT" rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$REPOSITORY_ROOT" ] || [ "$(readlink -f "$REPOSITORY_ROOT")" != "$APP_ROOT" ]; then
  echo "NEUROPREP_AGENTIC_PROJECT must be the exact git repository root" >&2
  exit 2
fi
export NEUROPREP_AGENTIC_PROJECT=$APP_ROOT
if ! [[ "$GPU_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU_WORKERS must be a positive integer" >&2
  exit 2
fi

LAUNCHER_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-runtime/launcher
export TMPDIR=$LAUNCHER_ROOT/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export XDG_CACHE_HOME=$LAUNCHER_ROOT/xdg-cache
export XDG_CONFIG_HOME=$LAUNCHER_ROOT/xdg-config
export XDG_DATA_HOME=$LAUNCHER_ROOT/xdg-data
export XDG_STATE_HOME=$LAUNCHER_ROOT/xdg-state
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$APP_ROOT/logs"

test -s "$PLAN"
test -x /usr/bin/singularity
test -x "$APP_ROOT/.venv/bin/neuroprep-agentic"

PLAN_AGENTIC_PROJECT=$(
  "$APP_ROOT/.venv/bin/python" -c \
    'import json,os,sys; print(os.path.realpath(json.load(open(sys.argv[1]))["layout"]["agentic_project"]))' \
    "$PLAN"
)
if [ "$PLAN_AGENTIC_PROJECT" != "$APP_ROOT" ]; then
  echo "Execution plan agentic project differs from launcher repository" >&2
  exit 2
fi
QUALIFICATION_REQUIRED=$(
  "$APP_ROOT/.venv/bin/python" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); print("true" if (p.get("backend_qualification_tests") or p.get("planned_smoke_tests")) else "false")' \
    "$PLAN"
)
if [ "$QUALIFICATION_REQUIRED" = true ] && \
    [ -z "${NEUROPREP_BACKEND_EVIDENCE_KEY:-}" ]; then
  echo "Signed backend evidence key is required for this execution plan" >&2
  exit 2
fi

STATE=$(
  "$APP_ROOT/.venv/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["layout"]["state_path"])' \
    "$PLAN"
)
RELEASE_ROOT=$(
  "$APP_ROOT/.venv/bin/python" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); print(p["layout"]["derivatives_root"]+"/release")' \
    "$PLAN"
)

# --export=ALL carries the validated evidence-key path without rendering it.
REVIEW_JOB=$(
  sbatch --parsable \
    --export=ALL,NEUROPREP_EXECUTION_PLAN_PATH="$PLAN" \
    "$APP_ROOT/scripts/slurm/neuroprep_execution_review_gpu.sbatch"
)

for _index in $(seq 1 "$GPU_WORKERS"); do
  JOB_ID=$(
    sbatch --parsable --dependency=afterok:"$REVIEW_JOB" \
      --export=ALL,NEUROPREP_STATE_PATH="$STATE" \
      "$APP_ROOT/scripts/slurm/neuroprep_execution_worker_gpu.sbatch"
  )
  WORKER_JOBS="$WORKER_JOBS $JOB_ID"
done

DEPENDENCY=$(echo "$WORKER_JOBS" | xargs | tr " " ":")
FINALIZE_JOB=$(
  sbatch --parsable --dependency=afterany:"$DEPENDENCY" \
    --export=ALL,NEUROPREP_EXECUTION_PLAN_PATH="$PLAN",NEUROPREP_RELEASE_ROOT="$RELEASE_ROOT" \
    "$APP_ROOT/scripts/slurm/neuroprep_execution_finalize_gpu.sbatch"
)

trap - EXIT
echo "launch_complete=true"
echo "review_job=$REVIEW_JOB"
echo "worker_jobs=$WORKER_JOBS"
echo "finalize_job=$FINALIZE_JOB"
echo "state=$STATE"
echo "release_root=$RELEASE_ROOT"
