# Running on Slurm

The simplest route is `auto` with `--execute slurm`: the pilot runs on the current
host (for example inside a GPU job) and the production workers are submitted to Slurm.
This page covers the launchers and the step-by-step workflow underneath.

## Example launchers

The source repository includes example Slurm launchers (`scripts/slurm/`) for a cluster
with Singularity and A100 GPUs. Site locations are set in one file,
`scripts/ciai_env.sh`; copy and edit it for another cluster.

```bash
cd /data/neuroprep/NeuroPrep-agentic
sbatch scripts/slurm/neuroprep_agentic_gpu.sbatch

# Compile a prompt and backend/smoke plan entirely on a scheduled GPU.
sbatch --export=ALL,\
NEUROPREP_REQUEST_FILE=/data/neuroprep/run/request.txt,\
NEUROPREP_POLICY_OUTPUT=/data/neuroprep/run/policy.json \
  scripts/slurm/neuroprep_prompt_policy_gpu.sbatch

# Run the complete prompt compiler on a scheduled GPU. Set
# NEUROPREP_SUBMIT=true only when the compiled run should be launched. Set
# NEUROPREP_AGENTIC_PROJECT when executing an immutable production clone.
sbatch --export=ALL,\
NEUROPREP_REQUEST_FILE=/data/neuroprep/run/request.txt,\
NEUROPREP_AGENTIC_PROJECT=/data/neuroprep/NeuroPrep-agentic,\
NEUROPREP_FREESURFER_LICENSE=/data/neuroprep/licenses/freesurfer.txt,\
NEUROPREP_SUBMIT=false \
  scripts/slurm/neuroprep_prompt_run_gpu.sbatch
```

For an agent review of a manifest, export `NEUROPREP_WORKFLOW=ingest_plan`,
`NEUROPREP_MANIFEST_PATH` and a strong `NEUROPREP_IDENTITY_KEY` before submitting with
`sbatch --export=ALL`.

## GPU-resident model

The launcher starts loopback-only Ollama in Singularity on the allocated GPU. It puts
XDG state, temporary directories, models, package/container caches and CrewAI state on
shared storage; preflights `/usr/bin/singularity`; binds NFS directories to `/tmp` and
`/var/tmp`; and disables Ollama cloud access.

## Workers and leases

Each compiled container command receives a task-specific NFS-backed container home,
`/tmp` and `/var/tmp`; the host `HOME` environment is never reassigned. Workers use
transactional claims and renewable leases; failed attempts retry up to the declared
limit, terminal failures block descendants, and completed artifacts are never silently
reopened.

## Step by step

`auto` and `prompt-run` chain these stages for you. Run them individually when you
need to inspect or edit an intermediate artifact:

```bash
neuroprep-agentic hardware-profile --output /data/neuroprep/run/hardware.json

neuroprep-agentic prompt-policy \
  --request-file /data/neuroprep/run/request.txt \
  --hardware-profile /data/neuroprep/run/hardware.json \
  --output /data/neuroprep/run/policy.json

neuroprep-agentic backend-plan \
  --policy /data/neuroprep/run/policy.json \
  --hardware-profile /data/neuroprep/run/hardware.json \
  --output /data/neuroprep/run/backend-plan.json

neuroprep-agentic compile-run MANIFEST.json \
  --identity-key 'replace-with-a-secret-at-least-12-characters' \
  --source-root /data/neuroprep/source \
  --bids-root /data/neuroprep/run/bids \
  --derivatives-root /data/neuroprep/run/derivatives \
  --work-root /data/neuroprep/run/work \
  --state-path /data/neuroprep/run/state/tasks.sqlite \
  --agentic-project /data/neuroprep/NeuroPrep-agentic \
  --freesurfer-license /data/neuroprep/licenses/freesurfer.txt \
  --policy /data/neuroprep/run/policy.json \
  --hardware-profile /data/neuroprep/run/hardware.json \
  --output /data/neuroprep/run/execution-plan.json

neuroprep-agentic prefetch-images /data/neuroprep/run/execution-plan.json --execute
neuroprep-agentic prepare-assets /data/neuroprep/run/execution-plan.json --execute
neuroprep-agentic preflight-run /data/neuroprep/run/execution-plan.json

# Submit central review, GPU workers, and the release gate.
scripts/launch_neuroprep_run.sh /data/neuroprep/run/execution-plan.json 2
```

Or bind the paths from a prompt, inventory the dataset, compile and preflight in one
step:

```bash
neuroprep-agentic prompt-run \
  --request 'My data is at /data/neuroprep/source; preprocess all modalities and longitudinal visits and save to /data/neuroprep/run' \
  --agentic-project /data/neuroprep/NeuroPrep-agentic \
  --freesurfer-license /data/neuroprep/licenses/freesurfer.txt
```

Add `--initialize` or `--submit` to launch central review, workers and the release
gate; omit them for a read-only planning and preflight run. Initialization or
submission requires [signed evidence](backend-evidence.md) when the backend plan has
smoke tests.

`finalize-run` is the only release gate; see [How it works](how-it-works.md#release).
