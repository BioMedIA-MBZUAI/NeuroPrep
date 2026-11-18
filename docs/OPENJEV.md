# OpenJev integration

NeuroPrep can use a local OpenJev helper for bounded decisions on unresolved
acquisitions. It is deliberately advisory: deterministic modality plugins,
preflight checks, task state, QC, and release validation remain authoritative.

Start the OpenJev helper using the serving instructions for the pinned model,
then run a prompt workflow with:

```bash
neuroprep-agentic prompt-run \
  --request "Process the dataset at /data/neuroprep/input and save to /data/neuroprep/runs/example" \
  --openjev-review
```

The review is written to `openjev-review.json`. It contains only de-identified
state summaries and the bounded choices returned by OpenJev. It never changes
the execution plan. Ambiguous or unsupported acquisitions still fail closed
unless the caller explicitly supplies the existing `--allow-quarantine` or
`--allow-unsupported` policy.

The endpoint and model can be overridden with:

```bash
export NEUROPREP_OPENJEV_ENDPOINT=http://127.0.0.1:3000/v1/systemone
export NEUROPREP_OPENJEV_MODEL=openjev
```

For direct diagnostics:

```bash
neuroprep-agentic jev-decision route /approved/state.json
neuroprep-agentic jev-decision boolean /approved/state.json \
  --instructions "Is this task safe to retry automatically?"
neuroprep-agentic jev-decision urgency /approved/state.json
neuroprep-agentic jev-decision next_action /approved/state.json
```

The state file must be a de-identified JSON object. Do not send patient names,
dates of birth, accession numbers, raw image data, or unrestricted filesystem
paths to the model.

## Slurm deployment

On the lab cluster, the reproducible service job is:

```bash
sbatch scripts/slurm/neuroprep_services_gpu.sbatch
```

The job starts user-local Ollama and the OpenJev shim on one GPU, verifies
`/api/tags` and `/v1/version`, and then runs `neuroprep-agentic runtime-check`.
The default cluster backend is the validated GGUF model served through Ollama;
the optional vLLM path is isolated in `~/openjev-venv` and should only be used
with a vLLM build compatible with the cluster driver. Set `OPENJEV_MODEL_DIR`,
`OPENJEV_GGUF`, or `OPENJEV_SHIM` if the assets are stored elsewhere.

The launcher also exports the validated site-tool roots by default:
`~/fsl-mrs-root` (FSL-MRS 2.5.0 entry points) and `~/ants-2.6.5.sif` (ANTsX
2.6.5). Override `NEUROPREP_FSL_MRS_ROOT` or `NEUROPREP_ANTS_IMAGE` when
deploying to another account or cluster.
