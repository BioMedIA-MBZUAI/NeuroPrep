# Quick start

This page takes a study from raw files to released derivatives with one command.
Install NeuroPrep first ([Installation](installation.md)).

## 1. Allow your data folders

Plans may only read and write under your home directory, `NEUROPREP_HOME`
(`~/.neuroprep`) or roots you list explicitly:

```bash
export NEUROPREP_ALLOWED_ROOTS=/data/neuroprep
```

## 2. Hand it the study in one sentence

Name the input folder and, optionally, the output folder:

```bash
neuroprep-agentic auto "My data is at /data/neuroprep/source; preprocess every modality and visit and save to /data/neuroprep/run"
```

Choose how far to go with `--execute`:

| Option | What happens |
|---|---|
| `--execute plan-only` | Qualify the backends on a pilot and stop with a ready production plan |
| `--execute local` | Run the pilot and the production run on this host |
| `--execute slurm` | Run the pilot, then submit production workers to Slurm |

## 3. What happens

1. One model call compiles the request into a typed, schema-validated policy.
2. NeuroPrep inventories the source, resolves identities and visits, maps to BIDS and
   compiles a task DAG.
3. It chooses the smallest set of participants that exercises every required backend
   and runs a complete **pilot** of them. When several backends can do the same job,
   all of them run on the pilot.
4. The pilot results are written to an HMAC-signed **evidence** document bound to the
   git commit. A failed pilot stops the run before any production task starts.
5. Each capability is routed to one backend, and the **production run** executes.
6. The completion contract is checked and the outputs are **released** atomically.

## 4. What you get

| Output | Where |
|---|---|
| Stage-by-stage report, with the backends compared and chosen | `autopilot-report.json` |
| BIDS dataset and derivatives | under the output folder |
| Release summary | `derivatives/release/release-summary.json` |
| Model-ready index of released files | `derivatives/release/model-ready-index.jsonl` |
| Tensor index | `derivatives/release/tensor-index.jsonl` |

## Next steps

- Prefer a browser? Use the [chat and live dashboard](interface.md).
- Pick a hosted model instead of local Ollama: [Language models](llm-providers.md).
- Run production on a cluster: [Running on Slurm](slurm.md).
- See the individual stages: [How it works](how-it-works.md) and the
  [command-line reference](cli.md).
