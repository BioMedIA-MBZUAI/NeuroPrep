# Chat and dashboard

The local interface turns a natural-language study request into the same typed policy,
acquisition graph, backend plan and preflight report used by the CLI.

```bash
neuroprep-agentic interface \
  --agentic-project /data/neuroprep/NeuroPrep-agentic \
  --port 8787
```

Open `http://127.0.0.1:8787`. The server binds only to loopback. When NeuroPrep runs on
a cluster, create a tunnel from the workstation and use the same URL:

```bash
ssh -N -L 8787:127.0.0.1:8787 USER@CLUSTER
```

The interface accepts only scientific request fields. Paths are validated by the
deterministic prompt-run boundary, and commands, containers, scheduler directives, task
state and release decisions are not part of its API.

## Planning page

The planning page (`/`) inventories the source and writes planning artifacts, but does
not initialize workers, submit jobs or release derivatives.

## Chat: one prompt, whole run

`http://127.0.0.1:8787/chat` is a chat page for `neuroprep-agentic auto`. Type a request
that names the input folder, for example *"Preprocess the structural T1-weighted MRI of
every subject with quality control. My data is at /data/bids."*, or attach the data with
**Attach folder** / **Attach files**.

- Uploads go to `$NEUROPREP_OUTPUT_ROOT/uploads/`, keeping the folder layout, at most
  `NEUROPREP_MAX_UPLOAD_MB` (default 20480) per file.
- Naming an output folder ("save outputs to /data/out") is optional: otherwise results
  go to a new folder named after the input under `NEUROPREP_OUTPUT_ROOT` (default
  `$NEUROPREP_HOME/runs`).
- When something is missing or unusable (no input, a folder that does not exist or is
  outside the allowed roots, an output folder that already has files) the page asks for
  it instead of failing, and the answer, typed or attached, completes the original
  request.

Choose how to run it:

| Choice | Equivalent | What happens |
|---|---|---|
| here, to completion | `--execute local` | Runs on the host serving the page, e.g. inside a GPU job; the released files are listed in the chat when done |
| pilot + plan only | `--execute plan-only` | Qualifies the tools on a pilot and stops at the production plan |
| submit to Slurm | `--execute slurm` | Submits the production workers |

Each request starts one `auto` process; the page shows its stages as they finish, the
backends chosen, the log tail and, at the end, the released files from
`derivatives/release/model-ready-index.jsonl`. The process log is written next to the
output folder (`<output>.chat-run.log`). Runs are tracked while the server is up; after
a restart the reports and outputs remain on disk.

!!! warning "The chat page executes runs"
    Unlike the planning page, the chat page executes the run, so start the server only
    where that is allowed. On a shared cluster, run it inside a compute job, never on
    the login node.

## Live dashboard

Each run in the chat links to a live dashboard (`/dashboard?run=<id>`) showing:

- the stage tracker and task counts;
- a timeline of every task (tool × subject, pilot and production), coloured by status
  and growing while it runs;
- the live log of the running task, or any task you click;
- GPU utilisation and memory, CPU load and RAM, sampled every 5 s on the serving node;
- the pilot runtime of each compared tool, with the one chosen.

It reads the run's own task database read-only and refreshes every 3 s.
