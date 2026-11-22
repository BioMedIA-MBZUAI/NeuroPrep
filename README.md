<p align="center">
  <img src="docs/assets/neuroprep-logo.svg" alt="NeuroPrep logo" width="132">
</p>

<h1 align="center">NeuroPrep</h1>

<p align="center">
  <b>One request in, validated multimodal neuroimaging derivatives out.</b><br>
  An agentic study compiler that turns a plain-language request into a pilot-qualified,<br>
  fully provenanced preprocessing run, from raw MRI, PET, CT, EEG/MEG and more to a released, model-ready dataset.
</p>

<p align="center">
  <img alt="Python 3.11 to 3.13" src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white">
  <img alt="Version 0.5.0" src="https://img.shields.io/badge/version-0.5.0-0E9F9A">
  <img alt="License: PolyForm Noncommercial 1.0.0" src="https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-3B3FC4">
  <img alt="Platforms: Linux and macOS" src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-555">
  <img alt="Runs on Slurm" src="https://img.shields.io/badge/HPC-Slurm%20%2B%20Singularity%2FApptainer-555">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#validated-backend-comparisons">Backends</a> ·
  <a href="#supported-modalities">Modalities</a> ·
  <a href="#chat-and-live-dashboard">Chat &amp; dashboard</a> ·
  <a href="#reference">Reference</a> ·
  <a href="#license-and-citation">Citation</a>
</p>

---

## Why NeuroPrep

Preprocessing a multimodal, longitudinal neuroimaging study usually means stitching
together a dozen tools by hand, guessing which backend suits your data and hardware,
and hoping nothing silently fails three days into a cluster run. NeuroPrep compiles the
whole study instead:

- **Plain-language requests.** Say what you want ("preprocess every modality and visit
  and save to /data/run"); a language model compiles it into a typed, allowlisted policy.
  It can choose objectives and preferred packages, never commands, containers or paths.
- **Evidence before production.** A one-participant pilot runs every competing backend
  on your own data and hardware. The results are written to an HMAC-signed evidence
  document bound to the exact git commit, and production only runs what the pilot qualified.
- **Automatic backend choice.** Failed backends are rejected; then a package you named
  wins; otherwise the fastest qualified backend is routed. Every decision is recorded with its reasons.
- **One engine, many datasets and modalities.** Dataset adapters and modality plugins
  handle structural, functional, diffusion and perfusion MRI, PET, CT, SPECT, EEG, iEEG,
  MEG, fNIRS, spectroscopy, microscopy and more, without per-cohort code.
- **Built for clusters.** A durable SQLite task DAG with leases and retries, GPU and CPU
  lanes, Slurm workers, offline containers and fail-closed preflight.
- **Nothing is "done" on an agent's say-so.** Completion is a declared contract: required
  tasks, artifacts, QC and provenance are checked before an atomic release.
- **Any model provider.** Local Ollama by default; OpenAI, Anthropic, OpenRouter, vLLM,
  LM Studio, llama.cpp or any OpenAI-compatible server.

## Quick start

NeuroPrep needs Python 3.11, 3.12 or 3.13 on Linux or macOS.

```bash
python -m venv .venv && . .venv/bin/activate
python -m pip install neuroprep-agentic        # or: pip install git+https://github.com/BioMedIA-MBZUAI/NeuroPrep
neuroprep-agentic --version
neuroprep-agentic runtime-check                # model provider, containers and tools on this host
```

Then hand it a whole study in one sentence:

```bash
neuroprep-agentic auto "My data is at /data/neuroprep/source; preprocess every modality and visit and save to /data/neuroprep/run"
```

Use `--execute local` to run on this host, `--execute slurm` to submit workers, or
`--execute plan-only` to stop after the pilot with a ready production plan. Scientific
applications (FreeSurfer, fMRIPrep, QSIPrep and so on) run from containers that are
checked, and can be prefetched, before a run starts; see [Configuring a host](#configuring-a-host).

## How it works

```mermaid
flowchart LR
    R["Plain-language request"] --> P["Typed policy<br/>(LLM, schema-validated)"]
    P --> I["Inventory, identity<br/>and BIDS mapping"]
    I --> C["Compiled task DAG"]
    C --> Q["Pilot: every competing<br/>backend on 1 participant"]
    Q --> E["Signed evidence<br/>(HMAC, git-bound)"]
    E --> S["Backend selection<br/>with recorded reasons"]
    S --> X["Production run<br/>local or Slurm"]
    X --> GContract met?
    G -->|yes| D["Atomic release<br/>+ model-ready index"]
    G -->|no| X
```

The model is called once, to compile the request. Everything after that is
deterministic code: identity mapping, BIDS conversion, dependencies, resources,
evidence and the release decision. A failed pilot stops the run before any production
task starts, and every stage is logged to `autopilot-report.json`.

## Validated backend comparisons

When a capability has more than one integrated, hardware-compatible backend, the pilot
runs all of them on the same participant. These comparisons were validated for v0.5.0
on public OpenNeuro subsets, on Slurm nodes with one NVIDIA A100-SXM4-40GB GPU, 16 CPU
cores and 64 GB of memory, with Qwen3 14B served locally by Ollama:

| Capability | Backends compared | Pilot result (1 participant) | Selected |
|---|---|---|---|
| Whole-brain segmentation (T1w) | DeepPrep 25.1.0, FastSurfer 2.6.1 | 25.1 vs 40.2 min, both passed | DeepPrep (fastest) |
| ... when speed or low memory is requested | + SynthSeg (FreeSurfer 7.4.1) | SynthSeg 1.7 min | SynthSeg (request) |
| ASL perfusion | ASLPrep 26.0.3, FSL BASIL | 16.8 vs 0.8 min, both passed | ASLPrep (request) |
| CT segmentation | ANTs (NeuroPrep spatial), TotalSegmentator 2.18.0 | 2.3 vs 1.1 min, both passed | TotalSegmentator (request) |
| Diffusion preprocessing | MRtrix3 3.0.8, QSIPrep 26.0.0 (+ QSIRecon) | 26.7 vs 27.2 min, both passed | MRtrix3 (request) |

When a backend fails its pilot it loses instead of failing the run: in an early ASL
validation, ASLPrep failed and BASIL was selected automatically. Pilots compare pass or
fail and runtime only; they do not rank segmentation or perfusion accuracy.


## Supported modalities

| Family | Inputs | Executor |
|---|---|---|
| Structural MRI | T1w, T2w, FLAIR, PDw | sMRIPrep, DeepPrep / FastSurfer / SynthSeg, NeuroPrep spatial |
| Functional MRI | BOLD task and rest | fMRIPrep + XCP-D |
| Diffusion MRI | DWI, DTI, DKI, HARDI, DSI | QSIPrep + QSIRecon, or MRtrix3 |
| Perfusion MRI | ASL | ASLPrep or FSL BASIL |
| Susceptibility and quantitative MRI | SWI, QSM, UNIT1, T2*, qMRI maps | NeuroPrep v4, QSMxT routing |
| Spectroscopy | SVS, MRSI, vendor raw | spec2nii + FSL-MRS |
| PET, SPECT | static and dynamic PET, SPECT | PETPrep; quantitative-preserving validation |
| CT | non-contrast, CTA, perfusion CT | NeuroPrep spatial (ANTs) or TotalSegmentator |
| Electrophysiology | EEG, iEEG, MEG | NeuroPrep v4 MNE pipelines |
| Optical and other | fNIRS, microscopy, ultrasound, physiology, behaviour, genetics | typed native normalization and QC |

The full routing table, with versions and BIDS targets, is in
[docs/modality-matrix.md](docs/modality-matrix.md). New modalities are plugins, not
changes to core ingestion.


## Chat and live dashboard

The local interface turns a natural-language study request into the same typed
policy, acquisition graph, backend plan and preflight report used by the CLI.
The planning page inventories the source and writes planning artifacts, but
does not initialize workers, submit jobs or release derivatives; the chat page
below runs the whole chain.

```bash
neuroprep-agentic interface \
  --agentic-project /data/neuroprep/NeuroPrep-agentic \
  --port 8787
```

Open `http://127.0.0.1:8787`. The server binds only to loopback. When NeuroPrep
runs on a cluster, create a tunnel from the workstation and use the same URL:

```bash
ssh -N -L 8787:127.0.0.1:8787 USER@CLUSTER
```

The interface accepts only scientific request fields. Paths are validated by
the deterministic prompt-run boundary, and commands, containers, scheduler
directives, task state and release decisions are not part of its API.

### Chat: one prompt, whole run

`http://127.0.0.1:8787/chat` is a chat page for `neuroprep-agentic auto`. Type a
request that names the input folder, for example *"Preprocess the structural
T1-weighted MRI of every subject with quality control. My data is at /data/bids."*,
or attach the data with **Attach folder** / **Attach files** (uploaded to
`$NEUROPREP_OUTPUT_ROOT/uploads/`, keeping the folder layout; at most
`NEUROPREP_MAX_UPLOAD_MB`, default 20480, per file). Naming an output folder
("save outputs to /data/out") is optional: otherwise results go to a new folder
named after the input under `NEUROPREP_OUTPUT_ROOT` (default `$NEUROPREP_HOME/runs`).
When something is missing or unusable (no input, a folder that does not exist or
is outside the allowed roots, an output folder that already has files) the page
asks for it instead of failing, and the answer, typed or attached, completes the
original request. Choose how to run it:

- **here, to completion** (`--execute local`): run on the host serving the page,
  e.g. inside a GPU job; the released files are listed in the chat when done.
- **pilot + plan only** (`--execute plan-only`): qualify the tools on a pilot and
  stop at the production plan.
- **submit to Slurm** (`--execute slurm`): submit the production workers.

Each run in the chat links to a live dashboard (`/dashboard?run=<id>`): the stage
tracker, task counts, a timeline of every task (tool × subject, pilot and production,
coloured by status and growing while it runs), the live log of the running task (or
any task you click), GPU utilisation and memory, CPU load and RAM sampled every 5 s
on the serving node, and the pilot runtime of each compared tool with the one chosen.
It reads the run's own task database read-only and refreshes every 3 s.

Each request starts one `auto` process; the page shows its stages as they finish,
the backends chosen, the log tail and, at the end, the released files from
`derivatives/release/model-ready-index.jsonl`. The process log is written next to
the output folder (`<output>.chat-run.log`). Runs are tracked while the server is
up; after a restart the reports and outputs remain on disk. Unlike the planning
page, the chat page executes the run, so start the server only where that is
allowed (on CIAI: inside a compute job, never on the login node).

## Reference

### Autonomous mode in detail

The autopilot calls the model once to compile the request into a typed policy, then
runs every remaining stage deterministically: inventory and plan compilation; choice
of the smallest set of participants that exercises every required backend; a
complete pilot run of those participants on this host; an audited, HMAC-signed
evidence document binding the pilot to the target plan; and the production run
(`--execute local` on this host, `--execute slurm` to submit workers, or
`--execute plan-only`) followed by release. A failed pilot stops the run before any
production task starts. Every stage is written to `autopilot-report.json` (with
`approval: autonomous`) next to the run and in `<run>.neuroprep-qualification/`.

Evidence binds the git commit of `NEUROPREP_AGENTIC_PROJECT`, so the checkout must be
clean. Without `--signing-key` a run-local `0600` key is created under the
qualification root.

**Comparing alternatives.** When a capability has more than one integrated,
hardware-compatible backend, the pilot runs all of them on the same subjects (as
optional tasks, so a backend that fails simply loses). The signed evidence records
each backend's pass/fail and measured runtime, and production is routed to one
backend per capability: failed backends are rejected; then a package named in the
prompt wins; then the shortest measured pilot runtime. A capability whose backend
another planned executor depends on (for example PETPrep on DeepPrep's
reconstruction) is pinned to it. `autopilot-report.json` lists the compared
backends, the losers and the reasons for each selection.

A backend only competes when it produces everything its capability promises
(`CAPABILITY_CONTRACTS` in `policy.py`), so a comparison never trades outputs for speed.

| Capability | Compared backends | Note |
|---|---|---|
| Whole-brain segmentation (T1w) | DeepPrep 25.1.0, FastSurfer 2.6.1 | segmentation + white surfaces |
| ... when the request asks for speed or low memory | + SynthSeg (FreeSurfer 7.4.1) | volumetric only, no surfaces |
| ASL perfusion | ASLPrep 26.0.3, FSL BASIL (oxford_asl 4.0.29) | BASIL needs an M0 |
| CT segmentation | NeuroPrep spatial (ANTs), TotalSegmentator 2.18.0 | free `total` task, brain and skull |
| Diffusion preprocessing | QSIPrep 26.0.0, MRtrix3 3.0.8 | MRtrix3 also fits the tensor itself |

Downstream apps read **derivative interfaces** rather than a fixed producer: XCP-D
consumes `fmriprep_bold`, QSIRecon consumes `qsiprep_dwi`, and the compiler fills the
path from whichever planned backend provides it. An alternative that does not provide
an interface a planned consumer reads is rejected, unless it takes over that consumer's
mandatory stages (MRtrix3 fits tensors instead of QSIRecon; QSIRecon's optional advanced
models are then dropped with a plan warning). DeepPrep 25.1.0's BOLD output is not an
alternative yet: XCP-D 26.1.1 needs an MNI-space brain mask and ANTs transforms that
DeepPrep does not write.


### Choosing a language model

The policy compiler and the advisory review crew work with any of these
providers. Name models as `provider/model`; no extra SDK is needed.

| Provider | Example `NEUROPREP_MANAGER_MODEL` | Required environment |
|---|---|---|
| Ollama (default) | `ollama/qwen3:14b` | `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`) |
| OpenAI | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic (Claude) | `anthropic/claude-sonnet-5-5` | `ANTHROPIC_API_KEY` |
| OpenRouter | `openrouter/anthropic/claude-sonnet-5.5` | `OPENROUTER_API_KEY` |
| vLLM | `vllm/Qwen/Qwen3-8B` | `NEUROPREP_LLM_BASE_URL` (default `http://127.0.0.1:8000/v1`) |
| LM Studio | `lmstudio/<loaded model>` | `NEUROPREP_LLM_BASE_URL` (default `http://127.0.0.1:1234/v1`) |
| llama.cpp server | `llamacpp/<model>` | `NEUROPREP_LLM_BASE_URL` (default `http://127.0.0.1:8080/v1`) |
| Any OpenAI-compatible server | `openai_compatible/<model>` | `NEUROPREP_LLM_BASE_URL`, optional `NEUROPREP_LLM_API_KEY` |

```bash
export NEUROPREP_MANAGER_MODEL=anthropic/claude-sonnet-5-5
export NEUROPREP_WORKER_MODEL=anthropic/claude-haiku-5-5
export ANTHROPIC_API_KEY=...
neuroprep-agentic runtime-check   # reports the provider and whether it is reachable
# one real, schema-validated call (a few hundred tokens) before a long run:
neuroprep-agentic prompt-policy --request "Preprocess the T1-weighted MRI" --output policy.json
```

`NEUROPREP_WORKER_MODEL` drives the advisory review agents. The older
`OLLAMA_MANAGER_MODEL`/`OLLAMA_WORKER_MODEL` variables still work. Hosted
providers never receive `NEUROPREP_LLM_BASE_URL`; use `OPENAI_BASE_URL`,
`OPENROUTER_BASE_URL` or `ANTHROPIC_BASE_URL` to route them through a proxy.

Structured output uses each provider's native mechanism: Ollama's schema
`format`, OpenAI-style `json_schema` (degrading to JSON mode, then a prompt,
on servers that reject it), and forced tool use for Claude. Every reply is
then validated against the `RunPolicy` schema, so a weaker model can fail
validation but cannot widen what NeuroPrep executes. Exact plan
reproducibility needs a pinned local model with a fixed seed; hosted APIs do
not guarantee deterministic output.

### Configuring a host

NeuroPrep has no built-in site paths. Every host-specific location comes from
the environment, with defaults under `NEUROPREP_HOME` (`~/.neuroprep`):

| Variable | Purpose | Default |
|---|---|---|
| `NEUROPREP_HOME` | Run state, logs and default asset locations | `~/.neuroprep` |
| `NEUROPREP_ALLOWED_ROOTS` | Extra data roots plans may read and write (`:`-separated); home and `NEUROPREP_HOME` are always allowed | none |
| `NEUROPREP_TOOLS_ROOT` | Native tools (FSL-MRS, dcm2niix, Nextflow, TemplateFlow seeds) | `$NEUROPREP_HOME/tools` |
| `NEUROPREP_CONTAINER_ROOT` | Singularity/Apptainer images | `$NEUROPREP_HOME/containers` |
| `NEUROPREP_V4_PROJECT` | NeuroPrep multimodal v4 code used by the CT/spatial, SWI/QSM, EEG/MEG and QSIPrep executors | the copy bundled with the package (`neuroprep_agentic/vendor/neuroprep_v4`, Apache-2.0) |
| `NEUROPREP_AGENTIC_PROJECT` | Git checkout whose commit is recorded in compiled plans | this checkout, else `$NEUROPREP_HOME/NeuroPrep-agentic` |
| `NEUROPREP_FSL_MRS_ROOT`, `NEUROPREP_DCM2NIIX`, `NEUROPREP_ANTS_IMAGE` | Individual tool overrides | under the tools/container roots |
| `NEUROPREP_CONTAINER_RUNTIME` | Singularity or Apptainer executable | `singularity`, then `apptainer`, on `PATH` |

Compiled execution plans record the git commit of `NEUROPREP_AGENTIC_PROJECT`
as provenance, so plan compilation and evidence binding need a git checkout of
this repository (a pip install alone is enough for planning, policy
compilation, the browser interface and the native executors). Bundled helper
files such as the Eddy wrapper always come from the installed package.

CrewAI usage telemetry and OpenTelemetry export are disabled by default; set
`CREWAI_TELEMETRY_OPT_OUT=false` to opt in.

Dataset adapters normalize source metadata; modality plugins detect and plan
acquisitions. The same engine can therefore handle OASIS4, OASIS3, ADNI,
ANMERGE, A4, IDEAS, AIBL, MR-RATE, and future datasets without cohort branches.

### Architecture

```mermaid
flowchart TB
    S["Immutable source datasets"] --> A["Dataset adapter registry"]
    A --> I["Identity and longitudinal resolver"]
    I --> C["Canonical acquisition graph"]
    C --> M["Modality plugin registry"]
    M --> Q{"Evidence-backed classification"}
    Q -->|recognized| B["BIDS/BIDS-extension mapping"]
    Q -->|ambiguous| X["Quarantine and review gate"]
    Q -->|unsupported| U["Unsupported inventory"]
    B --> P["Deterministic task DAG planner"]
    PI["Natural-language scientific request"] --> PC["CrewAI/Ollama typed policy compiler"]
    PC --> PG["Pydantic allowlist and safety gate"]
    PG --> BS["Hardware-aware backend selector"]
    BS --> ST["Bounded representative smoke-test plan"]
    ST --> P
    B --> BV["BIDS initialization and official validation gate"]
    P --> K["Versioned executor catalog"]
    K --> F["Fingerprinting execution compiler"]
    F --> O["CrewAI/Ollama central review (advisory)"]
    F --> R["Fail-closed image, tool, input, and metadata preflight"]
    R --> SDB["Durable SQLite DAG, leases, retries"]
    SDB --> SL["Independent GPU-resident Slurm workers"]
    SL --> E["Containerized and native scientific executors"]
    E --> V["Artifact validators, QC, provenance"]
    V --> G{"Required tasks and artifacts complete?"}
    G -->|no| SDB
    G -->|yes| D["Atomic release, model-ready index, tensor index"]
```

CrewAI and Ollama are the general instruction and planning layer. OpenJev is an
optional local typed-decision layer for high-volume routing and bounded recovery
decisions. Neither reasoning layer guesses modalities, edits source data, or
replaces scientific software. Deterministic code creates identity mappings,
BIDS mappings, dependencies, resource requests, evidence, and completion
decisions. OpenJev reviews only unresolved acquisitions when
`prompt-run --openjev-review` is enabled; its output is advisory and is recorded
in `openjev-review.json`.

Prompt instructions are supported, but deliberately constrained. A prompt can
select an objective, desired output families, and allowlisted package
preferences. The `prompt-run` command separately extracts the source and output
roots with deterministic code, accepts only the configured data roots,
inventories the source, builds a manifest, compiles the full execution DAG, and
can initialize or submit it. The LLM cannot introduce commands, images, metadata
values, scheduler directives, or bypass BIDS/artifact/QC gates. Both the typed
policy and its digest are embedded in every compiled execution plan.

### Guarantees

- Source files are immutable.
- Patient identifiers use a keyed deterministic pseudonym.
- Cross-dataset identities merge only through an explicit crosswalk.
- Longitudinal acquisitions are grouped by patient and ordered from baseline.
- Ambiguous acquisitions are quarantined; an LLM may not choose a modality.
- Missing optional modalities never make an otherwise valid patient incomplete.
- Core BIDS, extensions, and provisional mappings remain distinguishable.
- DTI derives from DWI; QSM derives from susceptibility acquisitions.
- Tasks declare dependencies and CPU/GPU/memory/time requirements.
- Completion uses a declared contract, not an agent narrative.

### Coverage details

The initial registry covers structural MRI (T1w, T2w, FLAIR, PDw, UNIT1,
T2-star, SWI, quantitative MRI), BOLD fMRI, ASL, DWI/DTI, field maps, MRS, PET,
CT, SPECT, EEG, iEEG, MEG, fNIRS, microscopy, physiology, behavior, genetics,
and neurosonography. New modalities are plugins, not changes to core ingestion.

The runtime creates `dataset_description.json`, isolates provisional extensions
with `.bidsignore`, and runs BIDS Validator 3.0.1 before scientific processing.

Vendor spectroscopy inputs are converted to anonymized NIfTI-MRS with
spec2nii before FSL-MRS processing. SPECT NIfTI is validated and canonicalized
without changing quantitative intensities; ultrasound video is normalized
losslessly to FFV1 with explicit stream QC.

Behavior JSON/CSV is normalized to TSV, EDF physiology is converted to FIF with
recording QC, and optional EEG features contain computed spectral band powers.

Every registered task route has a concrete deterministic executor definition.
Scientific qualification remains evidence-based: a route is released for a
dataset only after its metadata, container/tool, artifacts, and QC contracts
pass on that dataset. Provisional modalities fail closed instead of being
presented as validated merely because they were detected.

### Command-line reference

All paths must be under the executing user's home directory, `NEUROPREP_HOME`
or a root listed in `NEUROPREP_ALLOWED_ROOTS`. The examples below use
`/data/neuroprep`, allowed with `export NEUROPREP_ALLOWED_ROOTS=/data/neuroprep`.

```bash
neuroprep-agentic hardware-profile \
  --output /data/neuroprep/run/hardware.json

# One prompt can bind X/Y, discover all supported inputs, consolidate patients
# and visits, compile the executable DAG, and preflight it. Initialization or
# submission requires signed evidence when the backend plan has smoke tests.
neuroprep-agentic prompt-run \
  --request 'My data is at /data/neuroprep/source; preprocess all modalities and longitudinal visits and save to /data/neuroprep/run' \
  --agentic-project /data/neuroprep/NeuroPrep-agentic \
  --freesurfer-license /data/neuroprep/licenses/freesurfer.txt

# Add --submit to launch central review, workers, and the final release gate.
# Omit --initialize/--submit for a read-only-source planning and preflight run.

neuroprep-agentic prompt-policy \
  --request-file /data/neuroprep/run/request.txt \
  --hardware-profile /data/neuroprep/run/hardware.json \
  --output /data/neuroprep/run/policy.json
neuroprep-agentic backend-plan \
  --policy /data/neuroprep/run/policy.json \
  --hardware-profile /data/neuroprep/run/hardware.json \
  --output /data/neuroprep/run/backend-plan.json

# Flat qualification evidence is retained for planning and compatibility only.
# Production initialization uses the signed workflow in docs/backend-evidence.md.
neuroprep-agentic qualification-evidence \
  /data/neuroprep/run/execution-plan.json \
  --output /data/neuroprep/run/backend-evidence.json

neuroprep-agentic coverage --output /data/neuroprep/results/coverage.json
neuroprep-agentic ingest-plan \
  /data/neuroprep/NeuroPrep-agentic/examples/synthetic_multimodal_manifest.json \
  --identity-key 'replace-with-a-secret-at-least-12-characters' \
  --output /data/neuroprep/results/plan.json
neuroprep-agentic schemas --output /data/neuroprep/results/schemas.json

neuroprep-agentic executor-coverage \
  --output /data/neuroprep/results/executor-coverage.json
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

scripts/install_cluster_tools.sh
neuroprep-agentic catalog-images --inspect
neuroprep-agentic prefetch-images /data/neuroprep/run/execution-plan.json --execute
neuroprep-agentic prepare-assets /data/neuroprep/run/execution-plan.json --execute
neuroprep-agentic preflight-run /data/neuroprep/run/execution-plan.json

# Submit central review, GPU workers, and the release gate.
scripts/launch_neuroprep_run.sh \
  /data/neuroprep/run/execution-plan.json 2
```

### Running on Slurm with a GPU-resident model

The source repository includes example Slurm launchers (`scripts/slurm/`) for a
cluster with Singularity and A100 GPUs. Site locations are set in one file,
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

The backend portfolio records multiple packages per capability. Only pinned,
integrated executors are eligible for selection. Other packages remain explicit
qualification candidates until their adapters and representative smoke tests
pass; the agent may plan those tests but cannot silently promote a candidate.

Diffusion planning also inspects measured q-space sampling. Shelled data uses
GPU Eddy; DSI-like data with more than six nominal nonzero b-value levels uses
QSIPrep SHORELine (`--hmc-model 3dSHORE`) because FSL Eddy does not support DSI.
The selected model and b-value evidence are stored in the task metadata.

For an agent review of a manifest, export `NEUROPREP_WORKFLOW=ingest_plan`,
`NEUROPREP_MANIFEST_PATH`, and a strong `NEUROPREP_IDENTITY_KEY` before
submitting with `sbatch --export=ALL`.

The launcher starts loopback-only Ollama in Singularity on the allocated GPU.
It puts XDG state, temporary directories, models, package/container caches, and
CrewAI state on shared storage; preflights `/usr/bin/singularity`; binds
NFS directories to `/tmp` and `/var/tmp`; and disables Ollama cloud access.
`prepare-assets` also integrity-checks the offline Nextflow 24.10.3 jar and the
ASLPrep MNI6-to-MNI2009c transform so large assets never spill into the small
contained root filesystem.

Each compiled container command receives a task-specific NFS-backed container
home, `/tmp`, and `/var/tmp`; the host `HOME` environment is never reassigned.
Workers use transactional claims and renewable leases; failed attempts retry up
to the declared limit, terminal failures block descendants,
and completed artifacts are never silently reopened.

`finalize-run` is the only release gate. It verifies plan/state identity, zero
incomplete required tasks, and the continued existence of every recorded
artifact before atomically writing `release-summary.json`,
`model-ready-index.jsonl`, and `tensor-index.jsonl`.

See `docs/architecture.md`, `docs/modality-matrix.md`, and
`docs/dataset-onboarding.md`.

### Further documentation

- [Architecture](docs/architecture.md)
- [Signed backend evidence](docs/backend-evidence.md)
- [Modality matrix](docs/modality-matrix.md)
- [Onboarding a new dataset](docs/dataset-onboarding.md)
- [Changelog](CHANGELOG.md)

## License and citation

NeuroPrep is released under the PolyForm Noncommercial License 1.0.0 (see the
`LICENSE` file, also included in every installed copy).
Research, teaching, personal study and use by academic, non-profit, public
health and government institutions are permitted. Commercial use requires a
separate written license from the copyright holder.

If NeuroPrep contributes to a publication, presentation or other output, please
cite it using `CITATION.cff` (GitHub shows this as "Cite this repository"). Redistributions must keep the `Required Notice:` lines
at the top of `LICENSE`.

The package bundles the NeuroPrep multimodal v4 code (`neuroprep_agentic/vendor/neuroprep_v4`)
under its own Apache License 2.0, included in that folder; it is unchanged from its source.
