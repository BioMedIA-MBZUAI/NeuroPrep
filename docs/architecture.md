# NeuroPrep-agentic architecture

The product separates probabilistic reasoning from scientific authority.
CrewAI/Ollama can explain evidence and propose recovery, but it cannot classify
ambiguous data, edit commands, claim work, mark completion, or submit arbitrary
scheduler actions.

## Prompt and adaptive backend control

Natural language enters through the GPU-resident CrewAI central orchestrator and
is compiled to a strict `RunPolicy`. The schema exposes scientific objectives,
output families, and an allowlist of package preferences only. There is no field
for shell commands, filesystem paths, container images, scheduler options, or
acquisition metadata.

The deterministic selector combines that policy with a measured
`HardwareProfile` and persistent smoke-test evidence. It filters candidates by
integration state, CPU/RAM/GPU requirements, and prior failures; ranks the
remaining backends; and emits a bounded representative-subject test for every
eligible integrated backend lacking evidence. This permits challenger packages
for the same capability to be qualified before policy, hardware, and persistent
evidence choose the production backend. A package is never promoted simply
because an LLM recommended it.

After a representative run becomes terminal, `qualification-evidence` reads
the durable ledger and rechecks recorded artifact paths. It emits only
`passed`/`failed` test IDs understood by the selector; incomplete work is
rejected. This closes the adaptive loop without giving the agent authority to
declare its own scientific success.

~~~mermaid
flowchart LR
    P["Prompt"] --> C["CrewAI/Ollama policy compiler"]
    C --> V{"RunPolicy schema valid?"}
    V -->|no| X["Reject"]
    V -->|yes| H["Measured hardware profile"]
    H --> S["Deterministic candidate filter and ranking"]
    E["Versioned smoke evidence"] --> S
    R["Backend portfolio"] --> S
    S --> T["Bounded smoke-test specifications"]
    T --> Q{"Artifacts and QC pass?"}
    Q -->|no| X
    Q -->|yes| D["Fingerprint policy and decisions into execution DAG"]
~~~

The portfolio currently integrates sMRIPrep, DeepPrep, fMRIPrep, XCP-D,
QSIPrep/QSIRecon, ASLPrep, PETPrep, QSMxT, ANTs-based CT, MNE, and FSL-MRS.
FastSurfer, FreeSurfer, SynthSeg, MRtrix3, DIPY, BASIL, ExploreASL, PETPVC,
SPM, TotalSegmentator, EEGLAB, and LCModel are recorded as candidates—not as
validated executors—until their adapters and reference qualification evidence
exist.

Within the QSIPrep route, the compiler makes a second deterministic adaptive
decision from the b-values. Single- and multi-shell data use Eddy, while
DSI-like sampling uses SHORELine/3dSHORE. The choice is fingerprinted with the
task and its nominal b-value evidence, so a prompt cannot force a scientifically
incompatible correction model.

~~~mermaid
flowchart TB
    SRC["Immutable files: DICOM, NIfTI, PET, CT, EEG, tables, omics"] --> AD["Dataset adapter"]
    AD --> ID["Keyed identity resolver"]
    ID --> LG["Longitudinal patient/session graph"]
    LG --> DET["Evidence-based modality detectors"]
    DET -->|recognized| BIDS["BIDS/BIDS-extension target"]
    DET -->|ambiguous| Q["Quarantine"]
    DET -->|unsupported| U["Unsupported inventory"]
    BIDS --> DAG["Modality-independent logical DAG"]
    BIDS --> BV["Dataset init plus BIDS Validator 3.0.1"]
    DAG --> CAT["Executor catalog: route + version + command + artifacts"]
    CAT --> CMP["Execution compiler"]
    CMP --> FP["Fingerprint every task and dependency"]
    FP --> PF["Fail-closed preflight"]
    FP --> CR["CrewAI execution review"]
    PF --> DB["SQLite state and attempt ledger"]
    DB --> W1["GPU worker 1"]
    DB --> W2["GPU worker 2"]
    DB --> WN["GPU worker N"]
    W1 --> VAL["Artifact and QC validation"]
    W2 --> VAL
    WN --> VAL
    VAL --> DB
    DB --> REL["Release gate"]
    REL --> IDX["Model-ready and tensor indexes"]
~~~

## Authority boundaries

1. Dataset adapters normalize metadata only; they never contain cohort-specific
   preprocessing logic.
2. Modality plugins detect acquisitions, create BIDS mappings, and declare
   logical task templates.
3. BIDS-core content must pass BIDS Validator 3.0.1 before scientific tasks.
4. The executor catalog maps every plugin/stage route to a versioned scientific
   implementation, resource class, required metadata, and artifact contract.
5. The compiler coalesces acquisition/session/subject work, resolves
   dependencies, injects NFS container isolation, and fingerprints the result.
6. Preflight checks catalog identity, approved roots, source files, metadata,
   executables, licenses, container images, and pinned large runtime assets.
7. SQLite owns task state. BEGIN IMMEDIATE claims, renewable leases, attempt
   records, bounded retries, and descendant blocking prevent double execution.
8. Slurm workers execute argv arrays without a shell and validate outputs before
   completion can be recorded.
9. The release gate recomputes plan/state consistency and artifact existence.

## Dataset independence

A dataset integration supplies only a manifest, an optional metadata adapter,
and an explicit identity crosswalk when the same person appears under different
source identifiers. OASIS4, OASIS3, ADNI, ANMERGE, A4, IDEAS, AIBL, MR-RATE,
and future cohorts therefore use the same modality plugins and executor catalog.

## Cluster topology

The launch workflow submits a GPU CrewAI/Ollama review job, then independent GPU
workers, then a GPU finalizer. Every job uses NFS chdir, excludes gpu-03, places
XDG/tmp/cache/state below approved cluster roots, and checks Singularity before
work starts. Scientific containers receive task-specific NFS-backed home, `/tmp`,
and `/var/tmp` mounts.

SmriPrep is the canonical subject structural producer. The compiler injects
subject-level prerequisites from SmriPrep into fMRIPrep, QSIPrep, PETPrep,
ASLPrep, and DeepPrep; XCP-D waits for fMRIPrep, and QSIRecon waits for QSIPrep.
Compilation fails if an anatomy-dependent task has no planned structural task.
