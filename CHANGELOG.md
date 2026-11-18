# Changelog

## 0.5.0 — first public release

### Added
- `neuroprep-agentic auto "<request with input and output paths>"`: fully autonomous
  runs. One model call compiles the policy; deterministic stages then plan, select the
  smallest pilot cohort covering every required backend, execute the pilot, sign target-
  bound evidence, and run and release production (`--execute local|slurm|plan-only`).
  A failed pilot stops before production; every stage is recorded in
  `autopilot-report.json` with `approval: autonomous`.
- Provider-neutral language-model layer (`neuroprep_agentic.llm`). The policy
  compiler and the advisory review crew accept `provider/model` names for Ollama,
  OpenAI, Anthropic (Claude), OpenRouter, vLLM, LM Studio, llama.cpp and any
  OpenAI-compatible server. Structured output uses each provider's native
  mechanism and is always re-validated against the `RunPolicy` schema.
- `neuroprep_agentic.paths`: every host-specific location (tools, containers,
  data roots, NeuroPrep v4 checkout) is configured through `NEUROPREP_*`
  environment variables with defaults under `NEUROPREP_HOME` (`~/.neuroprep`).
- Optional site TemplateFlow cache: when `$NEUROPREP_TOOLS_ROOT/templateflow` exists, its
  files are copied into every run's TemplateFlow cache, so BIDS Apps need no S3 access at
  run time (on CIAI, TemplateFlow downloads failed with connection resets).
- Backend comparison. Executors can be declared as alternatives for a stage; the
  pilot runs every compared alternative as an optional qualification candidate,
  signed evidence records pass/fail and measured runtime per backend, and
  production routes each capability to the selected backend (failed → rejected,
  prompt preference, then fastest). Backends required by other planned executors
  are pinned. The execution gates require every comparison to be decided and every
  backend that runs to have passed; a routed plan is re-validated against the signed
  pilot before it initializes.
- FastSurfer 2.6.1 (`deepmi/fastsurfer:cu126-v2.6.1`) as an alternative to DeepPrep
  for whole-brain segmentation and surfaces.
- SynthSeg (`freesurfer/freesurfer:7.4.1`, `mri_synthseg --parc --robust`), FSL BASIL
  (`vnmd/fsl_6.0.7.18`, BIDS-ASL metadata mapped to `oxford_asl` flags) and
  TotalSegmentator 2.18.0 (free `total` task) as alternative backends.
- Derivative interfaces: executors declare what they `provide` and `consume`; XCP-D
  and QSIRecon read their input from whichever planned backend provides it instead of
  a hard-coded fMRIPrep/QSIPrep path. Selection rejects an alternative that would leave
  a planned consumer without input unless it takes over the consumer's mandatory
  stages; optional consumers without a provider are dropped with a plan warning.
- MRtrix3 3.0.8 (`mrtrix3/mrtrix3:3.0.8`) as an alternative to QSIPrep: denoise,
  Gibbs removal, `dwifslpreproc` (topup with an opposite-phase b0 series when present),
  bias correction, mask and tensor fit.
- Capability contracts: a backend competes only if it declares every output its
  capability requires; `fast`/`low_memory` requests accept volumetric segmentation.
- Backend capabilities activate by plan stage, not only by executor, and an executor
  that also does non-compared work stays required in a comparison pilot.
- `NEUROPREP_CONTAINER_RUNTIME` selects the Singularity or Apptainer executable;
  otherwise `singularity`, then `apptainer`, is found on `PATH`. Executor commands
  use a `{container_runtime}` placeholder instead of `/usr/bin/singularity`.
- Python 3.13 support (tested: full suite passes on 3.11, 3.12 and 3.13).
- `neuroprep-agentic --version`.
- `runtime-check` reports the configured LLM provider; hosted providers are
  checked for credentials without spending tokens.
- PolyForm Noncommercial 1.0.0 license and `CITATION.cff`.
- Example Slurm launchers read site locations from `scripts/ciai_env.sh`.

### Changed
- No built-in site paths: the data-root allowlist is the user's home,
  `NEUROPREP_HOME` and `NEUROPREP_ALLOWED_ROOTS`.
- Bundled helper files (Eddy wrapper, PETPrep site customization) are bound from
  the installed package instead of `<checkout>/src/...`, so container commands
  work from a pip install. In a source checkout the resolved paths are unchanged.
  The executor catalog digest changes accordingly.
- An installed package keeps state under `NEUROPREP_HOME` rather than inside the
  Python environment.
- CrewAI usage telemetry and OpenTelemetry export are off by default.
- Dependency `mne-nirs` (unused) replaced by `h5py`, which SNIRF reading needs;
  installs no longer pull nilearn, scikit-learn or seaborn.
- Version is single-sourced from `neuroprep_agentic.__version__`.
- Executor signatures in evidence no longer include the task's `required` flag
  (a scheduling choice, not executor behaviour). Evidence and catalog digests change.

### Fixed
- Policy compilation failed outright when a model listed packages outside the allowlist
  (seen with qwen3:14b listing nilearn, nipype, scikit-learn). The allowlist now reaches
  the model as a schema enum, and one repair attempt shows the model its validation error.
- The BIDS validator image was pinned to `bids/validator:v3.0.1`, a tag that does not
  exist on Docker Hub; it is now `3.0.1`. The executor catalog digest changes.
- The CT and EEG backend portfolio entries named executors that do not exist
  (`neuroprep-ct`, `neuroprep-eeg`), so those capabilities never got a decision.
- Settings rejected every model that was not `ollama/...`.
- The review crew could not use Claude without the `crewai[anthropic]` extra.
