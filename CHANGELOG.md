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

### Fixed
- Policy compilation failed outright when a model listed packages outside the allowlist
  (seen with qwen3:14b listing nilearn, nipype, scikit-learn). The allowlist now reaches
  the model as a schema enum, and one repair attempt shows the model its validation error.
- The BIDS validator image was pinned to `bids/validator:v3.0.1`, a tag that does not
  exist on Docker Hub; it is now `3.0.1`. The executor catalog digest changes.
- Settings rejected every model that was not `ollama/...`.
- The review crew could not use Claude without the `crewai[anthropic]` extra.
