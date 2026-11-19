# Vendored: NeuroPrep multimodal v4 (`neuroprep` 0.5.0)

Copied unchanged from the NeuroPrep v4 source tree (`src/neuroprep`), Apache License 2.0
(see `LICENSE` here). The original checkout is no longer published, so this copy is what
the following executors run, from inside their containers or with the package's Python:

- `neuroprep-spatial` and `neuroprep-susceptibility` (`python -m neuroprep.cli ...`)
- `neuroprep-sensor-eeg`, `-ieeg`, `-meg` (`python -m neuroprep.cli preprocess-sensor`)
- QSIPrep's `resources/qsiprep_primary_t1.json` and `resources/eddy_cuda.json`

The folder keeps a checkout's layout (`src/neuroprep`), so `NEUROPREP_V4_PROJECT` can still
point at an external v4 checkout instead. It is not imported by `neuroprep_agentic`.
