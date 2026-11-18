# Dataset onboarding contract

NeuroPrep-agentic is dataset-agnostic at the execution layer. A new cohort
provides source evidence; it does not add cohort-specific preprocessing code.

## Required inputs

- One or more immutable source roots under a configured data root (home,
  `NEUROPREP_HOME` or `NEUROPREP_ALLOWED_ROOTS`).
- The cohort adapter identifier (`generic` is valid when no named adapter fits).
- A strong private identity key and, only when cross-cohort identity merging is
  intended, an explicit source-ID crosswalk.
- The FreeSurfer license path for T1/fMRI workflows.
- MRS basis-set paths and water-reference roles for spectroscopy acquisitions.
- Any cohort tables needed to resolve subjects, visits, acquisition dates,
  tracers, units, and modality-specific metadata.

## Manifest record

Every immutable acquisition is represented by a `SourceRecord` containing its
absolute source path, source subject, optional source visit, acquisition time or
days from baseline, file format, checksum when available, and extracted metadata.
The adapter normalizes field names; deterministic modality plugins classify the
record from evidence. Close classifications are quarantined rather than guessed.

## Execution gates

1. Identity resolution consolidates subjects and orders longitudinal sessions.
2. BIDS mappings and the logical task DAG are compiled and fingerprinted.
3. Source/output roots must be disjoint and located on approved cluster storage.
4. BIDS-core staging passes BIDS Validator 3.0.1; provisional extensions remain
   explicit and isolated.
5. Required images, executables, licenses, runtime assets, inputs, and metadata
   pass preflight.
6. CrewAI/Ollama reviews evidence but cannot change commands or completion state.
7. Slurm workers execute the durable DAG and validate declared artifacts.
8. The finalizer releases model-ready and tensor indexes only at zero incomplete
   required tasks.

## Qualification boundary

Registry coverage is not the same as dataset qualification. After source paths
are supplied, each cohort must first run a representative qualification slice
covering every observed modality, vendor, protocol, and longitudinal pattern.
Only routes whose real artifacts and QC contracts pass are released at scale.

See `examples/synthetic_multimodal_manifest.json` for the machine-readable
manifest shape.
