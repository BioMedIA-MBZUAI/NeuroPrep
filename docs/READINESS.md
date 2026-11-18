# Operational readiness

## Verified in a clean Python 3.11 environment

- `uv build --wheel` succeeds.
- The wheel contains the OpenJev client and the `neuroprep-agentic` console entry point.
- `uv lock --check` succeeds.
- `ruff check` and `python -m compileall` succeed.
- 120 tests pass when excluding tests that require site-specific FSL-MRS and ANTs
  installations.
- The policy, control, planning, ingestion, execution, provenance, and OpenJev
  test groups pass.
- The OpenJev client has been exercised against a local mock HTTP server.
- `neuroprep-agentic runtime-check` reports each deployment dependency separately
  and exits non-zero until all required services and assets are available.

## Cluster qualification completed

The lab-cluster service job was run on a GPU allocation and verified:

- `spec2nii`, `fsl_mrs_preproc`, and `fsl_mrs` from FSL-MRS 2.5.0;
- the ANTsX 2.6.5 Singularity image;
- a live OpenJev shim backed by the OpenJev GGUF model through Ollama;
- the configured Ollama model and `/v1/systemone` route decision;
- a `runtime-check` result with `ready: true` for all required services.

The qualification was an infrastructure and request-path proof, not a claim
that every dataset has already been processed on the cluster. A representative
dataset run should still be scheduled before a study-specific release.

Those checks must be run on the deployment host with the approved tool roots and
container images. The system remains fail-closed when any required runtime asset
is unavailable.
