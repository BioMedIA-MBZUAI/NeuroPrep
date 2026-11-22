# Configuring a host

NeuroPrep has no built-in site paths. Every host-specific location comes from the
environment, with defaults under `NEUROPREP_HOME` (`~/.neuroprep`).

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

Model provider variables are listed in [Language models](llm-providers.md), and the
chat page's variables in [Chat and dashboard](interface.md).

## Allowed roots

All paths must be under the executing user's home directory, `NEUROPREP_HOME` or a root
listed in `NEUROPREP_ALLOWED_ROOTS`. The examples in these docs use `/data/neuroprep`:

```bash
export NEUROPREP_ALLOWED_ROOTS=/data/neuroprep
```

## Provenance needs a git checkout

Compiled execution plans record the git commit of `NEUROPREP_AGENTIC_PROJECT` as
provenance, so plan compilation and evidence binding need a git checkout of the
repository (a pip install alone is enough for planning, policy compilation, the browser
interface and the native executors). Bundled helper files such as the Eddy wrapper
always come from the installed package.

## Offline TemplateFlow

When `$NEUROPREP_TOOLS_ROOT/templateflow` exists, its files are copied into every run's
TemplateFlow cache, so BIDS Apps need no network access to TemplateFlow at run time.

## Preparing tools and images

```bash
scripts/install_cluster_tools.sh                     # from the source repository
neuroprep-agentic catalog-images --inspect
neuroprep-agentic prefetch-images /data/neuroprep/run/execution-plan.json --execute
neuroprep-agentic prepare-assets /data/neuroprep/run/execution-plan.json --execute
neuroprep-agentic preflight-run /data/neuroprep/run/execution-plan.json
```

`prepare-assets` also integrity-checks the offline Nextflow 24.10.3 jar and the ASLPrep
MNI6-to-MNI2009c transform. `preflight-run` fails closed when any image, tool, input or
metadata requirement is missing.
