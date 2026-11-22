# Installation

## Requirements

- Python 3.11, 3.12 or 3.13
- Linux or macOS
- For processing (not needed for planning): Singularity or Apptainer to run the
  scientific applications, and a language model provider (local
  [Ollama](https://ollama.com) by default; see [Language models](llm-providers.md))

## Install the package

=== "pip"

    ```bash
    python -m venv .venv && . .venv/bin/activate
    python -m pip install neuroprep-agentic
    ```

=== "uv"

    ```bash
    uv venv && . .venv/bin/activate
    uv pip install neuroprep-agentic
    ```

=== "From GitHub"

    ```bash
    python -m pip install git+https://github.com/BioMedIA-MBZUAI/NeuroPrep
    ```

Check the installation:

```bash
neuroprep-agentic --version
```

## Check the host

`runtime-check` reports each deployment dependency separately (model provider,
OpenJev, scientific tools, containers and runtime assets) and exits non-zero until
everything required is available:

```bash
neuroprep-agentic runtime-check
```

Scientific applications (FreeSurfer, fMRIPrep, QSIPrep and so on) run from containers
that are checked, and can be prefetched, before a run starts:

```bash
neuroprep-agentic catalog-images --inspect
neuroprep-agentic prefetch-images /data/neuroprep/run/execution-plan.json --execute
```

FreeSurfer-based backends need a FreeSurfer license file, passed with
`--freesurfer-license`. Where tools and images live is set through environment
variables; see [Configuring a host](configuration.md).

## Planning versus executing

A pip install is enough for planning, policy compilation, the browser interface and
the native executors. Compiled execution plans and signed evidence record the git
commit of the code that produced them, so **execution needs a git checkout** of the
repository, pointed to by `NEUROPREP_AGENTIC_PROJECT` (the checkout must be clean):

```bash
git clone https://github.com/BioMedIA-MBZUAI/NeuroPrep /data/neuroprep/NeuroPrep-agentic
export NEUROPREP_AGENTIC_PROJECT=/data/neuroprep/NeuroPrep-agentic
```

The source repository also contains the example Slurm launchers and cluster tool
installer (`scripts/`).

## Development install

```bash
git clone https://github.com/BioMedIA-MBZUAI/NeuroPrep
cd NeuroPrep
python -m pip install -e ".[dev]"
pytest
```
