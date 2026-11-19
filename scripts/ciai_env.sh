# CIAI cluster locations for NeuroPrep. Sourced by the Slurm launchers in this directory.
# The package itself has no built-in cluster paths (see src/neuroprep_agentic/paths.py);
# every value here can still be overridden by exporting it before submitting a job.
# Storage policy: /nfs-stor only. Do not add /l/users here.

export NEUROPREP_ALLOWED_ROOTS="${NEUROPREP_ALLOWED_ROOTS:-/nfs-stor/salma.hassan}"
export NEUROPREP_TOOLS_ROOT="${NEUROPREP_TOOLS_ROOT:-/nfs-stor/salma.hassan/NeuroPrep-agentic-tools}"
export NEUROPREP_CONTAINER_ROOT="${NEUROPREP_CONTAINER_ROOT:-/nfs-stor/salma.hassan/NeuroPrep/containers}"
# NEUROPREP_V4_PROJECT: unset, so the NeuroPrep v4 copy bundled with the package is used.
export NEUROPREP_HOME="${NEUROPREP_HOME:-/nfs-stor/salma.hassan/NeuroPrep-agentic-home}"
export CREWAI_TELEMETRY_OPT_OUT="${CREWAI_TELEMETRY_OPT_OUT:-true}"
export OTEL_SDK_DISABLED="${OTEL_SDK_DISABLED:-true}"
