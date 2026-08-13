#!/bin/bash

set -euo pipefail

TOOL_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-tools
RUNTIME_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-runtime/tool-installs
CACHE_ROOT=/nfs-stor/salma.hassan/NeuroPrep-agentic-cache/tool-installs
DCM_VERSION=v1.0.20260416
DCM_SHA256=e88b40f6ebbcf9f47ebfdd7bb5f0127297cb7e8b06266a91a4642b5814031bd0
DCM_URL=https://github.com/rordenlab/dcm2niix/releases/download/$DCM_VERSION/dcm2niix_lnx.zip
DCM_ROOT=$TOOL_ROOT/dcm2niix-$DCM_VERSION
NEXTFLOW_VERSION=24.10.3
NEXTFLOW_SHA256=835f5a178ac7c5c4fc12a3407aa381c05d8b3f5750c8c3292a5389c12721d12a
NEXTFLOW_SIZE=34362800
NEXTFLOW_URL=https://www.nextflow.io/releases/v$NEXTFLOW_VERSION/nextflow-$NEXTFLOW_VERSION-one.jar
NEXTFLOW_ROOT=$TOOL_ROOT/nextflow/framework/$NEXTFLOW_VERSION
NEXTFLOW_JAR=$NEXTFLOW_ROOT/nextflow-$NEXTFLOW_VERSION-one.jar
FSL_MRS_VERSION=2.4.17
FSL_MRS_ROOT=$TOOL_ROOT/fsl-mrs-$FSL_MRS_VERSION
VERIFY=0

if [ "${1:-}" = "--verify" ]; then
  VERIFY=1
elif [ "$#" -ne 0 ]; then
  echo "usage: $0 [--verify]" >&2
  exit 2
fi

export TMPDIR=$RUNTIME_ROOT/tmp
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export XDG_CONFIG_HOME=$RUNTIME_ROOT/xdg-config
export XDG_DATA_HOME=$RUNTIME_ROOT/xdg-data
export XDG_STATE_HOME=$RUNTIME_ROOT/xdg-state
export CONDA_PKGS_DIRS=$CACHE_ROOT/conda-pkgs
export CONDA_ENVS_PATH=$TOOL_ROOT/conda-envs
mkdir -p "$TOOL_ROOT" "$RUNTIME_ROOT" "$CACHE_ROOT" "$TMPDIR" \
  "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" \
  "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH"

install_dcm2niix() {
  if [ ! -x "$DCM_ROOT/dcm2niix" ]; then
    local build_root
    build_root=$(mktemp -d "$RUNTIME_ROOT/dcm2niix.XXXXXX")
    trap 'rm -rf "$build_root"' RETURN
    curl -fL --retry 5 -o "$build_root/dcm2niix_lnx.zip" "$DCM_URL"
    echo "$DCM_SHA256  $build_root/dcm2niix_lnx.zip" | sha256sum -c -
    mkdir -p "$build_root/unpacked" "$DCM_ROOT"
    unzip -q "$build_root/dcm2niix_lnx.zip" -d "$build_root/unpacked"
    install -m 0755 "$build_root/unpacked/dcm2niix" "$DCM_ROOT/dcm2niix"
    rm -rf "$build_root"
    trap - RETURN
  fi
  local version_output
  version_output=$("$DCM_ROOT/dcm2niix" --version 2>&1 || true)
  echo "$version_output" | grep -F "$DCM_VERSION" >/dev/null
  echo "dcm2niix=$DCM_ROOT/dcm2niix"
}

install_nextflow() {
  local valid=0
  if [ -f "$NEXTFLOW_JAR" ]; then
    if [ "$(stat -c %s "$NEXTFLOW_JAR")" = "$NEXTFLOW_SIZE" ] && \
      echo "$NEXTFLOW_SHA256  $NEXTFLOW_JAR" | sha256sum -c - >/dev/null; then
      valid=1
    fi
  fi
  if [ "$valid" -ne 1 ]; then
    local build_root
    build_root=$(mktemp -d "$RUNTIME_ROOT/nextflow.XXXXXX")
    trap 'rm -rf "$build_root"' RETURN
    curl -fL --retry 5 -o "$build_root/nextflow.jar" "$NEXTFLOW_URL"
    test "$(stat -c %s "$build_root/nextflow.jar")" = "$NEXTFLOW_SIZE"
    echo "$NEXTFLOW_SHA256  $build_root/nextflow.jar" | sha256sum -c -
    mkdir -p "$NEXTFLOW_ROOT"
    chmod u+w "$NEXTFLOW_JAR" 2>/dev/null || true
    install -m 0444 "$build_root/nextflow.jar" "$NEXTFLOW_JAR"
    rm -rf "$build_root"
    trap - RETURN
  fi
  echo "nextflow=$NEXTFLOW_JAR"
}

install_fsl_mrs() {
  if [ ! -x "$FSL_MRS_ROOT/bin/fsl_mrs" ] || \
    [ ! -x "$FSL_MRS_ROOT/bin/fsl_mrs_preproc" ]; then
    command -v conda >/dev/null
    conda create -y -p "$FSL_MRS_ROOT" --override-channels \
      -c conda-forge \
      -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \
      "fsl_mrs=$FSL_MRS_VERSION=with_fsl"
  fi
  export PATH=$FSL_MRS_ROOT/bin:$PATH
  export FSLDIR=$FSL_MRS_ROOT
  export FSLOUTPUTTYPE=NIFTI_GZ
  test "$(fsl_mrs --version)" = "$FSL_MRS_VERSION"
  test "$(fsl_mrs_preproc --version)" = "$FSL_MRS_VERSION"
  if [ "$VERIFY" -eq 1 ]; then
    local verify_root
    verify_root=$(mktemp -d "$RUNTIME_ROOT/fsl-mrs-verify.XXXXXX")
    trap 'rm -rf "$verify_root"' RETURN
    (cd "$verify_root" && fsl_mrs_verify)
    touch "$FSL_MRS_ROOT/.neuroprep-verified"
    rm -rf "$verify_root"
    trap - RETURN
  fi
  echo "fsl_mrs=$FSL_MRS_ROOT/bin/fsl_mrs"
}

install_dcm2niix
install_nextflow
install_fsl_mrs
