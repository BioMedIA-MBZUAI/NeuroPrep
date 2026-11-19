"""Installation-specific locations, configured through the environment.

Every host-specific path NeuroPrep uses is resolved here, so a deployment is described by a
handful of environment variables instead of edits to the code. Defaults live under
``NEUROPREP_HOME`` (``~/.neuroprep``) and never point at a particular user's cluster.

=============================  ===============================================================
Variable                       Default
=============================  ===============================================================
``NEUROPREP_HOME``             ``~/.neuroprep``
``NEUROPREP_ALLOWED_ROOTS``    (none) extra data roots, ``os.pathsep``-separated; home and
                               ``NEUROPREP_HOME`` are always allowed
``NEUROPREP_TOOLS_ROOT``       ``$NEUROPREP_HOME/tools``
``NEUROPREP_OUTPUT_ROOT``      ``$NEUROPREP_HOME/runs`` (chat runs whose prompt names no output)
``NEUROPREP_CONTAINER_ROOT``   ``$NEUROPREP_HOME/containers``
``NEUROPREP_V4_PROJECT``       the bundled copy (``neuroprep_agentic/vendor/neuroprep_v4``)
``NEUROPREP_AGENTIC_PROJECT``  this source checkout, else ``$NEUROPREP_HOME/NeuroPrep-agentic``
``NEUROPREP_FSL_MRS_ROOT``     ``$NEUROPREP_TOOLS_ROOT/fsl-mrs-2.4.17``
``NEUROPREP_DCM2NIIX``         ``$NEUROPREP_TOOLS_ROOT/dcm2niix-v1.0.20260416/dcm2niix``
``NEUROPREP_ANTS_IMAGE``       ``$NEUROPREP_CONTAINER_ROOT/ants-2.6.5.sif``
=============================  ===============================================================
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_SOURCE_ROOT = _PACKAGE_DIR.parents[1]

FSL_MRS_DIRNAME = "fsl-mrs-2.4.17"
DCM2NIIX_RELATIVE = Path("dcm2niix-v1.0.20260416") / "dcm2niix"
ANTS_IMAGE_NAME = "ants-2.6.5.sif"


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def container_runtime() -> Path:
    """Absolute Singularity/Apptainer executable: NEUROPREP_CONTAINER_RUNTIME, else PATH."""
    configured = _env_path("NEUROPREP_CONTAINER_RUNTIME")
    if configured:
        return configured
    for name in ("singularity", "apptainer"):
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    return Path("/usr/bin/singularity")


def neuroprep_home() -> Path:
    return _env_path("NEUROPREP_HOME") or Path.home() / ".neuroprep"


def extra_allowed_roots() -> tuple[Path, ...]:
    raw = os.environ.get("NEUROPREP_ALLOWED_ROOTS", "")
    return tuple(Path(item).expanduser() for item in raw.split(os.pathsep) if item.strip())


def allowed_roots() -> tuple[Path, ...]:
    """Data roots a plan may read or write: home, NEUROPREP_HOME and configured extras."""

    roots: list[Path] = []
    for root in (Path.home(), neuroprep_home(), *extra_allowed_roots()):
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def output_root() -> Path:
    return _env_path("NEUROPREP_OUTPUT_ROOT") or neuroprep_home() / "runs"


def tools_root() -> Path:
    return _env_path("NEUROPREP_TOOLS_ROOT") or neuroprep_home() / "tools"


def container_root() -> Path:
    return _env_path("NEUROPREP_CONTAINER_ROOT") or neuroprep_home() / "containers"


def neuroprep_v4_project() -> Path:
    """The NeuroPrep v4 code; the copy bundled with this package unless overridden."""

    return _env_path("NEUROPREP_V4_PROJECT") or _PACKAGE_DIR / "vendor" / "neuroprep_v4"


def source_checkout() -> Path | None:
    """The git checkout this code runs from, or ``None`` for an installed package."""

    if (_SOURCE_ROOT / "pyproject.toml").is_file() and (_SOURCE_ROOT / ".git").exists():
        return _SOURCE_ROOT
    return None


def default_agentic_project() -> Path:
    """Fallback for NEUROPREP_AGENTIC_PROJECT (read by settings at call time)."""

    return source_checkout() or neuroprep_home() / "NeuroPrep-agentic"


def package_resources() -> Path:
    """Bundled helper files; identical for a source checkout and a pip install."""

    return _PACKAGE_DIR / "resources"


def fsl_mrs_root() -> Path:
    return _env_path("NEUROPREP_FSL_MRS_ROOT") or tools_root() / FSL_MRS_DIRNAME


def dcm2niix_path() -> Path:
    return _env_path("NEUROPREP_DCM2NIIX") or tools_root() / DCM2NIIX_RELATIVE


def ants_image() -> Path:
    return _env_path("NEUROPREP_ANTS_IMAGE") or container_root() / ANTS_IMAGE_NAME
