"""Dataset-agnostic agentic control plane for multimodal neurodata."""

import os
from pathlib import Path

from crewai_core import settings as _crewai_settings
from crewai_core.token_manager import TokenManager as _CrewAITokenManager

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
# In a source checkout keep state beside the repository. In an installed package
# parents[2] is the environment's lib/ directory, which must never receive state or output.
_PROJECT_ROOT = (
    _SOURCE_ROOT
    if (_SOURCE_ROOT / "pyproject.toml").is_file()
    else Path(os.environ.get("NEUROPREP_HOME", Path.home() / ".neuroprep")).resolve()
)
_STATE_ROOT = Path(
    os.environ.get("NEUROPREP_AGENTIC_STATE_ROOT", _PROJECT_ROOT / "state")
).resolve()

# CrewAI and several of its transitive dependencies follow the XDG defaults and
# otherwise create state below ~/.config or ~/.cache during a harmless CLI
# import. Keep the default control-plane footprint self-contained. Cluster
# launchers may override these with run-specific NFS locations.
os.environ.setdefault("CREWAI_STORAGE_DIR", str(_STATE_ROOT / "crewai"))
os.environ.setdefault("XDG_CONFIG_HOME", str(_STATE_ROOT / "xdg-config"))
os.environ.setdefault("XDG_CACHE_HOME", str(_STATE_ROOT / "xdg-cache"))
os.environ.setdefault("XDG_DATA_HOME", str(_STATE_ROOT / "xdg-data"))
os.environ.setdefault("XDG_STATE_HOME", str(_STATE_ROOT / "xdg-state"))
# Research data stays local by default: CrewAI's anonymous usage telemetry and OpenTelemetry
# export are off unless the user explicitly sets these variables.
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

# CrewAI 1.15.13 hard-codes two locations below Path.home() and ignores XDG
# variables for them. Redirect those pinned APIs before importing CrewAI itself.
_crewai_settings.DEFAULT_CONFIG_PATH = _STATE_ROOT / "crewai" / "config" / "settings.json"
_CREWAI_CREDENTIAL_DIR = _STATE_ROOT / "crewai" / "credentials"


def _crewai_secure_storage_path() -> Path:
    _CREWAI_CREDENTIAL_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    _CREWAI_CREDENTIAL_DIR.chmod(0o700)
    return _CREWAI_CREDENTIAL_DIR


_CrewAITokenManager._get_secure_storage_path = staticmethod(_crewai_secure_storage_path)

from .registry import PluginRegistry, builtin_registry  # noqa: E402

__all__ = ["PluginRegistry", "builtin_registry"]
__version__ = "0.5.0"
