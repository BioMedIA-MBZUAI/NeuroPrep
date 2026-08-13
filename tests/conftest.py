from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path

import pytest

_TEST_SESSION_ROOT = Path(
    "/nfs-stor/salma.hassan/NeuroPrep-agentic-runtime/tests"
) / f"pytest-session-{os.getpid()}"
os.environ["XDG_CACHE_HOME"] = str(_TEST_SESSION_ROOT / "xdg-cache")
os.environ["XDG_CONFIG_HOME"] = str(_TEST_SESSION_ROOT / "xdg-config")
os.environ["XDG_DATA_HOME"] = str(_TEST_SESSION_ROOT / "xdg-data")
os.environ["XDG_STATE_HOME"] = str(_TEST_SESSION_ROOT / "xdg-state")
os.environ["CREWAI_STORAGE_DIR"] = str(_TEST_SESSION_ROOT / "crewai")
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

# Import the package before test modules import CrewAI directly. Its package
# initializer redirects the two CrewAI 1.15.13 Path.home()-hardcoded settings
# locations to approved project state on NFS.
import neuroprep_agentic as _neuroprep_agentic  # noqa: E402, F401


@pytest.fixture(scope="session", autouse=True)
def nfs_test_environment():
    _TEST_SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    yield
    for _attempt in range(20):
        try:
            shutil.rmtree(_TEST_SESSION_ROOT)
            break
        except OSError:
            time.sleep(0.1)
    else:
        shutil.rmtree(_TEST_SESSION_ROOT, ignore_errors=True)


@pytest.fixture
def nfs_tmp() -> Path:
    root = Path("/nfs-stor/salma.hassan/NeuroPrep-agentic-runtime/tests") / uuid.uuid4().hex
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root)
