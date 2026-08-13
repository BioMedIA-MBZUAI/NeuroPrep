from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_import_keeps_default_runtime_state_out_of_home(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    state_root = tmp_path / "state"
    fake_home.mkdir()

    environment = os.environ.copy()
    for name in (
        "CREWAI_STORAGE_DIR",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
    ):
        environment.pop(name, None)
    environment["HOME"] = str(fake_home)
    environment["NEUROPREP_AGENTIC_STATE_ROOT"] = str(state_root)

    code = """
import json
import os
import neuroprep_agentic.control
print(json.dumps({name: os.environ[name] for name in (
    'CREWAI_STORAGE_DIR', 'XDG_CONFIG_HOME', 'XDG_CACHE_HOME',
    'XDG_DATA_HOME', 'XDG_STATE_HOME')}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    paths = json.loads(completed.stdout.strip())

    assert not any(fake_home.iterdir())
    assert paths == {
        "CREWAI_STORAGE_DIR": str(state_root / "crewai"),
        "XDG_CONFIG_HOME": str(state_root / "xdg-config"),
        "XDG_CACHE_HOME": str(state_root / "xdg-cache"),
        "XDG_DATA_HOME": str(state_root / "xdg-data"),
        "XDG_STATE_HOME": str(state_root / "xdg-state"),
    }
