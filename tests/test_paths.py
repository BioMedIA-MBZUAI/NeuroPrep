"""Host-specific locations come from the environment, never from one user's cluster."""

from __future__ import annotations

import re
from pathlib import Path

from neuroprep_agentic import paths

PACKAGE = Path(paths.__file__).resolve().parent
PERSONAL = re.compile(r"salma\.hassan|/nfs-stor/[A-Za-z0-9]|/l/users/[A-Za-z0-9]")


def test_package_source_contains_no_personal_cluster_paths():
    offenders = [
        f"{path.relative_to(PACKAGE)}:{number}"
        for path in sorted(PACKAGE.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".jsonc", ".yaml", ".js", ".html", ""}
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1)
        if PERSONAL.search(line)
    ]
    assert offenders == []


def test_defaults_live_under_neuroprep_home(monkeypatch, tmp_path):
    for name in (
        "NEUROPREP_TOOLS_ROOT",
        "NEUROPREP_CONTAINER_ROOT",
        "NEUROPREP_V4_PROJECT",
        "NEUROPREP_FSL_MRS_ROOT",
        "NEUROPREP_DCM2NIIX",
        "NEUROPREP_ANTS_IMAGE",
        "NEUROPREP_ALLOWED_ROOTS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NEUROPREP_HOME", str(tmp_path))
    assert paths.tools_root() == tmp_path / "tools"
    assert paths.container_root() == tmp_path / "containers"
    assert paths.neuroprep_v4_project() == tmp_path / "NeuroPrep-multimodal-v4"
    assert paths.fsl_mrs_root() == tmp_path / "tools" / paths.FSL_MRS_DIRNAME
    assert paths.dcm2niix_path() == tmp_path / "tools" / paths.DCM2NIIX_RELATIVE
    assert paths.ants_image() == tmp_path / "containers" / paths.ANTS_IMAGE_NAME
    assert paths.allowed_roots() == (Path.home(), tmp_path)


def test_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("NEUROPREP_TOOLS_ROOT", str(tmp_path / "t"))
    monkeypatch.setenv("NEUROPREP_CONTAINER_ROOT", str(tmp_path / "c"))
    monkeypatch.setenv("NEUROPREP_ANTS_IMAGE", str(tmp_path / "ants.sif"))
    monkeypatch.setenv("NEUROPREP_ALLOWED_ROOTS", f"{tmp_path / 'a'}:{tmp_path / 'b'}")
    assert paths.fsl_mrs_root() == tmp_path / "t" / paths.FSL_MRS_DIRNAME
    assert paths.ants_image() == tmp_path / "ants.sif"
    assert paths.allowed_roots()[-2:] == (tmp_path / "a", tmp_path / "b")


def test_run_layout_defaults_follow_environment(monkeypatch, tmp_path):
    from neuroprep_agentic.execution_models import RunLayout

    monkeypatch.setenv("NEUROPREP_CONTAINER_ROOT", str(tmp_path / "containers"))
    monkeypatch.setenv("NEUROPREP_V4_PROJECT", str(tmp_path / "v4"))
    layout = RunLayout(
        dataset_id="d",
        source_root=str(Path.home() / "src"),
        bids_root=str(Path.home() / "bids"),
        derivatives_root=str(Path.home() / "deriv"),
        work_root=str(Path.home() / "work"),
        state_path=str(Path.home() / "state.sqlite"),
    )
    assert layout.container_root == str(tmp_path / "containers")
    assert layout.neuroprep_project == str(tmp_path / "v4")


def test_bundled_resources_ship_with_the_package():
    resources = paths.package_resources()
    for name in ("eddy_cuda10.2", "petprep_sitecustomize.py", "petprep_nfs_linear.yaml"):
        assert (resources / name).is_file(), name


def test_container_runtime_is_configurable(monkeypatch, tmp_path):
    runtime = tmp_path / "bin" / "apptainer"
    monkeypatch.setenv("NEUROPREP_CONTAINER_RUNTIME", str(runtime))
    assert paths.container_runtime() == runtime
    monkeypatch.delenv("NEUROPREP_CONTAINER_RUNTIME")
    monkeypatch.setenv("PATH", str(runtime.parent))
    runtime.parent.mkdir()
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    assert paths.container_runtime() == runtime.resolve()
