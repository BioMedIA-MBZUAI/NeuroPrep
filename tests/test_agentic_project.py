from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from neuroprep_agentic import settings
from neuroprep_agentic.main import cli


def _git_repository(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path.resolve()


def test_agentic_project_cli_env_default_precedence(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    default = _git_repository(nfs_tmp / "default-project")
    environment = _git_repository(nfs_tmp / "environment-project")
    explicit_cli = _git_repository(nfs_tmp / "cli-project")
    monkeypatch.setattr(settings, "DEFAULT_AGENTIC_PROJECT", str(default))

    monkeypatch.delenv("NEUROPREP_AGENTIC_PROJECT", raising=False)
    assert settings.validated_agentic_project() == default

    monkeypatch.setenv("NEUROPREP_AGENTIC_PROJECT", str(environment))
    assert settings.validated_agentic_project() == environment
    assert settings.validated_agentic_project(str(explicit_cli)) == explicit_cli


def test_agentic_project_is_resolved_and_must_be_exact_repository_root(
    nfs_tmp: Path,
) -> None:
    repository = _git_repository(nfs_tmp / "project")
    link = nfs_tmp / "project-link"
    link.symlink_to(repository, target_is_directory=True)

    assert settings.validated_agentic_project(str(link)) == repository

    nested = repository / "src"
    nested.mkdir()
    with pytest.raises(ValueError, match="exact git repository root"):
        settings.validated_agentic_project(str(nested))

    unrelated = nfs_tmp / "not-a-repository"
    unrelated.mkdir()
    with pytest.raises(ValueError, match="git repository root"):
        settings.validated_agentic_project(str(unrelated))

    file_path = nfs_tmp / "not-a-directory"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="existing directory"):
        settings.validated_agentic_project(str(file_path))


def test_compile_run_cli_records_cli_env_default_project_precedence(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    source = nfs_tmp / "source"
    image = source / "sub-P001" / "ses-V1" / "anat" / "T1w.nii.gz"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"nifti-placeholder")
    manifest = nfs_tmp / "manifest.json"
    manifest.write_text(
        (
            "{\n"
            '  "dataset": {"dataset_id": "project-precedence", '
            '"adapter_id": "generic", "required_modalities": ["structural_mri_t1w"]},\n'
            '  "records": [{"dataset_id": "project-precedence", '
            f'"source_path": "{image}", "source_subject_id": "P001", '
            '"source_session_id": "V1", "file_format": "NIfTI", '
            '"metadata": {"Modality": "MR", "SeriesDescription": "MPRAGE T1w"}}]\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    default = _git_repository(nfs_tmp / "default-cli-project")
    environment = _git_repository(nfs_tmp / "environment-cli-project")
    explicit_cli = _git_repository(nfs_tmp / "explicit-cli-project")
    monkeypatch.setattr(settings, "DEFAULT_AGENTIC_PROJECT", str(default))

    cases = (
        ("default", None, None, default),
        ("environment", str(environment), None, environment),
        ("explicit", str(environment), str(explicit_cli), explicit_cli),
    )
    for label, environment_value, explicit_value, expected in cases:
        if environment_value is None:
            monkeypatch.delenv("NEUROPREP_AGENTIC_PROJECT", raising=False)
        else:
            monkeypatch.setenv("NEUROPREP_AGENTIC_PROJECT", environment_value)
        root = nfs_tmp / f"run-{label}"
        output = root / "execution-plan.json"
        arguments = [
            "compile-run",
            str(manifest),
            "--identity-key",
            "strong-project-test-key",
            "--source-root",
            str(source),
            "--bids-root",
            str(root / "bids"),
            "--derivatives-root",
            str(root / "derivatives"),
            "--work-root",
            str(root / "work"),
            "--state-path",
            str(root / "state" / "tasks.sqlite"),
            "--output",
            str(output),
        ]
        if explicit_value is not None:
            arguments.extend(["--agentic-project", explicit_value])

        cli(arguments)

        plan = json.loads(output.read_text(encoding="utf-8"))
        assert plan["layout"]["agentic_project"] == str(expected)
