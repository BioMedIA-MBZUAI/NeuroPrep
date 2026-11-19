from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from neuroprep_agentic import main as main_module
from neuroprep_agentic import prompt_run
from neuroprep_agentic.backend_evidence import (
    BackendEvidenceBinding,
    LoadedBackendEvidence,
)
from neuroprep_agentic.execution_models import PreflightReport
from neuroprep_agentic.policy import HardwareProfile, RunPolicy
from neuroprep_agentic.prompt_run import (
    discover_source_manifest,
    prepare_prompt_run,
    resolve_prompt_paths,
)


def _git_repository(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path.resolve()


def test_prompt_paths_bind_natural_language_x_and_y(nfs_tmp: Path) -> None:
    source = nfs_tmp / "incoming"
    output = nfs_tmp / "prepared"
    source.mkdir()
    request = (
        f"My data is at {source}; preprocess all modalities and longitudinal visits "
        f"and save the results to {output}."
    )

    binding = resolve_prompt_paths(request)

    assert binding.source_root == str(source.resolve())
    assert binding.output_root == str(output.resolve())


def test_prompt_paths_reject_unapproved_or_overlapping_roots(nfs_tmp: Path) -> None:
    source = nfs_tmp / "incoming"
    source.mkdir()

    with pytest.raises(ValueError, match="Path must be under one of"):
        resolve_prompt_paths(f"My data is at {source}", output_root="/tmp/neuroprep-results")
    with pytest.raises(ValueError, match="must not overlap"):
        resolve_prompt_paths(f"My data is at {source} and save to {source / 'results'}")


def test_explicit_roots_cannot_silently_override_prompt(nfs_tmp: Path) -> None:
    source = nfs_tmp / "incoming"
    different = nfs_tmp / "different"
    output = nfs_tmp / "prepared"
    source.mkdir()
    different.mkdir()

    with pytest.raises(ValueError, match="conflicts with the prompt"):
        resolve_prompt_paths(
            f"Data is at {source} and save to {output}",
            source_root=str(different),
        )


def test_inventory_discovers_nifti_sidecars_and_longitudinal_sensor_data(
    nfs_tmp: Path,
) -> None:
    source = nfs_tmp / "incoming"
    t1w = source / "sub-P001" / "ses-V1" / "anat" / "sub-P001_ses-V1_T1w.nii.gz"
    eeg = source / "sub-P001" / "ses-V2" / "eeg" / "sub-P001_ses-V2_task-rest_eeg.edf"
    t1w.parent.mkdir(parents=True)
    eeg.parent.mkdir(parents=True)
    t1w.write_bytes(b"nifti-placeholder")
    t1w.with_name("sub-P001_ses-V1_T1w.json").write_text(
        json.dumps({"Modality": "MR", "SeriesDescription": "MPRAGE"}),
        encoding="utf-8",
    )
    eeg.write_bytes(b"edf-placeholder")

    manifest, report = discover_source_manifest(str(source), dataset_id="cohort-a")

    assert report["records"] == 2
    assert manifest.dataset.dataset_id == "cohort-a"
    assert manifest.dataset.adapter_id == "generic"
    assert {
        (record.source_subject_id, record.source_session_id) for record in manifest.records
    } == {
        ("P001", "V1"),
        ("P001", "V2"),
    }
    assert {record.file_format for record in manifest.records} == {
        "NIFTI",
        "ELECTROPHYSIOLOGY",
    }
    assert all(not record.source_path.endswith(".json") for record in manifest.records)


def test_inventory_groups_dicom_files_as_one_acquisition(nfs_tmp: Path) -> None:
    source = nfs_tmp / "incoming"
    series = source / "sub-P002" / "ses-V1" / "anat" / "T1_MPRAGE"
    series.mkdir(parents=True)
    payload = bytearray(132)
    payload[128:132] = b"DICM"
    (series / "IM0001").write_bytes(payload)
    (series / "IM0002.dcm").write_bytes(b"second")

    manifest, report = discover_source_manifest(str(source))

    assert report["dicom_directories"] == 1
    assert len(manifest.records) == 1
    assert manifest.records[0].source_path == str(series.resolve())
    assert manifest.records[0].file_format == "DICOM"
    assert manifest.records[0].source_subject_id == "P002"
    assert manifest.records[0].source_session_id == "V1"


def test_submit_uses_selected_agentic_project_launcher_and_environment(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    project = nfs_tmp / "selected-project"
    plan = nfs_tmp / "execution-plan.json"
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "launch_complete=true\nreview_job=1\nworker_jobs=2\n"
                "finalize_job=3\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(prompt_run.subprocess, "run", fake_run)

    result = prompt_run._submit_workers(plan, 3, str(project))

    assert observed["command"] == [
        str(project / "scripts/launch_neuroprep_run.sh"),
        str(plan),
        "3",
    ]
    assert observed["environment"]["NEUROPREP_AGENTIC_PROJECT"] == str(project)
    assert result["launch_complete"] == "true"


def test_submission_key_cli_value_overrides_environment_without_command_disclosure(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    inherited = str(nfs_tmp / "inherited.key")
    cli_only = str(nfs_tmp / "cli-only.key")
    observed: dict[str, object] = {}
    monkeypatch.setenv("NEUROPREP_BACKEND_EVIDENCE_KEY", inherited)

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="launch_complete=true\nreview_job=1\nworker_jobs=2\nfinalize_job=3\n",
            stderr="",
        )

    monkeypatch.setattr(prompt_run.subprocess, "run", fake_run)

    result = prompt_run._submit_workers(
        nfs_tmp / "plan.json",
        2,
        str(nfs_tmp / "project"),
        backend_evidence_key=cli_only,
    )

    assert observed["environment"]["NEUROPREP_BACKEND_EVIDENCE_KEY"] == cli_only
    assert cli_only not in observed["command"]
    assert cli_only not in json.dumps(result, sort_keys=True)


@pytest.mark.parametrize("inherited", ["environment", None])
def test_submission_key_environment_or_absent_default_is_preserved(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path, inherited: str | None
) -> None:
    expected = str(nfs_tmp / "environment.key") if inherited else None
    if expected is None:
        monkeypatch.delenv("NEUROPREP_BACKEND_EVIDENCE_KEY", raising=False)
    else:
        monkeypatch.setenv("NEUROPREP_BACKEND_EVIDENCE_KEY", expected)
    observed: dict[str, str] = {}

    def fake_run(command, **kwargs):
        observed.update(kwargs["env"])
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="launch_complete=true\n",
            stderr="",
        )

    monkeypatch.setattr(prompt_run.subprocess, "run", fake_run)

    prompt_run._submit_workers(
        nfs_tmp / "plan.json", 1, str(nfs_tmp / "project")
    )

    assert observed.get("NEUROPREP_BACKEND_EVIDENCE_KEY") == expected


@pytest.mark.parametrize(
    ("environment_value", "cli_value", "expected"),
    [
        (None, "cli-only.key", "cli-only.key"),
        ("environment.key", "cli-wins.key", "cli-wins.key"),
        ("environment.key", None, "environment.key"),
        (None, None, None),
    ],
)
def test_prompt_run_cli_backend_key_precedence_reaches_submit(
    monkeypatch: pytest.MonkeyPatch,
    nfs_tmp: Path,
    environment_value: str | None,
    cli_value: str | None,
    expected: str | None,
) -> None:
    environment_path = str(nfs_tmp / environment_value) if environment_value else None
    cli_path = str(nfs_tmp / cli_value) if cli_value else None
    expected_path = str(nfs_tmp / expected) if expected else None
    if environment_path is None:
        monkeypatch.delenv("NEUROPREP_BACKEND_EVIDENCE_KEY", raising=False)
    else:
        monkeypatch.setenv("NEUROPREP_BACKEND_EVIDENCE_KEY", environment_path)
    observed: dict[str, object] = {}

    def fake_prepare(request, **kwargs):
        observed.update(kwargs)
        return {"ready": True, "submitted": kwargs["submit"]}

    monkeypatch.setattr(main_module, "prepare_prompt_run", fake_prepare)
    monkeypatch.setattr(main_module, "load_hardware_profile", lambda _path: None)
    arguments = ["prompt-run", "--request", "test request", "--submit"]
    if cli_path is not None:
        arguments.extend(["--backend-evidence-key", cli_path])

    main_module.cli(arguments)

    assert observed["backend_evidence_key"] == expected_path
    assert observed["submit"] is True


def test_prepare_prompt_run_compiles_a_complete_handoff(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    source = nfs_tmp / "incoming"
    output = nfs_tmp / "prepared"
    t1w = source / "sub-P001" / "ses-V1" / "anat" / "sub-P001_ses-V1_T1w.nii.gz"
    t1w.parent.mkdir(parents=True)
    t1w.write_bytes(b"nifti-placeholder")
    t1w.with_name("sub-P001_ses-V1_T1w.json").write_text(
        json.dumps({"Modality": "MR", "SeriesDescription": "MPRAGE"}),
        encoding="utf-8",
    )
    agentic_project = _git_repository(nfs_tmp / "NeuroPrep-agentic-review")
    monkeypatch.setattr(
        prompt_run,
        "compile_prompt_policy",
        lambda request, profile: RunPolicy(source="prompt", source_prompt=request),
    )
    monkeypatch.setattr(prompt_run, "prepare_runtime_assets", lambda plan: {"ready": True})
    monkeypatch.setattr(prompt_run, "image_inventory", lambda plan, catalog: {"ready": True})
    monkeypatch.setattr(
        prompt_run,
        "preflight_execution_plan",
        lambda plan, catalog: PreflightReport(ready=True, checks=[]),
    )
    hardware = HardwareProfile(
        hostname="gpu-test",
        cpus=8,
        memory_gb=64,
        gpus=1,
        gpu_memory_gb=40,
        gpu_models=["A100"],
        cuda_version="12.8",
        container_runtime="singularity",
    )

    unqualified_output = nfs_tmp / "unqualified-submit"
    with pytest.raises(ValueError, match="signed, target-bound backend evidence"):
        prepare_prompt_run(
            f"My data is at {source}; preprocess it and save to {unqualified_output}",
            hardware_profile=hardware,
            agentic_project=str(agentic_project),
            submit=True,
        )
    assert not (unqualified_output / "state" / "tasks.sqlite").exists()

    initialize_output = nfs_tmp / "unqualified-initialize"
    with pytest.raises(ValueError, match="signed, target-bound backend evidence"):
        prepare_prompt_run(
            f"My data is at {source}; preprocess it and save to {initialize_output}",
            hardware_profile=hardware,
            agentic_project=str(agentic_project),
            initialize=True,
        )
    assert not (initialize_output / "state" / "tasks.sqlite").exists()

    planning_output = nfs_tmp / "planning-only"
    planning = prepare_prompt_run(
        f"My data is at {source}; preprocess it and save to {planning_output}",
        hardware_profile=hardware,
        agentic_project=str(agentic_project),
    )
    assert planning["state_summary"] is None
    planning_plan = json.loads(
        Path(planning["execution_plan"]).read_text(encoding="utf-8")
    )
    required_test_ids = sorted(
        item["test_id"] for item in planning_plan["planned_smoke_tests"]
    )
    assert required_test_ids

    def verified_evidence(**kwargs):
        observed = sorted(item["test_id"] for item in kwargs["target_smoke_tests"])
        assert observed == required_test_ids
        return LoadedBackendEvidence(
            statuses={test_id: "passed" for test_id in observed},
            binding=BackendEvidenceBinding(
                trust="verified",
                source_path=str(nfs_tmp / "signed-evidence.json"),
                document_sha256="2" * 64,
                catalog_digest="3" * 64,
                target_git_commit="4" * 40,
                target_agentic_project=str(agentic_project),
                key_id="test-key",
                pilot_plan_sha256="5" * 64,
                target_protocol_sha256="6" * 64,
                required_test_ids=observed,
                matched_test_ids=observed,
            ),
        )

    monkeypatch.delenv("NEUROPREP_BACKEND_EVIDENCE_KEY", raising=False)
    monkeypatch.setattr(prompt_run, "read_signing_key", lambda _path: b"x" * 32)
    monkeypatch.setattr(prompt_run, "current_git_commit", lambda _path: "4" * 40)
    monkeypatch.setattr(prompt_run, "validate_backend_evidence", verified_evidence)
    revalidated: list[set[str]] = []
    monkeypatch.setattr(
        prompt_run,
        "revalidate_execution_evidence",
        lambda *, plan, signing_key, target_git_commit: revalidated.append(
            {task["executor_id"] for task in plan["tasks"]}
        ),
    )
    submitted: dict[str, object] = {}

    def fake_submit(plan_path, gpu_workers, agentic_project, backend_evidence_key=None):
        submitted.update(
            {
                "plan_path": plan_path,
                "gpu_workers": gpu_workers,
                "agentic_project": agentic_project,
                "backend_evidence_key": backend_evidence_key,
            }
        )
        return {
            "launch_complete": "true",
            "review_job": "1",
            "worker_jobs": "2",
            "finalize_job": "3",
        }

    monkeypatch.setattr(prompt_run, "_submit_workers", fake_submit)
    cli_only_key = str(nfs_tmp / "signing.key")
    result = prepare_prompt_run(
        f"My data is at {source}; preprocess it and save to {output}",
        hardware_profile=hardware,
        agentic_project=str(agentic_project),
        backend_evidence=str(nfs_tmp / "signed-evidence.json"),
        backend_evidence_key=cli_only_key,
        submit=True,
    )

    assert result["ready"] is True
    assert len(revalidated) == 1 and "deepprep" in revalidated[0]
    assert result["submitted"] is True
    assert submitted["backend_evidence_key"] == cli_only_key
    assert cli_only_key not in json.dumps(result["launch"], sort_keys=True)
    assert result["record_count"] == 1
    assert result["task_count"] > 0
    assert Path(result["execution_plan"]).is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "logical-plan.json").is_file()
    assert (output / "state" / "identity.key").stat().st_mode & 0o777 == 0o600
    assert (output / "state" / "tasks.sqlite").is_file()
    assert result["state_summary"]["total"] == result["task_count"]
    assert result["state_summary"]["counts"]["pending"] > 0
    backend_plan = json.loads((output / "backend-plan.json").read_text(encoding="utf-8"))
    assert backend_plan["backend_evidence"]["trust"] == "verified"
    assert backend_plan["backend_evidence"]["target_agentic_project"] == str(
        agentic_project
    )
    assert backend_plan["smoke_tests"] == []
    execution_plan = json.loads(Path(result["execution_plan"]).read_text(encoding="utf-8"))
    assert execution_plan["layout"]["agentic_project"] == str(agentic_project)
    assert execution_plan["backend_evidence"]["document_sha256"] == "2" * 64
    assert {item["capability"] for item in backend_plan["decisions"]} == {
        "structural_preprocess",
        "structural_segment",
    }



def test_fetch_images_pulls_missing_images_before_preflight(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    monkeypatch.setenv("NEUROPREP_CONTAINER_ROOT", str(nfs_tmp / "no-images-yet"))
    source = nfs_tmp / "incoming"
    t1w = source / "sub-P001" / "anat" / "sub-P001_T1w.nii.gz"
    t1w.parent.mkdir(parents=True)
    t1w.write_bytes(b"nifti-placeholder")
    agentic_project = _git_repository(nfs_tmp / "NeuroPrep-agentic-fetch")
    monkeypatch.setattr(
        prompt_run,
        "compile_prompt_policy",
        lambda request, profile: RunPolicy(source="prompt", source_prompt=request),
    )
    monkeypatch.setattr(prompt_run, "prepare_runtime_assets", lambda plan: {"ready": True})
    monkeypatch.setattr(
        prompt_run, "preflight_execution_plan",
        lambda plan, catalog: PreflightReport(ready=True, checks=[]),
    )
    pulls = []
    monkeypatch.setattr(prompt_run, "prefetch_images", lambda plan, catalog: pulls.append(1))
    hardware = HardwareProfile(
        hostname="gpu-test", cpus=8, memory_gb=64, gpus=1, gpu_memory_gb=40,
        gpu_models=["A100"], cuda_version="12.8", container_runtime="singularity",
    )

    def run(name: str, fetch: bool) -> None:
        prepare_prompt_run(
            f"My data is at {source}; preprocess it and save to {nfs_tmp / name}",
            hardware_profile=hardware, agentic_project=str(agentic_project), fetch_images=fetch,
        )

    run("no-fetch", False)
    assert pulls == []  # images are missing, but nobody asked to fetch them
    run("fetch", True)
    assert pulls == [1]
