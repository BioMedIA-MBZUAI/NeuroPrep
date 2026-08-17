from __future__ import annotations

import json
from pathlib import Path

import pytest

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
        resolve_prompt_paths(f"My data is at {source}", output_root="/home/salma.hassan/results")
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
            submit=True,
        )
    assert not (unqualified_output / "state" / "tasks.sqlite").exists()

    initialize_output = nfs_tmp / "unqualified-initialize"
    with pytest.raises(ValueError, match="signed, target-bound backend evidence"):
        prepare_prompt_run(
            f"My data is at {source}; preprocess it and save to {initialize_output}",
            hardware_profile=hardware,
            initialize=True,
        )
    assert not (initialize_output / "state" / "tasks.sqlite").exists()

    planning_output = nfs_tmp / "planning-only"
    planning = prepare_prompt_run(
        f"My data is at {source}; preprocess it and save to {planning_output}",
        hardware_profile=hardware,
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
                key_id="test-key",
                pilot_plan_sha256="5" * 64,
                target_protocol_sha256="6" * 64,
                required_test_ids=observed,
                matched_test_ids=observed,
            ),
        )

    monkeypatch.setattr(prompt_run, "read_signing_key", lambda _path: b"x" * 32)
    monkeypatch.setattr(prompt_run, "current_git_commit", lambda _path: "4" * 40)
    monkeypatch.setattr(prompt_run, "validate_backend_evidence", verified_evidence)
    result = prepare_prompt_run(
        f"My data is at {source}; preprocess it and save to {output}",
        hardware_profile=hardware,
        backend_evidence=str(nfs_tmp / "signed-evidence.json"),
        backend_evidence_key=str(nfs_tmp / "signing.key"),
        initialize=True,
    )

    assert result["ready"] is True
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
    assert backend_plan["smoke_tests"] == []
    execution_plan = json.loads(Path(result["execution_plan"]).read_text(encoding="utf-8"))
    assert execution_plan["backend_evidence"]["document_sha256"] == "2" * 64
    assert {item["capability"] for item in backend_plan["decisions"]} == {
        "structural_preprocess",
        "structural_segment",
    }
