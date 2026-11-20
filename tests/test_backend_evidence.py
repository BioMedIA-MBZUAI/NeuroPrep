from __future__ import annotations

import json
import sqlite3
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from neuroprep_agentic.backend_evidence import (
    EvidenceValidationError,
    build_backend_evidence,
    create_signing_key,
    evidence_allows_execution,
    evidence_document_sha256,
    read_signing_key,
    revalidate_execution_evidence,
    validate_backend_evidence,
    write_evidence_document,
)

CATALOG = "0" * 64
COMMIT = "1" * 40
KEY = b"evidence-test-key-material-32-bytes-minimum"


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _task(root: Path, dataset: str, subject: str, image: Path) -> dict[str, object]:
    task_id = f"exec--smriprep--sub-{subject}--abcdef123456"
    work_dir = root / "work" / "tasks" / task_id
    artifact = root / "derivatives" / "smriprep" / f"sub-{subject}_desc-preproc_T1w.nii.gz"
    return {
        "task_id": task_id,
        "logical_task_ids": [f"{dataset}:anat"],
        "executor_id": "smriprep",
        "plugin_ids": ["mri.t1w"],
        "stages": ["anat_preprocess"],
        "scope": "subject",
        "subject_id": f"sub-{subject}",
        "session_id": "ses-01",
        "acquisition_ids": [f"acq-{subject}"],
        "depends_on": [],
        "command": [
            "/usr/bin/singularity",
            "run",
            "--bind",
            f"{work_dir}/tmp:/tmp",
            str(image),
            "/data",
            "/out",
            "participant",
            "--participant-label",
            subject,
            "--work-dir",
            f"/work/smriprep/{subject}",
            "--fs-no-reconall",
        ],
        "environment": {},
        "input_paths": [str(root / "bids" / f"sub-{subject}_T1w.nii.gz")],
        "output_root": str(root / "derivatives"),
        "work_dir": str(work_dir),
        "resource_class": "cpu",
        "cpus": 8,
        "memory_gb": 24,
        "gpus": 0,
        "walltime_minutes": 180,
        "required": True,
        "artifacts": [
            {
                "kind": "preprocessed_t1w",
                "patterns": [str(artifact)],
                "minimum_count": 1,
                "required": True,
                "validator": "exists_nonempty",
            }
        ],
        "fingerprint": f"fingerprint-{dataset}",
        "metadata": {
            "executor_version": "0.19.1",
            "executor_source": "https://example.test/smriprep",
            "image_source": "docker://nipreps/smriprep:0.19.1",
            "container_image": str(image),
        },
    }


def _plan(
    root: Path,
    dataset: str,
    subject: str,
    image: Path,
    smoke: bool,
    agentic_project: Path,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "dataset_id": dataset,
        "generated_at": "2026-08-15T22:10:30+00:00",
        "catalog_digest": CATALOG,
        "layout": {
            "dataset_id": dataset,
            "source_root": str(root / "source"),
            "bids_root": str(root / "bids"),
            "derivatives_root": str(root / "derivatives"),
            "work_root": str(root / "work"),
            "state_path": str(root / "state" / "tasks.sqlite"),
            "agentic_project": str(agentic_project),
        },
        "tasks": [_task(root, dataset, subject, image)],
        "required_executor_ids": ["smriprep"],
        "optional_executor_ids": [],
        "warnings": [],
        "run_policy": {},
        "policy_digest": None,
        "backend_decisions": [],
        "planned_smoke_tests": (
            [
                {
                    "test_id": "smoke--structural_preprocess--smriprep",
                    "capability": "structural_preprocess",
                    "package_id": "smriprep",
                    "executor_id": "smriprep",
                    "level": "representative_subject",
                    "description": "representative structural smoke",
                    "expected_evidence": ["artifact_contract=pass"],
                    "bounded_minutes": 720,
                }
            ]
            if smoke
            else []
        ),
    }


def _fixture(nfs_tmp: Path) -> dict[str, object]:
    image = nfs_tmp / "containers" / "smriprep.sif"
    image.parent.mkdir()
    image.write_bytes(b"immutable-smriprep-image")
    pilot_root = nfs_tmp / "openneuro-case01"
    target_root = nfs_tmp / "anmerge-full-v2-probe"
    repository = nfs_tmp / "NeuroPrep-agentic-target"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    repository = repository.resolve()
    pilot = _plan(
        pilot_root,
        "openneuro-case01",
        "pilot",
        image,
        smoke=False,
        agentic_project=repository,
    )
    target = _plan(
        target_root,
        "anmerge-full-v2",
        "target",
        image,
        smoke=True,
        agentic_project=repository,
    )
    pilot_plan = pilot_root / "execution-plan.json"
    target_plan = target_root / "execution-plan.json"
    _json(pilot_plan, pilot)
    _json(target_plan, target)

    pilot_task = pilot["tasks"][0]
    artifact = Path(pilot_task["artifacts"][0]["patterns"][0])
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"preprocessed-t1")
    log = Path(pilot_task["work_dir"]) / "attempt.log"
    log.parent.mkdir(parents=True)
    log.write_text("completed\n", encoding="utf-8")
    state = pilot_root / "state" / "tasks.sqlite"
    state.parent.mkdir(parents=True)
    connection = sqlite3.connect(state)
    connection.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE tasks(
            task_id TEXT PRIMARY KEY,status TEXT,executor_id TEXT,fingerprint TEXT,
            task_json TEXT,attempts INTEGER,return_code INTEGER,log_path TEXT,
            artifacts_json TEXT
        );
        """
    )
    connection.executemany(
        "INSERT INTO metadata(key,value) VALUES (?,?)",
        {
            "dataset_id": pilot["dataset_id"],
            "catalog_digest": CATALOG,
            "plan_path": str(pilot_plan),
        }.items(),
    )
    connection.execute(
        "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?)",
        (
            pilot_task["task_id"],
            "complete",
            "smriprep",
            pilot_task["fingerprint"],
            json.dumps(pilot_task),
            1,
            0,
            str(log),
            json.dumps([str(artifact)]),
        ),
    )
    connection.commit()
    connection.close()

    artifact_index = pilot_root / "derivatives" / "release" / "model-ready-index.jsonl"
    tensor_index = pilot_root / "derivatives" / "release" / "tensor-index.jsonl"
    artifact_index.parent.mkdir(parents=True)
    index_header = {
        "record_type": "index_metadata",
        "dataset_id": pilot["dataset_id"],
        "artifact_count": 1,
    }
    index_row = {
        "artifact_type": "file",
        "dataset_id": pilot["dataset_id"],
        "path": str(artifact),
        "size": artifact.stat().st_size,
    }
    artifact_index.write_text(
        "\n".join(json.dumps(row) for row in (index_header, index_row)) + "\n",
        encoding="utf-8",
    )
    tensor_header = {
        "record_type": "index_metadata",
        "dataset_id": pilot["dataset_id"],
        "tensor_count": 1,
    }
    tensor_index.write_text(
        "\n".join(json.dumps(row) for row in (tensor_header, index_row)) + "\n",
        encoding="utf-8",
    )
    release = pilot_root / "derivatives" / "release" / "release-summary.json"
    _json(
        release,
        {
            "ready": True,
            "dataset_id": pilot["dataset_id"],
            "catalog_digest": CATALOG,
            "generated_at": "2026-08-15T22:10:30+00:00",
            "task_counts": {"complete": 1},
            "artifact_count": 1,
            "tensor_count": 1,
            "artifact_index": str(artifact_index),
            "tensor_index": str(tensor_index),
        },
    )
    return {
        "image": image,
        "pilot": pilot,
        "pilot_plan": pilot_plan,
        "target": target,
        "target_plan": target_plan,
        "state": state,
        "release": release,
        "repository": repository,
    }


def _build(monkeypatch: pytest.MonkeyPatch, fixture: dict[str, object]):
    monkeypatch.setattr(
        "neuroprep_agentic.backend_evidence.current_git_commit", lambda _path: COMMIT
    )
    return build_backend_evidence(
        pilot_plan_path=fixture["pilot_plan"],
        target_plan_path=fixture["target_plan"],
        pilot_state_path=fixture["state"],
        pilot_release_path=fixture["release"],
        git_repository=fixture["repository"],
        signing_key=KEY,
        key_id="campaign-20260817",
    )


def test_openneuro_current_pilot_builds_deterministic_anmerge_evidence(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    fixture = _fixture(nfs_tmp)
    first = _build(monkeypatch, fixture)
    second = _build(monkeypatch, fixture)

    assert first == second
    assert first.catalog_digest == CATALOG
    assert first.target_git_commit == COMMIT
    assert first.target.agentic_project == str(fixture["repository"])
    assert first.pilot.git.verification == "unknown"
    assert first.pilot.git.commit is None
    assert first.target.dataset_id == "anmerge-full-v2"
    assert [item.test_id for item in first.tests] == [
        "smoke--structural_preprocess--smriprep"
    ]
    assert first.tests[0].artifact_count == 1
    assert evidence_document_sha256(first) == evidence_document_sha256(second)

    evidence_path = nfs_tmp / "evidence" / "anmerge.json"
    second_path = nfs_tmp / "evidence" / "anmerge-second.json"
    write_evidence_document(first, evidence_path)
    write_evidence_document(second, second_path)
    assert evidence_path.read_bytes() == second_path.read_bytes()
    loaded = validate_backend_evidence(
        evidence_path=evidence_path,
        signing_key=KEY,
        target_plan=fixture["target"],
        target_smoke_tests=fixture["target"]["planned_smoke_tests"],
        catalog_digest=CATALOG,
        target_git_commit=COMMIT,
    )

    assert loaded.statuses == {"smoke--structural_preprocess--smriprep": "passed"}
    assert loaded.binding.trust == "verified"
    assert loaded.binding.target_agentic_project == str(fixture["repository"])
    assert evidence_allows_execution(
        fixture["target"]["planned_smoke_tests"], loaded.binding
    )
    execution_plan = deepcopy(fixture["target"])
    execution_plan["backend_qualification_tests"] = execution_plan[
        "planned_smoke_tests"
    ]
    execution_plan["planned_smoke_tests"] = []
    execution_plan["backend_evidence"] = loaded.binding.model_dump(mode="json")
    assert (
        revalidate_execution_evidence(
            plan=execution_plan,
            signing_key=KEY,
            target_git_commit=COMMIT,
        ).binding
        == loaded.binding
    )


def test_builder_rejects_empty_target_smoke_intersection(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    fixture = _fixture(nfs_tmp)
    target = deepcopy(fixture["target"])
    target["planned_smoke_tests"] = []
    _json(fixture["target_plan"], target)

    with pytest.raises(EvidenceValidationError, match="intersection is empty"):
        _build(monkeypatch, fixture)


def test_builder_rejects_catalog_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    fixture = _fixture(nfs_tmp)
    target = deepcopy(fixture["target"])
    target["catalog_digest"] = "f" * 64
    _json(fixture["target_plan"], target)

    with pytest.raises(EvidenceValidationError, match="catalog digests differ"):
        _build(monkeypatch, fixture)


def test_builder_rejects_executor_command_mismatch(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    fixture = _fixture(nfs_tmp)
    target = deepcopy(fixture["target"])
    target["tasks"][0]["command"].append("--changed-science-flag")
    _json(fixture["target_plan"], target)

    with pytest.raises(EvidenceValidationError, match="command mismatch"):
        _build(monkeypatch, fixture)


def test_validator_rejects_executor_command_mismatch(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    fixture = _fixture(nfs_tmp)
    document = _build(monkeypatch, fixture)
    evidence_path = nfs_tmp / "evidence.json"
    write_evidence_document(document, evidence_path)
    target = deepcopy(fixture["target"])
    target["tasks"][0]["command"].append("--changed-after-signing")

    with pytest.raises(EvidenceValidationError, match="command mismatch"):
        validate_backend_evidence(
            evidence_path=evidence_path,
            signing_key=KEY,
            target_plan=target,
            target_smoke_tests=target["planned_smoke_tests"],
            catalog_digest=CATALOG,
            target_git_commit=COMMIT,
        )


def test_validator_rejects_image_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    fixture = _fixture(nfs_tmp)
    document = _build(monkeypatch, fixture)
    evidence_path = nfs_tmp / "evidence.json"
    write_evidence_document(document, evidence_path)
    fixture["image"].write_bytes(b"replaced-image-content")

    with pytest.raises(EvidenceValidationError, match="image hash mismatch"):
        validate_backend_evidence(
            evidence_path=evidence_path,
            signing_key=KEY,
            target_plan=fixture["target"],
            target_smoke_tests=fixture["target"]["planned_smoke_tests"],
            catalog_digest=CATALOG,
            target_git_commit=COMMIT,
        )


def test_legacy_flat_evidence_is_planning_only(nfs_tmp: Path) -> None:
    fixture = _fixture(nfs_tmp)
    legacy = nfs_tmp / "legacy.json"
    _json(legacy, {"smoke--structural_preprocess--smriprep": "passed"})

    loaded = validate_backend_evidence(
        evidence_path=legacy,
        signing_key=None,
        target_plan=fixture["target"],
        target_smoke_tests=fixture["target"]["planned_smoke_tests"],
        catalog_digest=CATALOG,
        target_git_commit=COMMIT,
    )

    assert loaded.binding.trust == "legacy_untrusted"
    assert not evidence_allows_execution([], loaded.binding)

    plan = deepcopy(fixture["target"])
    plan["backend_qualification_tests"] = plan["planned_smoke_tests"]
    plan["planned_smoke_tests"] = []
    plan["backend_evidence"] = loaded.binding.model_dump(mode="json")
    with pytest.raises(EvidenceValidationError, match="verified signed"):
        revalidate_execution_evidence(
            plan=plan,
            signing_key=KEY,
            target_git_commit=COMMIT,
        )


def test_empty_evidence_file_fails_closed(nfs_tmp: Path) -> None:
    fixture = _fixture(nfs_tmp)
    empty = nfs_tmp / "empty.json"
    _json(empty, {})

    with pytest.raises(EvidenceValidationError, match="cannot be empty"):
        validate_backend_evidence(
            evidence_path=empty,
            signing_key=None,
            target_plan=fixture["target"],
            target_smoke_tests=fixture["target"]["planned_smoke_tests"],
            catalog_digest=CATALOG,
            target_git_commit=COMMIT,
        )


def test_no_target_smoke_tests_need_no_evidence() -> None:
    assert evidence_allows_execution([], None)
    assert (
        revalidate_execution_evidence(
            plan={
                "catalog_digest": CATALOG,
                "backend_qualification_tests": [],
                "planned_smoke_tests": [],
                "backend_evidence": None,
            },
            signing_key=None,
            target_git_commit="",
        )
        is None
    )


def test_validator_rejects_target_git_commit_mismatch(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    fixture = _fixture(nfs_tmp)
    document = _build(monkeypatch, fixture)
    evidence_path = nfs_tmp / "evidence.json"
    write_evidence_document(document, evidence_path)

    with pytest.raises(EvidenceValidationError, match="target git commit mismatch"):
        validate_backend_evidence(
            evidence_path=evidence_path,
            signing_key=KEY,
            target_plan=fixture["target"],
            target_smoke_tests=fixture["target"]["planned_smoke_tests"],
            catalog_digest=CATALOG,
            target_git_commit="f" * 40,
        )


def test_builder_rejects_target_agentic_project_repository_mismatch(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    fixture = _fixture(nfs_tmp)
    different_repository = nfs_tmp / "different-builder-project"
    different_repository.mkdir()
    subprocess.run(["git", "init", "-q", str(different_repository)], check=True)
    target = deepcopy(fixture["target"])
    target["layout"]["agentic_project"] = str(different_repository.resolve())
    _json(fixture["target_plan"], target)

    with pytest.raises(EvidenceValidationError, match="differs from the evidence git repository"):
        _build(monkeypatch, fixture)


def test_validator_rejects_target_agentic_project_path_mismatch(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    fixture = _fixture(nfs_tmp)
    document = _build(monkeypatch, fixture)
    evidence_path = nfs_tmp / "evidence.json"
    write_evidence_document(document, evidence_path)
    different_repository = nfs_tmp / "different-validator-project"
    different_repository.mkdir()
    subprocess.run(["git", "init", "-q", str(different_repository)], check=True)
    target = deepcopy(fixture["target"])
    target["layout"]["agentic_project"] = str(different_repository.resolve())

    with pytest.raises(EvidenceValidationError, match="target agentic project mismatch"):
        validate_backend_evidence(
            evidence_path=evidence_path,
            signing_key=KEY,
            target_plan=target,
            target_smoke_tests=target["planned_smoke_tests"],
            catalog_digest=CATALOG,
            target_git_commit=COMMIT,
        )


def test_pilot_git_provenance_tamper_breaks_signature(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    fixture = _fixture(nfs_tmp)
    document = _build(monkeypatch, fixture)
    tampered = document.model_dump(mode="json")
    tampered["pilot"]["git"] = {
        "commit": "a" * 40,
        "verification": "embedded",
        "sources": ["pilot_plan"],
        "detail": "forged",
    }
    evidence_path = nfs_tmp / "tampered.json"
    _json(evidence_path, tampered)

    with pytest.raises(EvidenceValidationError, match="payload digest mismatch"):
        validate_backend_evidence(
            evidence_path=evidence_path,
            signing_key=KEY,
            target_plan=fixture["target"],
            target_smoke_tests=fixture["target"]["planned_smoke_tests"],
            catalog_digest=CATALOG,
            target_git_commit=COMMIT,
        )


def test_builder_rejects_disagreeing_embedded_pilot_commits(
    monkeypatch: pytest.MonkeyPatch, nfs_tmp: Path
) -> None:
    fixture = _fixture(nfs_tmp)
    pilot = deepcopy(fixture["pilot"])
    pilot["git_commit"] = "a" * 40
    _json(fixture["pilot_plan"], pilot)
    release = json.loads(fixture["release"].read_text(encoding="utf-8"))
    release["git_commit"] = "b" * 40
    _json(fixture["release"], release)

    with pytest.raises(EvidenceValidationError, match="git commits disagree"):
        _build(monkeypatch, fixture)


def test_signing_key_is_nfs_scoped_mode_0600_and_non_overwriting(nfs_tmp: Path) -> None:
    key_dir = nfs_tmp / "keys"
    key_dir.mkdir()
    key_path = key_dir / "campaign.key"

    assert create_signing_key(key_path) == key_path
    assert key_path.stat().st_mode & 0o777 == 0o600
    assert len(read_signing_key(key_path)) == 64
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_signing_key(key_path)

    key_path.chmod(0o640)
    with pytest.raises(EvidenceValidationError, match="mode 0600"):
        read_signing_key(key_path)


def test_execution_gate_allows_a_losing_alternative_but_not_a_failed_selection() -> None:
    from neuroprep_agentic.backend_evidence import BackendEvidenceBinding, evidence_allows_execution

    tests = [
        {"test_id": "smoke--structural_segment--deepprep", "executor_id": "deepprep"},
        {"test_id": "smoke--structural_segment--fastsurfer", "executor_id": "fastsurfer"},
    ]
    binding = BackendEvidenceBinding(
        trust="verified",
        source_path="/x/evidence.json",
        document_sha256="0" * 64,
        required_test_ids=[t["test_id"] for t in tests],
        matched_test_ids=["smoke--structural_segment--deepprep"],
        failed_test_ids=["smoke--structural_segment--fastsurfer"],
    )
    assert evidence_allows_execution(tests, binding, {"deepprep", "smriprep"})
    assert not evidence_allows_execution(tests, binding, {"fastsurfer"})
    assert not evidence_allows_execution(tests, binding)  # legacy rule: all must pass
    undecided = binding.model_copy(update={"failed_test_ids": []})
    assert not evidence_allows_execution(tests, undecided, {"deepprep"})


def test_signature_folds_per_subject_input_file_names() -> None:
    from neuroprep_agentic.backend_evidence import executor_signature

    def task(subject: str, acq: str) -> dict:
        t1 = f"/data/bids/{subject}/anat/{subject}_acq-{acq[:8]}_T1w.nii.gz"
        return {
            "executor_id": "synthseg",
            "subject_id": subject,
            "acquisition_ids": [f"acq-{acq}"],
            "command": ["mri_synthseg", "--i", "/source-data" + t1.removeprefix("/data/bids")],
            "metadata": {"primary_anatomical_bids_input": t1, "bids_inputs": [t1]},
        }

    plan = {
        "layout": {"bids_root": "/data/bids"},
        "tasks": [task("sub-aa", "1111111122"), task("sub-bb", "3333333344")],
    }
    executor_signature(plan, "synthseg")  # raises if the two subjects differ


def test_signature_ignores_stages_and_resources_an_alternative_took_over() -> None:
    from neuroprep_agentic.backend_evidence import executor_signature

    def plan(stages: list[str], resource_class: str, gpus: int, memory_gb: int) -> dict:
        task = {
            "executor_id": "neuroprep-spatial",
            "stages": stages,
            "command": ["run", "ct"],
            "resource_class": resource_class,
            "cpus": 8,
            "gpus": gpus,
            "memory_gb": memory_gb,
        }
        return {"layout": {}, "tasks": [task]}

    # CIAI 419890: in the pilot the spatial task also served ct_segment, so its resources
    # were merged with TotalSegmentator's GPU needs; the target routed ct_segment away.
    pilot = plan(["ct_preprocess", "ct_segment"], "gpu", 1, 32)
    target = plan(["ct_preprocess"], "cpu", 0, 24)
    assert executor_signature(pilot, "neuroprep-spatial") == (
        executor_signature(target, "neuroprep-spatial")
    )
