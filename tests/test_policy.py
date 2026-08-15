from pydantic import ValidationError

from neuroprep_agentic.policy import (
    BackendCandidate,
    HardwareProfile,
    ProcessingObjective,
    RunPolicy,
    detect_hardware_profile,
    plan_backends,
)


def _gpu() -> HardwareProfile:
    return HardwareProfile(
        hostname="gpu-test",
        cpus=16,
        memory_gb=96,
        gpus=1,
        gpu_memory_gb=24,
        gpu_models=["test-gpu"],
        cuda_version="12.4",
        container_runtime="singularity",
    )


def test_prompt_policy_schema_cannot_contain_commands_or_unknown_fields() -> None:
    try:
        RunPolicy.model_validate(
            {
                "source": "prompt",
                "source_prompt": "Use a high-quality workflow.",
                "objective": "high_quality",
                "shell_command": "rm -rf /",
            }
        )
    except ValidationError as exc:
        assert "shell_command" in str(exc)
    else:
        raise AssertionError("unknown prompt fields must fail closed")


def test_package_preferences_are_allowlisted() -> None:
    policy = RunPolicy(
        source="prompt",
        source_prompt="Prefer QSIPrep for diffusion.",
        preferred_packages=["QSIPrep", "qsiprep"],
    )
    assert policy.preferred_packages == ["qsiprep"]
    try:
        RunPolicy(source_prompt="test", preferred_packages=["mystery-package"])
    except ValidationError as exc:
        assert "unknown package" in str(exc)
    else:
        raise AssertionError("unknown package must fail closed")


def test_backend_selection_is_hardware_aware_and_plans_bounded_smokes() -> None:
    policy = RunPolicy(
        source="prompt",
        source_prompt="Process every modality; prefer QSIPrep.",
        preferred_packages=["qsiprep"],
    )
    plan = plan_backends(policy, _gpu())
    assert plan.ready
    dwi = next(item for item in plan.decisions if item.capability == "dwi_preprocess")
    assert dwi.selected_package == "qsiprep"
    assert dwi.selected_executor_id == "qsiprep"
    assert dwi.requires_smoke_test
    assert dwi.rejected_candidates["mrtrix3"] == "adapter_not_integrated_or_qualified"
    smoke = next(item for item in plan.smoke_tests if item.capability == "dwi_preprocess")
    assert smoke.bounded_minutes <= 720
    assert "artifact_contract=pass" in smoke.expected_evidence


def test_backend_selection_can_be_scoped_to_required_executors() -> None:
    plan = plan_backends(
        RunPolicy(),
        _gpu(),
        required_executor_ids={"smriprep", "deepprep", "neuroprep-bids-stage"},
    )

    assert {item.capability for item in plan.decisions} == {
        "structural_preprocess",
        "structural_segment",
    }
    assert {item.capability for item in plan.smoke_tests} == {
        "structural_preprocess",
        "structural_segment",
    }


def test_failed_smoke_evidence_rejects_backend_fail_closed() -> None:
    policy = RunPolicy()
    plan = plan_backends(policy, _gpu(), {"smoke--dwi_preprocess--qsiprep": "failed"})
    dwi = next(item for item in plan.decisions if item.capability == "dwi_preprocess")
    assert dwi.selected_executor_id is None
    assert dwi.rejected_candidates["qsiprep"] == "smoke_test_failed"
    assert not plan.ready


def test_multiple_qualified_backends_are_ranked_by_requirement_and_evidence(
    monkeypatch,
) -> None:
    def backend(package: str, objective: ProcessingObjective) -> BackendCandidate:
        return BackendCandidate(
            capability="dwi_preprocess",
            package_id=package,
            executor_id=f"executor-{package}",
            integration_state="integrated",
            objective_tags={objective},
            min_cpus=4,
            min_memory_gb=16,
            smoke_description="bounded representative smoke",
            expected_evidence=["artifact_contract=pass", "qc_contract=pass"],
        )

    monkeypatch.setattr(
        "neuroprep_agentic.policy.builtin_backend_portfolio",
        lambda: (
            backend("qsiprep", ProcessingObjective.HIGH_QUALITY),
            backend("mrtrix3", ProcessingObjective.FAST),
        ),
    )
    evidence = {
        "smoke--dwi_preprocess--qsiprep": "passed",
        "smoke--dwi_preprocess--mrtrix3": "passed",
    }

    plan = plan_backends(
        RunPolicy(objective=ProcessingObjective.FAST),
        _gpu(),
        evidence,
    )

    assert plan.ready
    assert plan.decisions[0].selected_package == "mrtrix3"
    assert "matches fast objective" in plan.decisions[0].selection_reasons
    assert not plan.decisions[0].requires_smoke_test
    assert plan.smoke_tests == []


def test_every_eligible_backend_gets_a_bounded_challenger_smoke(monkeypatch) -> None:
    def backend(package: str) -> BackendCandidate:
        return BackendCandidate(
            capability="structural_segment",
            package_id=package,
            executor_id=f"executor-{package}",
            integration_state="integrated",
            objective_tags={ProcessingObjective.STANDARD},
            min_cpus=4,
            min_memory_gb=16,
            smoke_description="bounded representative smoke",
            expected_evidence=["artifact_contract=pass", "qc_contract=pass"],
        )

    monkeypatch.setattr(
        "neuroprep_agentic.policy.builtin_backend_portfolio",
        lambda: (backend("deepprep"), backend("fastsurfer")),
    )

    plan = plan_backends(RunPolicy(), _gpu())

    assert plan.decisions[0].selected_package == "deepprep"
    assert plan.decisions[0].requires_smoke_test
    assert {item.package_id for item in plan.smoke_tests} == {
        "deepprep",
        "fastsurfer",
    }
    assert all(item.bounded_minutes <= 720 for item in plan.smoke_tests)


def test_gpu_backend_is_rejected_on_cpu_only_hardware() -> None:
    cpu = HardwareProfile(cpus=16, memory_gb=96, container_runtime="singularity")
    decision = next(
        item
        for item in plan_backends(RunPolicy(), cpu).decisions
        if item.capability == "structural_segment"
    )
    assert decision.selected_executor_id is None
    assert decision.rejected_candidates["deepprep"] == "insufficient_gpus"


def test_hardware_detection_honors_slurm_allocation(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "65536")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "1")
    monkeypatch.setattr("neuroprep_agentic.policy.shutil.which", lambda _name: None)
    profile = detect_hardware_profile()
    assert profile.cpus == 8
    assert profile.memory_gb == 64
    assert profile.gpus == 0
