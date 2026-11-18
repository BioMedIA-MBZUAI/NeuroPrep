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
    assert dwi.rejected_candidates["fsl"] == "adapter_not_integrated_or_qualified"
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
    fallback = plan_backends(policy, _gpu(), {"smoke--dwi_preprocess--qsiprep": "failed"})
    dwi = next(item for item in fallback.decisions if item.capability == "dwi_preprocess")
    assert dwi.selected_executor_id == "mrtrix3"  # the qualified alternative takes over
    plan = plan_backends(
        policy,
        _gpu(),
        {"smoke--dwi_preprocess--qsiprep": "failed", "smoke--dwi_preprocess--mrtrix3": "failed"},
    )
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


SEGMENT = "structural_segment"
DEEPPREP, FASTSURFER = f"smoke--{SEGMENT}--deepprep", f"smoke--{SEGMENT}--fastsurfer"
T1_ONLY = {"smriprep", "deepprep"}


def _segmenter(plan) -> str | None:
    return next(d for d in plan.decisions if d.capability == SEGMENT).selected_package


def test_alternatives_are_both_qualified_before_evidence() -> None:
    plan = plan_backends(RunPolicy(), _gpu(), required_executor_ids=T1_ONLY)
    assert {t.test_id for t in plan.smoke_tests} >= {DEEPPREP, FASTSURFER}


def test_fastest_passing_alternative_wins() -> None:
    evidence = {DEEPPREP: "passed", FASTSURFER: "passed"}
    fast_deepprep = plan_backends(
        RunPolicy(), _gpu(), evidence, required_executor_ids=T1_ONLY,
        runtimes={DEEPPREP: 1440.0, FASTSURFER: 2700.0},
    )
    assert _segmenter(fast_deepprep) == "deepprep"
    fast_fastsurfer = plan_backends(
        RunPolicy(), _gpu(), evidence, required_executor_ids=T1_ONLY,
        runtimes={DEEPPREP: 1440.0, FASTSURFER: 600.0},
    )
    decision = next(d for d in fast_fastsurfer.decisions if d.capability == SEGMENT)
    assert decision.selected_package == "fastsurfer"
    assert "fastest among equally qualified alternatives" in decision.selection_reasons
    assert not [t for t in fast_fastsurfer.smoke_tests if t.capability == SEGMENT]


def test_preference_beats_speed_and_failure_rejects() -> None:
    evidence = {DEEPPREP: "passed", FASTSURFER: "passed"}
    preferred = plan_backends(
        RunPolicy(source_prompt="use DeepPrep", preferred_packages=["deepprep"]),
        _gpu(), evidence, required_executor_ids=T1_ONLY,
        runtimes={DEEPPREP: 1440.0, FASTSURFER: 600.0},
    )
    assert _segmenter(preferred) == "deepprep"
    failed = plan_backends(
        RunPolicy(), _gpu(), {DEEPPREP: "failed", FASTSURFER: "passed"},
        required_executor_ids=T1_ONLY,
    )
    decision = next(d for d in failed.decisions if d.capability == SEGMENT)
    assert decision.selected_package == "fastsurfer"
    assert decision.rejected_candidates["deepprep"] == "smoke_test_failed"


def test_a_backend_another_executor_requires_is_pinned() -> None:
    plan = plan_backends(RunPolicy(), _gpu(), required_executor_ids=T1_ONLY | {"petprep"})
    decision = next(d for d in plan.decisions if d.capability == SEGMENT)
    assert decision.selected_package == "deepprep"
    assert decision.rejected_candidates["fastsurfer"] == "petprep_requires_deepprep"
    assert FASTSURFER not in {t.test_id for t in plan.smoke_tests}


SYNTHSEG = f"smoke--{SEGMENT}--synthseg"


def test_synthseg_lacks_surfaces_so_it_only_competes_for_fast_requests() -> None:
    standard = plan_backends(RunPolicy(), _gpu(), required_executor_ids=T1_ONLY)
    decision = next(d for d in standard.decisions if d.capability == SEGMENT)
    assert decision.rejected_candidates["synthseg"] == (
        "missing_outputs_left_white_surface+right_white_surface"
    )
    assert SYNTHSEG not in {t.test_id for t in standard.smoke_tests}

    fast = RunPolicy(source_prompt="segment quickly", objective=ProcessingObjective.FAST)
    planned = plan_backends(fast, _gpu(), required_executor_ids=T1_ONLY)
    assert {DEEPPREP, FASTSURFER, SYNTHSEG} <= {t.test_id for t in planned.smoke_tests}
    passed = plan_backends(
        fast, _gpu(), {DEEPPREP: "passed", FASTSURFER: "passed", SYNTHSEG: "passed"},
        required_executor_ids=T1_ONLY,
    )
    assert _segmenter(passed) == "synthseg"  # matches the fast objective


def test_capabilities_activate_by_stage_not_just_executor() -> None:
    flair_only = plan_backends(
        RunPolicy(), _gpu(),
        required_executor_ids={"neuroprep-spatial"},
        required_stages={"flair_preprocess", "anat_preprocess"},
    )
    assert not {d.capability for d in flair_only.decisions} & {"ct", "ct_segment"}
    with_ct = plan_backends(
        RunPolicy(), _gpu(),
        required_executor_ids={"neuroprep-spatial"},
        required_stages={"ct_preprocess", "ct_segment"},
    )
    assert {"ct", "ct_segment"} <= {d.capability for d in with_ct.decisions}


def test_every_integrated_backend_names_a_catalog_executor_that_meets_its_contract() -> None:
    from neuroprep_agentic.executors import builtin_executor_catalog
    from neuroprep_agentic.policy import (
        CAPABILITY_CONTRACTS,
        builtin_backend_portfolio,
        capability_contract,
    )

    catalog = builtin_executor_catalog()
    for item in builtin_backend_portfolio():
        if item.integration_state != "integrated":
            continue
        definition = catalog.by_id(str(item.executor_id))  # KeyError = dangling portfolio entry
        kinds = {a.kind for a in definition.artifacts}
        objectives = item.objective_tags or {ProcessingObjective.STANDARD}
        assert any(
            capability_contract(item.capability, objective) <= kinds for objective in objectives
        ), (item.capability, item.package_id)
    assert set(CAPABILITY_CONTRACTS) <= {item.capability for item in builtin_backend_portfolio()}


DWI_PLAN = {"smriprep", "qsiprep", "qsirecon"}


def test_mrtrix3_competes_when_it_takes_over_the_mandatory_tensor_fit() -> None:
    plan = plan_backends(
        RunPolicy(), _gpu(),
        required_executor_ids=DWI_PLAN,
        mandatory_stages={"anat_preprocess", "dwi_preprocess", "dti_fit"},
    )
    assert "smoke--dwi_preprocess--mrtrix3" in {t.test_id for t in plan.smoke_tests}
    all_mandatory = plan_backends(
        RunPolicy(), _gpu(),
        required_executor_ids=DWI_PLAN,
        mandatory_stages={"dwi_preprocess", "dti_fit", "advanced_diffusion_models"},
    )
    dwi = next(d for d in all_mandatory.decisions if d.capability == "dwi_preprocess")
    assert dwi.rejected_candidates["mrtrix3"] == "qsirecon_needs_qsiprep_dwi"


def test_dwi_plugin_leaves_tractography_optional_so_mrtrix3_can_compete() -> None:
    from neuroprep_agentic.plugins.modalities import builtin_modality_plugins

    dwi = next(p for p in builtin_modality_plugins() if p.plugin_id == "mri.dwi")
    mandatory = {"anat_preprocess"} | {t.stage for t in dwi.tasks if t.required}
    plan = plan_backends(
        RunPolicy(), _gpu(), required_executor_ids=DWI_PLAN, mandatory_stages=mandatory
    )
    assert "smoke--dwi_preprocess--mrtrix3" in {t.test_id for t in plan.smoke_tests}
