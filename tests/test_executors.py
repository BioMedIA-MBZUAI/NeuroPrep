from neuroprep_agentic.execution_models import ExecutorBackend
from neuroprep_agentic.executors import (
    ExecutorCatalog,
    builtin_executor_catalog,
    builtin_executor_definitions,
)
from neuroprep_agentic.registry import builtin_registry


def test_all_modality_and_core_routes_have_deterministic_executors() -> None:
    catalog = builtin_executor_catalog()
    assert catalog.audit(builtin_registry()) == []
    assert len(catalog.definitions) >= 27


def test_container_executors_are_versioned_and_pullable() -> None:
    for definition in builtin_executor_definitions():
        assert definition.version
        assert definition.command
        assert definition.artifacts
        if definition.backend in {ExecutorBackend.BIDS_APP, ExecutorBackend.CONTAINER}:
            assert definition.image
            assert definition.image_source
            assert definition.image_source.startswith(("docker://", "library://"))


def test_catalog_digest_is_stable_and_route_unique() -> None:
    first = builtin_executor_catalog()
    second = builtin_executor_catalog()
    assert first.digest() == second.digest()

    duplicate = builtin_executor_definitions()[0]
    try:
        ExecutorCatalog((duplicate, duplicate))
    except ValueError as exc:
        assert "duplicate executor id" in str(exc)
    else:
        raise AssertionError("duplicate executor must fail closed")


def test_qsiprep_preserves_proven_gradient_and_geometry_safeguards() -> None:
    definition = builtin_executor_catalog().by_id("qsiprep")
    command = " ".join(definition.command)
    assert "qsiprep_gradients_26_0.py" in command
    assert "/site-packages/qsiprep/interfaces/gradients.py:ro" in command
    assert "eddy_cuda10.2" in command
    assert "/neuroprep-resources/qsiprep_primary_t1.json" in command
    assert "/neuroprep-resources/eddy_cuda.json" in command
    assert "--skip-anat-based-spatial-normalization" in definition.command
    assert "--separate-all-dwis" in definition.command
    assert "--derivatives" not in definition.command
    assert "{agentic_project}/src/neuroprep_agentic/resources/eddy_cuda10.2" in command
    assert definition.environment["LD_LIBRARY_PATH"].startswith("/lib/x86_64-linux-gnu")


def test_deepprep_is_offline_and_uses_work_scoped_nextflow() -> None:
    definition = builtin_executor_catalog().by_id("deepprep")
    assert "--network" in definition.command
    network_index = definition.command.index("--network")
    assert definition.command[network_index + 1] == "none"
    assert definition.environment["NXF_HOME"] == "{work_root}/cache/nextflow"
    assert "--anat_only" in definition.command
    assert "--bold_confounds" not in definition.command
    assert "{task_work_dir}/deepprep-bids:/data:ro" in definition.command
    assert "{bids_root}:/source-data:ro" in definition.command
    assert definition.environment["LD_LIBRARY_PATH"].startswith("/lib/x86_64-linux-gnu")


def test_smriprep_019_uses_only_supported_cli_flags() -> None:
    definition = builtin_executor_catalog().by_id("smriprep")
    assert "--skip-bids-validation" not in definition.command
    assert "--fs-no-reconall" in definition.command
    output_index = definition.command.index("/data") + 1
    assert definition.command[output_index] == "/out"
    assert "{task_work_dir}/smriprep-bids:/data:ro" in definition.command
    assert "{bids_root}:/source-data:ro" in definition.command


def test_native_spatial_routes_use_pinned_container_toolchain() -> None:
    catalog = builtin_executor_catalog()
    for executor_id in ("neuroprep-spatial", "neuroprep-susceptibility"):
        definition = catalog.by_id(executor_id)
        assert definition.backend is ExecutorBackend.CONTAINER
        assert definition.image == "{container_root}/qsiprep-26.0.0.sif"
        assert "{bids_root}:/data:ro" in definition.command
        assert "{neuroprep_project}/src:/neuroprep-src:ro" in definition.command
        assert "PYTHONPATH=/neuroprep-src" in definition.command


def test_structural_derivatives_gate_every_anatomy_dependent_app() -> None:
    catalog = builtin_executor_catalog()
    assert catalog.by_id("smriprep").stages == ("anat_preprocess",)
    assert catalog.by_id("deepprep").stages == ("anat_segment",)
    for executor_id in ("deepprep", "fmriprep", "qsiprep", "petprep", "aslprep"):
        assert catalog.by_id(executor_id).requires_executor_ids == ("smriprep",)
    assert catalog.by_id("xcp-d").requires_executor_ids == ("fmriprep",)
    assert "/out/fmriprep" in catalog.by_id("xcp-d").command
    qsirecon = catalog.by_id("qsirecon")
    assert qsirecon.requires_executor_ids == ("qsiprep",)
    assert "--recon-spec" in qsirecon.command
    assert "dsi_studio_gqi" in qsirecon.command


def test_pet_and_asl_reuse_structural_derivatives() -> None:
    catalog = builtin_executor_catalog()
    for executor_id in ("petprep", "aslprep"):
        command = catalog.by_id(executor_id).command
        assert "smriprep=/out/smriprep" in command
        assert "{freesurfer_license}:/fs_license.txt:ro" in command
        assert "--fs-license-file" in command
        assert "--level" in command
        assert "full" in command
        assert "--output-spaces" in command


def test_conventional_t2star_does_not_require_quantitative_units() -> None:
    catalog = builtin_executor_catalog()
    definition = catalog.resolve("mri.t2star", "t2star_preprocess")
    assert definition.executor_id == "neuroprep-anatomical-maps"
    assert definition.required_metadata == ()
    assert catalog.resolve("mri.qmri", "qmri_preprocess").required_metadata == ("Units",)


def test_mrs_uses_verified_official_native_toolchain() -> None:
    definition = builtin_executor_catalog().by_id("fsl-mrs")
    assert definition.backend is ExecutorBackend.NATIVE
    assert definition.version == "2.4.17"
    assert "BasisPath" in definition.required_metadata
    assert not definition.image
