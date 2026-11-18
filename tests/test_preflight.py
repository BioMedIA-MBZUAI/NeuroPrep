from __future__ import annotations

import os
from pathlib import Path

import pytest

from neuroprep_agentic import paths
from neuroprep_agentic.execution_models import (
    ExecutableTask,
    ExecutionPlan,
    ExecutionScope,
    RunLayout,
)
from neuroprep_agentic.executors import builtin_executor_catalog
from neuroprep_agentic.preflight import preflight_execution_plan


def _plan(root: Path, metadata: dict[str, dict]) -> ExecutionPlan:
    catalog = builtin_executor_catalog()
    layout = RunLayout(
        dataset_id="preflight-test",
        source_root=str(root / "source"),
        bids_root=str(root / "bids"),
        derivatives_root=str(root / "derivatives"),
        work_root=str(root / "work"),
        state_path=str(root / "state" / "tasks.sqlite"),
    )
    task = ExecutableTask(
        task_id="qsiprep-test",
        logical_task_ids=["logical"],
        executor_id="qsiprep",
        plugin_ids=["mri.dwi"],
        stages=["dwi_preprocess"],
        scope=ExecutionScope.SUBJECT,
        subject_id="sub-01",
        acquisition_ids=list(metadata),
        command=["true"],
        output_root=layout.derivatives_root,
        work_dir=str(root / "work" / "task"),
        resource_class="gpu",
        cpus=1,
        memory_gb=4,
        gpus=1,
        walltime_minutes=10,
        fingerprint="test",
        metadata={"source_metadata": metadata},
    )
    return ExecutionPlan(
        dataset_id="preflight-test",
        generated_at="2026-01-01T00:00:00Z",
        catalog_digest=catalog.digest(),
        layout=layout,
        tasks=[task],
        required_executor_ids=["qsiprep"],
        optional_executor_ids=[],
    )


def test_required_metadata_must_exist_on_every_grouped_acquisition(nfs_tmp: Path) -> None:
    catalog = builtin_executor_catalog()
    incomplete = _plan(
        nfs_tmp,
        {
            "acq-a": {"PhaseEncodingDirection": "j-"},
            "acq-b": {},
        },
    )
    report = preflight_execution_plan(incomplete, catalog, check_inputs=False)
    assert report.missing_metadata["qsiprep-test"] == ["PhaseEncodingDirection"]

    complete = _plan(
        nfs_tmp,
        {
            "acq-a": {"PhaseEncodingDirection": "j-"},
            "acq-b": {"PhaseEncodingDirection": "j"},
        },
    )
    report = preflight_execution_plan(complete, catalog, check_inputs=False)
    assert "qsiprep-test" not in report.missing_metadata


def test_qsiprep_preflight_checks_the_agentic_eddy_wrapper(nfs_tmp: Path) -> None:
    catalog = builtin_executor_catalog()
    plan = _plan(
        nfs_tmp,
        {"acq-a": {"PhaseEncodingDirection": "j-"}},
    )
    # The wrapper ships inside the package (source checkout and pip install alike) and
    # must keep its executable bit, so preflight finds it wherever the package lives.
    wrapper = paths.package_resources() / "eddy_cuda10.2"
    assert wrapper.is_file() and os.access(wrapper, os.X_OK)

    report = preflight_execution_plan(plan, catalog, check_inputs=False)

    assert str(wrapper) not in report.missing_runtime_assets
    assert not any(
        path.endswith("NeuroPrep-multimodal-v4/src/neuroprep/resources/eddy_cuda10.2")
        for path in report.missing_runtime_assets
    )


def test_run_layout_rejects_source_output_overlap(nfs_tmp: Path) -> None:
    with pytest.raises(ValueError, match="source and work roots must not overlap"):
        RunLayout(
            dataset_id="unsafe-layout",
            source_root=str(nfs_tmp / "source"),
            bids_root=str(nfs_tmp / "bids"),
            derivatives_root=str(nfs_tmp / "derivatives"),
            work_root=str(nfs_tmp / "source" / "work"),
            state_path=str(nfs_tmp / "state" / "tasks.sqlite"),
        )
