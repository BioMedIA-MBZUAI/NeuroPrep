from __future__ import annotations

import hashlib
from pathlib import Path

import neuroprep_agentic.runtime_assets as runtime_assets
from neuroprep_agentic.execution_models import (
    ExecutableTask,
    ExecutionPlan,
    ExecutionScope,
    RunLayout,
)


def _plan(
    root: Path, *, with_deepprep: bool = True, with_aslprep: bool = False
) -> ExecutionPlan:
    layout = RunLayout(
        dataset_id="asset-test",
        source_root=str(root / "source"),
        bids_root=str(root / "bids"),
        derivatives_root=str(root / "derivatives"),
        work_root=str(root / "work"),
        state_path=str(root / "state" / "tasks.sqlite"),
    )
    tasks = []
    if with_deepprep:
        tasks.append(
            ExecutableTask(
                task_id="deepprep-test",
                logical_task_ids=["logical"],
                executor_id="deepprep",
                plugin_ids=["mri.t1w"],
                stages=["anat_segment"],
                scope=ExecutionScope.SUBJECT,
                subject_id="sub-01",
                command=["true"],
                output_root=layout.derivatives_root,
                work_dir=str(root / "work" / "task"),
                resource_class="gpu",
                cpus=1,
                memory_gb=4,
                gpus=1,
                walltime_minutes=10,
                fingerprint="test",
            )
        )
    if with_aslprep:
        tasks.append(
            ExecutableTask(
                task_id="aslprep-test",
                logical_task_ids=["logical-asl"],
                executor_id="aslprep",
                plugin_ids=["mri.asl"],
                stages=["asl_preprocess"],
                scope=ExecutionScope.SUBJECT,
                subject_id="sub-01",
                command=["true"],
                output_root=layout.derivatives_root,
                work_dir=str(root / "work" / "asl-task"),
                resource_class="cpu",
                cpus=1,
                memory_gb=4,
                gpus=0,
                walltime_minutes=10,
                fingerprint="test",
            )
        )
    return ExecutionPlan(
        dataset_id="asset-test",
        generated_at="2026-01-01T00:00:00Z",
        catalog_digest="test",
        layout=layout,
        tasks=tasks,
        required_executor_ids=[task.executor_id for task in tasks],
        optional_executor_ids=[],
    )


def test_runtime_assets_are_plan_scoped_and_integrity_checked(monkeypatch, nfs_tmp: Path) -> None:
    payload = b"pinned-nextflow-test"
    seed = nfs_tmp / "seed.jar"
    seed.write_bytes(payload)
    monkeypatch.setattr(runtime_assets, "NEXTFLOW_SEED", seed)
    monkeypatch.setattr(runtime_assets, "NEXTFLOW_SIZE", len(payload))
    monkeypatch.setattr(runtime_assets, "NEXTFLOW_SHA256", hashlib.sha256(payload).hexdigest())

    plan = _plan(nfs_tmp)
    assert runtime_assets.runtime_asset_inventory(plan)["ready"] is False
    prepared = runtime_assets.prepare_runtime_assets(plan)
    assert prepared["ready"] is True
    destination = runtime_assets.nextflow_destination(plan)
    assert destination.read_bytes() == payload
    assert destination.stat().st_mode & 0o777 == 0o444


def test_runtime_assets_are_not_required_without_deepprep(nfs_tmp: Path) -> None:
    inventory = runtime_assets.runtime_asset_inventory(_plan(nfs_tmp, with_deepprep=False))
    assert inventory == {"ready": True, "assets": []}


def test_aslprep_patches_are_generated_from_the_pinned_image(
    monkeypatch, nfs_tmp: Path
) -> None:
    cbf_before, cbf_after = runtime_assets.ASLPREP_PATCHES["cbf.py"][1][0]
    base_before, base_after = runtime_assets.ASLPREP_PATCHES["base.py"][1][0]
    sources = {
        runtime_assets.ASLPREP_PATCHES["cbf.py"][0]: "header\n" + cbf_before + "footer\n",
        runtime_assets.ASLPREP_PATCHES["base.py"][0]: "header\n" + base_before + "footer\n",
    }
    image = nfs_tmp / "containers" / runtime_assets.ASLPREP_IMAGE_NAME
    image.parent.mkdir(parents=True)
    image.write_bytes(b"pinned-image")
    monkeypatch.setattr(
        runtime_assets,
        "_extract_container_text",
        lambda _image, source: sources[source],
    )
    transform_payload = b"pinned-aslprep-templateflow-transform"
    transform_seed = nfs_tmp / "aslprep-transform-seed.h5"
    transform_seed.write_bytes(transform_payload)
    monkeypatch.setattr(runtime_assets, "ASLPREP_TEMPLATEFLOW_TRANSFORM_SEED", transform_seed)
    monkeypatch.setattr(
        runtime_assets, "ASLPREP_TEMPLATEFLOW_TRANSFORM_SIZE", len(transform_payload)
    )
    monkeypatch.setattr(
        runtime_assets,
        "ASLPREP_TEMPLATEFLOW_TRANSFORM_SHA256",
        hashlib.sha256(transform_payload).hexdigest(),
    )
    plan = _plan(nfs_tmp, with_deepprep=False, with_aslprep=True)
    plan = plan.model_copy(
        update={"layout": plan.layout.model_copy(update={"container_root": str(image.parent)})}
    )

    assert runtime_assets.runtime_asset_inventory(plan)["ready"] is False
    prepared = runtime_assets.prepare_runtime_assets(plan)
    assert prepared["ready"] is True
    root = runtime_assets.aslprep_patch_root(plan)
    assert cbf_after in (root / "cbf.py").read_text()
    assert base_after in (root / "base.py").read_text()
    assert (root / "manifest.json").is_file()
    transform = runtime_assets.aslprep_templateflow_transform_destination(plan)
    assert transform.read_bytes() == transform_payload
    assert transform.stat().st_mode & 0o777 == 0o444


def test_templateflow_seed_is_copied_without_overwriting(monkeypatch, nfs_tmp: Path) -> None:
    seed = nfs_tmp / "tools" / "templateflow"
    (seed / "tpl-OASIS30ANTs").mkdir(parents=True)
    (seed / "tpl-OASIS30ANTs" / "T1w.nii.gz").write_bytes(b"seed")
    (seed / "tpl-OASIS30ANTs" / "kept.txt").write_bytes(b"seed")
    monkeypatch.setattr(runtime_assets, "TEMPLATEFLOW_SEED", seed)
    plan = _plan(nfs_tmp, with_deepprep=False)
    cache = Path(plan.layout.work_root) / "cache" / "templateflow" / "tpl-OASIS30ANTs"
    cache.mkdir(parents=True)
    (cache / "kept.txt").write_bytes(b"run")

    runtime_assets.prepare_runtime_assets(plan)
    assert (cache / "T1w.nii.gz").read_bytes() == b"seed"
    assert (cache / "kept.txt").read_bytes() == b"run"


def test_templateflow_seed_is_optional(monkeypatch, nfs_tmp: Path) -> None:
    monkeypatch.setattr(runtime_assets, "TEMPLATEFLOW_SEED", nfs_tmp / "absent")
    assert runtime_assets.seed_templateflow(_plan(nfs_tmp, with_deepprep=False)) is None


def test_totalsegmentator_weights_download_once_into_the_run_cache(
    monkeypatch, nfs_tmp: Path
) -> None:
    plan = _plan(nfs_tmp, with_deepprep=False)
    plan.layout.container_root = str(nfs_tmp / "containers")
    template = _plan(nfs_tmp).tasks[0]
    plan.tasks.append(
        template.model_copy(update={"task_id": "ts", "executor_id": "totalsegmentator"})
    )
    image = Path(plan.layout.container_root) / runtime_assets.TOTALSEG_IMAGE_NAME
    image.parent.mkdir(parents=True, exist_ok=True)
    image.touch()
    calls = []
    monkeypatch.setattr(runtime_assets.subprocess, "run", lambda cmd, **kw: calls.append(cmd))

    assert runtime_assets.runtime_asset_inventory(plan)["ready"] is False
    assert runtime_assets.prepare_runtime_assets(plan)["ready"] is True
    runtime_assets.prepare_runtime_assets(plan)

    assert len(calls) == 1 and calls[0][-3:] == ["totalseg_download_weights", "-t", "total"]
    assert f"{runtime_assets.totalsegmentator_home(plan)}:/totalseg-home" in calls[0]
