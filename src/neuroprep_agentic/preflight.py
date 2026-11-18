from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import paths
from .execution_models import ExecutionPlan, PreflightCheck, PreflightReport
from .executors import ExecutorCatalog
from .runtime_assets import runtime_asset_inventory
from .settings import validated_cluster_path


def preflight_execution_plan(
    plan: ExecutionPlan,
    catalog: ExecutorCatalog,
    *,
    check_inputs: bool = True,
) -> PreflightReport:
    checks: list[PreflightCheck] = []
    missing_images: set[str] = set()
    missing_executables: set[str] = set()
    missing_runtime_assets: set[str] = set()
    missing_metadata: dict[str, list[str]] = {}

    digest_matches = plan.catalog_digest == catalog.digest()
    checks.append(
        PreflightCheck(
            name="catalog_digest",
            status="pass" if digest_matches else "fail",
            detail=(
                plan.catalog_digest
                if digest_matches
                else f"plan={plan.catalog_digest} current={catalog.digest()}"
            ),
        )
    )

    for name, value in (
        ("source_root", plan.layout.source_root),
        ("bids_root", plan.layout.bids_root),
        ("derivatives_root", plan.layout.derivatives_root),
        ("work_root", plan.layout.work_root),
        ("state_path", plan.layout.state_path),
    ):
        try:
            validated_cluster_path(value)
        except ValueError as exc:
            checks.append(PreflightCheck(name=name, status="fail", detail=str(exc)))
        else:
            checks.append(PreflightCheck(name=name, status="pass", detail=value))

    singularity = paths.container_runtime()
    singularity_ready = singularity.is_file() and os.access(singularity, os.X_OK)
    checks.append(
        PreflightCheck(
            name="singularity",
            status="pass" if singularity_ready else "fail",
            detail=str(singularity),
        )
    )

    used = {
        task.executor_id for task in plan.tasks if plan.runs(task)
    }
    for executor_id in sorted(used):
        definition = catalog.by_id(executor_id)
        values = {
            "container_root": plan.layout.container_root,
            "neuroprep_project": plan.layout.neuroprep_project,
        }
        if definition.image:
            image = Path(definition.image.format(**values))
            if not image.is_file() or image.stat().st_size == 0:
                missing_images.add(str(image))
        if definition.executable:
            executable = Path(definition.executable.format(**values))
            if not executable.is_file() or not os.access(executable, os.X_OK):
                missing_executables.add(str(executable))

    runtime_assets = runtime_asset_inventory(plan)
    for item in runtime_assets["assets"]:
        if not item["ready"]:
            missing_runtime_assets.add(str(item["path"]))

    if any(task.executor_id == "qsiprep" for task in plan.tasks):
        agentic_resources = paths.package_resources()
        resources = Path(plan.layout.neuroprep_project) / "src/neuroprep/resources"
        resource_requirements = (
            (agentic_resources / "eddy_cuda10.2", True),
            (resources / "qsiprep_gradients_26_0.py", False),
            (resources / "qsiprep_primary_t1.json", False),
            (resources / "eddy_cuda.json", False),
        )
        for path, executable in resource_requirements:
            valid = path.is_file() and path.stat().st_size > 0
            if executable:
                valid = valid and os.access(path, os.X_OK)
            if not valid:
                missing_runtime_assets.add(str(path))

    if any(task.executor_id == "petprep" for task in plan.tasks):
        sitecustomize = paths.package_resources() / "petprep_sitecustomize.py"
        if not sitecustomize.is_file() or sitecustomize.stat().st_size == 0:
            missing_runtime_assets.add(str(sitecustomize))

    if any(task.executor_id == "fsl-mrs" for task in plan.tasks):
        fsl_mrs_root = paths.fsl_mrs_root()
        for name in ("fsl_mrs", "fsl_mrs_preproc", "spec2nii"):
            executable = fsl_mrs_root / "bin" / name
            if not executable.is_file() or not os.access(executable, os.X_OK):
                missing_executables.add(str(executable))

    if any(task.executor_id == "neuroprep-ultrasound" for task in plan.tasks):
        for name in ("ffmpeg", "ffprobe"):
            executable = shutil.which(name)
            if not executable:
                missing_executables.add(name)

    bids_stage_tasks = [task for task in plan.tasks if task.executor_id == "neuroprep-bids-stage"]
    needs_dcm2niix = any(
        str(task.metadata.get("source_format", "")).lower() == "dicom"
        or any(Path(path).is_dir() or path.lower().endswith(".dcm") for path in task.input_paths)
        for task in bids_stage_tasks
    )
    if needs_dcm2niix:
        dcm2niix = paths.dcm2niix_path()
        if not dcm2niix.is_file() or not os.access(dcm2niix, os.X_OK):
            missing_executables.add(str(dcm2niix))

    if any(
        task.executor_id in {"neuroprep-multimodal-registration", "neuroprep-longitudinal-template"}
        for task in plan.tasks
    ):
        ants = Path(plan.layout.container_root) / "ants-2.6.5.sif"
        if not ants.is_file() or ants.stat().st_size == 0:
            missing_images.add(str(ants))

    freesurfer_executors = {"deepprep", "fastsurfer", "smriprep", "fmriprep"}
    if any(task.executor_id in freesurfer_executors for task in plan.tasks):
        license_path = plan.layout.freesurfer_license
        if not license_path or not Path(license_path).is_file():
            missing_executables.add("FreeSurfer license file")

    if check_inputs:
        missing_inputs = sorted(
            {
                path
                for task in plan.tasks
                if task.executor_id == "neuroprep-bids-stage"
                for path in task.input_paths
                if not Path(path).exists()
            }
        )
        checks.append(
            PreflightCheck(
                name="source_inputs",
                status="pass" if not missing_inputs else "fail",
                detail=(
                    "all source inputs exist"
                    if not missing_inputs
                    else f"{len(missing_inputs)} inputs are missing; first={missing_inputs[0]}"
                ),
            )
        )

    for task in plan.tasks:
        definition = catalog.by_id(task.executor_id)
        if not definition.required_metadata:
            continue
        observed = task.metadata.get("source_metadata", {})
        records = list(observed.values()) if isinstance(observed, dict) else []
        if not records and task.executor_id == "neuroprep-bids-stage":
            records = [observed]
        missing: set[str] = set()
        for key in definition.required_metadata:
            if not records or not all(
                isinstance(record, dict) and record.get(key) is not None and record.get(key) != ""
                for record in records
            ):
                missing.add(key)
        if task.executor_id == "fsl-mrs" and "BasisPath" not in missing:
            basis_paths = [
                str(record["BasisPath"])
                for record in records
                if isinstance(record, dict) and record.get("BasisPath")
            ]
            if len(set(basis_paths)) != 1:
                missing.add("BasisPath(single consistent path)")
            try:
                for basis_path in basis_paths:
                    validated_cluster_path(basis_path, must_exist=True)
            except (FileNotFoundError, ValueError):
                missing.add("BasisPath(existing NFS path)")
            has_water_role = any(
                isinstance(record, dict)
                and any(
                    marker
                    in str(record.get("MRSRole") or record.get("AcquisitionRole") or "").lower()
                    for marker in ("water", "reference", "wref", "h2o")
                )
                for record in records
            )
            explicit_water = [
                str(record["WaterReferencePath"])
                for record in records
                if isinstance(record, dict) and record.get("WaterReferencePath")
            ]
            if not has_water_role and not explicit_water:
                missing.add("MRS water reference (MRSRole or WaterReferencePath)")
            try:
                for water_path in explicit_water:
                    validated_cluster_path(water_path, must_exist=True)
            except (FileNotFoundError, ValueError):
                missing.add("WaterReferencePath(existing NFS path)")
        if missing:
            missing_metadata[task.task_id] = sorted(missing)

    checks.extend(
        (
            PreflightCheck(
                name="executor_images",
                status="pass" if not missing_images else "fail",
                detail=(
                    "all required images are installed"
                    if not missing_images
                    else f"{len(missing_images)} required images are missing"
                ),
            ),
            PreflightCheck(
                name="executables",
                status="pass" if not missing_executables else "fail",
                detail=(
                    "all required executables are available"
                    if not missing_executables
                    else f"{len(missing_executables)} executables or licenses are missing"
                ),
            ),
            PreflightCheck(
                name="runtime_assets",
                status="pass" if not missing_runtime_assets else "fail",
                detail=(
                    "all pinned runtime assets passed integrity checks"
                    if not missing_runtime_assets
                    else f"{len(missing_runtime_assets)} runtime assets are missing or invalid"
                ),
            ),
            PreflightCheck(
                name="required_metadata",
                status="pass" if not missing_metadata else "fail",
                detail=(
                    "required scientific metadata are present"
                    if not missing_metadata
                    else f"{len(missing_metadata)} tasks have missing metadata"
                ),
            ),
        )
    )
    ready = all(check.status == "pass" for check in checks)
    return PreflightReport(
        ready=ready,
        checks=checks,
        missing_images=sorted(missing_images),
        missing_executables=sorted(missing_executables),
        missing_runtime_assets=sorted(missing_runtime_assets),
        missing_metadata=missing_metadata,
    )
