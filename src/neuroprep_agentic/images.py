from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import paths
from .execution_models import ExecutionPlan
from .executors import ExecutorCatalog
from .settings import validated_cluster_path


@dataclass(frozen=True)
class ImageSpec:
    executor_id: str
    path: str
    source: str


def required_images(
    plan: ExecutionPlan,
    catalog: ExecutorCatalog,
) -> tuple[ImageSpec, ...]:
    specs: dict[str, ImageSpec] = {}
    selected = {task.executor_id for task in plan.tasks if plan.runs(task)}
    # Backends a comparison pilot will qualify, so prefetching a target plan covers them.
    selected |= {
        str(item["executor_id"])
        for item in (*plan.backend_qualification_tests, *plan.planned_smoke_tests)
        if item.get("executor_id")
    }
    for executor_id in sorted(selected):
        definition = catalog.by_id(executor_id)
        if not definition.image:
            continue
        if not definition.image_source:
            raise ValueError(f"executor image has no immutable source: {executor_id}")
        path = definition.image.format(container_root=plan.layout.container_root)
        specs[path] = ImageSpec(executor_id, path, definition.image_source)

    if any(
        task.executor_id in {"neuroprep-multimodal-registration", "neuroprep-longitudinal-template"}
        for task in plan.tasks
    ):
        path = str(Path(plan.layout.container_root) / "ants-2.6.5.sif")
        specs[path] = ImageSpec(
            "neuroprep-registration-runtime",
            path,
            "docker://antsx/ants:v2.6.5",
        )
    return tuple(specs[path] for path in sorted(specs))


def catalog_images(catalog: ExecutorCatalog) -> tuple[ImageSpec, ...]:
    specs: dict[str, ImageSpec] = {}
    for definition in catalog.definitions:
        if not definition.image:
            continue
        if not definition.image_source:
            raise ValueError(f"executor image has no immutable source: {definition.executor_id}")
        path = definition.image.format(container_root=str(paths.container_root()))
        specs[path] = ImageSpec(definition.executor_id, path, definition.image_source)
    ants_path = str(paths.ants_image())
    specs[ants_path] = ImageSpec(
        "neuroprep-registration-runtime", ants_path, "docker://antsx/ants:v2.6.5"
    )
    return tuple(specs[path] for path in sorted(specs))


def _spec_inventory(specs: tuple[ImageSpec, ...], *, inspect: bool = False) -> dict[str, Any]:
    singularity = paths.container_runtime()
    rows = []
    for spec in specs:
        path = validated_cluster_path(spec.path)
        installed = path.is_file() and path.stat().st_size > 0
        valid_sif: bool | None = None
        inspection_error = ""
        if inspect and installed:
            completed = subprocess.run(
                [str(singularity), "inspect", "--json", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            valid_sif = completed.returncode == 0
            if not valid_sif:
                inspection_error = completed.stderr[-500:]
        rows.append(
            {
                **asdict(spec),
                "installed": installed,
                "size": path.stat().st_size if installed else 0,
                "valid_sif": valid_sif,
                "inspection_error": inspection_error,
            }
        )
    return {
        "ready": all(row["installed"] and (row["valid_sif"] is not False) for row in rows),
        "image_count": len(rows),
        "missing_count": sum(not row["installed"] for row in rows),
        "invalid_count": sum(row["valid_sif"] is False for row in rows),
        "images": rows,
    }


def catalog_image_inventory(catalog: ExecutorCatalog, *, inspect: bool = False) -> dict[str, Any]:
    return _spec_inventory(catalog_images(catalog), inspect=inspect)


def image_inventory(
    plan: ExecutionPlan,
    catalog: ExecutorCatalog,
) -> dict[str, Any]:
    rows = []
    for spec in required_images(plan, catalog):
        path = validated_cluster_path(spec.path)
        installed = path.is_file() and path.stat().st_size > 0
        rows.append(
            {
                **asdict(spec),
                "installed": installed,
                "size": path.stat().st_size if installed else 0,
            }
        )
    return {
        "ready": all(row["installed"] for row in rows),
        "image_count": len(rows),
        "missing_count": sum(not row["installed"] for row in rows),
        "images": rows,
    }


def prefetch_catalog_images(catalog: ExecutorCatalog) -> dict[str, Any]:
    singularity = paths.container_runtime()
    if not singularity.is_file() or not os.access(singularity, os.X_OK):
        raise FileNotFoundError(singularity)
    for spec in catalog_images(catalog):
        path = validated_cluster_path(spec.path)
        if path.is_file() and path.stat().st_size > 0:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".pull-{os.getpid()}")
        try:
            completed = subprocess.run(
                [
                    str(singularity),
                    "pull",
                    "--disable-cache",
                    str(temporary),
                    spec.source,
                ],
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"image pull failed for {spec.executor_id}: {completed.returncode}"
                )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f"image pull produced an empty file: {temporary}")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return catalog_image_inventory(catalog, inspect=True)


def prefetch_images(
    plan: ExecutionPlan,
    catalog: ExecutorCatalog,
) -> dict[str, Any]:
    singularity = paths.container_runtime()
    if not singularity.is_file() or not os.access(singularity, os.X_OK):
        raise FileNotFoundError(singularity)
    for spec in required_images(plan, catalog):
        path = validated_cluster_path(spec.path)
        if path.is_file() and path.stat().st_size > 0:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".pull-{os.getpid()}")
        try:
            completed = subprocess.run(
                [
                    str(singularity),
                    "pull",
                    "--disable-cache",
                    str(temporary),
                    spec.source,
                ],
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"image pull failed for {spec.executor_id}: {completed.returncode}"
                )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f"image pull produced an empty file: {temporary}")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return image_inventory(plan, catalog)
