from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .execution_models import ExecutionPlan
from .settings import validated_cluster_path

NEXTFLOW_VERSION = "24.10.3"
NEXTFLOW_SIZE = 34_362_800
NEXTFLOW_SHA256 = "835f5a178ac7c5c4fc12a3407aa381c05d8b3f5750c8c3292a5389c12721d12a"
NEXTFLOW_SEED = Path(
    "/nfs-stor/salma.hassan/NeuroPrep-agentic-tools/nextflow/framework/24.10.3/"
    "nextflow-24.10.3-one.jar"
)

ASLPREP_PATCH_VERSION = "26.0.3-neuroprep-2"
ASLPREP_IMAGE_NAME = "aslprep-26.0.3.sif"
ASLPREP_TEMPLATEFLOW_TRANSFORM_SIZE = 204_735_104
ASLPREP_TEMPLATEFLOW_TRANSFORM_SHA256 = (
    "2e3869a07b96aec406e0419ca2e434afc54882d37cc212b933b139d1b63a4dfe"
)
ASLPREP_TEMPLATEFLOW_TRANSFORM_NAME = (
    "tpl-MNI152NLin2009cAsym_from-MNI152NLin6Asym_mode-image_xfm.h5"
)
ASLPREP_TEMPLATEFLOW_TRANSFORM_SEED = Path(
    "/nfs-stor/salma.hassan/NeuroPrep-agentic-tools/templateflow/"
    "tpl-MNI152NLin2009cAsym/"
    + ASLPREP_TEMPLATEFLOW_TRANSFORM_NAME
)
ASLPREP_PATCHES: dict[str, tuple[str, tuple[tuple[str, str], ...], str]] = {
    "cbf.py": (
        "/app/.pixi/envs/aslprep/lib/python3.12/site-packages/aslprep/workflows/asl/cbf.py",
        (
            (
                """        workflow.__desc__ += f\"\"\"\\
: *ASLPrep* calculated cerebral blood flow (CBF) from the single-delay PASL
using a single-compartment general kinetic model [@buxton1998general]
using the {bcut} modification, as described in {singlepld_pasl_strs[bcut]}.
{m0_str}
\"\"\"
""",
                """        if bcut is None and metadata.get('BolusCutOffFlag') is False:
            workflow.__desc__ += f\"\"\"\\
: *ASLPrep* calculated cerebral blood flow (CBF) from the single-delay PASL
using a single-compartment general kinetic model [@buxton1998general]
without a bolus cut-off modification.
{m0_str}
\"\"\"
        else:
            workflow.__desc__ += f\"\"\"\\
: *ASLPrep* calculated cerebral blood flow (CBF) from the single-delay PASL
using a single-compartment general kinetic model [@buxton1998general]
using the {bcut} modification, as described in {singlepld_pasl_strs[bcut]}.
{m0_str}
\"\"\"
""",
            ),
        ),
        "without a bolus cut-off modification",
    ),
    "base.py": (
        "/app/.pixi/envs/aslprep/lib/python3.12/site-packages/aslprep/workflows/base.py",
        (
            (
                """            anatomical_cache.update(temp_cache)

    config.loggers.workflow.info(f'Anatomical cache: {pformat(anatomical_cache)}')
""",
                """            # ASLPrep 26.0.3 may return both a valid derivative and a
            # non-existent legacy-layout candidate for scalar anatomical files.
            # Collapse only scalar fields and only when one existing file remains.
            for scalar_key in ('t1w_preproc', 't1w_mask', 't1w_dseg', 't2w_preproc'):
                scalar_value = temp_cache.get(scalar_key)
                if isinstance(scalar_value, list):
                    existing = [item for item in scalar_value if os.path.isfile(item)]
                    if len(existing) == 1:
                        temp_cache[scalar_key] = existing[0]
            anatomical_cache.update(temp_cache)

    config.loggers.workflow.info(f'Anatomical cache: {pformat(anatomical_cache)}')
""",
            ),
        ),
        "Collapse only scalar fields",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _needs_nextflow(plan: ExecutionPlan) -> bool:
    return any(
        task.executor_id.startswith("deepprep") and (task.required or plan.layout.execute_optional)
        for task in plan.tasks
    )


def _needs_aslprep_patches(plan: ExecutionPlan) -> bool:
    return any(
        task.executor_id == "aslprep" and (task.required or plan.layout.execute_optional)
        for task in plan.tasks
    )


def nextflow_destination(plan: ExecutionPlan) -> Path:
    return (
        Path(plan.layout.work_root)
        / "cache"
        / "nextflow"
        / "framework"
        / NEXTFLOW_VERSION
        / f"nextflow-{NEXTFLOW_VERSION}-one.jar"
    )


def _valid_nextflow(path: Path) -> bool:
    return (
        path.is_file() and path.stat().st_size == NEXTFLOW_SIZE and _sha256(path) == NEXTFLOW_SHA256
    )


def aslprep_patch_root(plan: ExecutionPlan) -> Path:
    return Path(plan.layout.work_root) / "cache" / "patches" / "aslprep-26.0.3"


def aslprep_templateflow_transform_destination(plan: ExecutionPlan) -> Path:
    return (
        Path(plan.layout.work_root)
        / "cache"
        / "templateflow"
        / "tpl-MNI152NLin2009cAsym"
        / ASLPREP_TEMPLATEFLOW_TRANSFORM_NAME
    )


def _valid_aslprep_templateflow_transform(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == ASLPREP_TEMPLATEFLOW_TRANSFORM_SIZE
        and _sha256(path) == ASLPREP_TEMPLATEFLOW_TRANSFORM_SHA256
    )


def _aslprep_patch_manifest(plan: ExecutionPlan) -> Path:
    return aslprep_patch_root(plan) / "manifest.json"


def _valid_aslprep_patches(plan: ExecutionPlan) -> bool:
    manifest_path = _aslprep_patch_manifest(plan)
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if manifest.get("patch_version") != ASLPREP_PATCH_VERSION:
        return False
    files = manifest.get("files")
    if not isinstance(files, dict):
        return False
    for filename, (_source, _replacements, marker) in ASLPREP_PATCHES.items():
        path = aslprep_patch_root(plan) / filename
        if not path.is_file() or path.stat().st_size == 0:
            return False
        content = path.read_text(encoding="utf-8")
        if marker not in content or files.get(filename) != _sha256(path):
            return False
    return True


def _extract_container_text(image: Path, source: str) -> str:
    completed = subprocess.run(
        [
            "/usr/bin/singularity",
            "exec",
            "--cleanenv",
            str(image),
            "/bin/cat",
            source,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _prepare_aslprep_patches(plan: ExecutionPlan) -> None:
    image = validated_cluster_path(
        str(Path(plan.layout.container_root) / ASLPREP_IMAGE_NAME), must_exist=True
    )
    root = validated_cluster_path(str(aslprep_patch_root(plan)))
    root.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for filename, (source, replacements, marker) in ASLPREP_PATCHES.items():
        content = _extract_container_text(image, source)
        for before, after in replacements:
            if content.count(before) != 1:
                raise ValueError(
                    f"ASLPrep {ASLPREP_PATCH_VERSION} patch anchor mismatch for {source}"
                )
            content = content.replace(before, after, 1)
        if marker not in content:
            raise ValueError(f"ASLPrep patch marker missing after patching {source}")
        destination = root / filename
        temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
        temporary.write_text(content, encoding="utf-8")
        temporary.chmod(0o444)
        temporary.replace(destination)
        digests[filename] = _sha256(destination)
    manifest = {
        "patch_version": ASLPREP_PATCH_VERSION,
        "image": str(image),
        "files": digests,
    }
    destination = _aslprep_patch_manifest(plan)
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o444)
    temporary.replace(destination)


def runtime_asset_inventory(plan: ExecutionPlan) -> dict[str, Any]:
    assets: list[dict[str, Any]] = []
    if _needs_nextflow(plan):
        destination = validated_cluster_path(str(nextflow_destination(plan)))
        valid = _valid_nextflow(destination)
        assets.append(
            {
                "asset_id": "nextflow-24.10.3-one-jar",
                "path": str(destination),
                "ready": valid,
                "expected_size": NEXTFLOW_SIZE,
                "expected_sha256": NEXTFLOW_SHA256,
            }
        )
    if _needs_aslprep_patches(plan):
        root = validated_cluster_path(str(aslprep_patch_root(plan)))
        ready = _valid_aslprep_patches(plan)
        assets.append(
            {
                "asset_id": f"aslprep-patches-{ASLPREP_PATCH_VERSION}",
                "path": str(root),
                "ready": ready,
                "patch_version": ASLPREP_PATCH_VERSION,
            }
        )
        transform = validated_cluster_path(
            str(aslprep_templateflow_transform_destination(plan))
        )
        assets.append(
            {
                "asset_id": "aslprep-mni6-to-mni2009c-transform",
                "path": str(transform),
                "ready": _valid_aslprep_templateflow_transform(transform),
                "expected_size": ASLPREP_TEMPLATEFLOW_TRANSFORM_SIZE,
                "expected_sha256": ASLPREP_TEMPLATEFLOW_TRANSFORM_SHA256,
            }
        )
    return {"ready": all(item["ready"] for item in assets), "assets": assets}


def prepare_runtime_assets(plan: ExecutionPlan) -> dict[str, Any]:
    if _needs_nextflow(plan):
        seed = validated_cluster_path(str(NEXTFLOW_SEED), must_exist=True)
        if not _valid_nextflow(seed):
            raise ValueError(f"Pinned Nextflow seed failed integrity validation: {seed}")
        destination = validated_cluster_path(str(nextflow_destination(plan)))
        if not _valid_nextflow(destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + f".copy-{os.getpid()}")
            try:
                shutil.copy2(seed, temporary)
                if not _valid_nextflow(temporary):
                    raise ValueError("Copied Nextflow asset failed integrity validation")
                temporary.chmod(0o444)
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
    if _needs_aslprep_patches(plan) and not _valid_aslprep_patches(plan):
        _prepare_aslprep_patches(plan)
    if _needs_aslprep_patches(plan):
        seed = validated_cluster_path(
            str(ASLPREP_TEMPLATEFLOW_TRANSFORM_SEED), must_exist=True
        )
        if not _valid_aslprep_templateflow_transform(seed):
            raise ValueError(
                f"Pinned ASLPrep TemplateFlow seed failed integrity validation: {seed}"
            )
        destination = validated_cluster_path(
            str(aslprep_templateflow_transform_destination(plan))
        )
        if not _valid_aslprep_templateflow_transform(destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + f".copy-{os.getpid()}")
            try:
                shutil.copy2(seed, temporary)
                if not _valid_aslprep_templateflow_transform(temporary):
                    raise ValueError(
                        "Copied ASLPrep TemplateFlow asset failed integrity validation"
                    )
                temporary.chmod(0o444)
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
    return runtime_asset_inventory(plan)
