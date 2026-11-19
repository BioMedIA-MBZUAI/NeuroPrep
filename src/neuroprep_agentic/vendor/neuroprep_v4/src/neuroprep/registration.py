from __future__ import annotations

import json
import hashlib
import re
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .models import Modality, Space


@dataclass(frozen=True)
class TransformRecord:
    subject: str
    source_image: str
    target_image: str
    source_space: str
    target_space: str
    transform_type: str
    forward_transforms: list[str]
    inverse_transforms: list[str]
    interpolation: str
    tool: str = "ANTs"
    created_at: str = ""
    dataset_id: str = ""
    session: str | None = None
    modality: str | None = None
    metadata: dict[str, str] | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        if not payload["created_at"]:
            payload["created_at"] = datetime.now(timezone.utc).isoformat()
        identity = {
            key: payload[key]
            for key in (
                "dataset_id",
                "subject",
                "session",
                "source_image",
                "target_image",
                "source_space",
                "target_space",
                "transform_type",
                "forward_transforms",
                "inverse_transforms",
                "interpolation",
                "tool",
            )
        }
        payload["transform_id"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return payload


class TransformGraph:
    def __init__(self, records: list[TransformRecord] | None = None):
        self.records: list[TransformRecord] = []
        self._ids: set[str] = set()
        for record in records or []:
            self.add(record)

    def add(self, record: TransformRecord) -> str:
        if not record.source_space or not record.target_space:
            raise ValueError("Transform source and target spaces are required")
        if record.source_space == record.target_space:
            raise ValueError("Transform source and target spaces must differ")
        if not record.forward_transforms:
            raise ValueError("At least one forward transform is required")
        transform_id = str(record.to_dict()["transform_id"])
        if transform_id not in self._ids:
            self.records.append(record)
            self._ids.add(transform_id)
        return transform_id

    def path(self, source_space: str, target_space: str) -> list[dict]:
        if source_space == target_space:
            return []
        adjacency: dict[str, list[tuple[str, TransformRecord, bool]]] = {}
        for record in self.records:
            adjacency.setdefault(record.source_space, []).append(
                (record.target_space, record, False)
            )
            if record.inverse_transforms:
                adjacency.setdefault(record.target_space, []).append(
                    (record.source_space, record, True)
                )
        queue = deque([(source_space, [])])
        visited = {source_space}
        while queue:
            space, route = queue.popleft()
            for next_space, record, inverse in adjacency.get(space, []):
                step = {
                    "transform_id": record.to_dict()["transform_id"],
                    "inverse": inverse,
                    "source_space": space,
                    "target_space": next_space,
                    "files": (
                        list(record.inverse_transforms)
                        if inverse
                        else list(record.forward_transforms)
                    ),
                }
                if next_space == target_space:
                    return [*route, step]
                if next_space not in visited:
                    visited.add(next_space)
                    queue.append((next_space, [*route, step]))
        raise ValueError(f"No transform path from {source_space} to {target_space}")

    def to_dict(self) -> dict:
        return {"version": "1.0", "transforms": [record.to_dict() for record in self.records]}

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)


def interpolation_for(modality: Modality, is_label: bool = False) -> str:
    if is_label:
        return "NearestNeighbor"
    if modality is Modality.DWI:
        raise ValueError(
            "Raw DWI/tensor data must be transformed by QSIPrep/QSirecon so gradients are reoriented"
        )
    if modality in {Modality.EEG, Modality.MEG, Modality.IEEG}:
        raise ValueError("Sensor time series use coordinate transforms, not voxel interpolation")
    if modality in {Modality.PET, Modality.CT, Modality.ASL}:
        return "Linear"
    return "LanczosWindowedSinc"


def rigid_registration_command(
    fixed: Path,
    moving: Path,
    output_prefix: Path,
    dimension: int = 3,
) -> list[str]:
    if dimension not in {2, 3}:
        raise ValueError("ANTs registration dimension must be 2 or 3")
    return [
        "antsRegistrationSyNQuick.sh",
        "-d",
        str(dimension),
        "-f",
        str(fixed),
        "-m",
        str(moving),
        "-t",
        "r",
        "-o",
        str(output_prefix),
    ]


def apply_transform_command(
    fixed: Path,
    moving: Path,
    transform: Path,
    output: Path,
    modality: Modality,
    is_label: bool = False,
) -> list[str]:
    return [
        "antsApplyTransforms",
        "-d",
        "3",
        "-i",
        str(moving),
        "-r",
        str(fixed),
        "-t",
        str(transform),
        "-n",
        interpolation_for(modality, is_label),
        "-o",
        str(output),
    ]


def write_transform_record(record: TransformRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(record.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def index_transform_files(subject: str, search_roots: list[Path], output: Path) -> TransformGraph:
    pattern = re.compile(r"(?:^|_)from-([^_]+)_to-([^_]+)(?:_|$)", re.I)
    extensions = (".h5", ".mat", ".txt", ".lta", ".tfm", ".nii.gz")
    graph = TransformGraph()
    grouped: dict[tuple[str, str, str], list[Path]] = {}
    for root in search_roots:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file() or not path.name.lower().endswith(extensions):
                continue
            match = pattern.search(path.name)
            if not match:
                continue
            source_space, target_space = match.groups()
            acquisition = path.name[: match.start()].rstrip("_") or "unspecified"
            grouped.setdefault((acquisition, source_space, target_space), []).append(path)
    for (acquisition, source_space, target_space), paths in sorted(grouped.items()):
        # ANTs composite application convention: deformation field, then affine. A single graph
        # edge must contain the whole chain rather than silently selecting one component.
        ordered = sorted(paths, key=_transform_order)
        graph.add(
            TransformRecord(
                subject=subject,
                source_image=acquisition,
                target_image="",
                source_space=source_space,
                target_space=target_space,
                transform_type="upstream-bids-transform-chain",
                forward_transforms=[str(path) for path in ordered],
                inverse_transforms=[],
                interpolation="defined-by-consuming-modality",
                tool="upstream-bids-derivative",
                metadata={"acquisition_transform_prefix": acquisition},
            )
        )
    if not graph.records:
        roots = ", ".join(str(path) for path in search_roots)
        raise ValueError(f"No BIDS-named transform files found under: {roots}")
    graph.write(output)
    return graph


def _transform_order(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if "warp" in name and "inverse" not in name:
        return (0, name)
    if "affine" in name or name.endswith(".mat"):
        return (1, name)
    return (2, name)


def subject_to_standard_space() -> Space:
    return Space.STANDARD
