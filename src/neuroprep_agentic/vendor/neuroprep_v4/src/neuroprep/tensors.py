from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np

from .normalization import normalize_image, normalize_sensor


DEFAULT_IMAGE_MODALITIES = (
    "t1w",
    "t2w",
    "flair",
    "pd",
    "mra",
    "ct",
    "pet",
    "asl",
    "swi",
    "dwi",
)
SENSOR_MODALITIES = ("eeg", "meg", "ieeg")


@dataclass(frozen=True)
class TensorPolicy:
    target_shape: tuple[int, int, int] = (96, 112, 96)
    image_modalities: tuple[str, ...] = DEFAULT_IMAGE_MODALITIES
    fmri_window: int = 100
    fmri_stride: int = 50
    sensor_window_seconds: float = 10.0
    sensor_stride_seconds: float = 5.0
    missing_value: float = 0.0
    dwi_shells: tuple[int, ...] = (0, 1000, 2000, 3000)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_visit_tensors(
    aligned_manifest: Path,
    output_root: Path,
    policy: TensorPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or TensorPolicy()
    aligned = _read_object(aligned_manifest)
    if aligned.get("status") != "pass":
        raise ValueError(f"Aligned visit is not releasable: {aligned_manifest}")
    participant = str(aligned["participant_id"])
    visit_id = str(aligned["visit_id"])
    destination = output_root / f"sub-{participant}" / visit_id
    destination.mkdir(parents=True, exist_ok=True)
    modalities = aligned.get("modalities", {})

    channels: list[np.ndarray] = []
    channel_names: list[str] = []
    normalizations: dict[str, Any] = {}
    source_paths: dict[str, Any] = {}
    crop_pad_records: dict[str, Any] = {}
    missing_mask = {name: 0 for name in (*policy.image_modalities, "fmri", *SENSOR_MODALITIES)}
    auxiliary_tensors: dict[str, str] = {}
    warnings: list[str] = []
    errors: list[str] = []
    channel_presence: list[int] = []

    for modality in policy.image_modalities:
        expected_names = _expected_channel_names(modality, policy)
        record = modalities.get(modality)
        if not isinstance(record, dict) or not record.get("present"):
            channels.extend(
                np.full(policy.target_shape, policy.missing_value, dtype=np.float32)
                for _ in expected_names
            )
            channel_names.extend(expected_names)
            channel_presence.extend(0 for _ in expected_names)
            continue
        try:
            if modality == "dwi":
                result, sources, dwi_metadata = _prepare_dwi(record, policy)
                normalizations[modality] = dwi_metadata
            else:
                source = Path(str(record["path"]))
                image = nib.load(str(source))
                data = np.asarray(image.dataobj, dtype=np.float32)
                if data.ndim == 4:
                    if modality == "pet":
                        temporal_path, metadata = _prepare_dynamic_image(
                            modality, record, data, destination, policy
                        )
                        auxiliary_tensors[f"{modality}_windows"] = str(temporal_path)
                        normalizations[modality] = metadata
                        source_paths[modality] = str(source)
                        mask = _load_mask(record, data.shape[:3])
                        normalized = normalize_image(
                            modality,
                            data,
                            mask,
                            _metadata(record),
                            _load_optional_reference_mask(record, data.shape[:3]),
                        )
                        result = np.nanmean(normalized.data, axis=3, dtype=np.float32)[
                            np.newaxis, ...
                        ]
                        sources = [str(source)]
                    else:
                        raise ValueError("unexpected 4D image in static channel stack")
                else:
                    mask = _load_mask(record, data.shape[:3])
                    metadata = _metadata(record)
                    reference_mask = _load_optional_reference_mask(record, data.shape[:3])
                    normalized = normalize_image(modality, data, mask, metadata, reference_mask)
                    result = normalized.data
                    if result.ndim == 3:
                        result = result[np.newaxis, ...]
                    normalizations[modality] = normalized.metadata()
                    sources = [str(source)]
            conformed, geometry = _crop_or_pad_channels(
                result, policy.target_shape, policy.missing_value
            )
            if conformed.shape[0] != len(expected_names):
                raise ValueError(
                    f"expected {len(expected_names)} channels but prepared {conformed.shape[0]}"
                )
            channels.extend(conformed[index] for index in range(conformed.shape[0]))
            channel_names.extend(expected_names)
            if modality == "dwi":
                channel_presence.extend(
                    int(value)
                    for value in normalizations[modality].get(
                        "shell_presence", [1] * len(expected_names)
                    )
                )
            else:
                channel_presence.extend(1 for _ in expected_names)
            source_paths[modality] = sources
            crop_pad_records[modality] = geometry
            missing_mask[modality] = 1
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{modality}:{exc}")
            channels.extend(
                np.full(policy.target_shape, policy.missing_value, dtype=np.float32)
                for _ in expected_names
            )
            channel_names.extend(expected_names)
            channel_presence.extend(0 for _ in expected_names)

    if channels:
        image_tensor = np.stack(channels, axis=0).astype(np.float32)
    else:
        image_tensor = np.empty((0, *policy.target_shape), dtype=np.float32)
    image_path = destination / f"sub-{participant}_{visit_id}_image-tensor.npz"
    _save_npz(
        image_path,
        image=image_tensor,
        channel_names=np.asarray(channel_names, dtype="U"),
        modality_presence=np.asarray(
            [missing_mask[name] for name in policy.image_modalities], dtype=np.uint8
        ),
        channel_presence=np.asarray(channel_presence, dtype=np.uint8),
        modality_order=np.asarray(policy.image_modalities, dtype="U"),
    )

    fmri = modalities.get("fmri")
    if isinstance(fmri, dict) and fmri.get("present"):
        try:
            path, metadata = _prepare_dynamic_image("fmri", fmri, None, destination, policy)
            auxiliary_tensors["fmri_windows"] = str(path)
            normalizations["fmri"] = metadata
            source_paths["fmri"] = str(fmri.get("path"))
            missing_mask["fmri"] = 1
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"fmri:{exc}")

    for modality in SENSOR_MODALITIES:
        record = modalities.get(modality)
        if not isinstance(record, dict) or not record.get("present"):
            continue
        try:
            path, metadata = _prepare_sensor(record, modality, destination, policy)
            auxiliary_tensors[f"{modality}_windows"] = str(path)
            normalizations[modality] = metadata
            source_paths[modality] = str(record.get("path"))
            missing_mask[modality] = 1
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            errors.append(f"{modality}:{exc}")

    tensor_manifest = {
        "contract_version": "3.0",
        "dataset_id": aligned.get("dataset_id"),
        "participant_id": participant,
        "visit_id": visit_id,
        "visit_key": aligned.get("visit_key"),
        "days_from_baseline": aligned.get("days_from_baseline"),
        "site_covariates": aligned.get("site_covariates", {}),
        "site_covariate_conflicts": aligned.get("site_covariate_conflicts", []),
        "status": "pass"
        if not errors and (any(missing_mask.values()) or auxiliary_tensors)
        else "fail",
        "image_tensor": str(image_path),
        "image_shape": list(image_tensor.shape),
        "image_channels": channel_names,
        "channel_presence": channel_presence,
        "auxiliary_tensors": auxiliary_tensors,
        "missing_modality_mask": missing_mask,
        "normalization": normalizations,
        "crop_pad": crop_pad_records,
        "sources": source_paths,
        "policy": policy.to_dict(),
        "warnings": warnings,
        "errors": errors,
        "training_contract": {
            "image_axis_order": "CXYZ",
            "fmri_axis_order": "WTXYZ",
            "sensor_axis_order": "WCT",
            "missing_modality_mask": "1=present, 0=missing",
            "source_quantitative_files_preserved": True,
        },
    }
    manifest_path = destination / f"sub-{participant}_{visit_id}_tensor-manifest.json"
    _write_json(manifest_path, tensor_manifest)
    tensor_manifest["manifest_path"] = str(manifest_path)
    return tensor_manifest


def index_tensor_manifests(root: Path, output: Path) -> dict[str, Any]:
    manifests = sorted(root.rglob("*_tensor-manifest.json"))
    rows = []
    for path in manifests:
        payload = _read_object(path)
        if payload.get("status") == "pass":
            payload["manifest_path"] = str(path)
            rows.append(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(output)
    return {
        "status": "pass" if rows else "fail",
        "visits": len(rows),
        "participants": len({row["participant_id"] for row in rows}),
        "index": str(output),
    }


def prepare_dataset_tensors(
    aligned_index: Path,
    output_root: Path,
    output_index: Path,
    policy: TensorPolicy | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(aligned_index)
    failures = []
    completed = 0
    for row in rows:
        manifest_value = row.get("manifest_path")
        if not manifest_value:
            failures.append(f"missing-manifest-path:{row.get('visit_key')}")
            continue
        try:
            result = prepare_visit_tensors(Path(str(manifest_value)), output_root, policy)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            failures.append(f"{row.get('visit_key')}:{exc}")
            continue
        completed += result["status"] == "pass"
        if result["status"] != "pass":
            failures.append(f"tensor-failed:{row.get('visit_key')}")
    index_summary = index_tensor_manifests(output_root, output_index)
    summary = {
        "status": "pass" if rows and not failures and index_summary["status"] == "pass" else "fail",
        "visits": len(rows),
        "completed": completed,
        "failures": failures,
        "index": str(output_index),
        "policy": (policy or TensorPolicy()).to_dict(),
    }
    _write_json(output_index.with_suffix(output_index.suffix + ".summary.json"), summary)
    return summary


class NeuroPrepDataset:
    """Small framework-neutral map-style dataset over tensor manifests."""

    def __init__(self, index: Path, include_auxiliary: bool = True):
        self.index = Path(index)
        self.records = _read_jsonl(self.index)
        self.include_auxiliary = include_auxiliary

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_archive = np.load(record["image_tensor"], allow_pickle=False)
        result: dict[str, Any] = {
            "image": image_archive["image"],
            "channel_names": image_archive["channel_names"],
            "modality_presence": image_archive["modality_presence"],
            "channel_presence": image_archive["channel_presence"],
            "participant_id": record["participant_id"],
            "visit_id": record["visit_id"],
            "visit_key": record.get("visit_key"),
        }
        if self.include_auxiliary:
            result["auxiliary"] = {
                name: {key: archive[key] for key in archive.files}
                for name, path in record.get("auxiliary_tensors", {}).items()
                for archive in [np.load(path, allow_pickle=False)]
            }
        return result


class TorchNeuroPrepDataset(NeuroPrepDataset):
    """PyTorch/MONAI-compatible dataset; torch is imported only when accessed."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("TorchNeuroPrepDataset requires NeuroPrep's torch extra") from exc
        sample = super().__getitem__(index)
        sample["image"] = torch.from_numpy(sample["image"])
        sample["modality_presence"] = torch.from_numpy(sample["modality_presence"])
        sample["channel_presence"] = torch.from_numpy(sample["channel_presence"])
        sample["auxiliary"] = {
            name: {
                key: torch.from_numpy(value)
                if isinstance(value, np.ndarray) and value.dtype.kind not in "OUS"
                else value
                for key, value in payload.items()
            }
            for name, payload in sample.get("auxiliary", {}).items()
        }
        return sample


def _prepare_dwi(
    record: dict[str, Any], policy: TensorPolicy
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    channels = record.get("shell_channels")
    if not isinstance(channels, list) or not channels:
        raise ValueError("DWI alignment did not provide gradient-safe shell maps")
    selected: list[np.ndarray] = []
    sources = []
    selected_targets: list[int] = []
    shell_presence: list[int] = []
    spatial_shape: tuple[int, ...] | None = None
    for target_shell in policy.dwi_shells:
        available = [
            row for row in channels if abs(float(row.get("shell_b", -9999)) - target_shell) <= 100
        ]
        if not available:
            shell_presence.append(0)
            continue
        best = min(available, key=lambda row: abs(float(row["shell_b"]) - target_shell))
        path = Path(str(best["path"]))
        image = nib.load(str(path))
        values = np.asarray(image.dataobj, dtype=np.float32)
        if spatial_shape is not None and values.shape != spatial_shape:
            raise ValueError("DWI shell maps do not share a common grid")
        spatial_shape = values.shape
        selected.append(values)
        sources.append(str(path))
        selected_targets.append(int(target_shell))
        shell_presence.append(1)
    if not selected:
        raise ValueError("No DWI shells matched the configured tensor policy")
    baseline_candidates = [
        index for index, shell in enumerate(selected_targets) if abs(shell) <= 100
    ]
    if not baseline_candidates:
        raise ValueError("DWI tensor normalization requires an acquired b0 shell")
    data = np.stack(selected, axis=3)
    mask = np.any(np.isfinite(data) & (data != 0), axis=3)
    normalized = normalize_image("dwi", data, mask, {"b0_index": baseline_candidates[0]})
    metadata = normalized.metadata()
    metadata["channel_names"] = [f"dwi-b{int(shell)}" for shell in policy.dwi_shells]
    metadata["shell_presence"] = shell_presence
    metadata["configured_shells"] = list(policy.dwi_shells)
    full = np.zeros((len(policy.dwi_shells), *normalized.data.shape[:3]), dtype=np.float32)
    for normalized_index, target_shell in enumerate(selected_targets):
        policy_index = policy.dwi_shells.index(target_shell)
        full[policy_index] = normalized.data[..., normalized_index]
    return full, sources, metadata


def _expected_channel_names(modality: str, policy: TensorPolicy) -> list[str]:
    if modality == "ct":
        return ["ct-brain", "ct-subdural", "ct-bone"]
    if modality == "dwi":
        return [f"dwi-b{int(shell)}" for shell in policy.dwi_shells]
    return [modality]


def _prepare_dynamic_image(
    modality: str,
    record: dict[str, Any],
    data: np.ndarray | None,
    destination: Path,
    policy: TensorPolicy,
) -> tuple[Path, dict[str, Any]]:
    source = Path(str(record["path"]))
    if data is None:
        data = np.asarray(nib.load(str(source)).dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"{modality} temporal windows require a 4D image")
    mask = _load_mask(record, data.shape[:3])
    normalized = normalize_image(
        modality,
        data,
        mask,
        _metadata(record),
        _load_optional_reference_mask(record, data.shape[:3]),
    )
    conformed, geometry = _crop_or_pad_time_last(normalized.data, policy.target_shape)
    windows, temporal_mask = _temporal_windows(
        np.moveaxis(conformed, -1, 0), policy.fmri_window, policy.fmri_stride
    )
    output = destination / f"{modality}_windows.npz"
    _save_npz(output, windows=windows, temporal_mask=temporal_mask)
    metadata = normalized.metadata()
    metadata.update(
        {
            "source": str(source),
            "window": policy.fmri_window,
            "stride": policy.fmri_stride,
            "windows": int(windows.shape[0]),
            "crop_pad": geometry,
        }
    )
    return output, metadata


def _prepare_sensor(
    record: dict[str, Any], modality: str, destination: Path, policy: TensorPolicy
) -> tuple[Path, dict[str, Any]]:
    source = Path(str(record["path"]))
    if not source.exists():
        raise FileNotFoundError(source)
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("Sensor tensor preparation requires the sensors extra") from exc
    raw = mne.io.read_raw_fif(source, preload=True, verbose="error")
    picks = mne.pick_types(raw.info, meg=True, eeg=True, seeg=True, ecog=True, exclude="bads")
    if len(picks) == 0:
        raise ValueError("No usable data channels remain after bad-channel exclusion")
    data = raw.get_data(picks=picks, reject_by_annotation="omit")
    channel_types = [raw.get_channel_types(picks=[index])[0] for index in picks]
    normalized = normalize_sensor(data, modality, channel_types)
    window = max(1, int(round(policy.sensor_window_seconds * raw.info["sfreq"])))
    stride = max(1, int(round(policy.sensor_stride_seconds * raw.info["sfreq"])))
    windows, temporal_mask = _temporal_windows(np.moveaxis(normalized.data, -1, 0), window, stride)
    windows = np.moveaxis(windows, -1, 1)
    output = destination / f"{modality}_windows.npz"
    _save_npz(
        output,
        windows=windows,
        temporal_mask=temporal_mask,
        channel_names=np.asarray([raw.ch_names[index] for index in picks], dtype="U"),
        sampling_frequency=np.asarray(float(raw.info["sfreq"]), dtype=np.float32),
    )
    metadata = normalized.metadata()
    metadata.update(
        {
            "source": str(source),
            "window_samples": window,
            "stride_samples": stride,
            "sampling_frequency_hz": float(raw.info["sfreq"]),
            "windows": int(windows.shape[0]),
        }
    )
    return output, metadata


def _crop_or_pad_channels(
    data: np.ndarray, target_shape: tuple[int, int, int], fill: float
) -> tuple[np.ndarray, dict[str, Any]]:
    if data.ndim != 4:
        raise ValueError("Static channel array must have shape CXYZ")
    result = np.full((data.shape[0], *target_shape), fill, dtype=np.float32)
    source_slices = [slice(None)]
    target_slices = [slice(None)]
    offsets = []
    for source_size, target_size in zip(data.shape[1:], target_shape, strict=True):
        if source_size >= target_size:
            start = (source_size - target_size) // 2
            source_slices.append(slice(start, start + target_size))
            target_slices.append(slice(0, target_size))
            offsets.append({"crop_start": start, "pad_start": 0})
        else:
            start = (target_size - source_size) // 2
            source_slices.append(slice(0, source_size))
            target_slices.append(slice(start, start + source_size))
            offsets.append({"crop_start": 0, "pad_start": start})
    result[tuple(target_slices)] = data[tuple(source_slices)]
    return result, {
        "source_shape": list(data.shape[1:]),
        "target_shape": list(target_shape),
        "axis_offsets": offsets,
    }


def _crop_or_pad_time_last(
    data: np.ndarray, target_shape: tuple[int, int, int]
) -> tuple[np.ndarray, dict[str, Any]]:
    channels = np.moveaxis(data, -1, 0)
    conformed, metadata = _crop_or_pad_channels(channels, target_shape, 0.0)
    return np.moveaxis(conformed, 0, -1), metadata


def _temporal_windows(data: np.ndarray, window: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    if data.ndim < 2 or window <= 0 or stride <= 0:
        raise ValueError("Invalid temporal window request")
    timepoints = data.shape[0]
    starts = list(range(0, max(1, timepoints - window + 1), stride))
    if timepoints >= window and starts[-1] != timepoints - window:
        starts.append(timepoints - window)
    windows = []
    masks = []
    for start in starts:
        chunk = data[start : start + window]
        mask = np.zeros(window, dtype=np.uint8)
        mask[: chunk.shape[0]] = 1
        if chunk.shape[0] < window:
            padded = np.zeros((window, *data.shape[1:]), dtype=np.float32)
            padded[: chunk.shape[0]] = chunk
            chunk = padded
        windows.append(chunk.astype(np.float32))
        masks.append(mask)
    return np.stack(windows), np.stack(masks)


def _load_mask(record: dict[str, Any], shape: Sequence[int]) -> np.ndarray:
    value = record.get("mask")
    if not value:
        raise ValueError("model tensor normalization requires a brain mask")
    mask = np.asarray(nib.load(str(value)).dataobj) > 0.5
    if mask.shape != tuple(shape[:3]):
        raise ValueError("brain mask does not share the aligned image grid")
    return mask


def _load_optional_reference_mask(
    record: dict[str, Any], shape: Sequence[int]
) -> np.ndarray | None:
    auxiliaries = record.get("auxiliaries")
    if not isinstance(auxiliaries, dict):
        return None
    value = auxiliaries.get("reference_mask") or auxiliaries.get("qsm_reference_mask")
    if not value or not Path(str(value)).exists():
        return None
    mask = np.asarray(nib.load(str(value)).dataobj) > 0.5
    if mask.shape != tuple(shape[:3]):
        raise ValueError("reference mask does not share the image grid")
    return mask


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata")
    if not value or not Path(str(value)).exists():
        return {}
    return _read_object(Path(str(value)))


def _union_mask(shape: Sequence[int]) -> np.ndarray:
    return np.ones(tuple(shape), dtype=bool)


def _save_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
