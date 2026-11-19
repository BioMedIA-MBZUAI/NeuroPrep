from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import DatasetConfig


class ConfigError(ValueError):
    """Raised when a NeuroPrep configuration is invalid."""


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
    except OSError as exc:
        raise ConfigError(f"Could not read configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    return payload


def _expand_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_dataset_configs(path: Path) -> list[DatasetConfig]:
    payload = load_yaml(path)
    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ConfigError("Configuration must contain a non-empty 'datasets' list")

    base = path.parent.resolve()
    configs: list[DatasetConfig] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_datasets):
        if not isinstance(raw, dict):
            raise ConfigError(f"datasets[{index}] must be a mapping")
        dataset_id = str(raw.get("id", "")).strip()
        if not dataset_id:
            raise ConfigError(f"datasets[{index}] is missing 'id'")
        if dataset_id in seen:
            raise ConfigError(f"Duplicate dataset id: {dataset_id}")
        seen.add(dataset_id)
        if "root" not in raw:
            raise ConfigError(f"Dataset {dataset_id} is missing 'root'")

        root = _expand_path(raw["root"], base)
        input_layout = str(raw.get("input_layout", "bids")).lower()
        derivatives = _expand_path(
            raw.get("derivatives_root", root / "derivatives" / "neuroprep"), base
        )
        work = _expand_path(raw.get("work_root", derivatives / "work"), base)
        bids_root = _expand_path(
            raw.get("bids_root", root if input_layout == "bids" else derivatives / "bids"), base
        )
        exclude = tuple(str(item) for item in raw.get("exclude", ("derivatives", ".git", "logs")))
        configs.append(
            DatasetConfig(
                dataset_id=dataset_id,
                root=root,
                bids_root=bids_root,
                derivatives_root=derivatives,
                work_root=work,
                adapter=str(raw.get("adapter", "bids")).lower(),
                input_layout=input_layout,
                subject_regex=raw.get("subject_regex"),
                session_regex=raw.get("session_regex"),
                exclude=exclude,
                metadata=dict(raw.get("metadata", {})),
            )
        )
    return configs


def select_dataset(configs: list[DatasetConfig], dataset_id: str | None) -> DatasetConfig:
    if dataset_id is None:
        if len(configs) != 1:
            available = ", ".join(config.dataset_id for config in configs)
            raise ConfigError(f"Select a dataset with --dataset-id. Available: {available}")
        return configs[0]
    for config in configs:
        if config.dataset_id == dataset_id:
            return config
    raise ConfigError(f"Unknown dataset id: {dataset_id}")
