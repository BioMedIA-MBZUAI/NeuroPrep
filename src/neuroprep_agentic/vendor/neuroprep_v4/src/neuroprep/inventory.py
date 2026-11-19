from __future__ import annotations

import os
from pathlib import Path

from .adapters import get_adapter
from .models import DatasetConfig, Inventory, Modality


_IMAGE_SUFFIXES = (".nii", ".nii.gz")
_ELECTROPHYS_SUFFIXES = (".edf", ".bdf", ".vhdr", ".set", ".fif", ".con", ".sqd", ".nwb", ".mefd")
_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


def _is_image(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(suffix) for suffix in _IMAGE_SUFFIXES)


def inventory_dataset(config: DatasetConfig) -> Inventory:
    if not config.root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {config.root}")

    adapter = get_adapter(config)
    images = []
    excluded = set(config.exclude)
    dicom_files = 0
    archive_files = 0

    for directory, dirnames, filenames in os.walk(config.root, followlinks=False):
        dataset_directories = [name for name in dirnames if name.lower().endswith(".ds")]
        for name in dataset_directories:
            images.append(adapter.image_record(Path(directory) / name, image_format="electrophysiology"))
        dirnames[:] = [name for name in dirnames if name not in dataset_directories]
        dirnames[:] = [name for name in dirnames if name not in excluded]
        base = Path(directory)
        for filename in filenames:
            if _is_image(filename):
                images.append(adapter.image_record(base / filename))
            elif filename.lower().endswith(_ELECTROPHYS_SUFFIXES):
                # BrainVision .eeg/.vmrk companions are intentionally not indexed
                # separately; the .vhdr file is the primary BIDS recording.
                images.append(
                    adapter.image_record(base / filename, image_format="electrophysiology")
                )
            elif filename.lower().endswith(_ARCHIVE_SUFFIXES):
                images.append(adapter.image_record(base / filename, image_format="archive"))
                archive_files += 1
            elif filename.lower().endswith((".dcm", ".ima")):
                dicom_files += 1

    images.sort(key=lambda image: image.path.as_posix())
    warnings: list[str] = []
    unknown_subjects = sum(image.subject == "__unknown__" for image in images)
    unknown_modalities = sum(image.modality is Modality.UNKNOWN for image in images)
    if unknown_subjects:
        warnings.append(f"{unknown_subjects} images could not be assigned to a subject")
    if unknown_modalities:
        warnings.append(f"{unknown_modalities} images have an unknown modality")
    if dicom_files and not images:
        warnings.append(
            f"Found {dicom_files} DICOM-like files and no NIfTI images; an ingest task is required"
        )
    if archive_files:
        warnings.append(f"Found {archive_files} imaging archives; ingest must extract and curate them")
    if not images and not dicom_files:
        warnings.append("No supported image files were found")

    return Inventory(dataset_id=config.dataset_id, root=config.root, images=images, warnings=warnings)
