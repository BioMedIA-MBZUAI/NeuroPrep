from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .experts import ExpertRegistry
from .inventory import inventory_dataset
from .models import DatasetConfig


DATATYPE_BY_MODALITY = {
    "t1w": "anat",
    "t2w": "anat",
    "flair": "anat",
    "mra": "anat",
    "swi": "anat",
    "fmri": "func",
    "dwi": "dwi",
    "fieldmap": "fmap",
    "asl": "perf",
    "pet": "pet",
}
COMPANION_EXTENSIONS = (".json", ".bval", ".bvec", "_events.tsv", "_aslcontext.tsv")
ROOT_DATASET_FILES = (
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "README",
    "CHANGES",
)


def read_selection(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"subject", "session", "modality", "selection_role", "source_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Selection manifest is missing columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def stage_selection(
    selection_path: Path,
    output_root: Path,
    modalities: Iterable[str],
    roles: Iterable[str] = ("primary-reference", "include"),
    execute: bool = False,
    link_mode: str = "hardlink",
    dataset_name: str = "NeuroPrep selected processing view",
    primary_only_modalities: Iterable[str] = (),
) -> dict[str, Any]:
    if link_mode not in {"hardlink", "symlink", "copy"}:
        raise ValueError("link_mode must be hardlink, symlink, or copy")
    modality_set = {value.lower() for value in modalities}
    primary_only_set = {value.lower() for value in primary_only_modalities}
    role_set = set(roles)
    unsupported = modality_set - set(DATATYPE_BY_MODALITY)
    if unsupported:
        raise ValueError(f"Unsupported staging modalities: {sorted(unsupported)}")
    invalid_primary_only = primary_only_set - modality_set
    if invalid_primary_only:
        raise ValueError(
            "Primary-only modalities must also be staged modalities: "
            f"{sorted(invalid_primary_only)}"
        )

    selected = [
        row
        for row in read_selection(selection_path)
        if row["modality"] in modality_set
        and row["selection_role"] in role_set
        and (
            row["modality"] not in primary_only_set
            or row["selection_role"] == "primary-reference"
        )
        and row.get("eligibility") == "eligible"
    ]
    actions: list[tuple[Path, Path]] = []
    for row in selected:
        source = Path(row["source_path"])
        if not source.exists():
            raise FileNotFoundError(f"Selected source does not exist: {source}")
        subject = _bids_label(row["subject"])
        session = _session_label(row["session"], source)
        datatype = DATATYPE_BY_MODALITY[row["modality"]]
        filename = _bids_filename(source.name, subject, session, row["modality"])
        destination = output_root / f"sub-{subject}" / f"ses-{session}" / datatype / filename
        actions.append((source, destination))
        actions.extend(_companion_actions(source, destination))

    _validate_destinations(actions, output_root)
    if execute:
        for source, destination in actions:
            _materialize(source, destination, link_mode)
        for _, destination in actions:
            if destination.suffix.lower() == ".json":
                _curate_sidecar(destination)
        _write_dataset_files(output_root, selected, dataset_name)
        _write_provenance(output_root, actions)

    return {
        "selection_manifest": str(selection_path),
        "output_root": str(output_root),
        "execute": execute,
        "link_mode": link_mode,
        "selected_images": len(selected),
        "materialized_files": len(actions),
        "subjects": len({row["subject"] for row in selected}),
        "sessions": len({(row["subject"], row["session"]) for row in selected}),
        "modality_counts": dict(sorted(Counter(row["modality"] for row in selected).items())),
        "role_counts": dict(sorted(Counter(row["selection_role"] for row in selected).items())),
        "primary_only_modalities": sorted(primary_only_set),
    }


def write_subject_list(bids_root: Path, datatype: str, output: Path) -> int:
    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS root does not exist: {bids_root}")
    subjects = sorted(
        {
            subject.name.removeprefix("sub-")
            for subject in bids_root.glob("sub-*")
            if subject.is_dir() and any(subject.glob(f"ses-*/{datatype}/*"))
        }
    )
    _atomic_text(output, "".join(f"{subject}\n" for subject in subjects))
    return len(subjects)


def stage_visit(
    bids_root: Path,
    output_root: Path,
    subject: str,
    visit_id: str,
    marker: Path,
    tolerance_days: int = 30,
    link_mode: str = "hardlink",
    primary_t1: Path | None = None,
) -> dict[str, Any]:
    """Create an immutable, one-visit BIDS execution view.

    Some upstream applications accept a participant selector but not a session
    selector.  A hard-linked visit view prevents them from mixing longitudinal
    acquisitions while preserving filenames, metadata, and source bytes.
    """

    if link_mode not in {"hardlink", "symlink", "copy"}:
        raise ValueError("link_mode must be hardlink, symlink, or copy")
    bids_root = bids_root.resolve()
    subject_token = subject.removeprefix("sub-")
    inventory = inventory_dataset(
        DatasetConfig(
            dataset_id="visit-staging",
            root=bids_root,
            bids_root=bids_root,
            derivatives_root=output_root / ".derivatives-unused",
            work_root=output_root / ".work-unused",
        )
    )
    contexts = ExpertRegistry().visit_contexts(
        [
            image
            for image in inventory.images
            if image.subject.removeprefix("sub-") == subject_token
        ],
        tolerance_days=tolerance_days,
    )
    matches = [context for context in contexts if context.visit_id == visit_id]
    if len(matches) != 1:
        observed = [context.visit_id for context in contexts]
        raise ValueError(
            f"Expected one visit {visit_id!r} for sub-{subject_token}; observed {observed}"
        )
    context = matches[0]
    sources: set[Path] = set()
    primary_t1_resolved = primary_t1.resolve() if primary_t1 else None
    if primary_t1_resolved is not None and not any(
        image.modality.value == "t1w"
        and image.path.resolve() == primary_t1_resolved
        for image in context.images
    ):
        raise ValueError(
            f"Primary T1w is not part of sub-{subject_token} visit {visit_id}: "
            f"{primary_t1}"
        )
    for image in context.images:
        if (
            primary_t1_resolved is not None
            and image.modality.value == "t1w"
            and image.path.resolve() != primary_t1_resolved
        ):
            continue
        sources.update(_record_files(image.path, image.sidecar))

    subject_root = bids_root / f"sub-{subject_token}"
    if subject_root.is_dir():
        sources.update(path for path in subject_root.glob("*") if path.is_file())
    for session in {
        image.session.removeprefix("ses-")
        for image in context.images
        if image.session
    }:
        session_root = subject_root / f"ses-{session}"
        sources.update(path for path in session_root.glob("*") if path.is_file())
    sources.update(
        bids_root / name
        for name in ROOT_DATASET_FILES
        if (bids_root / name).is_file()
    )
    if not sources:
        raise FileNotFoundError(
            f"No files selected for sub-{subject_token} visit {visit_id}"
        )

    actions: list[tuple[Path, Path]] = []
    for source in sorted(sources):
        try:
            relative = source.resolve().relative_to(bids_root)
        except ValueError as exc:
            raise ValueError(f"Visit input escapes BIDS root: {source}") from exc
        actions.append((source, output_root / relative))
    _validate_destinations(actions, output_root)
    for source, destination in actions:
        _materialize(source, destination, link_mode)
    summary = {
        "status": "pass",
        "subject": subject_token,
        "visit_id": visit_id,
        "visit_index": context.visit_index,
        "visit_count": context.visit_count,
        "acquisition_mode": context.acquisition_mode,
        "acquisition_time": context.acquisition_time,
        "modalities": sorted({image.modality.value for image in context.images}),
        "source_bids_root": str(bids_root),
        "output_root": str(output_root),
        "link_mode": link_mode,
        "primary_t1": str(primary_t1_resolved) if primary_t1_resolved else None,
        "files": len(actions),
    }
    _atomic_text(marker, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def write_visit_execution_index(
    roots: Iterable[Path],
    marker: Path,
    subject: str,
    pipeline: str,
) -> dict[str, Any]:
    """Verify every visit executor output and write a subject-level barrier."""

    resolved = [Path(root).resolve() for root in roots]
    missing = [str(root) for root in resolved if not root.exists()]
    if missing:
        raise FileNotFoundError(f"Visit executor outputs are missing: {missing}")
    summary = {
        "status": "pass",
        "subject": subject.removeprefix("sub-"),
        "pipeline": pipeline,
        "visit_count": len(resolved),
        "visit_outputs": [str(root) for root in resolved],
    }
    _atomic_text(marker, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _record_files(path: Path, sidecar: Path | None) -> set[Path]:
    files = {path}
    if sidecar and sidecar.is_file():
        files.add(sidecar)
    name = path.name
    stem = name[:-7] if name.endswith(".nii.gz") else path.stem
    prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
    files.update(candidate for candidate in path.parent.glob(prefix + "_*") if candidate.is_file())
    return files


def _companion_actions(source: Path, destination: Path) -> list[tuple[Path, Path]]:
    source_stem = source.name[:-7] if source.name.endswith(".nii.gz") else source.stem
    destination_stem = (
        destination.name[:-7] if destination.name.endswith(".nii.gz") else destination.stem
    )
    actions: list[tuple[Path, Path]] = []
    for extension in COMPANION_EXTENSIONS:
        candidate = source.parent / f"{source_stem}{extension}"
        if candidate.exists():
            actions.append((candidate, destination.parent / f"{destination_stem}{extension}"))
    return actions


def _bids_filename(name: str, subject: str, session: str, modality: str) -> str:
    renamed = re.sub(r"^sub-[A-Za-z0-9]+", f"sub-{subject}", name)
    renamed = re.sub(r"_sess-[A-Za-z0-9]+", f"_ses-{session}", renamed)
    if f"_ses-{session}" not in renamed:
        renamed = renamed.replace(f"sub-{subject}_", f"sub-{subject}_ses-{session}_", 1)

    # OASIS legacy exports contain several scientifically meaningful NIfTI
    # series whose names pre-date BIDS.  Normalize only the staged processing
    # view; source names and bytes remain immutable.
    if modality == "t1w":
        renamed = re.sub(r"_echo-([0-9]+)(?=_)", r"_acq-echo\1", renamed)
        renamed = re.sub(r"_TSE(?=\.nii(?:\.gz)?$)", "_acq-TSE_T1w", renamed)
        renamed = re.sub(r"_run-([0-9]+)_acq-([A-Za-z0-9]+)", r"_acq-\2_run-\1", renamed)
    elif modality == "swi":
        renamed = re.sub(r"_part-phase(?P<run>_run-[0-9]+)?_GRE_ph(?=\.nii(?:\.gz)?$)", r"\g<run>_part-phase_T2starw", renamed)
        renamed = re.sub(r"_part-mag(?P<run>_run-[0-9]+)?_GRE(?=\.nii(?:\.gz)?$)", r"\g<run>_part-mag_T2starw", renamed)
        renamed = re.sub(r"_swi(?=\.nii(?:\.gz)?$)", "_rec-SWI_T2starw", renamed)
        renamed = re.sub(r"_minIP(?=\.nii(?:\.gz)?$)", "_rec-minIP_T2starw", renamed)
        renamed = re.sub(r"_run-([0-9]+)_rec-([A-Za-z0-9]+)", r"_rec-\2_run-\1", renamed)
    elif modality == "mra":
        renamed = re.sub(r"_echo-[0-9]+", "", renamed)
        renamed = re.sub(
            r"_run-([0-9]+)_angio_i([0-9]+)(?=\.nii(?:\.gz)?$)",
            r"_rec-i\2_run-\1_angio",
            renamed,
        )
    elif modality == "fieldmap":
        renamed = re.sub(
            r"_echo-1_run-01_fieldmap(?=\.nii(?:\.gz)?$)",
            "_run-01_magnitude1",
            renamed,
        )
        renamed = re.sub(
            r"_echo-2_run-01_fieldmap(?=\.nii(?:\.gz)?$)",
            "_run-01_magnitude2",
            renamed,
        )
        renamed = re.sub(
            r"_run-02_fieldmap_e2_ph(?=\.nii(?:\.gz)?$)",
            "_run-01_phasediff",
            renamed,
        )
    return renamed


def _session_label(value: str, source: Path) -> str:
    if match := re.search(r"_sess-([A-Za-z0-9]+)", source.name):
        return _bids_label(match.group(1))
    if match := re.search(r"_(?:MR|PET)_([A-Za-z0-9]+)$", value):
        return _bids_label(match.group(1))
    return _bids_label(value)


def _bids_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9]+", "", value)
    if not label:
        raise ValueError(f"Cannot create a BIDS label from: {value!r}")
    return label


def _validate_destinations(actions: list[tuple[Path, Path]], output_root: Path) -> None:
    resolved_root = output_root.resolve()
    seen: dict[Path, Path] = {}
    for source, destination in actions:
        resolved_destination = destination.resolve(strict=False)
        if resolved_root not in resolved_destination.parents:
            raise ValueError(f"Staging destination escapes output root: {destination}")
        previous = seen.get(destination)
        if previous is not None and previous != source:
            raise ValueError(
                f"Two sources map to the same BIDS destination: {previous}, {source} -> {destination}"
            )
        seen[destination] = source


def _materialize(source: Path, destination: Path, link_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if _same_file(source, destination):
            return
        raise FileExistsError(f"Refusing to replace staged file: {destination}")
    if link_mode == "hardlink":
        os.link(source, destination)
    elif link_mode == "symlink":
        destination.symlink_to(source)
    else:
        import shutil

        shutil.copy2(source, destination)


def _same_file(source: Path, destination: Path) -> bool:
    try:
        return source.samefile(destination)
    except OSError:
        return False


def _write_dataset_files(root: Path, rows: list[dict[str, str]], name: str) -> None:
    payload = {
        "Name": name,
        "BIDSVersion": "1.11.1",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "NeuroPrep", "Version": "0.1.0"}],
        "SourceDatasets": [{"URL": "https://www.oasis-brains.org/"}],
    }
    _atomic_text(root / "dataset_description.json", json.dumps(payload, indent=2) + "\n")
    participants = sorted({_bids_label(row["subject"]) for row in rows})
    participant_text = "participant_id\n" + "".join(f"sub-{value}\n" for value in participants)
    _atomic_text(root / "participants.tsv", participant_text)
    _atomic_text(
        root / "README",
        "Curated processing view generated by NeuroPrep. Source files remain immutable.\n",
    )


def _write_provenance(root: Path, actions: list[tuple[Path, Path]]) -> None:
    path = root / "sourcedata" / "neuroprep_staging.tsv"
    rows = [
        f"{source}\t{destination.relative_to(root)}\t{_actual_mode(source, destination)}\n"
        for source, destination in actions
    ]
    _atomic_text(path, "source_path\tbids_path\tmode\n" + "".join(rows))


def _curate_sidecar(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    changed = False
    if match := re.search(r"_task-([A-Za-z0-9+]+).*_bold\.json$", path.name):
        if not payload.get("TaskName"):
            payload["TaskName"] = match.group(1)
            changed = True
    if re.search(r"_(?:magnitude[12]|phasediff)\.json$", path.name):
        if payload.get("B0FieldIdentifier") != "fmap1":
            payload["B0FieldIdentifier"] = "fmap1"
            changed = True
    if path.name.endswith("_part-phase_T2starw.json") and not payload.get("Units"):
        # DICOM-converted phase values in these legacy exports have no physical
        # unit declaration.  BIDS explicitly permits the honest "arbitrary"
        # value, while downstream QSM policy can still require reviewed scaling.
        payload["Units"] = "arbitrary"
        changed = True
    if path.name.endswith("_phasediff.json"):
        echo_times = _fieldmap_echo_times(path.parent, path.name.replace("_phasediff.json", ""))
        if echo_times:
            if payload.get("EchoTime1") != echo_times[0]:
                payload["EchoTime1"] = echo_times[0]
                changed = True
            if payload.get("EchoTime2") != echo_times[1]:
                payload["EchoTime2"] = echo_times[1]
                changed = True
        for key in ("EchoTime", "EchoNumber", "Units"):
            if key in payload:
                del payload[key]
                changed = True
        intended_for = _fieldmap_targets(path)
        if intended_for and payload.get("IntendedFor") != intended_for:
            payload["IntendedFor"] = intended_for
            changed = True
    if re.search(r"_(?:bold|dwi)\.json$", path.name):
        fmap_dir = path.parent.parent / "fmap"
        if any(fmap_dir.glob("*_phasediff.nii.gz")) and payload.get("B0FieldSource") != "fmap1":
            payload["B0FieldSource"] = "fmap1"
            changed = True
    if changed:
        _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _fieldmap_echo_times(directory: Path, prefix: str) -> tuple[float, float] | None:
    values: list[float] = []
    for suffix in ("_magnitude1.json", "_magnitude2.json"):
        candidate = directory / f"{prefix}{suffix}"
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            values.append(float(payload["EchoTime"]))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
    return values[0], values[1]


def _fieldmap_targets(path: Path) -> list[str]:
    session_dir = path.parent.parent
    targets: list[str] = []
    for datatype, pattern in (("func", "*_bold.nii.gz"), ("dwi", "*_dwi.nii.gz")):
        for candidate in sorted((session_dir / datatype).glob(pattern)):
            targets.append(
                f"bids::{session_dir.parent.name}/{session_dir.name}/{datatype}/{candidate.name}"
            )
    return targets


def _actual_mode(source: Path, destination: Path) -> str:
    if destination.is_symlink():
        return "symlink"
    try:
        return "hardlink" if source.samefile(destination) else "copy"
    except OSError:
        return "unknown"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
