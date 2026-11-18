from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from . import __version__, paths
from . import settings as _settings
from .execution_models import ExecutableTask
from .settings import validated_cluster_path

PHI_KEYS = {
    "AccessionNumber",
    "AcquisitionDate",
    "AcquisitionDateTime",
    "AcquisitionTime",
    "ContentDate",
    "ContentTime",
    "OperatorsName",
    "PatientAddress",
    "PatientAge",
    "PatientBirthDate",
    "PatientID",
    "PatientName",
    "PatientSex",
    "PatientTelephoneNumbers",
    "PerformingPhysicianName",
    "ReferringPhysicianName",
    "SeriesDate",
    "SeriesInstanceUID",
    "SeriesTime",
    "SOPInstanceUID",
    "StudyDate",
    "StudyID",
    "StudyInstanceUID",
    "StudyTime",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    if source.name.endswith(".nii") and target.name.endswith(".nii.gz"):
        with source.open("rb") as source_stream, gzip.open(temporary, "wb") as target_stream:
            shutil.copyfileobj(source_stream, target_stream)
    else:
        shutil.copy2(source, temporary)
    temporary.replace(target)


def _sidecar_path(path: Path) -> Path:
    if path.name.endswith(".nii.gz"):
        return path.with_name(path.name[:-7] + ".json")
    return path.with_suffix(".json")


def _safe_sidecar(payload: dict[str, Any]) -> dict[str, Any]:
    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: sanitize(item)
                for key, item in value.items()
                if key not in PHI_KEYS
                and not (key.startswith("Patient") and key != "PatientPosition")
            }
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        return value

    return sanitize(payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _paired_sidecars(source: Path, target: Path, plugin_id: str = "") -> None:
    source_stem = source.name[:-7] if source.name.endswith(".nii.gz") else source.stem
    target_stem = target.name[:-7] if target.name.endswith(".nii.gz") else target.stem
    extensions = (".bval", ".bvec", ".tsv", ".tsv.gz")
    if plugin_id in {"pet", "mri.asl"}:
        extensions = (".bval", ".bvec")
    for extension in extensions:
        candidate = source.with_name(source_stem + extension)
        if candidate.is_file():
            _atomic_copy(candidate, target.with_name(target_stem + extension))


def stage_bids(task: ExecutableTask) -> dict[str, Any]:
    source = validated_cluster_path(str(task.metadata["source_path"]), must_exist=True)
    target = validated_cluster_path(str(task.metadata["bids_path"]))
    plugin_id = str(task.metadata.get("selected_plugin_id") or "")
    if plugin_id == "mri.mrs" and (
        source.is_dir() or not source.name.lower().endswith((".nii", ".nii.gz"))
    ):
        return _stage_mrs_source(source, target, task)
    if plugin_id == "pet" and source.suffix.lower() == ".ecat":
        return _stage_nifti_convert(source, target, task)
    if (
        plugin_id in {"behavior", "physiology"}
        and target.suffix == ".tsv"
        and source.suffix.lower() in {".csv", ".json"}
    ):
        return _stage_tabular_source(source, target, task)
    if source.is_dir() or source.suffix.lower() == ".dcm":
        result = _stage_dicom(source, target, task)
        sidecar_path = _sidecar_path(target)
        sidecar: dict[str, Any] = {}
        if sidecar_path.is_file():
            loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"Converted sidecar is not a JSON object: {sidecar_path}")
            sidecar.update(_safe_sidecar(loaded))
        sidecar.update(_safe_sidecar(dict(task.metadata.get("source_metadata") or {})))
        sidecar.update(_safe_sidecar(dict(task.metadata.get("bids_sidecar") or {})))
        sidecar.update(
            {
                "SourceAcquisitionID": task.acquisition_ids[0],
                "NeuroPrepAgenticVersion": __version__,
            }
        )
        if plugin_id == "mri.asl":
            result["asl_context"] = _derive_asl_context(target, sidecar)
            sidecar["TotalAcquiredPairs"] = result["asl_context"]["pairs"]
        _write_json(sidecar_path, sidecar)
        return result
    if not source.is_file():
        raise ValueError(f"BIDS staging requires a regular file or DICOM directory: {source}")
    _atomic_copy(source, target)
    _paired_sidecars(source, target, plugin_id)

    source_sidecar = _sidecar_path(source)
    sidecar: dict[str, Any] = {}
    if source_sidecar.is_file():
        loaded = json.loads(source_sidecar.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Source sidecar is not a JSON object: {source_sidecar}")
        sidecar.update(_safe_sidecar(loaded))
    sidecar.update(_safe_sidecar(dict(task.metadata.get("source_metadata") or {})))
    sidecar.update(_safe_sidecar(dict(task.metadata.get("bids_sidecar") or {})))
    sidecar.update(
        {
            "SourceAcquisitionID": task.acquisition_ids[0],
            "NeuroPrepAgenticVersion": __version__,
        }
    )
    if target.name.endswith((".nii", ".nii.gz")) or sidecar:
        _write_json(_sidecar_path(target), sidecar)
    result = {
        "status": "pass",
        "source": str(source),
        "target": str(target),
        "size": target.stat().st_size,
        "sha256": _sha256(target),
    }
    if plugin_id == "mri.asl":
        result["asl_context"] = _derive_asl_context(target, sidecar)
        sidecar["TotalAcquiredPairs"] = result["asl_context"]["pairs"]
        _write_json(_sidecar_path(target), sidecar)
    return result


def _derive_asl_context(target: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    import nibabel as nib
    import numpy as np

    image = nib.load(target)
    if len(image.shape) != 4 or image.shape[3] < 2:
        raise ValueError("ASL context derivation requires a multi-volume 4D NIfTI")
    volume_count = image.shape[3]
    included_m0 = metadata.get("M0Type") == "Included"
    offset = 1 if included_m0 else 0
    if (volume_count - offset) < 2 or (volume_count - offset) % 2:
        raise ValueError("ASL series is not an M0 plus complete control-label pairs")
    data = np.asanyarray(image.dataobj, dtype=np.float64)
    means = data.reshape((-1, volume_count)).mean(axis=0)
    first = means[offset::2]
    second = means[offset + 1 :: 2]
    paired_delta = second - first
    standard_error = paired_delta.std(ddof=1) / np.sqrt(len(paired_delta))
    if not np.isfinite(standard_error):
        raise ValueError("ASL control-label order cannot be established from signal evidence")
    if standard_error == 0:
        if paired_delta.mean() == 0:
            raise ValueError("ASL control-label order cannot be established from signal evidence")
        t_statistic = float(np.copysign(np.inf, paired_delta.mean()))
    else:
        t_statistic = float(paired_delta.mean() / standard_error)
    if abs(t_statistic) < 3.0:
        raise ValueError("ASL control-label order lacks sufficient signal evidence")
    first_type, second_type = (
        ("label", "control") if paired_delta.mean() > 0 else ("control", "label")
    )
    rows = (["m0scan"] if included_m0 else []) + [
        item
        for _ in range((volume_count - offset) // 2)
        for item in (first_type, second_type)
    ]
    stem = target.name[:-7] if target.name.endswith(".nii.gz") else target.stem
    if not stem.endswith("_asl"):
        raise ValueError(f"ASL target lacks _asl suffix: {target}")
    context = target.with_name(stem[:-4] + "_aslcontext.tsv")
    temporary = context.with_suffix(context.suffix + ".tmp")
    temporary.write_text("volume_type\n" + "\n".join(rows) + "\n", encoding="utf-8")
    temporary.replace(context)
    return {
        "path": str(context),
        "volumes": volume_count,
        "pairs": (volume_count - offset) // 2,
        "first_pair": [first_type, second_type],
        "order_t_statistic": t_statistic,
    }


def _stage_tabular_source(source: Path, target: Path, task: ExecutableTask) -> dict[str, Any]:
    import pandas as pd

    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        frame = pd.json_normalize(payload if isinstance(payload, list) else [payload])
    if frame.empty or not len(frame.columns):
        raise ValueError(f"Tabular source is empty: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False, na_rep="n/a")
    temporary.replace(target)
    sidecar = {
        **_safe_sidecar(dict(task.metadata.get("source_metadata") or {})),
        **_safe_sidecar(dict(task.metadata.get("bids_sidecar") or {})),
        "SourceAcquisitionID": task.acquisition_ids[0],
        "NeuroPrepAgenticVersion": __version__,
        "ConversionSoftware": "pandas",
    }
    _write_json(_sidecar_path(target), sidecar)
    return {
        "status": "pass",
        "source": str(source),
        "target": str(target),
        "rows": len(frame),
        "columns": list(map(str, frame.columns)),
        "size": target.stat().st_size,
        "sha256": _sha256(target),
    }


def _stage_nifti_convert(source: Path, target: Path, task: ExecutableTask) -> dict[str, Any]:
    import nibabel as nib
    import numpy as np

    image = nib.load(source)
    data = np.asanyarray(image.dataobj)
    if data.size == 0 or not np.isfinite(data).any():
        raise ValueError(f"Source image has no finite voxels: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.as_closest_canonical(image), target)
    _write_json(
        _sidecar_path(target),
        {
            **_safe_sidecar(dict(task.metadata.get("source_metadata") or {})),
            **_safe_sidecar(dict(task.metadata.get("bids_sidecar") or {})),
            "SourceAcquisitionID": task.acquisition_ids[0],
            "NeuroPrepAgenticVersion": __version__,
            "ConversionSoftware": "nibabel",
        },
    )
    return {
        "status": "pass",
        "source": str(source),
        "target": str(target),
        "size": target.stat().st_size,
        "sha256": _sha256(target),
    }


def _stage_mrs_source(source: Path, target: Path, task: ExecutableTask) -> dict[str, Any]:
    configured_root = str(paths.fsl_mrs_root())
    tool_root = validated_cluster_path(configured_root)
    candidate = tool_root / "bin/spec2nii"
    # Preflight normally verifies the pinned executable. Keeping a command-name
    # fallback here makes the native function testable with an injected runner
    # while still failing clearly when a real process is launched without the tool.
    spec2nii = candidate if candidate.is_file() else Path(shutil.which("spec2nii") or "spec2nii")
    target.parent.mkdir(parents=True, exist_ok=True)
    stem = target.name[:-7] if target.name.endswith(".nii.gz") else target.stem
    subcommand = "dicom" if source.is_dir() or source.suffix.lower() == ".dcm" else "auto"
    command = [
        str(spec2nii),
        subcommand,
        "-j",
        "--anon",
        "-f",
        stem,
        "-o",
        str(target.parent),
    ]
    metadata = task.metadata.get("source_metadata") or {}
    if isinstance(metadata, dict):
        nucleus = metadata.get("ResonantNucleus")
        frequency = metadata.get("SpectrometerFrequency")
        if nucleus:
            values = nucleus if isinstance(nucleus, list) else [nucleus]
            command.extend(["--override_nucleus", *map(str, values)])
        if frequency:
            values = frequency if isinstance(frequency, list) else [frequency]
            command.extend(["--override_frequency", *map(str, values)])
    command.append(str(source))
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{tool_root / 'bin'}:/usr/bin:/bin",
            "FSLDIR": str(tool_root),
            "FSLOUTPUTTYPE": "NIFTI_GZ",
            "MPLCONFIGDIR": str(Path(task.work_dir) / "mplconfig"),
            "XDG_CACHE_HOME": str(Path(task.work_dir) / "xdg-cache"),
        }
    )
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(f"spec2nii failed for {task.task_id} with {completed.returncode}")
    if not target.is_file():
        candidates = sorted(target.parent.glob(f"{stem}*.nii.gz"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"spec2nii produced {len(candidates)} candidate NIfTI-MRS files for {task.task_id}"
            )
        candidates[0].replace(target)
    if target.stat().st_size == 0:
        raise RuntimeError("spec2nii produced an empty NIfTI-MRS file")
    generated_sidecar = _sidecar_path(target)
    sidecar: dict[str, Any] = {}
    if generated_sidecar.is_file():
        loaded = json.loads(generated_sidecar.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"spec2nii sidecar is not an object: {generated_sidecar}")
        sidecar.update(_safe_sidecar(loaded))
    sidecar.update(_safe_sidecar(dict(task.metadata.get("bids_sidecar") or {})))
    sidecar.update(
        {
            "SourceAcquisitionID": task.acquisition_ids[0],
            "NeuroPrepAgenticVersion": __version__,
            "ConversionSoftware": "spec2nii",
            "ConversionSoftwareVersion": "0.8.15",
        }
    )
    _write_json(generated_sidecar, sidecar)
    return {
        "status": "pass",
        "source": str(source),
        "target": str(target),
        "size": target.stat().st_size,
        "sha256": _sha256(target),
        "converter": "spec2nii 0.8.15",
    }


def _stage_dicom(source: Path, target: Path, task: ExecutableTask) -> dict[str, Any]:
    configured = str(
        task.metadata.get("dcm2niix_executable")
        or os.environ.get("NEUROPREP_DCM2NIIX", "")
    )
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        [
            paths.dcm2niix_path(),
            Path("/usr/bin/dcm2niix"),
        ]
    )
    executable = next((candidate for candidate in candidates if candidate.is_file()), None)
    if executable is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"dcm2niix executable is missing; searched: {searched}")
    if source.is_file():
        raise ValueError(
            "A DICOM SourceRecord must identify a complete acquisition directory, not one file"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    stem = target.name[:-7] if target.name.endswith(".nii.gz") else target.stem
    command = [
        str(executable),
        "-b",
        "y",
        "-ba",
        "y",
        "-z",
        "y",
        "-f",
        stem,
        "-o",
        str(target.parent),
        str(source),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0 or not target.is_file():
        raise RuntimeError(f"dcm2niix failed for {task.task_id} with {completed.returncode}")
    plugin_id = str(task.metadata.get("selected_plugin_id") or "")
    if plugin_id.startswith("mri.") and plugin_id != "mri.mrs":
        import nibabel as nib

        shape = nib.load(target).shape
        if len(shape) < 3 or any(size <= 1 for size in shape[:3]):
            for generated in (
                target,
                _sidecar_path(target),
                target.with_name(stem + ".bval"),
                target.with_name(stem + ".bvec"),
            ):
                generated.unlink(missing_ok=True)
            raise ValueError(
                f"Converted MRI output has invalid spatial shape {shape}; no artifact released"
            )
    sidecar_path = _sidecar_path(target)
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        _write_json(sidecar_path, _safe_sidecar(sidecar))
    return {"status": "pass", "target": str(target), "size": target.stat().st_size}


def _bids_init(task: ExecutableTask) -> dict[str, Any]:
    bids_root = validated_cluster_path(task.output_root)
    bids_root.mkdir(parents=True, exist_ok=True)
    description = bids_root / "dataset_description.json"
    _write_json(
        description,
        {
            "Name": str(task.metadata.get("dataset_id") or "NeuroPrep dataset"),
            "BIDSVersion": str(task.metadata.get("bids_version") or "1.11.1"),
            "DatasetType": "raw",
            "GeneratedBy": [
                {
                    "Name": "NeuroPrep-agentic",
                    "Version": __version__,
                }
            ],
        },
    )
    ignored = bids_root / ".bidsignore"
    ignored.write_text(
        "\n".join(
            (
                "**/ct/",
                "**/ct/**",
                "**/x-nuclear/",
                "**/x-nuclear/**",
                "**/x-ultrasound/",
                "**/x-ultrasound/**",
                "**/genetic/",
                "**/genetic/**",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "pass",
        "dataset_description": str(description),
        "bidsignore": str(ignored),
    }


def _bids_validate(task: ExecutableTask) -> dict[str, Any]:
    bids_root = validated_cluster_path(str(task.metadata["bids_root"]), must_exist=True)
    image = validated_cluster_path(str(task.metadata["container_image"]), must_exist=True)
    report_root = validated_cluster_path(str(Path(task.output_root) / "bids-validation"))
    report_root.mkdir(parents=True, exist_ok=True)
    report = report_root / "bids-validator.json"
    provisional_datatypes = {"ct", "genetic", "x-nuclear", "x-ultrasound"}
    ignored_directories = sorted(
        {
            "/" + path.relative_to(bids_root).as_posix().rstrip("/") + "/"
            for path in bids_root.rglob("*")
            if path.is_dir() and path.name in provisional_datatypes
        }
    )
    validator_config = report_root / "bids-validator-config.json"
    _write_json(
        validator_config,
        {
            "ignore": [
                {"code": "NOT_INCLUDED", "location": location}
                for location in ignored_directories
            ],
            "warning": [],
            "error": [],
        },
    )
    command = [
        str(paths.container_runtime()),
        "run",
        "--cleanenv",
        "--containall",
        "--bind",
        f"{bids_root}:/data:ro",
        "--bind",
        f"{report_root}:/reports",
        str(image),
        "/data",
        "--config",
        "/reports/bids-validator-config.json",
        "--format",
        "json_pp",
        "--outfile",
        "/reports/bids-validator.json",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-2000:]
        raise RuntimeError(f"BIDS validation failed with {completed.returncode}: {detail}")
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError("BIDS validator produced no machine-readable report")
    payload = json.loads(report.read_text(encoding="utf-8"))
    return {
        "status": "pass",
        "report": str(report),
        "validator": "bids-validator 3.0.1",
        "validator_config": str(validator_config),
        "ignored_provisional_directories": ignored_directories,
        "summary": payload.get("summary", {}),
    }


def _native_output(task: ExecutableTask, kind: str) -> Path:
    subject = task.subject_id or "dataset"
    session = task.session_id or "single"
    return validated_cluster_path(str(Path(task.output_root) / "native" / subject / session / kind))


def _bids_inputs(task: ExecutableTask) -> list[Path]:
    return [
        validated_cluster_path(str(value), must_exist=True)
        for value in task.metadata.get("bids_inputs", [])
    ]


def _file_evidence(paths: list[Path]) -> list[dict[str, Any]]:
    evidence = []
    for path in paths:
        if path.is_file():
            evidence.append(
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                    "suffix": "".join(path.suffixes),
                }
            )
    return evidence


def _quantitative_imaging(task: ExecutableTask, output: Path) -> dict[str, Any]:
    import nibabel as nib
    import numpy as np

    inputs = [path for path in _bids_inputs(task) if path.name.endswith((".nii", ".nii.gz"))]
    if not inputs:
        raise ValueError("quantitative imaging requires at least one NIfTI input")
    results = []
    output.mkdir(parents=True, exist_ok=True)
    for source in inputs:
        image = nib.load(source)
        data = np.asanyarray(image.dataobj)
        finite = np.isfinite(data)
        if not finite.any():
            raise ValueError(f"No finite voxels in {source}")
        target = output / f"{source.name.split('.nii')[0]}_desc-preproc.nii.gz"
        nib.save(image, target)
        results.append(
            {
                "source": str(source),
                "target": str(target),
                "shape": list(data.shape),
                "finite_fraction": float(finite.mean()),
                "minimum": float(data[finite].min()),
                "maximum": float(data[finite].max()),
                "mean": float(data[finite].mean()),
                "units_preserved": True,
            }
        )
    return {"status": "pass", "images": results}


def _tabular(task: ExecutableTask, output: Path, kind: str) -> dict[str, Any]:
    import pandas as pd

    inputs = _bids_inputs(task)
    if not inputs:
        raise ValueError(f"{kind} requires an input file")
    output.mkdir(parents=True, exist_ok=True)
    tables = []
    signals = []
    for source in inputs:
        lowered = source.name.lower()
        if not lowered.endswith((".csv", ".tsv", ".tsv.gz", ".json", ".edf", ".bdf")):
            continue
        if kind == "physiology" and lowered.endswith((".edf", ".bdf")):
            import mne

            raw = mne.io.read_raw_edf(source, preload=True, verbose="error")
            if not raw.ch_names or raw.n_times == 0:
                raise ValueError(f"Empty physiological recording: {source}")
            target = output / f"{source.stem}_{kind}-preproc_raw.fif"
            raw.save(target, overwrite=True, verbose="error")
            signals.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "channels": len(raw.ch_names),
                    "samples": int(raw.n_times),
                    "sampling_frequency": float(raw.info["sfreq"]),
                }
            )
            continue
        if lowered.endswith(".csv"):
            frame = pd.read_csv(source)
        elif lowered.endswith((".tsv", ".tsv.gz")):
            frame = pd.read_csv(source, sep="\t")
        elif lowered.endswith(".json"):
            payload = json.loads(source.read_text(encoding="utf-8"))
            frame = pd.json_normalize(payload if isinstance(payload, list) else [payload])
        else:
            continue
        if frame.empty or not len(frame.columns):
            raise ValueError(f"Empty tabular input: {source}")
        target = output / f"{source.stem}_{kind}.tsv"
        frame.to_csv(target, sep="\t", index=False, na_rep="n/a")
        tables.append(
            {
                "source": str(source),
                "target": str(target),
                "rows": len(frame),
                "columns": list(map(str, frame.columns)),
                "missing_fraction": float(frame.isna().mean().mean()),
            }
        )
    if not tables and not signals:
        raise ValueError(f"No supported input for {kind}")
    return {"status": "pass", "tables": tables, "signals": signals}


def _sensor_features(task: ExecutableTask, output: Path) -> dict[str, Any]:
    import mne
    import numpy as np
    import pandas as pd

    readers = {
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".vhdr": mne.io.read_raw_brainvision,
        ".set": mne.io.read_raw_eeglab,
        ".fif": mne.io.read_raw_fif,
    }
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 80.0),
    }
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for source in _bids_inputs(task):
        reader = readers.get(source.suffix.lower())
        if reader is None:
            continue
        raw = reader(source, preload=True, verbose="error")
        data = raw.get_data(reject_by_annotation="omit")
        if data.size == 0 or not np.isfinite(data).all():
            raise ValueError(f"Invalid sensor recording: {source}")
        upper = min(80.0, float(raw.info["sfreq"]) / 2.0)
        if upper <= 0.5:
            raise ValueError(f"Sampling frequency is too low for spectral features: {source}")
        psd, frequencies = mne.time_frequency.psd_array_welch(
            data,
            sfreq=float(raw.info["sfreq"]),
            fmin=0.5,
            fmax=upper,
            verbose="error",
        )
        rows = []
        for channel, spectrum in zip(raw.ch_names, psd, strict=True):
            row: dict[str, Any] = {"channel": channel}
            for band, (low, high) in bands.items():
                mask = (frequencies >= low) & (frequencies < min(high, upper + 1e-9))
                if mask.sum() > 1:
                    row[f"{band}_power"] = float(np.trapezoid(spectrum[mask], frequencies[mask]))
                else:
                    row[f"{band}_power"] = 0.0
            rows.append(row)
        target = output / f"{source.stem}_sensor-bandpower.tsv"
        pd.DataFrame(rows).to_csv(target, sep="\t", index=False)
        results.append(
            {
                "source": str(source),
                "target": str(target),
                "channels": len(rows),
                "frequency_max": upper,
            }
        )
    if not results:
        raise ValueError(
            "Sensor feature extraction requires EDF, BDF, BrainVision, EEGLAB, or FIF input"
        )
    return {"status": "pass", "features": results}


def _genetics(task: ExecutableTask, output: Path) -> dict[str, Any]:
    inputs = _bids_inputs(task)
    output.mkdir(parents=True, exist_ok=True)
    validated = []
    for source in inputs:
        if source.suffix.lower() not in {".vcf", ".bcf", ".tsv", ".gz"}:
            continue
        if source.suffix.lower() == ".vcf":
            with source.open("r", encoding="utf-8") as stream:
                header = [line.rstrip() for line in stream if line.startswith("#")]
            if not header or not header[0].startswith("##fileformat=VCF"):
                raise ValueError(f"Invalid VCF header: {source}")
        target = output / f"{source.stem}_genetics-validated{source.suffix}"
        _atomic_copy(source, target)
        validated.append({"source": str(source), "target": str(target)})
    if not validated:
        raise ValueError("No VCF/BCF/TSV genetics input was found")
    return {"status": "pass", "files": validated}


def _microscopy(task: ExecutableTask, output: Path) -> dict[str, Any]:
    import numpy as np
    import tifffile
    from scipy.ndimage import median_filter

    inputs = _bids_inputs(task)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for source in inputs:
        if not source.name.lower().endswith((".tif", ".tiff", ".ome.tif")):
            continue
        image = tifffile.imread(source)
        if image.size == 0 or not np.isfinite(image).any():
            raise ValueError(f"Invalid microscopy image: {source}")
        denoised = median_filter(image, size=3)
        target = output / f"{source.stem}_microscopy-preproc.ome.tif"
        tifffile.imwrite(target, denoised, metadata={"axes": "YX" if image.ndim == 2 else None})
        results.append({"source": str(source), "target": str(target), "shape": list(image.shape)})
    if not results:
        raise ValueError("No TIFF microscopy input was found")
    return {"status": "pass", "images": results}


def _fnirs(task: ExecutableTask, output: Path) -> dict[str, Any]:
    import mne
    from mne.preprocessing.nirs import beer_lambert_law, optical_density, scalp_coupling_index

    inputs = _bids_inputs(task)
    source = next((path for path in inputs if path.suffix.lower() == ".snirf"), None)
    if source is None:
        raise ValueError("fNIRS preprocessing requires a SNIRF recording")
    raw = mne.io.read_raw_snirf(source, preload=True, verbose="error")
    optical = optical_density(raw)
    sci = scalp_coupling_index(optical)
    haemoglobin = beer_lambert_law(optical)
    haemoglobin.filter(0.01, 0.2, verbose="error")
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{source.stem}_fnirs-preproc_raw.fif"
    haemoglobin.save(target, overwrite=True, verbose="error")
    return {
        "status": "pass",
        "target": str(target),
        "channels": len(haemoglobin.ch_names),
        "sampling_frequency": float(haemoglobin.info["sfreq"]),
        "scalp_coupling_median": float(__import__("numpy").median(sci)),
    }


def _mrs(task: ExecutableTask, output: Path) -> dict[str, Any]:
    configured_root = str(paths.fsl_mrs_root())
    tool_root = validated_cluster_path(configured_root)
    fsl_mrs_preproc = tool_root / "bin/fsl_mrs_preproc"
    fsl_mrs = tool_root / "bin/fsl_mrs"
    if not fsl_mrs_preproc.is_file():
        fsl_mrs_preproc = Path(shutil.which("fsl_mrs_preproc") or "fsl_mrs_preproc")
    if not fsl_mrs.is_file():
        fsl_mrs = Path(shutil.which("fsl_mrs") or "fsl_mrs")
    metadata_by_id = task.metadata.get("source_metadata", {})
    if not isinstance(metadata_by_id, dict):
        raise ValueError("MRS source metadata must be keyed by acquisition id")
    inputs = _bids_inputs(task)
    if len(inputs) != len(task.acquisition_ids):
        raise ValueError("Every MRS acquisition must have a staged BIDS input")

    paired = list(zip(task.acquisition_ids, inputs, strict=True))
    water: Path | None = None
    metabolite: Path | None = None
    basis_paths: set[str] = set()
    optional: dict[str, str] = {}
    for acquisition_id, path in paired:
        record = metadata_by_id.get(acquisition_id, {})
        if not isinstance(record, dict):
            raise ValueError(f"Invalid MRS metadata for {acquisition_id}")
        role = str(record.get("MRSRole") or record.get("AcquisitionRole") or "").lower()
        if "water" in role or "reference" in role or role in {"wref", "h2o"}:
            if water is not None and water != path:
                raise ValueError("MRS session has multiple water-reference acquisitions")
            water = path
        elif metabolite is None:
            metabolite = path
        if record.get("BasisPath"):
            basis_paths.add(str(record["BasisPath"]))
        for key in ("WaterReferencePath", "T1Path", "TissueFractionPath"):
            if record.get(key):
                observed = str(record[key])
                if key in optional and optional[key] != observed:
                    raise ValueError(f"MRS metadata disagree on {key}")
                optional[key] = observed

    if metabolite is None:
        raise ValueError("MRS session has no metabolite acquisition")
    if water is None and optional.get("WaterReferencePath"):
        water = validated_cluster_path(optional["WaterReferencePath"], must_exist=True)
    if water is None:
        raise ValueError("MRS preprocessing requires an explicit water-reference acquisition")
    if len(basis_paths) != 1:
        raise ValueError("MRS session must identify exactly one BasisPath")
    basis = validated_cluster_path(basis_paths.pop(), must_exist=True)

    output.mkdir(parents=True, exist_ok=True)
    preprocessed = output / "preprocessed"
    fit = output / "fit"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{tool_root / 'bin'}:/usr/bin:/bin",
            "FSLDIR": str(tool_root),
            "FSLOUTPUTTYPE": "NIFTI_GZ",
            "MPLCONFIGDIR": str(Path(task.work_dir) / "mplconfig"),
            "XDG_CACHE_HOME": str(Path(task.work_dir) / "xdg-cache"),
        }
    )
    preproc_command = [
        str(fsl_mrs_preproc),
        "--data",
        str(metabolite),
        "--reference",
        str(water),
        "--quant",
        str(water),
        "--output",
        str(preprocessed),
        "--overwrite",
        "--report",
    ]
    completed = subprocess.run(preproc_command, check=False, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(f"FSL-MRS preprocessing failed with {completed.returncode}")

    processed_data = preprocessed / "metab.nii.gz"
    processed_water = preprocessed / "wref.nii.gz"
    if not processed_data.is_file() or processed_data.stat().st_size == 0:
        raise RuntimeError("FSL-MRS preprocessing produced no metabolite NIfTI-MRS file")
    if not processed_water.is_file() or processed_water.stat().st_size == 0:
        processed_water = water
    fit_command = [
        str(fsl_mrs),
        "--data",
        str(processed_data),
        "--basis",
        str(basis),
        "--output",
        str(fit),
        "--h2o",
        str(processed_water),
        "--overwrite",
        "--report",
    ]
    if optional.get("T1Path"):
        fit_command.extend(
            ["--t1", str(validated_cluster_path(optional["T1Path"], must_exist=True))]
        )
    if optional.get("TissueFractionPath"):
        fit_command.extend(
            [
                "--tissue_frac",
                str(validated_cluster_path(optional["TissueFractionPath"], must_exist=True)),
            ]
        )
    completed = subprocess.run(fit_command, check=False, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(f"FSL-MRS quantification failed with {completed.returncode}")
    concentrations = list(fit.glob("concentrations*.csv"))
    if not concentrations:
        raise RuntimeError("FSL-MRS produced no concentration table")
    return {
        "status": "pass",
        "metabolite": str(metabolite),
        "water_reference": str(water),
        "basis": str(basis),
        "preprocessed": str(processed_data),
        "fit_root": str(fit),
        "concentration_tables": list(map(str, concentrations)),
        "fsl_mrs_version": "2.4.17",
    }


def _validated_nifti_series(task: ExecutableTask, output: Path, kind: str) -> list[dict[str, Any]]:
    import nibabel as nib
    import numpy as np

    results = []
    output.mkdir(parents=True, exist_ok=True)
    for source in _bids_inputs(task):
        if not source.name.lower().endswith((".nii", ".nii.gz")):
            continue
        image = nib.load(source)
        data = np.asanyarray(image.dataobj)
        finite = np.isfinite(data)
        if data.size == 0 or not finite.all():
            raise ValueError(f"{kind} input has empty or non-finite image data: {source}")
        base = source.name[:-7] if source.name.endswith(".nii.gz") else source.stem
        target = output / f"{base}_desc-preproc_{kind}.nii.gz"
        nib.save(nib.as_closest_canonical(image), target)
        results.append(
            {
                "source": str(source),
                "target": str(target),
                "shape": list(data.shape),
                "minimum": float(data.min()),
                "maximum": float(data.max()),
                "mean": float(data.mean()),
                "nonnegative_fraction": float((data >= 0).mean()),
                "intensity_units_preserved": True,
                "canonical_orientation": True,
            }
        )
    return results


def _spect(task: ExecutableTask, output: Path) -> dict[str, Any]:
    images = _validated_nifti_series(task, output, "spect")
    if not images:
        raise ValueError("SPECT preprocessing requires at least one NIfTI image")
    return {"status": "pass", "images": images}


def _angio(task: ExecutableTask, output: Path) -> dict[str, Any]:
    images = _validated_nifti_series(task, output, "angio")
    if not images:
        raise ValueError("MR angiography preprocessing requires at least one NIfTI image")
    return {
        "status": "pass",
        "images": images,
        "normalization": "closest-canonical orientation; native angiographic intensities preserved",
    }


def _ultrasound(task: ExecutableTask, output: Path) -> dict[str, Any]:
    images = _validated_nifti_series(task, output, "ultrasound")
    videos = []
    output.mkdir(parents=True, exist_ok=True)
    for source in _bids_inputs(task):
        if source.suffix.lower() != ".mp4":
            continue
        ffprobe = shutil.which("ffprobe")
        ffmpeg = shutil.which("ffmpeg")
        if not ffprobe or not ffmpeg:
            raise FileNotFoundError("ultrasound video preprocessing requires ffmpeg and ffprobe")
        probe_command = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration,pix_fmt",
            "-of",
            "json",
            str(source),
        ]
        probed = subprocess.run(probe_command, check=False, capture_output=True, text=True)
        if probed.returncode != 0:
            raise RuntimeError(f"ffprobe failed for {source}: {probed.stderr[-500:]}")
        payload = json.loads(probed.stdout)
        streams = payload.get("streams") or []
        if len(streams) != 1:
            raise ValueError(f"ultrasound video must contain exactly one video stream: {source}")
        target = output / f"{source.stem}_desc-lossless_ultrasound.mkv"
        encode_command = [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            str(target),
        ]
        encoded = subprocess.run(encode_command, check=False, capture_output=True, text=True)
        if encoded.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg failed for {source}: {encoded.stderr[-500:]}")
        videos.append(
            {
                "source": str(source),
                "target": str(target),
                "stream": streams[0],
                "normalization": "lossless FFV1; audio removed",
            }
        )
    if not images and not videos:
        raise ValueError("ultrasound preprocessing requires NIfTI or MP4 input")
    return {"status": "pass", "images": images, "videos": videos}


def _copy_modality(task: ExecutableTask, output: Path, kind: str) -> dict[str, Any]:
    inputs = _bids_inputs(task)
    output.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in inputs:
        target = output / f"{source.stem}_{kind}{''.join(source.suffixes)}"
        _atomic_copy(source, target)
        copied.append(str(target))
    if not copied:
        raise ValueError(f"No inputs for {kind}")
    return {"status": "pass", "outputs": copied, "evidence": _file_evidence(inputs)}


def _registration_suffix(path: Path) -> str:
    name = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    return name.rsplit("_", 1)[-1].lower()


def _registration_candidates(inputs: list[Path], longitudinal: bool) -> list[Path]:
    excluded = {
        "epi",
        "fieldmap",
        "magnitude",
        "magnitude1",
        "magnitude2",
        "phase1",
        "phase2",
        "phasediff",
        "sbref",
    }
    candidates = [path for path in inputs if _registration_suffix(path) not in excluded]
    if longitudinal:
        t1_inputs = [
            path for path in candidates if _registration_suffix(path) in {"t1w", "unit1"}
        ]
        structural_inputs = [
            path
            for path in candidates
            if _registration_suffix(path)
            in {"angio", "flair", "swi", "t1map", "t1w", "t2map", "t2starw", "t2w", "unit1"}
        ]
        if len(t1_inputs) >= 2:
            candidates = t1_inputs
        elif len(structural_inputs) >= 2:
            candidates = structural_inputs
    if len(candidates) < 2:
        raise ValueError(
            "registration requires at least two non-fieldmap NIfTI inputs; "
            f"found {len(candidates)} from {len(inputs)}"
        )
    rank = {
        "t1w": 0,
        "unit1": 1,
        "t2w": 2,
        "flair": 3,
        "t2starw": 4,
        "swi": 5,
        "angio": 6,
        "ct": 7,
        "pet": 8,
        "asl": 9,
        "dwi": 10,
        "bold": 11,
    }
    return sorted(
        candidates,
        key=lambda path: (rank.get(_registration_suffix(path), 50), str(path)),
    )


def _registration_volume_indices(source: Path, volumes: int) -> tuple[list[int], str]:
    suffix = _registration_suffix(source)
    stem = source.name[:-7] if source.name.endswith(".nii.gz") else source.stem
    if suffix == "dwi":
        bval_path = source.with_name(stem + ".bval")
        if bval_path.is_file():
            try:
                bvals = [float(value) for value in bval_path.read_text(encoding="utf-8").split()]
            except ValueError as exc:
                raise ValueError(f"Invalid diffusion b-values: {bval_path}") from exc
            if len(bvals) == volumes:
                b0_indices = [index for index, value in enumerate(bvals) if abs(value) <= 50]
                if b0_indices:
                    return b0_indices, "b0mean"
                return [min(range(volumes), key=lambda index: abs(bvals[index]))], "minb"
    if suffix == "asl":
        context_path = source.with_name(stem + "context.tsv")
        if context_path.is_file():
            labels = [
                line.strip().lower()
                for line in context_path.read_text(encoding="utf-8").splitlines()[1:]
            ]
            if len(labels) == volumes:
                control_indices = [
                    index for index, label in enumerate(labels) if label in {"control", "m0scan"}
                ]
                if control_indices:
                    return control_indices, "controlmean"
    return list(range(volumes)), "tmean"


def _registration_reference(source: Path, reference_root: Path) -> tuple[Path, dict[str, Any]]:
    import nibabel as nib
    import numpy as np

    image = nib.load(source)
    if len(image.shape) < 3:
        raise ValueError(f"Registration input is not a 3D spatial image: {source}")
    if len(image.shape) == 3:
        return source, {
            "source": str(source),
            "reference": str(source),
            "strategy": "native-3d",
            "source_shape": list(image.shape),
        }
    volumes = int(np.prod(image.shape[3:]))
    indices, strategy = _registration_volume_indices(source, volumes)
    data = np.asanyarray(image.dataobj, dtype=np.float32).reshape((*image.shape[:3], volumes))
    reference_data = np.nanmean(data[..., indices], axis=3, dtype=np.float32)
    if not np.isfinite(reference_data).any():
        raise ValueError(f"Registration reference contains no finite values: {source}")
    reference_root.mkdir(parents=True, exist_ok=True)
    stem = source.name[:-7] if source.name.endswith(".nii.gz") else source.stem
    target = reference_root / f"{stem}_desc-{strategy}_registration-reference.nii.gz"
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(reference_data, image.affine, header), target)
    return target, {
        "source": str(source),
        "reference": str(target),
        "strategy": strategy,
        "source_shape": list(image.shape),
        "selected_volumes": indices,
    }


def _ants_registration(task: ExecutableTask, output: Path, longitudinal: bool) -> dict[str, Any]:
    inputs = [path for path in _bids_inputs(task) if path.name.endswith((".nii", ".nii.gz"))]
    candidates = _registration_candidates(inputs, longitudinal)
    configured = str(
        task.metadata.get("ants_image") or os.environ.get("NEUROPREP_ANTS_IMAGE", "")
    )
    image_candidates = [Path(configured)] if configured else []
    image_candidates.extend(
        [
            paths.ants_image(),
            Path.home() / "ants-2.6.5.sif",
        ]
    )
    image = next((candidate for candidate in image_candidates if candidate.is_file()), None)
    if image is None:
        searched = ", ".join(str(candidate) for candidate in image_candidates)
        raise FileNotFoundError(f"ANTs image is missing; searched: {searched}")
    output.mkdir(parents=True, exist_ok=True)
    prepared = [
        _registration_reference(source, output / "registration-references")
        for source in candidates
    ]
    references = [item[0] for item in prepared]
    reference_evidence = [item[1] for item in prepared]
    bind_roots = set(_settings.ALLOWED_ROOTS)
    approved_binds = [
        item
        for root in sorted(bind_roots, key=str)
        if root.is_dir()
        for item in ("--bind", f"{root}:{root}")
    ]
    if longitudinal:
        prefix = output / "longitudinal-template-"
        command = [
            str(paths.container_runtime()),
            "exec",
            "--cleanenv",
            "--containall",
            *approved_binds,
            str(image),
            "antsMultivariateTemplateConstruction2.sh",
            "-d",
            "3",
            "-o",
            str(prefix),
            *map(str, references),
        ]
    else:
        fixed = references[0]
        transforms = []
        for index, moving in enumerate(references[1:], start=1):
            prefix = output / f"multimodal-registration-{index}-"
            command = [
                str(paths.container_runtime()),
                "exec",
                "--cleanenv",
                "--containall",
                *approved_binds,
                str(image),
                "antsRegistrationSyNQuick.sh",
                "-d",
                "3",
                "-t",
                "r",
                "-f",
                str(fixed),
                "-m",
                str(moving),
                "-o",
                str(prefix),
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"ANTs registration failed with {completed.returncode}")
            transforms.extend(map(str, output.glob(f"{prefix.name}*GenericAffine.mat")))
        if not transforms:
            raise RuntimeError("ANTs produced no multimodal affine transforms")
        return {
            "status": "pass",
            "fixed_reference": str(fixed),
            "references": reference_evidence,
            "transforms": transforms,
        }
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ANTs template construction failed with {completed.returncode}")
    products = sorted(map(str, output.glob("longitudinal-template-*")))
    if not products:
        raise RuntimeError("ANTs produced no longitudinal template artifacts")
    return {"status": "pass", "references": reference_evidence, "artifacts": products}


def oxford_asl_arguments(
    metadata: dict[str, Any], volume_types: list[str]
) -> tuple[list[str], list[int], list[int]]:
    """Map BIDS-ASL metadata to FSL ``oxford_asl`` flags.

    Returns (flags, data volume indices, M0 volume indices). PASL PostLabelingDelay is
    already the inversion time in BIDS, so it is passed as ``--tis``; (P)CASL delays go
    to ``--plds`` with ``--bolus`` = LabelingDuration.
    """
    data = [i for i, kind in enumerate(volume_types) if kind in {"label", "control", "deltam"}]
    m0 = [i for i, kind in enumerate(volume_types) if kind == "m0scan"]
    if not data:
        raise ValueError("ASL series has no label, control or deltam volumes")
    if not m0:
        raise ValueError("BASIL adapter needs an M0 (m0scan volumes) for calibrated CBF")
    first = volume_types[data[0]]
    flags = ["--iaf", {"label": "tc", "control": "ct", "deltam": "diff"}[first]]
    labeling = str(metadata.get("ArterialSpinLabelingType", "")).upper()
    if labeling not in {"PCASL", "CASL", "PASL"}:
        raise ValueError(f"unsupported ArterialSpinLabelingType: {labeling or 'missing'}")
    delay = metadata.get("PostLabelingDelay")
    if delay is None:
        raise ValueError("ASL sidecar lacks PostLabelingDelay")
    per_volume = list(delay) if isinstance(delay, list) else [delay] * len(volume_types)
    step = 1 if first == "deltam" else 2
    sequence = [float(per_volume[index]) for index in data[::step]]
    unique = list(dict.fromkeys(sequence))
    if len(unique) > 1:
        cyclic = sequence == unique * (len(sequence) // len(unique))
        grouped = all(
            sequence.index(value) + sequence.count(value) - 1
            == len(sequence) - 1 - sequence[::-1].index(value)
            for value in unique
        )
        if not (cyclic or grouped):
            raise ValueError("multi-PLD ordering is neither repeated nor grouped")
        flags += ["--ibf", "rpt" if cyclic else "tis"]
    delays = ",".join(f"{value:g}" for value in unique)
    if labeling == "PASL":
        bolus = metadata.get("BolusCutOffDelayTime")
        if bolus is None:
            raise ValueError("PASL needs BolusCutOffDelayTime for the bolus duration")
        flags += ["--tis", delays, "--bolus", f"{float(bolus):g}"]
    else:
        duration = metadata.get("LabelingDuration")
        if duration is None:
            raise ValueError("(P)CASL needs LabelingDuration")
        flags += ["--casl", "--plds", delays, "--bolus", f"{float(duration):g}"]
    repetition = metadata.get("RepetitionTimePreparation") or metadata.get("RepetitionTime")
    flags += ["--tr", f"{float(repetition):g}", "--cmethod", "voxel"]
    timing = metadata.get("SliceTiming")
    if metadata.get("MRAcquisitionType") == "2D" and isinstance(timing, list) and len(timing) > 1:
        ordered = sorted(float(value) for value in timing)
        gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False) if b > a]
        if gaps:
            flags += ["--slicedt", f"{sum(gaps) / len(gaps):g}"]
    return flags, data, m0


def _basil(task: ExecutableTask, output: Path, image: Path) -> dict[str, Any]:
    asl = next(path for path in _bids_inputs(task) if path.name.endswith("_asl.nii.gz"))
    stem = asl.name[: -len("_asl.nii.gz")]
    metadata = json.loads(asl.with_name(stem + "_asl.json").read_text(encoding="utf-8"))
    rows = asl.with_name(stem + "_aslcontext.tsv").read_text(encoding="utf-8").splitlines()
    volume_types = [row.strip().split("\t")[0] for row in rows[1:] if row.strip()]
    flags, data, m0 = oxford_asl_arguments(metadata, volume_types)
    binds = [
        item
        for root in sorted(set(_settings.ALLOWED_ROOTS), key=str)
        if root.is_dir()
        for item in ("--bind", f"{root}:{root}")
    ]
    fsl = [str(paths.container_runtime()), "exec", "--cleanenv", "--containall", *binds, str(image)]
    output.mkdir(parents=True, exist_ok=True)
    asl_data, m0_image, basil = output / "asl_data.nii.gz", output / "m0.nii.gz", output / "basil"
    steps = [
        [*fsl, "fslselectvols", "-i", str(asl), "-o", str(asl_data),
         "--vols=" + ",".join(map(str, data))],
        [*fsl, "fslselectvols", "-i", str(asl), "-o", str(m0_image),
         "--vols=" + ",".join(map(str, m0)), "-m"],
        [*fsl, "oxford_asl", "-i", str(asl_data), "-o", str(basil), "-c", str(m0_image), *flags],
    ]
    for command in steps:
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"BASIL step failed ({completed.returncode}): {command[-4:]}")
    cbf = basil / "native_space" / "perfusion_calib.nii.gz"
    if not cbf.is_file():
        raise RuntimeError(f"oxford_asl wrote no calibrated perfusion map: {cbf}")
    return {"oxford_asl_flags": flags, "cbf_map": str(cbf), "m0_volumes": len(m0)}


def run_native(task: ExecutableTask, kind: str, image: str | None = None) -> dict[str, Any]:
    if kind == "bids-init":
        return _bids_init(task)
    if kind == "bids-validate":
        return _bids_validate(task)
    if kind == "bids-stage":
        return stage_bids(task)
    output = _native_output(task, kind)
    if kind == "quantitative-imaging":
        result = _quantitative_imaging(task, output)
    elif kind in {"behavior", "physiology"}:
        result = _tabular(task, output, kind)
    elif kind == "genetics":
        result = _genetics(task, output)
    elif kind == "microscopy":
        result = _microscopy(task, output)
    elif kind == "fnirs":
        result = _fnirs(task, output)
    elif kind == "sensor-features":
        result = _sensor_features(task, output)
    elif kind == "mrs":
        result = _mrs(task, output)
    elif kind == "spect":
        result = _spect(task, output)
    elif kind == "angio":
        result = _angio(task, output)
    elif kind == "ultrasound":
        result = _ultrasound(task, output)
    elif kind == "multimodal-registration":
        result = _ants_registration(task, output, False)
    elif kind == "longitudinal-template":
        result = _ants_registration(task, output, True)
    elif kind == "basil":
        if not image:
            raise ValueError("basil requires --image")
        result = _basil(task, output, validated_cluster_path(image, must_exist=True))
    else:
        result = _copy_modality(task, output, kind)
    report = output / f"{_safe_filename(task.task_id)}_{kind}_qc.json"
    result.update(
        {
            "task_id": task.task_id,
            "executor_id": task.executor_id,
            "input_evidence": _file_evidence(_bids_inputs(task)),
        }
    )
    _write_json(report, result)
    return result


def _safe_filename(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "-" for character in value
    )


def cli() -> None:
    parser = argparse.ArgumentParser(description="Deterministic NeuroPrep native executor")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--kind", required=True)
    run_parser.add_argument("--context", required=True)
    run_parser.add_argument("--image", help="Container image for container-backed kinds")
    arguments = parser.parse_args()
    context = validated_cluster_path(arguments.context, must_exist=True)
    task = ExecutableTask.model_validate_json(context.read_text(encoding="utf-8"))
    result = run_native(task, arguments.kind, arguments.image)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    cli()
