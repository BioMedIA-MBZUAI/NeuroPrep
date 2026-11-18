#!/usr/bin/env python3
"""Run a pseudonymized prospective multimodal ingestion/preprocessing benchmark.

The input specification remains at the restricted-data site. Published output contains
only caller-supplied case labels and modalities, never source member names or subject IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from neuroprep_agentic.execution_models import ExecutableTask, ExecutionScope
from neuroprep_agentic.native import PHI_KEYS, stage_bids

PLUGIN_BY_MODALITY = {
    "T1w": "mri.t1w",
    "FLAIR": "mri.flair",
    "dwi": "mri.dwi",
    "bold": "mri.bold",
    "asl": "mri.asl",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], log_path: Path) -> float:
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps({"command": command, "return_code": completed.returncode})
        + "\nSTDOUT\n"
        + completed.stdout
        + "\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"command failed with {completed.returncode}; see {log_path}")
    return elapsed


def _phi_keys(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in PHI_KEYS or (key.startswith("Patient") and key != "PatientPosition"):
                found.append(key)
            found.extend(_phi_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_phi_keys(item))
    return sorted(set(found))


def _image_metrics(path: Path) -> dict[str, Any]:
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    return {
        "shape": list(image.shape),
        "finite_fraction": float(np.isfinite(data).mean()),
        "nonzero_fraction": float(np.count_nonzero(data) / data.size),
    }


def _task(case: str, modality: str, source: Path, target: Path, run_root: Path) -> ExecutableTask:
    return ExecutableTask(
        task_id=f"benchmark-{case}-{modality}",
        logical_task_ids=[f"benchmark-{case}-{modality}"],
        executor_id="neuroprep-bids-stage",
        plugin_ids=[PLUGIN_BY_MODALITY[modality]],
        stages=["bids_stage"],
        scope=ExecutionScope.ACQUISITION,
        subject_id="sub-benchmark",
        session_id="ses-01",
        acquisition_ids=[f"{case}-{modality}"],
        command=["benchmark"],
        input_paths=[str(source)],
        output_root=str(target.parent),
        work_dir=str(run_root / "work" / case),
        resource_class="cpu",
        cpus=1,
        memory_gb=4,
        gpus=0,
        walltime_minutes=60,
        fingerprint="benchmark",
        metadata={
            "source_path": str(source),
            "bids_path": str(target),
            "selected_plugin_id": PLUGIN_BY_MODALITY[modality],
            "source_metadata": {},
            "bids_sidecar": {},
        },
    )


def _extract(spec: dict[str, str], destination: Path, log_path: Path) -> tuple[Path, float]:
    archive = Path(spec["archive"])
    prefix = spec["member_prefix"].rstrip("/")
    destination.mkdir(parents=True, exist_ok=True)
    elapsed = _run(
        ["/usr/bin/unzip", "-q", str(archive), f"{prefix}/*", "-d", str(destination)],
        log_path,
    )
    extracted = destination / prefix
    if not extracted.is_dir() or not any(extracted.glob("*.dcm")):
        raise FileNotFoundError(f"no DICOM files extracted for case {spec['case']}")
    return extracted, elapsed


def _direct_convert(source: Path, target: Path, log_path: Path) -> float:
    target.parent.mkdir(parents=True, exist_ok=True)
    stem = target.name[:-7]
    return _run(
        [
            "/usr/bin/dcm2niix",
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
        ],
        log_path,
    )


def _ants_transform(
    source: Path,
    output: Path,
    modality: str,
    run_root: Path,
    ants_image: Path,
    log_path: Path,
) -> tuple[str, float]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if modality in {"T1w", "FLAIR"}:
        command = [
            "/usr/bin/singularity",
            "exec",
            "--cleanenv",
            "--bind",
            f"{run_root}:{run_root}",
            str(ants_image),
            "N4BiasFieldCorrection",
            "-d",
            "3",
            "-i",
            str(source),
            "-o",
            str(output),
        ]
        method = "N4 bias-field correction"
    elif modality in {"dwi", "bold"}:
        image = nib.load(source)
        data = np.asanyarray(image.dataobj)
        if data.ndim != 4:
            raise ValueError(f"{modality} input is not four-dimensional")
        reference = output.with_name(output.name.replace("_processed", "_reference"))
        if modality == "dwi":
            bval_path = source.with_name(source.name[:-7] + ".bval")
            b_values = np.loadtxt(bval_path) if bval_path.is_file() else np.arange(data.shape[3])
            volume_index = int(np.argmin(np.abs(np.atleast_1d(b_values))))
            reference_data = data[..., volume_index]
            method = f"adaptive non-local-means denoising (b=0 reference volume {volume_index})"
        else:
            subset_count = min(data.shape[3], 20)
            reference_data = np.mean(data[..., :subset_count], axis=3, dtype=np.float32)
            method = (
                "adaptive non-local-means denoising "
                f"({subset_count}-volume temporal-mean reference)"
            )
        nib.save(nib.Nifti1Image(reference_data, image.affine, image.header), reference)
        command = [
            "/usr/bin/singularity",
            "exec",
            "--cleanenv",
            "--bind",
            f"{run_root}:{run_root}",
            str(ants_image),
            "DenoiseImage",
            "-d",
            "3",
            "-i",
            str(reference),
            "-o",
            str(output),
        ]
    else:
        image = nib.load(source)
        data = np.asanyarray(image.dataobj, dtype=np.float32)
        if data.ndim != 4 or data.shape[3] < 2:
            raise ValueError("ASL input is not a paired four-dimensional series")
        stem = source.name[:-7]
        sidecar = source.with_name(stem.removesuffix("_asl") + "_aslcontext.tsv")
        rows = sidecar.read_text(encoding="utf-8").splitlines()[1:]
        labels = [index for index, value in enumerate(rows) if value == "label"]
        controls = [index for index, value in enumerate(rows) if value == "control"]
        pair_count = min(len(labels), len(controls))
        if not pair_count:
            raise ValueError("ASL context contains no complete control-label pairs")
        perfusion = np.mean(
            data[..., controls[:pair_count]] - data[..., labels[:pair_count]], axis=3
        )
        nib.save(nib.Nifti1Image(perfusion, image.affine, image.header), output)
        return f"mean control-label perfusion difference ({pair_count} pairs)", 0.0
    return method, _run(command, log_path)


def run_benchmark(spec_path: Path, run_root: Path, ants_image: Path) -> dict[str, Any]:
    specs = json.loads(spec_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for spec in specs:
        case = spec["case"]
        modality = spec["modality"]
        expected_outcome = spec.get("expected_outcome", "processed")
        if modality not in PLUGIN_BY_MODALITY:
            raise ValueError(f"unsupported modality: {modality}")
        source, extraction_seconds = _extract(
            spec, run_root / "raw" / case, run_root / "logs" / f"{case}-extract.log"
        )
        stem = f"sub-benchmark_ses-01_{modality}"
        direct_target = run_root / "direct" / case / f"{stem}.nii.gz"
        neuroprep_target = run_root / "neuroprep" / case / f"{stem}.nii.gz"
        direct_seconds = _direct_convert(
            source, direct_target, run_root / "logs" / f"{case}-direct.log"
        )
        started = time.perf_counter()
        try:
            stage_bids(_task(case, modality, source, neuroprep_target, run_root))
        except ValueError as exc:
            neuroprep_seconds = time.perf_counter() - started
            if expected_outcome != "fail_closed" or "invalid spatial shape" not in str(exc):
                raise
            metrics = _image_metrics(direct_target)
            rows.append(
                {
                    "case": case,
                    "modality": modality,
                    "dicom_files": len(list(source.glob("*.dcm"))),
                    "extraction_seconds": extraction_seconds,
                    "direct_conversion_seconds": direct_seconds,
                    "neuroprep_conversion_seconds": neuroprep_seconds,
                    "neuroprep_overhead_seconds": neuroprep_seconds - direct_seconds,
                    "image_sha256_equal": "not_released",
                    "phi_keys_remaining": [],
                    **{f"input_{key}": value for key, value in metrics.items()},
                    "preprocess_method": "fail-closed invalid-mosaic rejection",
                    "preprocess_seconds": 0.0,
                    "output_shape": [],
                    "output_finite_fraction": "not_released",
                    "output_nonzero_fraction": "not_released",
                    "outcome": "fail_closed",
                    "released_artifact_count": 0,
                    "status": "pass",
                }
            )
            continue
        neuroprep_seconds = time.perf_counter() - started
        if expected_outcome != "processed":
            raise AssertionError(f"{case} was expected to fail closed but produced an artifact")
        processed = run_root / "processed" / case / f"{stem}_processed.nii.gz"
        method, preprocess_seconds = _ants_transform(
            neuroprep_target,
            processed,
            modality,
            run_root,
            ants_image,
            run_root / "logs" / f"{case}-preprocess.log",
        )
        sidecar = neuroprep_target.with_name(neuroprep_target.name[:-7] + ".json")
        phi_remaining = _phi_keys(json.loads(sidecar.read_text(encoding="utf-8")))
        direct_hash = _sha256(direct_target)
        neuroprep_hash = _sha256(neuroprep_target)
        rows.append(
            {
                "case": case,
                "modality": modality,
                "dicom_files": len(list(source.glob("*.dcm"))),
                "extraction_seconds": extraction_seconds,
                "direct_conversion_seconds": direct_seconds,
                "neuroprep_conversion_seconds": neuroprep_seconds,
                "neuroprep_overhead_seconds": neuroprep_seconds - direct_seconds,
                "image_sha256_equal": direct_hash == neuroprep_hash,
                "phi_keys_remaining": phi_remaining,
                **{
                    f"input_{key}": value
                    for key, value in _image_metrics(neuroprep_target).items()
                },
                "preprocess_method": method,
                "preprocess_seconds": preprocess_seconds,
                **{f"output_{key}": value for key, value in _image_metrics(processed).items()},
                "outcome": "processed",
                "released_artifact_count": 1,
                "status": "pass",
            }
        )
    summary = {
        "schema_version": "1.0",
        "benchmark": "prospective-pseudonymized-multimodal-smoke",
        "case_count": len(rows),
        "modalities": sorted(row["modality"] for row in rows),
        "all_passed": all(row["status"] == "pass" for row in rows),
        "all_images_exact": all(
            row["image_sha256_equal"] is True
            for row in rows
            if row["outcome"] == "processed"
        ),
        "all_sidecars_phi_free": all(not row["phi_keys_remaining"] for row in rows),
        "all_outputs_finite": all(
            row["output_finite_fraction"] == 1.0
            for row in rows
            if row["outcome"] == "processed"
        ),
        "processed_count": sum(row["outcome"] == "processed" for row in rows),
        "fail_closed_count": sum(row["outcome"] == "fail_closed" for row in rows),
        "rows": rows,
    }
    evidence = run_root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (evidence / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ants-image", type=Path, required=True)
    arguments = parser.parse_args()
    summary = run_benchmark(
        arguments.spec.resolve(), arguments.run_root.resolve(), arguments.ants_image.resolve()
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
