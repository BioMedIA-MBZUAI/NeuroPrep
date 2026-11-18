#!/usr/bin/env python3
"""Balanced paired timing of direct dcm2niix and NeuroPrep DICOM staging."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import time
from pathlib import Path

from prospective_multimodal_benchmark import _direct_convert, _phi_keys, _sha256, _task

from neuroprep_agentic.native import stage_bids


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name[:-7] + ".json")


def _neuroprep_convert(
    case: str,
    modality: str,
    source: Path,
    target: Path,
    repeat_root: Path,
) -> float:
    started = time.perf_counter()
    stage_bids(_task(case, modality, source, target, repeat_root))
    return time.perf_counter() - started


def run_timing(spec_path: Path, run_root: Path, repetitions: int) -> dict:
    specs = [
        row
        for row in json.loads(spec_path.read_text(encoding="utf-8"))
        if row.get("expected_outcome", "processed") == "processed"
    ]
    records = []
    timing_root = run_root / "timing-work"
    for spec in specs:
        case = spec["case"]
        modality = spec["modality"]
        source = run_root / "raw" / case / spec["member_prefix"]
        for repeat in range(1, repetitions + 1):
            repeat_root = timing_root / case / f"repeat-{repeat:02d}"
            direct_target = repeat_root / "direct" / f"sub-benchmark_ses-01_{modality}.nii.gz"
            neuroprep_target = (
                repeat_root / "neuroprep" / f"sub-benchmark_ses-01_{modality}.nii.gz"
            )

            direct_log = run_root / "logs" / f"timing-{case}-{repeat:02d}-direct.log"
            if repeat % 2:
                order = "direct-first"
                direct_seconds = _direct_convert(source, direct_target, direct_log)
                neuroprep_seconds = _neuroprep_convert(
                    case, modality, source, neuroprep_target, repeat_root
                )
            else:
                order = "neuroprep-first"
                neuroprep_seconds = _neuroprep_convert(
                    case, modality, source, neuroprep_target, repeat_root
                )
                direct_seconds = _direct_convert(source, direct_target, direct_log)
            sidecar_payload = json.loads(_sidecar(neuroprep_target).read_text(encoding="utf-8"))
            records.append(
                {
                    "case": case,
                    "modality": modality,
                    "repeat": repeat,
                    "order": order,
                    "direct_seconds": direct_seconds,
                    "neuroprep_seconds": neuroprep_seconds,
                    "paired_overhead_seconds": neuroprep_seconds - direct_seconds,
                    "image_sha256_equal": _sha256(direct_target)
                    == _sha256(neuroprep_target),
                    "phi_keys_remaining": _phi_keys(sidecar_payload),
                }
            )
            shutil.rmtree(repeat_root)
    overheads = [row["paired_overhead_seconds"] for row in records]
    direct_times = [row["direct_seconds"] for row in records]
    neuroprep_times = [row["neuroprep_seconds"] for row in records]
    summary = {
        "schema_version": "1.0",
        "benchmark": "balanced-paired-ingestion-timing",
        "case_count": len(specs),
        "repetitions_per_case": repetitions,
        "paired_run_count": len(records),
        "direct_median_seconds": statistics.median(direct_times),
        "neuroprep_median_seconds": statistics.median(neuroprep_times),
        "paired_overhead_median_seconds": statistics.median(overheads),
        "paired_overhead_mean_seconds": statistics.fmean(overheads),
        "all_images_exact": all(row["image_sha256_equal"] for row in records),
        "all_sidecars_phi_free": all(not row["phi_keys_remaining"] for row in records),
        "records": records,
    }
    evidence = run_root / "evidence"
    (evidence / "paired_ingestion_timing.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (evidence / "paired_ingestion_timing.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    shutil.rmtree(timing_root, ignore_errors=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=6)
    arguments = parser.parse_args()
    if arguments.repetitions < 2 or arguments.repetitions % 2:
        raise ValueError("repetitions must be a positive even integer of at least two")
    result = run_timing(
        arguments.spec.resolve(), arguments.run_root.resolve(), arguments.repetitions
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
