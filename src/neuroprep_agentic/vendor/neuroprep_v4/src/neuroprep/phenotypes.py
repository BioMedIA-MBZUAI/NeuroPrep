from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def join_phenotypes(
    visit_index: Path,
    phenotype_table: Path,
    output: Path,
    report: Path,
    participant_column: str = "participant_id",
    session_column: str | None = "session_id",
    date_column: str | None = None,
    label_columns: tuple[str, ...] = (),
    covariate_columns: tuple[str, ...] = (),
    tolerance_days: int = 60,
) -> dict[str, Any]:
    """Link an allow-listed phenotype view to visits without copying arbitrary EHR fields."""

    visits = _read_jsonl(visit_index)
    phenotypes = _read_table(phenotype_table)
    allowed = [participant_column]
    if session_column:
        allowed.append(session_column)
    if date_column:
        allowed.append(date_column)
    allowed.extend((*label_columns, *covariate_columns))
    missing_columns = sorted(set(allowed) - set(phenotypes[0] if phenotypes else []))
    if missing_columns:
        raise ValueError(f"Phenotype table is missing columns: {missing_columns}")

    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in phenotypes:
        participant = _participant(row.get(participant_column, ""))
        if participant:
            by_subject[participant].append(row)
    relative_dates = _relative_phenotype_days(by_subject, date_column)
    joined: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for visit in visits:
        participant = _participant(visit.get("participant_id", ""))
        candidates = by_subject.get(participant, [])
        exact: list[dict[str, str]] = []
        visit_id = str(visit.get("visit_id", ""))
        if session_column:
            exact = [
                row
                for row in candidates
                if _session(row.get(session_column, "")) == _session(visit_id)
            ]
        if len(exact) == 1:
            selected = exact[0]
            method = "participant+session"
            delta = 0
        elif not exact and date_column and visit.get("days_from_baseline") is not None:
            target = int(visit["days_from_baseline"])
            dated = [
                (abs(relative_dates[id(row)] - target), row)
                for row in candidates
                if id(row) in relative_dates
            ]
            dated.sort(key=lambda item: (item[0], _row_digest(item[1])))
            if dated and dated[0][0] <= tolerance_days and (
                len(dated) == 1 or dated[1][0] != dated[0][0]
            ):
                delta, selected = dated[0]
                method = "participant+nearest-relative-date"
            elif dated and dated[0][0] <= tolerance_days:
                ambiguous.append(
                    {
                        "participant_id": participant,
                        "visit_id": visit_id,
                        "reason": "equidistant-phenotype-visits",
                        "distance_days": dated[0][0],
                    }
                )
                continue
            else:
                unmatched.append(
                    {
                        "participant_id": participant,
                        "visit_id": visit_id,
                        "reason": "no-phenotype-within-tolerance",
                    }
                )
                continue
        elif len(candidates) == 1:
            selected = candidates[0]
            method = "participant-only-single-candidate"
            delta = None
        elif len(exact) > 1 or len(candidates) > 1:
            ambiguous.append(
                {
                    "participant_id": participant,
                    "visit_id": visit_id,
                    "reason": "multiple-phenotype-candidates",
                    "candidate_count": len(exact or candidates),
                }
            )
            continue
        else:
            unmatched.append(
                {
                    "participant_id": participant,
                    "visit_id": visit_id,
                    "reason": "participant-not-found",
                }
            )
            continue
        linked = {
            **visit,
            "phenotype_match": {
                "method": method,
                "distance_days": delta,
                "source_row_digest": _row_digest(selected),
            },
            "labels": {column: selected.get(column) for column in label_columns},
            "covariates": {column: selected.get(column) for column in covariate_columns},
        }
        joined.append(linked)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in joined:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(output)
    summary = {
        "status": "pass" if joined and not ambiguous else "review" if joined else "fail",
        "visits": len(visits),
        "matched": len(joined),
        "unmatched": unmatched,
        "ambiguous": ambiguous,
        "participant_column": participant_column,
        "session_column": session_column,
        "date_column": date_column,
        "label_columns": list(label_columns),
        "covariate_columns": list(covariate_columns),
        "privacy_policy": "only explicitly allow-listed labels/covariates are copied",
        "output": str(output),
    }
    _write_json(report, summary)
    return summary


def create_grouped_splits(
    joined_index: Path,
    output: Path,
    report: Path,
    folds: int = 5,
    seed: int = 42,
    site_column: str = "site",
    label_column: str | None = None,
    strategy: str = "stratified-group",
) -> dict[str, Any]:
    """Create fold assignments with all visits from a participant kept together."""

    if folds < 3:
        raise ValueError("At least three folds are required for train/validation/test rotation")
    rows = _read_jsonl(joined_index)
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[_participant(row.get("participant_id", ""))].append(row)
    subject_site: dict[str, str] = {}
    subject_labels: dict[str, tuple[str, ...]] = {}
    for participant, visits in by_subject.items():
        sites = {
            str(
                _nested(row, "covariates", site_column)
                or _nested(row, "site_covariates", site_column)
                or row.get(site_column)
                or "unknown"
            )
            for row in visits
        }
        if len(sites) != 1:
            raise ValueError(f"Participant {participant} maps to multiple sites: {sorted(sites)}")
        subject_site[participant] = next(iter(sites))
        subject_labels[participant] = tuple(
            sorted(
                {
                    str(_nested(row, "labels", label_column) or row.get(label_column) or "missing")
                    for row in visits
                }
            )
        ) if label_column else ()

    assignments: dict[str, int] = {}
    if strategy == "stratified-group":
        strata: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
        for participant in by_subject:
            strata[(subject_site[participant], subject_labels[participant])].append(participant)
        generator = random.Random(seed)
        for stratum in sorted(strata):
            participants = sorted(strata[stratum])
            generator.shuffle(participants)
            start = int(
                hashlib.sha256(
                    f"{seed}:{stratum}".encode("utf-8")
                ).hexdigest()[:8],
                16,
            ) % folds
            for index, participant in enumerate(participants):
                assignments[participant] = (start + index) % folds
    elif strategy == "leave-one-site-out":
        sites = sorted(set(subject_site.values()))
        if len(sites) < 3:
            raise ValueError("Leave-one-site-out requires at least three sites")
        folds = len(sites)
        site_fold = {site: index for index, site in enumerate(sites)}
        assignments = {
            participant: site_fold[site] for participant, site in subject_site.items()
        }
    else:
        raise ValueError("Split strategy must be stratified-group or leave-one-site-out")

    output_rows: list[dict[str, Any]] = []
    for fold in range(folds):
        test_fold = fold
        validation_fold = (fold + 1) % folds
        for row in rows:
            participant = _participant(row.get("participant_id", ""))
            assigned = assignments[participant]
            split = "test" if assigned == test_fold else "validation" if assigned == validation_fold else "train"
            output_rows.append({**row, "fold": fold, "split": split})
    _write_jsonl(output, output_rows)
    audit = _audit_splits(output_rows, folds)
    summary = {
        "status": "pass",
        "strategy": strategy,
        "seed": seed,
        "folds": folds,
        "participants": len(by_subject),
        "visits": len(rows),
        "site_column": site_column,
        "label_column": label_column,
        "audit": audit,
        "output": str(output),
    }
    _write_json(report, summary)
    return summary


def _audit_splits(rows: list[dict[str, Any]], folds: int) -> dict[str, Any]:
    summaries = []
    for fold in range(folds):
        subset = [row for row in rows if int(row["fold"]) == fold]
        participants_by_split = {
            split: {
                _participant(row["participant_id"])
                for row in subset
                if row["split"] == split
            }
            for split in ("train", "validation", "test")
        }
        if (
            participants_by_split["train"] & participants_by_split["validation"]
            or participants_by_split["train"] & participants_by_split["test"]
            or participants_by_split["validation"] & participants_by_split["test"]
        ):
            raise RuntimeError(f"Participant leakage detected in fold {fold}")
        summaries.append(
            {
                "fold": fold,
                "participants": {
                    split: len(values) for split, values in participants_by_split.items()
                },
                "visits": {
                    split: sum(row["split"] == split for row in subset)
                    for split in participants_by_split
                },
            }
        )
    return {"participant_disjoint": True, "folds": summaries}


def _relative_phenotype_days(
    by_subject: dict[str, list[dict[str, str]]], date_column: str | None
) -> dict[int, int]:
    if not date_column:
        return {}
    result: dict[int, int] = {}
    for rows in by_subject.values():
        parsed = [(row, _parse_date(row.get(date_column, ""))) for row in rows]
        baseline = min((date for _, date in parsed if date), default=None)
        if baseline is None:
            continue
        for row, date in parsed:
            if date:
                result[id(row)] = int(round((date - baseline).total_seconds() / 86400))
    return result


def _parse_date(value: str) -> datetime | None:
    normalized = value.strip().replace("Z", "+00:00")
    for format_string in (None, "%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            parsed = datetime.fromisoformat(normalized) if format_string is None else datetime.strptime(normalized, format_string)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _participant(value: Any) -> str:
    return str(value).strip().removeprefix("sub-")


def _session(value: Any) -> str:
    return str(value).strip().removeprefix("ses-").removeprefix("visit-")


def _nested(row: dict[str, Any], section: str, key: str | None) -> Any:
    payload = row.get(section)
    return payload.get(key) if key and isinstance(payload, dict) else None


def _row_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
