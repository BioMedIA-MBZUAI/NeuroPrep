from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .models import DatasetConfig


@dataclass(frozen=True)
class IngestRecord:
    source: Path
    participant: str
    session: str | None
    config: Path | None
    converter: str = "dcm2bids"
    command: tuple[str, ...] = ()


def read_ingest_manifest(path: Path) -> list[IngestRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Ingest manifest does not exist: {path}")
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    records: list[IngestRecord] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        required = {"source", "participant"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Ingest manifest is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            source = Path(row["source"]).expanduser()
            converter = (row.get("converter") or "dcm2bids").strip().lower()
            config_value = (row.get("config") or "").strip()
            config = Path(config_value).expanduser() if config_value else None
            participant = row["participant"].removeprefix("sub-").strip()
            session = (row.get("session") or "").removeprefix("ses-").strip() or None
            if not participant:
                raise ValueError(f"Ingest manifest row {row_number} has no participant")
            if not source.exists():
                raise FileNotFoundError(f"Ingest source does not exist: {source}")
            if converter == "dcm2bids":
                if config is None:
                    raise ValueError(f"Ingest manifest row {row_number} requires a config")
                if not config.exists():
                    raise FileNotFoundError(f"Dcm2Bids config does not exist: {config}")
            command_value = (row.get("command_json") or "").strip()
            command: tuple[str, ...] = ()
            if command_value:
                try:
                    decoded = json.loads(command_value)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Ingest manifest row {row_number} has invalid command_json"
                    ) from exc
                if not isinstance(decoded, list) or not decoded:
                    raise ValueError(
                        f"Ingest manifest row {row_number} command_json must be a non-empty list"
                    )
                command = tuple(str(value) for value in decoded)
            if converter != "dcm2bids" and not command:
                raise ValueError(
                    f"Ingest manifest row {row_number} converter {converter!r} requires command_json"
                )
            records.append(
                IngestRecord(
                    source.resolve(),
                    participant,
                    session,
                    config.resolve() if config else None,
                    converter,
                    command,
                )
            )
    if not records:
        raise ValueError("Ingest manifest has no records")
    identities = [(record.participant, record.session) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Ingest manifest contains duplicate participant/session rows")
    return records


def dcm2bids_command(record: IngestRecord, dataset: DatasetConfig) -> list[str]:
    if record.converter != "dcm2bids" or record.config is None:
        raise ValueError("dcm2bids_command requires a Dcm2Bids ingest record")
    command = [
        "dcm2bids",
        "-d",
        str(record.source),
        "-p",
        record.participant,
        "-c",
        str(record.config),
        "-o",
        str(dataset.bids_root),
        "--force_dcm2bids",
    ]
    if record.session:
        command.extend(("-s", record.session))
    return command


def ingest_command(record: IngestRecord, dataset: DatasetConfig) -> list[str]:
    if record.converter == "dcm2bids":
        return dcm2bids_command(record, dataset)
    bindings = {
        "source": str(record.source),
        "participant": record.participant,
        "session": record.session or "",
        "bids_root": str(dataset.bids_root),
        "work_root": str(dataset.work_root),
    }
    return [value.format_map(bindings) for value in record.command]
