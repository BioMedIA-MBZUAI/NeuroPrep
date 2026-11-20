from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from .bids import bids_relative_path
from .identity import IdentityResolver
from .models import (
    AcquisitionRecord,
    CanonicalAcquisitionGraph,
    ClassificationStatus,
    DatasetSpec,
    SessionRecord,
    SourceRecord,
    SubjectRecord,
)
from .registry import PluginRegistry


class IngestionEngine:
    """Rule-first, non-destructive conversion from source records to a canonical graph."""

    def __init__(
        self,
        registry: PluginRegistry,
        identity: IdentityResolver,
        minimum_score: float = 0.45,
        ambiguity_margin: float = 0.08,
    ) -> None:
        self.registry = registry
        self.identity = identity
        self.minimum_score = minimum_score
        self.ambiguity_margin = ambiguity_margin

    def build_graph(
        self,
        dataset: DatasetSpec,
        records: Iterable[SourceRecord],
    ) -> CanonicalAcquisitionGraph:
        source_records = list(records)
        for record in source_records:
            if record.dataset_id != dataset.dataset_id:
                raise ValueError(
                    f"Record dataset {record.dataset_id!r} does not match {dataset.dataset_id!r}"
                )

        baselines = self._baselines(source_records)
        subjects: dict[str, SubjectRecord] = {}
        sessions: dict[tuple[str, str], SessionRecord] = {}
        quarantined: list[AcquisitionRecord] = []
        unsupported: list[AcquisitionRecord] = []
        status_counts: Counter[str] = Counter()

        adapter_key = dataset.adapter_id if dataset.adapter_id != "generic" else dataset.dataset_id
        adapter = self.registry.dataset_adapter(adapter_key)

        for raw_record in source_records:
            record = adapter.normalize(raw_record)
            subject_id = self.identity.subject_id(dataset.dataset_id, record.source_subject_id)
            baseline = baselines.get(record.source_subject_id)
            session_id = self.identity.session_id(record, baseline)
            acquisition_id = self.identity.acquisition_id(record)
            candidates = self.registry.detect(record)
            status, issue = self._classification_status(candidates)

            selected = candidates[0] if status is ClassificationStatus.RECOGNIZED else None
            mapping = None
            if selected is not None:
                plugin = self.registry.modality_plugin(selected.plugin_id)
                mapping = plugin.build_bids_mapping(
                    record,
                    {
                        "subject": subject_id.removeprefix("sub-"),
                        "session": session_id.removeprefix("ses-"),
                        "acquisition": acquisition_id.removeprefix("acq-")[:8],
                    },
                )

            acquisition = AcquisitionRecord(
                acquisition_id=acquisition_id,
                source=record,
                subject_id=subject_id,
                session_id=session_id,
                status=status,
                candidates=candidates[:5],
                selected_plugin_id=selected.plugin_id if selected else None,
                modality=selected.modality if selected else None,
                bids=mapping,
                issues=[issue] if issue else [],
            )

            subject = subjects.setdefault(
                subject_id,
                SubjectRecord(
                    subject_id=subject_id,
                    dataset_id=dataset.dataset_id,
                    source_subject_ids=set(),
                ),
            )
            subject.source_subject_ids.add(record.source_subject_id)
            session = sessions.setdefault(
                (subject_id, session_id),
                SessionRecord(
                    session_id=session_id,
                    days_from_baseline=record.days_from_baseline,
                    acquisition_time=record.acquisition_time,
                ),
            )
            if record.source_session_id:
                session.source_session_ids.add(record.source_session_id)
            session.acquisitions.append(acquisition)

            if status is ClassificationStatus.AMBIGUOUS:
                quarantined.append(acquisition)
            elif status is ClassificationStatus.UNSUPPORTED:
                unsupported.append(acquisition)
            status_counts[str(status)] += 1

        for (subject_id, _), session in sessions.items():
            self._normalize_pet_entities(session)
            self._enrich_fieldmap_collection(session)
            self._stage_reverse_b0_as_fieldmap(session)
            subjects[subject_id].sessions.append(session)
        for subject in subjects.values():
            subject.sessions.sort(
                key=lambda item: (
                    item.days_from_baseline is None,
                    item.days_from_baseline or 0,
                    item.acquisition_time or datetime.max,
                    item.session_id,
                )
            )

        return CanonicalAcquisitionGraph(
            dataset=dataset,
            subjects=sorted(subjects.values(), key=lambda item: item.subject_id),
            quarantined=quarantined,
            unsupported=unsupported,
            provenance={
                "dataset_adapter": adapter.adapter_id,
                "classification_engine": "deterministic-rules-v1",
                "status_counts": dict(status_counts),
            },
        )

    @staticmethod
    def _normalize_pet_entities(session: SessionRecord) -> None:
        groups: dict[tuple[str, str, str], list[AcquisitionRecord]] = defaultdict(list)
        for item in session.acquisitions:
            if item.selected_plugin_id != "pet" or item.bids is None:
                continue
            entities = dict(item.bids.entities)
            entities.pop("acquisition", None)
            tracer = str(
                entities.get("tracer")
                or item.source.metadata.get("TracerName")
                or item.source.metadata.get("Tracer.Name")
                or ""
            )
            tracer = re.sub(r"[^A-Za-z0-9]+", "", tracer)
            if tracer:
                entities["tracer"] = tracer
            item.bids = item.bids.model_copy(update={"entities": entities})
            groups[(
                item.bids.suffix,
                str(entities.get("tracer") or ""),
                str(entities.get("task") or ""),
            )].append(item)
        for members in groups.values():
            if len(members) < 2:
                continue
            for index, item in enumerate(
                sorted(members, key=lambda member: member.source.source_path), start=1
            ):
                if item.bids is None:
                    continue
                entities = dict(item.bids.entities)
                entities["run"] = str(index)
                item.bids = item.bids.model_copy(update={"entities": entities})

    @staticmethod
    def _enrich_fieldmap_collection(session: SessionRecord) -> None:
        fieldmaps = [
            item
            for item in session.acquisitions
            if item.selected_plugin_id == "mri.fmap" and item.bids is not None
        ]
        echo_times = sorted(
            {
                float(item.source.metadata["EchoTime"])
                for item in fieldmaps
                if item.source.metadata.get("EchoTime") not in (None, "")
            }
        )
        if len(echo_times) < 2:
            return
        for item in fieldmaps:
            if item.bids is None or item.bids.suffix != "phasediff":
                continue
            additions = {"EchoTime1": echo_times[0], "EchoTime2": echo_times[-1]}
            item.source.metadata.update(additions)
            item.bids.sidecar.update(additions)

    @staticmethod
    def _stage_reverse_b0_as_fieldmap(session: SessionRecord) -> None:
        """Stage a b0-only DWI series with an opposite-PE partner as ``fmap/*_epi``.

        Datasets often keep the reverse-phase b0 pair for TOPUP under ``dwi/``. Processed
        as a DWI run of its own it breaks QSIPrep (Eddy cannot read an all-zero gradient
        table, CIAI 422645); BIDS represents it as an EPI fieldmap with ``IntendedFor``.
        """

        def bvals(item: AcquisitionRecord) -> list[float] | None:
            source = Path(item.source.source_path)
            stem = source.name[:-7] if source.name.endswith(".nii.gz") else source.stem
            path = source.with_name(stem + ".bval")
            try:
                return [float(value) for value in path.read_text(encoding="utf-8").split()]
            except (OSError, ValueError):
                return None

        def opposite(direction: str) -> str:
            return direction[:-1] if direction.endswith("-") else direction + "-"

        series = [
            (item, bvals(item), str(item.source.metadata.get("PhaseEncodingDirection") or ""))
            for item in session.acquisitions
            if item.selected_plugin_id == "mri.dwi"
            and item.bids is not None
            and item.bids.datatype == "dwi"
        ]
        weighted = [
            (item, direction)
            for item, values, direction in series
            if values and any(value >= 50 for value in values) and direction
        ]
        for item, values, direction in series:
            if not values or any(value >= 50 for value in values) or not direction:
                continue
            targets = [main for main, pe in weighted if pe == opposite(direction)]
            if not targets or item.bids is None:
                continue
            intended = [
                str(bids_relative_path(main).relative_to(f"{main.subject_id}"))
                for main in targets
            ]
            item.bids.datatype = "fmap"
            item.bids.suffix = "epi"
            item.bids.entities["direction"] = direction.replace("-", "neg")
            item.bids.sidecar["IntendedFor"] = intended

    def _classification_status(self, candidates):
        # Scores are additive decimal weights. Compare with a small tolerance so
        # a conceptually exact threshold such as 0.30 + 0.15 is not rejected by
        # binary floating-point representation.
        if not candidates or candidates[0].score + 1e-9 < self.minimum_score:
            return ClassificationStatus.UNSUPPORTED, "No modality reached the evidence threshold."
        if len(candidates) > 1:
            gap = candidates[0].score - candidates[1].score
            if gap < self.ambiguity_margin and candidates[0].plugin_id != candidates[1].plugin_id:
                return (
                    ClassificationStatus.AMBIGUOUS,
                    "Top modality candidates are too close; human or dataset-rule "
                    "review is required.",
                )
        return ClassificationStatus.RECOGNIZED, ""

    @staticmethod
    def _baselines(records: list[SourceRecord]) -> dict[str, datetime]:
        by_subject: dict[str, list[datetime]] = defaultdict(list)
        for record in records:
            if record.acquisition_time:
                by_subject[record.source_subject_id].append(record.acquisition_time)
        return {subject: min(times) for subject, times in by_subject.items() if times}
