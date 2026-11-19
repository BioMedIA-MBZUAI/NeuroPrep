from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from .contracts import audit_image
from .models import ImageRecord, Modality


OUTPUT_CONTRACT_VERSION = "3.1"
VISIT_TOLERANCE_DAYS = 30


@dataclass(frozen=True)
class VisitContext:
    """The acquisition context passed to every modality specialist."""

    subject: str
    visit_id: str
    visit_index: int
    visit_count: int
    acquisition_mode: str
    images: tuple[ImageRecord, ...]
    acquisition_time: str | None = None

    @property
    def is_longitudinal(self) -> bool:
        return self.visit_count > 1

    def to_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "visit_id": self.visit_id,
            "visit_index": self.visit_index,
            "visit_count": self.visit_count,
            "acquisition_mode": self.acquisition_mode,
            "is_longitudinal": self.is_longitudinal,
            "acquisition_time": self.acquisition_time,
            "modalities": sorted({image.modality.value for image in self.images}),
        }


@dataclass(frozen=True)
class ExpertSpec:
    """A deterministic, single-modality specialist-agent contract.

    The agent audits its modality and selects a pinned scientific executor.
    Numerical transforms remain in reviewed applications; the coordinator
    cannot invent transforms or silently relax acquisition requirements.
    """

    name: str
    modality: Modality
    pipeline: str
    output_kind: str
    native_space: str
    supports_subject_alignment: bool = True
    requires_dataset_policy: bool = False
    high_throughput_pipeline: str | None = None
    reference_pipeline: str | None = None
    longitudinal_strategy: str = "visit-local-independent-common-grid"
    missing_policy: str = "emit-explicit-missing-mask"

    @property
    def modalities(self) -> tuple[Modality, ...]:
        """Compatibility view for callers from the pre-agent API."""

        return (self.modality,)

    def pipeline_for(self, profile: str) -> str:
        if profile == "high-throughput" and self.high_throughput_pipeline:
            return self.high_throughput_pipeline
        if profile == "reference" and self.reference_pipeline:
            return self.reference_pipeline
        return self.pipeline

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["modality"] = self.modality.value
        payload["modalities"] = [self.modality.value]
        payload["agent_kind"] = "modality-specialist"
        payload["contract_version"] = OUTPUT_CONTRACT_VERSION
        return payload


# Public semantic alias: old integrations may still import ExpertSpec.
ModalitySpecialist = ExpertSpec


@dataclass(frozen=True)
class ExpertDecision:
    expert: str
    pipeline: str
    status: str
    modalities: tuple[str, ...]
    inputs: tuple[str, ...]
    output_kind: str
    native_space: str
    policy: str | None
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    subject: str | None = None
    visit_id: str | None = None
    visit_index: int | None = None
    visit_count: int | None = None
    visit_mode: str | None = None
    longitudinal_strategy: str | None = None
    missing_policy: str | None = None
    present: bool = True
    contract_version: str = OUTPUT_CONTRACT_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


EXPERTS: tuple[ExpertSpec, ...] = (
    ExpertSpec(
        "t1w",
        Modality.T1W,
        "smriprep",
        "spatial-image",
        "subject-anat",
        high_throughput_pipeline="deepprep",
    ),
    ExpertSpec(
        "t2w",
        Modality.T2W,
        "neuroprep-structural",
        "spatial-image",
        "t2w-native",
    ),
    ExpertSpec(
        "flair",
        Modality.FLAIR,
        "neuroprep-structural",
        "spatial-image",
        "flair-native",
    ),
    ExpertSpec(
        "pd",
        Modality.PD,
        "neuroprep-structural",
        "spatial-image",
        "pd-native",
    ),
    ExpertSpec(
        "fmri",
        Modality.FMRI,
        "fmriprep",
        "spatiotemporal-image",
        "bold-native",
        high_throughput_pipeline="deepprep",
    ),
    ExpertSpec(
        "dwi",
        Modality.DWI,
        "qsiprep",
        "gradient-image",
        "dwi-native",
        longitudinal_strategy="visit-local-gradient-safe",
    ),
    ExpertSpec(
        "pet",
        Modality.PET,
        "petprep",
        "quantitative-spatiotemporal-image",
        "pet-native",
    ),
    ExpertSpec("asl", Modality.ASL, "aslprep", "quantitative-image", "asl-native"),
    ExpertSpec(
        "swi",
        Modality.SWI,
        "neuroprep-susceptibility",
        "spatial-or-quantitative-image",
        "swi-native",
    ),
    ExpertSpec("mra", Modality.MRA, "neuroprep-mra", "spatial-image", "mra-native"),
    ExpertSpec(
        "ct",
        Modality.CT,
        "neuroprep-ct",
        "quantitative-image",
        "ct-native",
    ),
    ExpertSpec(
        "eeg",
        Modality.EEG,
        "mne-bids-pipeline-eeg",
        "sensor-timeseries",
        "sensor-native",
        longitudinal_strategy="visit-local-coordinate-alignment",
    ),
    ExpertSpec(
        "meg",
        Modality.MEG,
        "mne-bids-pipeline-meg",
        "sensor-timeseries",
        "sensor-native",
        longitudinal_strategy="visit-local-coordinate-alignment",
    ),
    ExpertSpec(
        "ieeg",
        Modality.IEEG,
        "mne-bids-pipeline-ieeg",
        "sensor-timeseries",
        "sensor-native",
        longitudinal_strategy="visit-local-coordinate-alignment",
    ),
    ExpertSpec(
        "fieldmap",
        Modality.FIELDMAP,
        "calibration-support",
        "calibration-image",
        "acquisition-native",
        supports_subject_alignment=False,
        longitudinal_strategy="attach-to-compatible-visit",
    ),
)


class ExpertRegistry:
    """Registry and deterministic coordinator for all modality specialists."""

    def __init__(self, experts: Iterable[ExpertSpec] = EXPERTS):
        self._experts = tuple(experts)
        self._by_modality: dict[Modality, ExpertSpec] = {}
        for expert in self._experts:
            if expert.modality is Modality.UNKNOWN:
                raise ValueError("Unknown inputs cannot have a processing specialist")
            if expert.modality in self._by_modality:
                raise ValueError(f"Multiple experts registered for {expert.modality.value}")
            self._by_modality[expert.modality] = expert
        missing = set(Modality) - {Modality.UNKNOWN} - set(self._by_modality)
        if missing:
            labels = ", ".join(sorted(modality.value for modality in missing))
            raise ValueError(f"Missing modality specialists: {labels}")

    def for_modality(self, modality: Modality) -> ExpertSpec | None:
        return self._by_modality.get(modality)

    def route(self, images: Iterable[ImageRecord]) -> list[ExpertSpec]:
        routed = {
            image.modality: self.for_modality(image.modality)
            for image in images
            if image.modality is not Modality.UNKNOWN
        }
        return [
            routed[modality]
            for modality in sorted(routed, key=lambda value: value.value)
            if routed[modality] is not None
        ]

    def describe(self) -> list[dict[str, object]]:
        return [expert.to_dict() for expert in self._experts]

    def visit_contexts(
        self,
        images: Iterable[ImageRecord],
        tolerance_days: int = VISIT_TOLERANCE_DAYS,
    ) -> list[VisitContext]:
        records = [
            image
            for image in images
            if image.subject != "__unknown__" and not image.is_derivative
        ]
        by_subject: dict[str, list[ImageRecord]] = {}
        for image in records:
            by_subject.setdefault(image.subject, []).append(image)
        contexts: list[VisitContext] = []
        for subject, subject_images in sorted(by_subject.items()):
            groups = _group_subject_visits(subject_images, tolerance_days)
            count = len(groups)
            mode = "longitudinal" if count > 1 else "single-visit"
            for index, (visit_id, visit_images, timestamp) in enumerate(groups):
                contexts.append(
                    VisitContext(
                        subject=subject,
                        visit_id=visit_id,
                        visit_index=index,
                        visit_count=count,
                        acquisition_mode=mode,
                        images=tuple(sorted(visit_images, key=lambda image: str(image.path))),
                        acquisition_time=timestamp.isoformat() if timestamp else None,
                    )
                )
        return contexts

    def visit_decisions(
        self,
        images: Iterable[ImageRecord],
        dataset_metadata: dict[str, Any] | None = None,
        include_missing: bool = True,
    ) -> list[ExpertDecision]:
        records = list(images)
        metadata = dataset_metadata or {}
        raw_policies = metadata.get("expert_policies", {})
        policies = raw_policies if isinstance(raw_policies, dict) else {}
        generated_policy_path = metadata.get("autoconfigure_policy")
        generated_policy = _generated_policy(generated_policy_path)
        profile = _profile_name(metadata)
        tolerance_days = int(metadata.get("visit_tolerance_days", VISIT_TOLERANCE_DAYS))
        decisions: list[ExpertDecision] = []
        for context in self.visit_contexts(records, tolerance_days):
            visit_modalities = {image.modality for image in context.images}
            for expert in self._experts:
                expert_images = [
                    image for image in context.images if image.modality is expert.modality
                ]
                if not expert_images and not include_missing:
                    continue
                pipeline = expert.pipeline_for(profile)
                policy_value = policies.get(
                    expert.name,
                    policies.get(expert.modality.value, policies.get(pipeline)),
                )
                policy = str(policy_value) if policy_value else None
                blockers: list[str] = []
                warnings: list[str] = []
                if expert_images:
                    issues = [
                        issue for image in expert_images for issue in audit_image(image)
                    ]
                    blockers.extend(
                        issue.code for issue in issues if issue.severity == "error"
                    )
                    warnings.extend(
                        issue.code for issue in issues if issue.severity != "error"
                    )
                    approved = set(generated_policy.get("auto_approved_modalities", []))
                    if (
                        not policy
                        and generated_policy_path
                        and expert.modality.value in approved
                    ):
                        policy = f"{generated_policy_path}#auto-approved"
                    if expert.requires_dataset_policy and not policy:
                        blockers.append("expert_policy_missing")
                    lacks_anatomical_anchor = not visit_modalities & {
                        Modality.T1W,
                        Modality.T2W,
                        Modality.CT,
                    }
                    if expert.modality is Modality.FMRI and lacks_anatomical_anchor:
                        # fMRIPrep 25.2.5 and DeepPrep both require a structural
                        # image. Never borrow one from another longitudinal
                        # visit: retain the visit as acquired-but-unreleased.
                        blockers.append("anatomical_anchor_missing_in_visit")
                    elif (
                        expert.modality
                        in {
                            Modality.DWI,
                            Modality.PET,
                            Modality.ASL,
                            Modality.SWI,
                            Modality.MRA,
                        }
                        and lacks_anatomical_anchor
                    ):
                        warnings.append("anatomical_anchor_missing_in_visit")
                    status = (
                        "blocked"
                        if blockers
                        else "review"
                        if warnings
                        else "ready"
                    )
                else:
                    status = "missing"
                decisions.append(
                    ExpertDecision(
                        expert=expert.name,
                        pipeline=pipeline,
                        status=status,
                        modalities=(expert.modality.value,),
                        inputs=tuple(sorted(str(image.path) for image in expert_images)),
                        output_kind=expert.output_kind,
                        native_space=expert.native_space,
                        policy=policy,
                        blockers=tuple(sorted(set(blockers))),
                        warnings=tuple(sorted(set(warnings))),
                        subject=context.subject,
                        visit_id=context.visit_id,
                        visit_index=context.visit_index,
                        visit_count=context.visit_count,
                        visit_mode=context.acquisition_mode,
                        longitudinal_strategy=expert.longitudinal_strategy,
                        missing_policy=expert.missing_policy,
                        present=bool(expert_images),
                    )
                )
        return decisions

    def decisions(
        self,
        images: Iterable[ImageRecord],
        dataset_metadata: dict[str, Any] | None = None,
    ) -> list[ExpertDecision]:
        """Return present visit-level decisions (legacy API name)."""

        return self.visit_decisions(
            images,
            dataset_metadata=dataset_metadata,
            include_missing=False,
        )


def _profile_name(metadata: dict[str, Any]) -> str:
    raw = str(metadata.get("execution_profile", "high-throughput"))
    name = raw.strip().lower().replace("_", "-")
    return {
        "fast": "high-throughput",
        "throughput": "high-throughput",
        "highthroughput": "high-throughput",
        "nipreps": "reference",
        "validated-reference": "reference",
    }.get(name, name)


def _generated_policy(path_value: object) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(str(path_value)).expanduser()
    if not path.exists():
        raise ValueError(f"Autoconfigure policy does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or str(payload.get("version")) != "3.0":
        raise ValueError(f"Invalid NeuroPrep autoconfigure policy: {path}")
    return payload


def _group_subject_visits(
    images: list[ImageRecord],
    tolerance_days: int,
) -> list[tuple[str, list[ImageRecord], datetime | None]]:
    explicit: dict[str, list[ImageRecord]] = {}
    unsessioned: list[ImageRecord] = []
    for image in images:
        if image.session:
            token = image.session.removeprefix("ses-")
            explicit.setdefault(f"ses-{token}", []).append(image)
        else:
            unsessioned.append(image)

    raw_groups: list[tuple[str | None, list[ImageRecord], datetime | None]] = [
        (session, session_images, _earliest_time(session_images))
        for session, session_images in explicit.items()
    ]

    # Partially curated datasets sometimes label only some acquisitions with a
    # BIDS session.  Attach an unlabeled acquisition to the nearest dated
    # explicit session when the match is unambiguous within the visit window.
    # With one explicit session, an undated acquisition is also safe to attach;
    # with multiple undated sessions it remains separate rather than guessing.
    residual: list[ImageRecord] = []
    for image in unsessioned:
        timestamp = _image_time(image)
        candidates = [
            (abs((timestamp - group_time).total_seconds()), index)
            for index, (_, _, group_time) in enumerate(raw_groups)
            if timestamp is not None
            and group_time is not None
            and _within_days(timestamp, group_time, tolerance_days)
        ]
        if candidates:
            _, target_index = min(candidates)
            session, grouped, group_time = raw_groups[target_index]
            grouped.append(image)
            raw_groups[target_index] = (
                session,
                grouped,
                min(timestamp, group_time) if group_time is not None else timestamp,
            )
        elif timestamp is None and len(raw_groups) == 1:
            raw_groups[0][1].append(image)
        else:
            residual.append(image)

    dated = sorted(
        residual,
        key=lambda image: (
            _image_time(image) or datetime.max.replace(tzinfo=timezone.utc),
            str(image.path),
        ),
    )
    clusters: list[list[ImageRecord]] = []
    for image in dated:
        if not clusters:
            clusters.append([image])
            continue
        anchor = _earliest_time(clusters[-1])
        timestamp = _image_time(image)
        if timestamp and anchor and _within_days(timestamp, anchor, tolerance_days):
            clusters[-1].append(image)
        elif timestamp is None and anchor is None:
            clusters[-1].append(image)
        else:
            clusters.append([image])
    for cluster in clusters:
        raw_groups.append((None, cluster, _earliest_time(cluster)))

    raw_groups.sort(
        key=lambda item: (
            item[2] or datetime.max.replace(tzinfo=timezone.utc),
            item[0] or "",
            str(min(image.path for image in item[1])),
        )
    )
    only_one_unsessioned = (
        len(raw_groups) == 1 and raw_groups[0][0] is None and not explicit
    )
    result: list[tuple[str, list[ImageRecord], datetime | None]] = []
    for index, (session, grouped, timestamp) in enumerate(raw_groups, start=1):
        visit_id = session or ("single" if only_one_unsessioned else f"visit-{index:03d}")
        result.append((visit_id, grouped, timestamp))
    return result


def _earliest_time(images: Iterable[ImageRecord]) -> datetime | None:
    return min(
        (timestamp for image in images if (timestamp := _image_time(image)) is not None),
        default=None,
    )


def _within_days(left: datetime, right: datetime, tolerance_days: int) -> bool:
    return abs((left - right).total_seconds()) <= max(0, tolerance_days) * 86_400


def _image_time(image: ImageRecord) -> datetime | None:
    if image.sidecar is None or not image.sidecar.exists():
        return None
    try:
        payload = json.loads(image.sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    candidates = (
        payload.get("AcquisitionDateTime"),
        payload.get("AcquisitionDatetime"),
        payload.get("StudyDateTime"),
        _date_and_time(payload.get("AcquisitionDate"), payload.get("AcquisitionTime")),
        _date_and_time(payload.get("StudyDate"), payload.get("StudyTime")),
    )
    for value in candidates:
        if not value:
            continue
        text = str(value).strip().replace("Z", "+00:00")
        for candidate in (text, text.replace(".", ":", 2) if "." in text[:10] else text):
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                continue
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _date_and_time(date: object, time: object) -> str | None:
    if not date:
        return None
    date_text = str(date).strip()
    if len(date_text) == 8 and date_text.isdigit():
        date_text = f"{date_text[:4]}-{date_text[4:6]}-{date_text[6:]}"
    time_text = str(time or "00:00:00").strip()
    if len(time_text) >= 6 and time_text[:6].isdigit():
        time_text = f"{time_text[:2]}:{time_text[2:4]}:{time_text[4:]}"
    return f"{date_text}T{time_text}"
