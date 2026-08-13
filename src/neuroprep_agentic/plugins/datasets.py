from __future__ import annotations

from dataclasses import dataclass

from ..models import SourceRecord
from .base import DeclarativeDatasetAdapter


@dataclass(frozen=True)
class Oasis3DatasetAdapter(DeclarativeDatasetAdapter):
    """Normalize released OASIS-3 legacy metadata using documented protocol evidence."""

    def normalize(self, record: SourceRecord) -> SourceRecord:
        normalized = super().normalize(record)
        metadata = dict(normalized.metadata)
        modality = str(metadata.get("Modality") or "").upper()
        derivations = dict(metadata.get("NeuroPrepMetadataDerivations") or {})

        if modality in {"PT", "PET"}:
            legacy_tracer = metadata.get("Tracer.Name")
            if legacy_tracer and not metadata.get("TracerName"):
                metadata["TracerName"] = legacy_tracer
                derivations["TracerName"] = "copied from OASIS-3 Tracer.Name"
            isotope = metadata.get("Tracer.Isotope")
            if isotope and not metadata.get("TracerRadionuclide"):
                metadata["TracerRadionuclide"] = isotope
            if not metadata.get("Units"):
                metadata["Units"] = "arbitrary"
                derivations["Units"] = "source image physical units are not declared"
            dosage = metadata.get("Dosage")
            dosage_units = str(metadata.get("DosageUnits") or "")
            if dosage not in (None, "") and dosage_units.lower() == "mci":
                metadata["InjectedRadioactivity"] = float(dosage) * 37.0
                metadata["InjectedRadioactivityUnits"] = "MBq"
            metadata["InjectedMass"] = metadata.get("InjectedMass") or "n/a"
            metadata["InjectedMassUnits"] = metadata.get("InjectedMassUnits") or "n/a"
            metadata["SpecificRadioactivity"] = metadata.get("SpecificRadioactivity") or "n/a"
            metadata["SpecificRadioactivityUnits"] = (
                metadata.get("SpecificRadioactivityUnits") or "n/a"
            )
            administration = metadata.get("Tracer.InjectionType")
            if administration:
                metadata["ModeOfAdministration"] = str(administration).lower()
            time_zero = metadata.get("InjectionTime") or metadata.get("ScanStartTime")
            if time_zero:
                metadata["TimeZero"] = time_zero
                metadata["InjectionStart"] = 0
                metadata["ScanStart"] = 0
            frame_times = metadata.get("Time", {}).get("FrameTimes", {})
            labels = frame_times.get("Labels", []) if isinstance(frame_times, dict) else []
            values = frame_times.get("Values", []) if isinstance(frame_times, dict) else []
            if labels and values:
                by_name = {name: index for index, name in enumerate(labels)}
                if "frameStart" in by_name:
                    metadata["FrameTimesStart"] = [
                        row[by_name["frameStart"]] for row in values
                    ]
                if "frameDuration" in by_name:
                    metadata["FrameDuration"] = [
                        row[by_name["frameDuration"]] for row in values
                    ]
            recon = metadata.get("Recon.MethodName")
            if recon:
                metadata["ReconMethodName"] = recon
            pet_reconstruction_defaults = {
                "AcquisitionMode": "n/a",
                "ReconMethodParameterLabels": ["n/a"],
                "ReconMethodParameterValues": [0],
                "ReconMethodParameterUnits": ["none"],
                "ReconFilterType": ["n/a"],
                "ReconFilterSize": [0],
            }
            missing_reconstruction = []
            for key, value in pet_reconstruction_defaults.items():
                if metadata.get(key) in (None, "", []):
                    metadata[key] = value
                    missing_reconstruction.append(key)
            if missing_reconstruction:
                derivations["PETReconstructionPlaceholders"] = (
                    "Required PET-BIDS reconstruction fields absent from released source "
                    "metadata; explicit n/a labels and zero-valued sizes/parameters encode "
                    "unavailability without inferring scanner settings: "
                    + ", ".join(missing_reconstruction)
                )
            if str(metadata.get("Data.DecayCorrectionTime") or "").lower() == "start":
                metadata["ImageDecayCorrected"] = True
                metadata["ImageDecayCorrectionTime"] = 0

        series = " ".join(
            str(metadata.get(key) or "")
            for key in ("SeriesDescription", "PulseSequenceDetails", "SequenceName")
        ).lower()
        if "asl" in series or "pasl" in series:
            scanner = str(metadata.get("DeviceSerialNumber") or "")
            software = str(metadata.get("SoftwareVersions") or "")
            if scanner == "51010" and software in {"syngo_MR_B18P", "syngo_MR_B20P"}:
                metadata.update(
                    {
                        "ArterialSpinLabelingType": "PASL",
                        "PostLabelingDelay": 1.9,
                        "BolusCutOffFlag": False,
                        "BackgroundSuppression": False,
                        "M0Type": "Included",
                        "RepetitionTimePreparation": metadata.get("RepetitionTime"),
                        "NeuroPrepM0VolumePosition": "first",
                    }
                )
                derivations["ASLProtocol"] = (
                    "OASIS-3 Imaging Data Dictionary v2.3 Table 7: scanner 51010, "
                    f"{software}, TI1=0.7 s, TI2=1.9 s, PASL, no background suppression"
                )

        if derivations:
            metadata["NeuroPrepMetadataDerivations"] = derivations
        return normalized.model_copy(update={"metadata": metadata})


def builtin_dataset_adapters() -> tuple[DeclarativeDatasetAdapter, ...]:
    """Known source-layout adapters; the core remains independent of this list."""

    return (
        DeclarativeDatasetAdapter(
            adapter_id="generic",
            aliases=("custom", "unknown"),
            notes="Metadata-driven fallback with no cohort assumptions.",
        ),
        DeclarativeDatasetAdapter(
            adapter_id="oasis4",
            aliases=("oasis-4",),
            subject_fields=("subject_id", "OASISID", "PatientID"),
            session_fields=("session_id", "days_to_visit", "StudyDate"),
        ),
        Oasis3DatasetAdapter(
            adapter_id="oasis3",
            aliases=("oasis-3",),
            subject_fields=("subject_id", "OASISID", "PatientID"),
            session_fields=("session_id", "days_to_visit", "StudyDate"),
        ),
        DeclarativeDatasetAdapter(
            adapter_id="adni",
            aliases=("adni1", "adnigo", "adni2", "adni3", "adni4"),
            subject_fields=("PTID", "RID", "PatientID"),
            session_fields=("VISCODE", "VISCODE2", "EXAMDATE", "StudyDate"),
        ),
        DeclarativeDatasetAdapter(
            adapter_id="anmerge",
            aliases=("addneuromed", "an-merge"),
            subject_fields=("subject_id", "PatientID"),
            session_fields=("visit", "StudyDate"),
        ),
        DeclarativeDatasetAdapter(
            adapter_id="a4",
            aliases=("anti-amyloid-treatment-asymptomatic-alzheimers",),
            subject_fields=("subject_id", "PatientID"),
            session_fields=("visit", "StudyDate"),
        ),
        DeclarativeDatasetAdapter(
            adapter_id="ideas",
            aliases=("imaging-dementia-evidence-amyloid-scanning",),
            subject_fields=("subject_id", "PatientID"),
            session_fields=("visit", "StudyDate"),
        ),
        DeclarativeDatasetAdapter(
            adapter_id="aibl",
            aliases=("australian-imaging-biomarkers-lifestyle",),
            subject_fields=("RID", "subject_id", "PatientID"),
            session_fields=("VISCODE", "EXAMDATE", "StudyDate"),
        ),
        DeclarativeDatasetAdapter(
            adapter_id="mr-rate",
            aliases=("mrrate", "mr_rate"),
            subject_fields=("subject_id", "PatientID"),
            session_fields=("visit", "StudyDate"),
        ),
    )
