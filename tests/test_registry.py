from neuroprep_agentic.models import SourceRecord, SupportLevel
from neuroprep_agentic.registry import builtin_registry


def source(path: str, modality: str, description: str) -> SourceRecord:
    return SourceRecord(
        dataset_id="custom-study",
        source_path=path,
        source_subject_id="source-1",
        metadata={"Modality": modality, "SeriesDescription": description},
    )


def test_registry_covers_requested_datasets_and_modalities():
    registry = builtin_registry()
    assert {
        "generic",
        "oasis4",
        "oasis3",
        "adni",
        "anmerge",
        "a4",
        "ideas",
        "aibl",
        "mr-rate",
    } <= set(registry.datasets)
    assert {
        "mri.t1w",
        "mri.t2w",
        "mri.flair",
        "mri.bold",
        "mri.dwi",
        "mri.swi",
        "mri.asl",
        "mri.angio",
        "mri.mrs",
        "pet",
        "ct",
        "eeg",
        "ieeg",
        "meg",
        "fnirs",
    } <= set(registry.modalities)


def test_ct_is_explicitly_provisional():
    plugin = builtin_registry().modality_plugin("ct")
    assert plugin.support_level is SupportLevel.PROVISIONAL
    assert plugin.schema_id == "bep024-provisional"


def test_non_nifti_native_modalities_preserve_their_source_container() -> None:
    registry = builtin_registry()
    ultrasound = registry.modality_plugin("ultrasound").build_bids_mapping(
        source("/raw/doppler.mp4", "US", "transcranial doppler ultrasound"),
        {"subject": "01"},
    )
    microscopy = registry.modality_plugin("microscopy").build_bids_mapping(
        source("/raw/slide.ome.tif", "SM", "histology microscopy"),
        {"subject": "01"},
    )
    genetics = registry.modality_plugin("genetics").build_bids_mapping(
        source("/raw/variants.vcf", "GENETIC", "genetic variants"), {"subject": "01"}
    )
    physiology = registry.modality_plugin("physiology").build_bids_mapping(
        source("/raw/cardiac.edf", "PHYSIO", "cardiac physiology"), {"subject": "01"}
    )
    assert ultrasound.extension == ".mp4"
    assert microscopy.extension == ".ome.tif"
    assert genetics.extension == ".vcf"
    assert physiology.extension == ".edf"


def test_detection_is_evidence_ranked():
    candidates = builtin_registry().detect(source("T1_MPRAGE.dcm", "MR", "T1w MPRAGE"))
    assert candidates[0].plugin_id == "mri.t1w"
    assert candidates[0].score > candidates[1].score


def test_oasis3_adapter_normalizes_documented_asl_and_legacy_pet_metadata() -> None:
    adapter = builtin_registry().dataset_adapter("oasis3")
    asl = adapter.normalize(
        SourceRecord(
            dataset_id="oasis3",
            source_path="/source/sub_asl.nii.gz",
            source_subject_id="OAS30001",
            metadata={
                "Modality": "MR",
                "SeriesDescription": "Axial ASL",
                "PulseSequenceDetails": "%SiemensSeq%_ep2d_pasl",
                "DeviceSerialNumber": "51010",
                "SoftwareVersions": "syngo_MR_B18P",
                "RepetitionTime": 3.4,
            },
        )
    )
    assert asl.metadata["ArterialSpinLabelingType"] == "PASL"
    assert asl.metadata["PostLabelingDelay"] == 1.9
    assert asl.metadata["M0Type"] == "Included"
    pet = adapter.normalize(
        SourceRecord(
            dataset_id="oasis3",
            source_path="/source/sub_pet.nii.gz",
            source_subject_id="OAS30001",
            metadata={
                "Modality": "PT",
                "Tracer.Name": "AV45",
                "Tracer.Isotope": "F18",
                "Tracer.InjectionType": "Bolus",
                "Dosage": 10.0,
                "DosageUnits": "mCi",
            },
        )
    )
    assert pet.metadata["TracerName"] == "AV45"
    assert pet.metadata["Units"] == "arbitrary"
    assert pet.metadata["InjectedRadioactivity"] == 370.0
    assert pet.metadata["AcquisitionMode"] == "n/a"
    assert pet.metadata["ReconMethodParameterLabels"] == ["n/a"]
    assert pet.metadata["ReconMethodParameterValues"] == [0]
    assert pet.metadata["ReconFilterType"] == ["n/a"]
