import json

from neuroprep_agentic.models import DatasetSpec, SourceRecord
from neuroprep_agentic.services import ManifestDocument, build_ingest_plan


def test_manifest_service_is_dataset_agnostic_and_pseudonymous():
    manifest = ManifestDocument(
        dataset=DatasetSpec(dataset_id="future-cohort"),
        records=[
            SourceRecord(
                dataset_id="future-cohort",
                source_path="/incoming/person-name/T1w.dcm",
                source_subject_id="person-name",
                source_session_id="baseline",
                metadata={"Modality": "MR", "SeriesDescription": "T1w MPRAGE"},
            ),
            SourceRecord(
                dataset_id="future-cohort",
                source_path="/incoming/person-name/eeg.edf",
                source_subject_id="person-name",
                source_session_id="baseline",
                metadata={"Modality": "EEG", "SeriesDescription": "EEG resting"},
            ),
        ],
    )
    result = build_ingest_plan(manifest, "a-strong-test-key")
    assert result["dataset_adapter"] == "generic"
    assert result["subjects"] == 1
    assert {item["plugin_id"] for item in result["mappings"]} == {"mri.t1w", "eeg"}
    assert all(item["subject_id"] != "person-name" for item in result["mappings"])
    serialized = json.dumps(result)
    assert "person-name" not in serialized
    assert "/incoming/" not in serialized


def test_agent_facing_issues_redact_source_paths():
    manifest = ManifestDocument(
        dataset=DatasetSpec(dataset_id="future-cohort"),
        records=[
            SourceRecord(
                dataset_id="future-cohort",
                source_path="/incoming/person-name/localizer.dcm",
                source_subject_id="person-name",
                metadata={"Modality": "MR", "SeriesDescription": "unknown localizer"},
            )
        ],
    )
    result = build_ingest_plan(manifest, "a-strong-test-key")
    assert result["quarantined"] == 1
    assert result["plan"]["issues"][0]["path"] is None
    serialized = json.dumps(result)
    assert "person-name" not in serialized
    assert "/incoming/" not in serialized


def test_explicit_crosswalk_is_the_only_merge_mechanism():
    manifest = ManifestDocument(
        dataset=DatasetSpec(dataset_id="study"),
        records=[
            SourceRecord(
                dataset_id="study",
                source_path="/raw/a_t1.dcm",
                source_subject_id="A",
                metadata={"Modality": "MR", "SeriesDescription": "T1w MPRAGE"},
            ),
            SourceRecord(
                dataset_id="study",
                source_path="/raw/b_t1.dcm",
                source_subject_id="B",
                metadata={"Modality": "MR", "SeriesDescription": "T1w MPRAGE"},
            ),
        ],
        explicit_identity_crosswalk={"study:A": "linked", "study:B": "linked"},
    )
    result = build_ingest_plan(manifest, "a-strong-test-key")
    assert result["subjects"] == 1
