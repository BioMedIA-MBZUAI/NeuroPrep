from datetime import datetime

import pytest

from neuroprep_agentic.identity import IdentityResolver
from neuroprep_agentic.models import SourceRecord


def test_dataset_namespace_prevents_accidental_cross_dataset_linking():
    resolver = IdentityResolver("a-strong-test-key")
    assert resolver.subject_id("adni", "001") != resolver.subject_id("aibl", "001")
    assert resolver.subject_id("adni", "001") == resolver.subject_id("adni", "001")


def test_cross_dataset_link_requires_explicit_crosswalk():
    resolver = IdentityResolver(
        "a-strong-test-key",
        {"adni:001": "shared0001", "aibl:A01": "shared0001"},
    )
    assert resolver.subject_id("adni", "001") == "sub-shared0001"
    assert resolver.subject_id("aibl", "A01") == "sub-shared0001"


def test_longitudinal_session_uses_relative_days():
    resolver = IdentityResolver("a-strong-test-key")
    record = SourceRecord(
        dataset_id="study",
        source_path="/source/file.dcm",
        source_subject_id="p1",
        acquisition_time=datetime(2025, 2, 1),
    )
    assert resolver.session_id(record, datetime(2025, 1, 1)) == "ses-d00031"


def test_negative_relative_day_has_bids_safe_label():
    resolver = IdentityResolver("a-strong-test-key")
    record = SourceRecord(
        dataset_id="study",
        source_path="/source/file.dcm",
        source_subject_id="p1",
        days_from_baseline=-30,
    )
    assert resolver.session_id(record) == "ses-m00030"


def test_weak_identity_key_is_rejected():
    with pytest.raises(ValueError, match="at least 12"):
        IdentityResolver("weak")
