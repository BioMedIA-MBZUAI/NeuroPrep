from pathlib import Path

import pytest
from crewai.project import load_crew

from neuroprep_agentic.control import (
    NeuroPrepControlFlow,
    authoritative_evidence,
    validate_control_payload,
)
from neuroprep_agentic.crews.control.crew import CREW_CONFIGS
from neuroprep_agentic.settings import AgenticSettings


def test_flow_definition_has_no_self_referential_listeners():
    flow = NeuroPrepControlFlow()
    assert flow.state.workflow == "coverage"
    assert flow.state.authoritative_artifact == ""


def test_authoritative_coverage_is_computed_without_agent_transcription():
    evidence = authoritative_evidence("coverage", AgenticSettings(workflow="coverage"))
    by_plugin = {item["plugin_id"]: item for item in evidence["modality_plugins"]}
    assert evidence["support_counts"] == {"bids_core": 23, "provisional": 3}
    assert by_plugin["ct"]["support_level"] == "provisional"
    assert by_plugin["mri.t1w"]["support_level"] == "bids_core"
    assert len(by_plugin) == 26


def test_control_payload_rejects_agent_narrative_as_authority():
    evidence = {"support_counts": {"bids_core": 22, "provisional": 3}}
    payload = {
        "workflow": "coverage",
        "narrative_authoritative": True,
        "authoritative_evidence": evidence,
    }
    with pytest.raises(ValueError, match="must never"):
        validate_control_payload(payload, "coverage", evidence)


def test_control_payload_rejects_tampered_evidence():
    evidence = {"support_counts": {"bids_core": 22, "provisional": 3}}
    payload = {
        "workflow": "coverage",
        "narrative_authoritative": False,
        "authoritative_evidence": {"support_counts": {"bids_core": 23}},
    }
    with pytest.raises(ValueError, match="does not match"):
        validate_control_payload(payload, "coverage", evidence)


def test_every_control_crew_configuration_loads():
    config_root = Path(__file__).parents[1] / "src" / "neuroprep_agentic" / "crews" / "control"
    for config_name in CREW_CONFIGS.values():
        crew, _defaults = load_crew(config_root / config_name)
        assert crew.agents
        assert crew.tasks
