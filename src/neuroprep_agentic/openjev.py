"""Local OpenJev typed decisions for bounded NeuroPrep orchestration.

OpenJev is advisory. It may choose among registered actions, but it cannot
create commands, alter a plan, bypass QC, or authorize release.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DecisionKind = Literal["route", "boolean", "urgency", "next_action"]


class OpenJevDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    kind: DecisionKind
    model: str
    advisory_only: Literal[True] = True


class OpenJevClient:
    """Small dependency-free client for a local OpenJev helper endpoint."""

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        timeout: float = 90.0,
    ) -> None:
        self.endpoint = (
            endpoint
            or os.getenv("NEUROPREP_OPENJEV_ENDPOINT", "http://127.0.0.1:3000/v1/systemone")
        ).rstrip("/")
        self.model = model or os.getenv("NEUROPREP_OPENJEV_MODEL", "openjev")
        self.timeout = timeout

    def decide(
        self,
        *,
        state: dict[str, Any],
        kind: DecisionKind,
        instructions: str,
        options: list[str],
    ) -> OpenJevDecision:
        if not options or len(options) > 52:
            raise ValueError("OpenJev requires 1-52 bounded options")
        payload = {
            "model": self.model,
            "state": json.dumps(state, sort_keys=True, separators=(",", ":")),
            "questions": {
                "decision": {
                    "type": (
                        "noul"
                        if kind == "boolean"
                        else "score"
                        if kind == "urgency"
                        else "choice"
                    ),
                    "instructions": instructions,
                    "criteria": (
                        {"true": options[0], "false": options[1]}
                        if kind == "boolean"
                        else options
                        if kind == "urgency"
                        else {option: None for option in options}
                    ),
                }
            },
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenJev request failed: {exc}") from exc
        answer = body.get("decision")
        if answer is None and isinstance(body.get("answers"), dict):
            answer = body["answers"].get("decision")
        if isinstance(answer, dict):
            selected = answer.get("choice") or answer.get("label") or answer.get("decision")
            probabilities = answer.get("probabilities") or answer.get("scores") or {}
        else:
            selected = answer
            probabilities = body.get("probabilities") or {}
        if selected not in options:
            if kind == "boolean" and isinstance(answer, dict) and "noul" in answer:
                selected = "true" if float(answer["noul"]) >= 0.5 else "false"
                probabilities = {selected: float(answer["noul"])}
            elif kind == "urgency" and isinstance(answer, dict) and "score" in answer:
                selected = str(answer["score"])
                probabilities = answer.get("probabilities") or {}
        if selected not in options and not (kind == "urgency" and selected is not None):
            raise RuntimeError("OpenJev returned an option outside the registered action set")
        return OpenJevDecision(
            decision=str(selected),
            probabilities={str(k): float(v) for k, v in probabilities.items()},
            kind=kind,
            model=self.model,
        )


def route_acquisition(
    state: dict[str, Any], client: OpenJevClient | None = None
) -> OpenJevDecision:
    return (client or OpenJevClient()).decide(
        state=state,
        kind="route",
        instructions="Select the NeuroPrep workflow route. Choose manual_review for ambiguity.",
        options=["anat", "func", "dwi", "pet", "eeg", "mrs", "manual_review"],
    )


def choose_next_action(
    state: dict[str, Any], client: OpenJevClient | None = None
) -> OpenJevDecision:
    return (client or OpenJevClient()).decide(
        state=state,
        kind="next_action",
        instructions="Choose the next bounded NeuroPrep action; never invent a command.",
        options=[
            "continue",
            "retry",
            "switch_backend",
            "quarantine",
            "request_metadata",
            "human_review",
            "stop",
        ],
    )


def check_boolean(
    state: dict[str, Any], instructions: str, client: OpenJevClient | None = None
) -> OpenJevDecision:
    return (client or OpenJevClient()).decide(
        state=state,
        kind="boolean",
        instructions=instructions,
        options=["true", "false"],
    )


def score_urgency(
    state: dict[str, Any], client: OpenJevClient | None = None
) -> OpenJevDecision:
    return (client or OpenJevClient()).decide(
        state=state,
        kind="urgency",
        instructions="Score the urgency for scheduling this NeuroPrep task.",
        options=["routine", "same_day", "priority", "critical"],
    )


def review_acquisitions(
    acquisitions: list[dict[str, Any]], client: OpenJevClient | None = None
) -> list[dict[str, Any]]:
    """Review only unresolved acquisitions and return provenance-safe decisions."""
    reviewer = client or OpenJevClient()
    results: list[dict[str, Any]] = []
    for acquisition in acquisitions:
        decision = route_acquisition(
            {
                "acquisition_id": acquisition.get("acquisition_id"),
                "subject_id": acquisition.get("subject_id"),
                "session_id": acquisition.get("session_id"),
                "status": acquisition.get("status"),
                "source": {
                    "format": (acquisition.get("source") or {}).get("format"),
                    "path": "redacted",
                },
                "metadata_keys": sorted((acquisition.get("source") or {}).get("metadata", {})),
            },
            reviewer,
        )
        results.append(
            {
                "acquisition_id": acquisition.get("acquisition_id"),
                "source_status": acquisition.get("status"),
                "decision": decision.model_dump(mode="json"),
            }
        )
    return results
