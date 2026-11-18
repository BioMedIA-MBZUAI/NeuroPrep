from __future__ import annotations

import json

from neuroprep_agentic.openjev import (
    OpenJevClient,
    check_boolean,
    choose_next_action,
    route_acquisition,
    score_urgency,
)


class _Response:
    def __init__(self, choice: str = "continue") -> None:
        self.choice = choice

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {
                "id": "shim-test",
                "model": "openjev",
                "answers": {
                    "decision": {
                        "type": "choice",
                        "choice": self.choice,
                        "probabilities": {self.choice: 0.91},
                    }
                },
            }
        ).encode()


def test_openjev_decision_is_bounded_and_advisory(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("neuroprep_agentic.openjev.urllib.request.urlopen", fake_urlopen)
    decision = choose_next_action({"task_id": "task-1"}, OpenJevClient(endpoint="http://jev"))
    assert decision.decision == "continue"
    assert decision.advisory_only is True
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["questions"]["decision"]["type"] == "choice"


def test_openjev_route_uses_redacted_state_contract(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode())
        assert payload["state"]
        return _Response("anat")

    monkeypatch.setattr("neuroprep_agentic.openjev.urllib.request.urlopen", fake_urlopen)
    decision = route_acquisition({"status": "ambiguous"}, OpenJevClient(endpoint="http://jev"))
    assert decision.kind == "route"


def test_openjev_boolean_and_score_answers_are_typed(monkeypatch) -> None:
    responses = iter(
        [
            {"answers": {"decision": {"type": "noul", "noul": 0.9}}},
            {
                "answers": {
                    "decision": {
                        "type": "score",
                        "score": 2.0,
                        "probabilities": {"0": 0.1, "1": 0.2, "2": 0.6, "3": 0.1},
                    }
                }
            },
        ]
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(next(responses)).encode()

    monkeypatch.setattr(
        "neuroprep_agentic.openjev.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    client = OpenJevClient(endpoint="http://jev")
    assert check_boolean({}, "Is it safe?", client).decision == "true"
    assert score_urgency({}, client).decision == "2.0"
