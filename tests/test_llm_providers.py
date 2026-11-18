"""Provider transports against a local stub server (no network, no keys)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import BaseModel

from neuroprep_agentic.llm import (
    LLMError,
    resolve_endpoint,
    split_model_spec,
    structured_chat,
)

POLICY = {"gpu": True, "packages": ["fmriprep"]}


class Schema(BaseModel):
    gpu: bool
    packages: list[str]


class _Stub:
    def __init__(self, responder):
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                # HTTP header names are case-insensitive; urllib title-cases them on the wire.
                headers = {key.lower(): value for key, value in self.headers.items()}
                outer.requests.append({"path": self.path, "headers": headers, "body": body})
                status, payload = responder(self.path, body, len(outer.requests))
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    created: list[_Stub] = []

    def make(responder):
        item = _Stub(responder)
        created.append(item)
        return item

    yield make
    for item in created:
        item.close()


MESSAGES = [
    {"role": "system", "content": "You compile policies."},
    {"role": "user", "content": "use a GPU"},
]


def test_model_spec_validation():
    assert split_model_spec("openrouter/anthropic/claude-sonnet-4.5") == (
        "openrouter",
        "anthropic/claude-sonnet-4.5",
    )
    for bad in ("qwen3:8b", "ollama/", "nope/model"):
        with pytest.raises(ValueError):
            split_model_spec(bad)


def test_hosted_providers_require_keys_and_never_use_the_local_base_url():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        resolve_endpoint("openai/gpt-4o-mini", environ={})
    env = {"OPENAI_API_KEY": "sk-test", "NEUROPREP_LLM_BASE_URL": "http://localhost:8000/v1"}
    endpoint = resolve_endpoint("openai/gpt-4o-mini", environ=env)
    assert endpoint.base_url == "https://api.openai.com/v1"
    assert resolve_endpoint("vllm/m", environ=env).base_url == "http://localhost:8000/v1"
    assert resolve_endpoint("openrouter/a/b", environ={"OPENROUTER_API_KEY": "k"}).base_url == (
        "https://openrouter.ai/api/v1"
    )


def test_ollama_native_chat(stub):
    server = stub(lambda path, body, n: (200, {"message": {"content": json.dumps(POLICY)}}))
    endpoint = resolve_endpoint("ollama/qwen3:8b", base_url=server.url, environ={})
    raw = structured_chat(endpoint, MESSAGES, Schema.model_json_schema(), seed=7)
    assert Schema.model_validate_json(raw).packages == ["fmriprep"]
    request = server.requests[0]
    assert request["path"] == "/api/chat"
    assert request["body"]["think"] is False
    assert request["body"]["format"]["type"] == "object"
    assert request["body"]["options"]["seed"] == 7


@pytest.mark.parametrize("provider", ["vllm", "openai_compatible", "lmstudio"])
def test_openai_compatible_json_schema(stub, provider):
    server = stub(
        lambda path, body, n: (
            200,
            {"choices": [{"message": {"content": "```json\n" + json.dumps(POLICY) + "\n```"}}]},
        )
    )
    endpoint = resolve_endpoint(f"{provider}/m", base_url=server.url + "/v1", environ={})
    raw = structured_chat(endpoint, MESSAGES, Schema.model_json_schema())
    assert Schema.model_validate_json(raw).gpu is True
    request = server.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["body"]["response_format"]["type"] == "json_schema"
    assert request["headers"]["authorization"] == "Bearer EMPTY"


def test_openrouter_sends_bearer_key(stub):
    server = stub(
        lambda path, body, n: (200, {"choices": [{"message": {"content": json.dumps(POLICY)}}]})
    )
    endpoint = resolve_endpoint(
        "openrouter/vendor/model",
        base_url=server.url,
        environ={"OPENROUTER_API_KEY": "or-key"},
    )
    structured_chat(endpoint, MESSAGES, Schema.model_json_schema())
    assert server.requests[0]["headers"]["authorization"] == "Bearer or-key"
    assert server.requests[0]["body"]["model"] == "vendor/model"


def test_openai_compatible_degrades_when_json_schema_unsupported(stub):
    def responder(path, body, n):
        if body.get("response_format", {}).get("type") == "json_schema":
            return 400, {"error": "response_format json_schema not supported"}
        return 200, {"choices": [{"message": {"content": json.dumps(POLICY)}}]}

    server = stub(responder)
    endpoint = resolve_endpoint("vllm/m", base_url=server.url + "/v1", environ={})
    raw = structured_chat(endpoint, MESSAGES, Schema.model_json_schema())
    assert json.loads(raw) == POLICY
    assert [r["body"].get("response_format", {}).get("type") for r in server.requests] == [
        "json_schema",
        "json_object",
    ]
    # the fallback prompt must carry the schema, since the server will not enforce it
    assert "JSON Schema" in server.requests[1]["body"]["messages"][-1]["content"]


def test_auth_failure_is_not_retried_or_degraded(stub):
    server = stub(lambda path, body, n: (401, {"error": "bad key"}))
    endpoint = resolve_endpoint("vllm/m", base_url=server.url + "/v1", environ={})
    with pytest.raises(LLMError, match="401"):
        structured_chat(endpoint, MESSAGES, Schema.model_json_schema())
    assert len(server.requests) == 1


def test_anthropic_forced_tool_use(stub):
    def responder(path, body, n):
        return 200, {"content": [{"type": "tool_use", "name": "emit_policy", "input": POLICY}]}

    server = stub(responder)
    endpoint = resolve_endpoint(
        "anthropic/claude-test", base_url=server.url, environ={"ANTHROPIC_API_KEY": "ak"}
    )
    raw = structured_chat(endpoint, MESSAGES, Schema.model_json_schema())
    assert json.loads(raw) == POLICY
    request = server.requests[0]
    assert request["path"] == "/v1/messages"
    assert request["headers"]["x-api-key"] == "ak"
    assert request["headers"]["anthropic-version"]
    assert request["body"]["tool_choice"] == {"type": "tool", "name": "emit_policy"}
    assert request["body"]["system"] == "You compile policies."
    assert [m["role"] for m in request["body"]["messages"]] == ["user"]


def test_anthropic_retries_without_rejected_temperature(stub):
    def responder(path, body, n):
        if "temperature" in body:
            return 400, {"error": {"message": "`temperature` is deprecated for this model"}}
        return 200, {"content": [{"type": "tool_use", "name": "emit_policy", "input": POLICY}]}

    server = stub(responder)
    endpoint = resolve_endpoint(
        "anthropic/claude-test", base_url=server.url, environ={"ANTHROPIC_API_KEY": "ak"}
    )
    assert json.loads(structured_chat(endpoint, MESSAGES, Schema.model_json_schema())) == POLICY
    assert len(server.requests) == 2


def test_transient_errors_are_retried(stub, monkeypatch):
    monkeypatch.setattr("neuroprep_agentic.llm.time.sleep", lambda _s: None)

    def responder(path, body, n):
        if n == 1:
            return 503, {"error": "overloaded"}
        return 200, {"message": {"content": json.dumps(POLICY)}}

    server = stub(responder)
    endpoint = resolve_endpoint("ollama/m", base_url=server.url, environ={})
    assert json.loads(structured_chat(endpoint, MESSAGES, Schema.model_json_schema())) == POLICY
    assert len(server.requests) == 2


def test_settings_accept_any_provider_and_reject_unknown(monkeypatch):
    from neuroprep_agentic.settings import AgenticSettings

    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
    settings = AgenticSettings(manager_model="anthropic/claude-x", worker_model="ollama/qwen3:8b")
    assert settings.endpoint(settings.manager_model).provider == "anthropic"
    with pytest.raises(ValueError):
        AgenticSettings(manager_model="qwen3:14b")


def test_policy_schema_exposes_package_allowlist_as_enum():
    from neuroprep_agentic.policy import KNOWN_PACKAGES, RunPolicy

    items = RunPolicy.model_json_schema()["properties"]["preferred_packages"]["items"]
    assert items["type"] == "string"
    assert set(items["enum"]) == set(KNOWN_PACKAGES)


def test_policy_compiler_repairs_one_rejected_reply(stub, monkeypatch):
    """Reproduces the live qwen3:14b failure: general Python libraries as package
    preferences. The first reply is rejected, the model sees the error, the second passes."""

    from neuroprep_agentic.policy import HardwareProfile, RunPolicy, compile_prompt_policy

    bad = {**RunPolicy().model_dump(mode="json"), "preferred_packages": ["nilearn", "nipype"]}
    good = {**RunPolicy().model_dump(mode="json"), "preferred_packages": ["fmriprep"]}

    def responder(path, body, n):
        policy = bad if n == 1 else good
        return 200, {"message": {"content": json.dumps(policy)}}

    server = stub(responder)
    monkeypatch.setenv("NEUROPREP_MANAGER_MODEL", "ollama/test-model")
    monkeypatch.setenv("OLLAMA_BASE_URL", server.url)
    hardware = HardwareProfile(
        hostname="h",
        cpus=8,
        memory_gb=32,
        gpus=0,
        gpu_memory_gb=0,
        gpu_models=[],
        cuda_version=None,
        container_runtime="none",
    )
    policy = compile_prompt_policy("Preprocess fMRI with fMRIPrep.", hardware)

    assert policy.preferred_packages == ["fmriprep"]
    assert len(server.requests) == 2
    repair_turn = server.requests[1]["body"]["messages"][-1]["content"]
    assert "rejected by the schema validator" in repair_turn
    assert "nilearn" in repair_turn


def test_policy_compiler_gives_up_after_one_repair(stub, monkeypatch):
    from neuroprep_agentic.policy import HardwareProfile, RunPolicy, compile_prompt_policy

    bad = {**RunPolicy().model_dump(mode="json"), "preferred_packages": ["nilearn"]}
    server = stub(lambda path, body, n: (200, {"message": {"content": json.dumps(bad)}}))
    monkeypatch.setenv("NEUROPREP_MANAGER_MODEL", "ollama/test-model")
    monkeypatch.setenv("OLLAMA_BASE_URL", server.url)
    hardware = HardwareProfile(
        hostname="h",
        cpus=8,
        memory_gb=32,
        gpus=0,
        gpu_memory_gb=0,
        gpu_models=[],
        cuda_version=None,
        container_runtime="none",
    )
    with pytest.raises(Exception, match="unknown package preferences"):
        compile_prompt_policy("Preprocess fMRI.", hardware)
    # CrewAI may re-invoke the LLM after a failure; what matters is that the
    # compiler never accepts the invalid object.
    assert len(server.requests) >= 2
