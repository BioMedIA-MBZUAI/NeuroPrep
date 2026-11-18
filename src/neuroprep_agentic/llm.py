"""Provider-neutral structured-output client for the NeuroPrep policy compiler.

Models are named ``provider/model`` (LiteLLM style):

==============  =========================================  ==========================
provider        example model                              credentials / endpoint
==============  =========================================  ==========================
``ollama``      ``ollama/qwen3:8b``                        ``OLLAMA_BASE_URL``
``openai``      ``openai/gpt-4o-mini``                     ``OPENAI_API_KEY``
``anthropic``   ``anthropic/claude-sonnet-5-5``            ``ANTHROPIC_API_KEY``
``openrouter``  ``openrouter/anthropic/claude-sonnet-5.5`` ``OPENROUTER_API_KEY``
``vllm``        ``vllm/meta-llama/Llama-3.1-8B-Instruct``  ``NEUROPREP_LLM_BASE_URL``
``lmstudio``    ``lmstudio/<loaded-model>``                ``NEUROPREP_LLM_BASE_URL``
``openai_compatible`` any OpenAI-style server              ``NEUROPREP_LLM_BASE_URL``
==============  =========================================  ==========================

The model only *proposes* a typed object. The caller validates it with Pydantic and the
deterministic compiler owns everything that matters, so a weaker or non-deterministic model
can fail validation but cannot widen what NeuroPrep will execute.

Only the standard library is used for transport so the package works without extra SDKs.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

Message = dict[str, str]

OPENAI_COMPATIBLE = {"openai", "openrouter", "vllm", "lmstudio", "llamacpp", "openai_compatible"}
KNOWN_PROVIDERS = {"ollama", "anthropic"} | OPENAI_COMPATIBLE

_DEFAULT_BASE_URL = {
    "ollama": "http://127.0.0.1:11434",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com",
    "vllm": "http://127.0.0.1:8000/v1",
    "lmstudio": "http://127.0.0.1:1234/v1",
    "llamacpp": "http://127.0.0.1:8080/v1",
}
_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
ANTHROPIC_VERSION = "2023-06-01"


class LLMError(RuntimeError):
    """Transport, provider or response-shape failure."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class LLMEndpoint:
    provider: str
    model: str
    base_url: str
    api_key: str | None = None

    @property
    def spec(self) -> str:
        return f"{self.provider}/{self.model}"


def split_model_spec(spec: str) -> tuple[str, str]:
    provider, sep, model = spec.partition("/")
    if not sep or not model:
        raise ValueError(
            f"model {spec!r} must look like provider/model "
            f"(providers: {', '.join(sorted(KNOWN_PROVIDERS))})"
        )
    if provider not in KNOWN_PROVIDERS:
        choices = ", ".join(sorted(KNOWN_PROVIDERS))
        raise ValueError(f"unsupported LLM provider {provider!r}; use one of {choices}")
    return provider, model


def resolve_endpoint(
    spec: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    environ: dict[str, str] | None = None,
) -> LLMEndpoint:
    """Resolve a ``provider/model`` string plus environment into a concrete endpoint."""

    env = os.environ if environ is None else environ
    provider, model = split_model_spec(spec)
    url = base_url or env.get("NEUROPREP_LLM_BASE_URL")
    if provider == "ollama":
        url = base_url or env.get("OLLAMA_BASE_URL") or env.get("NEUROPREP_LLM_BASE_URL")
    elif provider in {"openai", "openrouter", "anthropic"} and base_url is None:
        # Hosted providers: only an explicit per-provider override applies, so a local
        # NEUROPREP_LLM_BASE_URL for vLLM can never receive a hosted API key by accident.
        url = env.get(f"{provider.upper()}_BASE_URL")
    url = url or _DEFAULT_BASE_URL.get(provider)
    if not url:
        raise ValueError(f"{provider} needs NEUROPREP_LLM_BASE_URL")
    key = api_key or env.get("NEUROPREP_LLM_API_KEY")
    if key is None and provider in _API_KEY_ENV:
        key = env.get(_API_KEY_ENV[provider])
    if provider in _API_KEY_ENV and not key:
        raise ValueError(f"{provider} requires {_API_KEY_ENV[provider]} (or NEUROPREP_LLM_API_KEY)")
    return LLMEndpoint(provider=provider, model=model, base_url=url.rstrip("/"), api_key=key)


def _post(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    retries: int = 2,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", **headers}, method="POST"
    )
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            if error.code in {429, 500, 502, 503, 504, 529} and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise LLMError(f"HTTP {error.code} from {url}: {detail}", error.code) from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt < retries and not isinstance(error, json.JSONDecodeError):
                time.sleep(delay)
                delay *= 2
                continue
            raise LLMError(f"request to {url} failed: {error}") from error
    raise LLMError(f"request to {url} failed")  # pragma: no cover


def _schema_instruction(schema: dict[str, Any]) -> str:
    return (
        "Reply with one JSON object and nothing else (no markdown, no commentary) that "
        "validates against this JSON Schema:\n" + json.dumps(schema, sort_keys=True)
    )


def _strip_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else ""
        raw = raw.rsplit("```", 1)[0].strip()
    return raw


def _ollama(ep, messages, schema, temperature, seed, max_tokens, timeout) -> str:
    base = ep.base_url[:-3] if ep.base_url.endswith("/v1") else ep.base_url
    result = _post(
        f"{base}/api/chat",
        {
            "model": ep.model,
            "stream": False,
            # Native endpoint: the OpenAI-compatible one does not reliably honor think=False
            # for Qwen3, which would let hidden reasoning consume the output budget.
            "think": False,
            "format": schema,
            "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens},
            "messages": messages,
        },
        {},
        timeout,
    )
    try:
        return str(result["message"]["content"])
    except (KeyError, TypeError) as error:
        raise LLMError("Ollama response omitted message content") from error


def _openai_compatible(ep, messages, schema, temperature, seed, max_tokens, timeout) -> str:
    headers = {"Authorization": f"Bearer {ep.api_key or 'EMPTY'}"}
    if ep.provider == "openrouter":
        headers["X-Title"] = "NeuroPrep"
    url = f"{ep.base_url}/chat/completions"
    base = {
        "model": ep.model,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    guided = [*messages, {"role": "system", "content": _schema_instruction(schema)}]
    attempts = (
        {
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "neuroprep_policy", "schema": schema, "strict": False},
            },
        },
        # Servers/models without json_schema support: ask for JSON mode, schema in the prompt.
        {"messages": guided, "response_format": {"type": "json_object"}},
        # Last resort: prompt-only, validated by the caller.
        {"messages": guided},
    )
    last: LLMError | None = None
    for extra in attempts:
        try:
            result = _post(url, {**base, **extra}, headers, timeout)
            content = result["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise LLMError("response contained no message content")
            return content
        except LLMError as error:
            last = error
            # Only a request-format rejection justifies degrading the request; auth, rate
            # limit and server errors are real failures.
            if error.status not in {400, 404, 415, 422}:
                raise
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError("unexpected chat-completions response shape") from error
    raise last or LLMError("no response")


def _anthropic(ep, messages, schema, temperature, seed, max_tokens, timeout) -> str:
    del seed  # the Messages API has no sampling seed
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [
        {"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]}
        for m in messages
        if m["role"] != "system"
    ]
    payload: dict[str, Any] = {
        "model": ep.model,
        "max_tokens": max_tokens,
        "messages": turns,
        # Forced tool use is the supported route to schema-shaped output.
        "tools": [
            {
                "name": "emit_policy",
                "description": "Return the policy object.",
                "input_schema": schema,
            }
        ],
        "tool_choice": {"type": "tool", "name": "emit_policy"},
        "temperature": temperature,
    }
    if system:
        payload["system"] = system
    headers = {"x-api-key": ep.api_key or "", "anthropic-version": ANTHROPIC_VERSION}
    url = f"{ep.base_url}/v1/messages"
    try:
        result = _post(url, payload, headers, timeout)
    except LLMError as error:
        # Some newer models reject an explicit temperature; retry once without it.
        if "temperature" not in str(error).lower():
            raise
        payload.pop("temperature")
        result = _post(url, payload, headers, timeout)
    for block in result.get("content", []):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return json.dumps(block.get("input", {}))
    for block in result.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            return str(block.get("text", ""))
    raise LLMError("Anthropic response contained no tool_use or text block")


def structured_chat(
    endpoint: LLMEndpoint,
    messages: list[Message],
    schema: dict[str, Any],
    *,
    temperature: float = 0.0,
    seed: int = 0,
    max_tokens: int = 2048,
    timeout: float = 90.0,
) -> str:
    """Return raw JSON text for ``schema`` from the selected provider."""

    if endpoint.provider == "ollama":
        raw = _ollama(endpoint, messages, schema, temperature, seed, max_tokens, timeout)
    elif endpoint.provider == "anthropic":
        raw = _anthropic(endpoint, messages, schema, temperature, seed, max_tokens, timeout)
    elif endpoint.provider in OPENAI_COMPATIBLE:
        raw = _openai_compatible(endpoint, messages, schema, temperature, seed, max_tokens, timeout)
    else:  # pragma: no cover - guarded by split_model_spec
        raise LLMError(f"unsupported provider {endpoint.provider}")
    return _strip_fence(raw)
