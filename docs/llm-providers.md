# Language models

The policy compiler and the advisory review crew work with any of these providers.
Name models as `provider/model`; no extra SDK is needed.

| Provider | Example `NEUROPREP_MANAGER_MODEL` | Required environment |
|---|---|---|
| Ollama (default) | `ollama/qwen3:14b` | `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`) |
| OpenAI | `openai/gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic (Claude) | `anthropic/claude-sonnet-5-5` | `ANTHROPIC_API_KEY` |
| OpenRouter | `openrouter/anthropic/claude-sonnet-5.5` | `OPENROUTER_API_KEY` |
| vLLM | `vllm/Qwen/Qwen3-8B` | `NEUROPREP_LLM_BASE_URL` (default `http://127.0.0.1:8000/v1`) |
| LM Studio | `lmstudio/<loaded model>` | `NEUROPREP_LLM_BASE_URL` (default `http://127.0.0.1:1234/v1`) |
| llama.cpp server | `llamacpp/<model>` | `NEUROPREP_LLM_BASE_URL` (default `http://127.0.0.1:8080/v1`) |
| Any OpenAI-compatible server | `openai_compatible/<model>` | `NEUROPREP_LLM_BASE_URL`, optional `NEUROPREP_LLM_API_KEY` |

## Example

```bash
export NEUROPREP_MANAGER_MODEL=anthropic/claude-sonnet-5-5
export NEUROPREP_WORKER_MODEL=anthropic/claude-haiku-5-5
export ANTHROPIC_API_KEY=...
neuroprep-agentic runtime-check   # reports the provider and whether it is reachable
```

Before a long run, make one real, schema-validated call (a few hundred tokens):

```bash
neuroprep-agentic prompt-policy --request "Preprocess the T1-weighted MRI" --output policy.json
```

## Variables

- `NEUROPREP_MANAGER_MODEL` compiles the policy.
- `NEUROPREP_WORKER_MODEL` drives the advisory review agents.
- The older `OLLAMA_MANAGER_MODEL` and `OLLAMA_WORKER_MODEL` variables still work.
- Hosted providers never receive `NEUROPREP_LLM_BASE_URL`; use `OPENAI_BASE_URL`,
  `OPENROUTER_BASE_URL` or `ANTHROPIC_BASE_URL` to route them through a proxy.

## Structured output and reproducibility

Structured output uses each provider's native mechanism: Ollama's schema `format`,
OpenAI-style `json_schema` (degrading to JSON mode, then a prompt, on servers that
reject it), and forced tool use for Claude. Every reply is then validated against the
`RunPolicy` schema, so a weaker model can fail validation but cannot widen what
NeuroPrep executes.

!!! tip "Reproducible plans"
    Exact plan reproducibility needs a pinned local model with a fixed seed; hosted
    APIs do not guarantee deterministic output.

CrewAI usage telemetry and OpenTelemetry export are disabled by default; set
`CREWAI_TELEMETRY_OPT_OUT=false` to opt in.
