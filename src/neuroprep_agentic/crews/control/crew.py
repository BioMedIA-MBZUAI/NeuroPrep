from __future__ import annotations

from pathlib import Path

from crewai import LLM
from crewai.project import load_crew

from ...settings import AgenticSettings, WorkflowName
from ...tools import (
    CoverageInventoryTool,
    ExecutionPlanReviewTool,
    ManifestIngestPlanTool,
    MutationPolicyTool,
)

CREW_CONFIGS: dict[WorkflowName, str] = {
    "coverage": "coverage_crew.jsonc",
    "ingest_plan": "ingest_plan_crew.jsonc",
    "execution_review": "execution_review_crew.jsonc",
}


def _llm(model: str, settings: AgenticSettings) -> LLM:
    endpoint = settings.endpoint(model)
    # LiteLLM spells every OpenAI-style server (vLLM, LM Studio, llama.cpp, ...) "openai/".
    litellm_model = (
        f"openai/{endpoint.model}"
        if endpoint.provider in {"vllm", "lmstudio", "llamacpp", "openai_compatible"}
        else endpoint.spec
    )
    local = endpoint.provider not in {"openai", "openrouter", "anthropic"}
    return LLM(
        model=litellm_model,
        # Hosted providers use LiteLLM's own endpoint unless a *_BASE_URL override was set.
        base_url=endpoint.base_url if local or endpoint.provider != "anthropic" else None,
        api_key=endpoint.api_key or ("EMPTY" if local and endpoint.provider != "ollama" else None),
        temperature=0,
        timeout=300.0,
        max_retries=2,
    )


def _configure_agents(crew, settings: AgenticSettings) -> None:
    worker = _llm(settings.worker_model, settings)
    manager = _llm(settings.manager_model, settings)
    for agent in crew.agents:
        role = agent.role.lower()
        selected = manager if "central" in role and "orchestrator" in role else worker
        agent.llm = selected
        agent.function_calling_llm = selected
        if "coverage architect" in role:
            agent.tools = [CoverageInventoryTool()]
        elif "ingestion curator" in role:
            agent.tools = [ManifestIngestPlanTool()]
        elif "recovery planner" in role:
            agent.tools = [MutationPolicyTool()]
        elif "execution auditor" in role:
            agent.tools = [ExecutionPlanReviewTool()]
        else:
            agent.tools = []


def kickoff_control_crew(workflow: WorkflowName, inputs: dict[str, object]):
    settings = AgenticSettings.from_env(workflow)
    config = Path(__file__).with_name(CREW_CONFIGS[workflow])
    crew, defaults = load_crew(config)
    _configure_agents(crew, settings)
    return crew.kickoff(inputs={**defaults, "workflow": workflow, **inputs})
