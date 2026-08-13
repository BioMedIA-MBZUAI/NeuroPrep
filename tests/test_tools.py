import json
from pathlib import Path

from neuroprep_agentic.execution_models import ExecutionPlan, RunLayout
from neuroprep_agentic.executors import builtin_executor_catalog
from neuroprep_agentic.tools import (
    CoverageInventoryTool,
    ExecutionPlanReviewTool,
    ManifestIngestPlanTool,
)


def test_coverage_tool_ignores_unapproved_extra_arguments():
    result = json.loads(CoverageInventoryTool()._run(manifest_path="ignored"))
    assert result["support_counts"] == {"bids_core": 23, "provisional": 3}


def test_manifest_tool_ignores_redundant_llm_arguments(monkeypatch):
    manifest = Path(__file__).parents[1] / "examples" / "synthetic_multimodal_manifest.json"
    monkeypatch.setenv("NEUROPREP_MANIFEST_PATH", str(manifest))
    monkeypatch.setenv("NEUROPREP_IDENTITY_KEY", "a-strong-test-key")
    result = json.loads(ManifestIngestPlanTool()._run(manifest_path="untrusted-value"))
    assert result["dataset_id"] == "synthetic-demo"
    assert result["quarantined"] == 1
    assert result["unsupported"] == 1


def test_execution_review_tool_is_deterministic_and_read_only(monkeypatch, nfs_tmp: Path) -> None:
    plan = ExecutionPlan(
        dataset_id="review-test",
        generated_at="2026-01-01T00:00:00Z",
        catalog_digest=builtin_executor_catalog().digest(),
        layout=RunLayout(
            dataset_id="review-test",
            source_root=str(nfs_tmp / "source"),
            bids_root=str(nfs_tmp / "bids"),
            derivatives_root=str(nfs_tmp / "derivatives"),
            work_root=str(nfs_tmp / "work"),
            state_path=str(nfs_tmp / "state" / "tasks.sqlite"),
        ),
        tasks=[],
        required_executor_ids=[],
        optional_executor_ids=[],
    )
    path = nfs_tmp / "execution-plan.json"
    path.write_text(plan.model_dump_json())
    monkeypatch.setenv("NEUROPREP_EXECUTION_PLAN_PATH", str(path))
    result = json.loads(ExecutionPlanReviewTool()._run(untrusted="ignored"))
    assert result["dataset_id"] == "review-test"
    assert result["task_count"] == 0
    assert result["dependency_edges"] == 0
