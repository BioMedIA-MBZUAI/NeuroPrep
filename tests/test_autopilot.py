"""Autopilot: prompt + paths in, qualified run out, with every gate still deciding."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import neuroprep_agentic.autopilot as autopilot
from neuroprep_agentic.autopilot import (
    AutopilotError,
    pilot_manifest,
    qualification_root,
    run_autopilot,
    select_pilot_subjects,
)
from neuroprep_agentic.policy import RunPolicy
from neuroprep_agentic.prompt_run import resolve_prompt_paths
from neuroprep_agentic.services import ManifestDocument


def _plan(*rows: tuple[str | None, str]) -> dict:
    return {"tasks": [{"subject_id": s, "executor_id": e} for s, e in rows]}


def test_pilot_selection_covers_every_backend_with_fewest_cheapest_subjects():
    plan = _plan(
        ("sub-a", "smriprep"),
        ("sub-a", "fmriprep"),
        ("sub-a", "qsiprep"),
        ("sub-b", "smriprep"),
        ("sub-b", "fmriprep"),
        ("sub-c", "petprep"),
        (None, "neuroprep-bids-init"),
    )
    subjects, uncovered = select_pilot_subjects(plan, {"smriprep", "qsiprep", "petprep"})
    assert subjects == ["sub-a", "sub-c"]
    assert uncovered == set()


def test_pilot_selection_ties_prefer_cheaper_then_lower_id():
    plan = _plan(
        ("sub-z", "smriprep"),
        ("sub-y", "smriprep"),
        ("sub-y", "xcp-d"),
        ("sub-x", "smriprep"),
    )
    subjects, _ = select_pilot_subjects(plan, {"smriprep"})
    assert subjects == ["sub-x"]


def test_pilot_selection_reports_what_the_budget_cannot_cover():
    plan = _plan(("sub-a", "smriprep"), ("sub-b", "qsiprep"), ("sub-c", "petprep"))
    subjects, uncovered = select_pilot_subjects(
        plan, {"smriprep", "qsiprep", "petprep", "aslprep"}, max_subjects=2
    )
    assert len(subjects) == 2
    assert "aslprep" in uncovered


def test_pilot_manifest_keeps_only_the_chosen_subjects_sources():
    manifest = ManifestDocument.model_validate(
        {
            "dataset": {"dataset_id": "d", "adapter_id": "generic"},
            "records": [
                {"dataset_id": "d", "source_path": "/s/p1/t1.nii.gz", "source_subject_id": "p1"},
                {"dataset_id": "d", "source_path": "/s/p2/t1.nii.gz", "source_subject_id": "p2"},
            ],
        }
    )
    graph = {
        "subjects": [
            {
                "subject_id": "sub-AAA",
                "sessions": [{"acquisitions": [{"source": {"source_path": "/s/p1/t1.nii.gz"}}]}],
            },
            {
                "subject_id": "sub-BBB",
                "sessions": [{"acquisitions": [{"source": {"source_path": "/s/p2/t1.nii.gz"}}]}],
            },
        ]
    }
    subset = pilot_manifest(manifest, graph, ["sub-BBB"])
    assert [r.source_subject_id for r in subset.records] == ["p2"]
    assert subset.dataset == manifest.dataset
    with pytest.raises(ValueError):
        pilot_manifest(manifest, graph, ["sub-missing"])


# --- orchestration with fakes --------------------------------------------------------


class _Harness:
    """Fakes for every heavy tool; records the order and arguments of each call."""

    def __init__(
        self, root: Path, monkeypatch, *, smoke=True, pilot_fails=False, compare=False, losing=None
    ):
        self.calls: list[tuple[str, dict]] = []
        self.root = root
        self.policy_calls = 0
        self.pilot_fails = pilot_fails
        self.smoke = smoke
        self.compare = compare
        self.losing = losing
        source = root / "source"
        source.mkdir()
        self.request = f"My data is at {source}; save to {root / 'run'}"

        def compile_policy(request, profile):
            self.policy_calls += 1
            return RunPolicy(source="prompt", source_prompt=request)

        def prepare(request, *, output_root, **kwargs):
            # same path contract as the real prepare_prompt_run: staging roots must be bound
            if not kwargs.get("bound_roots"):
                resolve_prompt_paths(
                    request, source_root=kwargs.get("source_root"), output_root=output_root
                )
            name = Path(output_root).name
            label = {"probe": "probe", "pilot": "pilot"}.get(name, "production")
            self.calls.append((label, {"output_root": output_root, **kwargs}))
            out = Path(output_root)
            out.mkdir(parents=True, exist_ok=True)
            tests = (
                [
                    {
                        "test_id": "smoke--structural_preprocess--smriprep",
                        "executor_id": "smriprep",
                        "capability": "structural_preprocess",
                    }
                ]
                if self.smoke
                else []
            )
            tasks = [
                {"subject_id": "sub-1", "executor_id": "smriprep", "resource_class": "gpu"},
                {"subject_id": "sub-2", "executor_id": "fmriprep", "resource_class": "gpu"},
            ]
            if self.compare:
                tests += [
                    {
                        "test_id": f"smoke--structural_segment--{name}",
                        "executor_id": name,
                        "capability": "structural_segment",
                    }
                    for name in ("deepprep", "fastsurfer")
                ]
                tasks.append(
                    {"subject_id": "sub-1", "executor_id": "deepprep", "resource_class": "gpu"}
                )
                if kwargs.get("qualify_alternatives"):
                    for task in tasks:
                        if task["executor_id"] == "deepprep":
                            task["metadata"] = {"qualification_candidate": True}
                    tasks.append(
                        {
                            "subject_id": "sub-1",
                            "executor_id": "fastsurfer",
                            "resource_class": "gpu",
                            "metadata": {"qualification_candidate": True},
                        }
                    )
            (out / "backend-plan.json").write_text(
                json.dumps(
                    {
                        "decisions": [
                            {
                                "capability": "structural_segment",
                                "selected_package": "deepprep",
                                "selection_reasons": ["fastest among equally qualified"],
                                "rejected_candidates": {"fastsurfer": "smoke_test_failed"},
                            }
                        ]
                    }
                )
            )
            plan = {
                "dataset_id": "demo",
                "tasks": tasks,
                "backend_qualification_tests": tests,
                "layout": {
                    "state_path": str(out / "state.sqlite"),
                    "derivatives_root": str(out / "derivatives"),
                },
            }
            (out / "execution-plan.json").write_text(json.dumps(plan))
            if label == "probe":
                (out / "manifest.json").write_text(
                    json.dumps(
                        {
                            "dataset": {"dataset_id": "demo", "adapter_id": "generic"},
                            "records": [
                                {
                                    "dataset_id": "demo",
                                    "source_path": str(source / "p1.nii.gz"),
                                    "source_subject_id": "p1",
                                },
                                {
                                    "dataset_id": "demo",
                                    "source_path": str(source / "p2.nii.gz"),
                                    "source_subject_id": "p2",
                                },
                            ],
                        }
                    )
                )
                (out / "acquisition-graph.json").write_text(
                    json.dumps(
                        {
                            "subjects": [
                                {
                                    "subject_id": "sub-1",
                                    "sessions": [
                                        {
                                            "acquisitions": [
                                                {
                                                    "source": {
                                                        "source_path": str(source / "p1.nii.gz")
                                                    }
                                                }
                                            ]
                                        }
                                    ],
                                },
                                {
                                    "subject_id": "sub-2",
                                    "sessions": [
                                        {
                                            "acquisitions": [
                                                {
                                                    "source": {
                                                        "source_path": str(source / "p2.nii.gz")
                                                    }
                                                }
                                            ]
                                        }
                                    ],
                                },
                            ]
                        }
                    )
                )
            return {
                "execution_plan": str(out / "execution-plan.json"),
                "ready": True,
                "task_count": 2,
                "record_count": 2,
                "preflight": "p",
                "launch": None,
            }

        class FakePlan:
            @staticmethod
            def model_validate_json(text):
                data = json.loads(text)
                return SimpleNamespace(
                    layout=SimpleNamespace(**data["layout"]),
                    tasks=[
                        SimpleNamespace(
                            resource_class=t["resource_class"],
                            executor_id=t["executor_id"],
                            metadata=t.get("metadata", {}),
                        )
                        for t in data["tasks"]
                    ],
                )

        class FakeState:
            def __init__(self, path):
                self.path = path

            def initialize(self, plan, plan_path):
                harness.calls.append(("initialize", {"plan_path": plan_path}))

            def summary(self):
                failing = harness.losing or ("smriprep" if harness.pilot_fails else None)
                failed = failing is not None and "pilot" in str(self.path)
                return {
                    "counts": {"failed": 1, "complete": 1} if failed else {"complete": 2},
                    "active": [{"status": "failed", "executor_id": failing}] if failed else [],
                }

        def worker(state_path, worker_id, classes, **kwargs):
            harness.calls.append(("execute", {"state_path": state_path, "classes": classes}))
            return {"claimed": 2, "completed": 2, "failed": 0}

        def finalize(plan, state, output_root):
            harness.calls.append(("finalize", {"output_root": output_root}))
            return {"ready": True}

        def evidence(**kwargs):
            harness.calls.append(("evidence", kwargs))
            return SimpleNamespace(
                tests=[
                    SimpleNamespace(
                        test_id="smoke--structural_preprocess--smriprep",
                        status="passed",
                        runtime_seconds=1920.0,
                    )
                ],
                target_git_commit="a" * 40,
            )

        def write_evidence(document, path):
            Path(path).write_text("{}")
            return Path(path)

        def make_key(path):
            Path(path).write_text("k" * 64)
            return Path(path)

        harness = self
        for name, value in {
            "compile_prompt_policy": compile_policy,
            "prepare_prompt_run": prepare,
            "ExecutionPlan": FakePlan,
            "RunState": FakeState,
            "finalize_execution_run": finalize,
            "build_backend_evidence": evidence,
            "write_evidence_document": write_evidence,
            "create_signing_key": make_key,
            "read_signing_key": lambda path: b"k" * 64,
            "validated_agentic_project": lambda value: root,
        }.items():
            monkeypatch.setattr(autopilot, name, value)
        self.worker = worker

    def run(self, **kwargs):
        return run_autopilot(self.request, worker=self.worker, idle_seconds=0, **kwargs)

    def labels(self):
        return [label for label, _ in self.calls]


def test_autopilot_runs_every_stage_unattended(nfs_tmp, monkeypatch):
    harness = _Harness(nfs_tmp, monkeypatch)
    report = harness.run()

    assert harness.policy_calls == 1  # one model decision shared by every stage
    assert harness.labels() == [
        "probe",
        "pilot",
        "initialize",
        "execute",
        "finalize",
        "evidence",
        "production",
        "execute",
        "finalize",
    ]
    pilot_kwargs = dict(harness.calls)["pilot"]
    assert Path(pilot_kwargs["manifest_path"]).name == "pilot-manifest.json"
    subset = json.loads(Path(pilot_kwargs["manifest_path"]).read_text())
    assert [r["source_subject_id"] for r in subset["records"]] == ["p1"]  # only sub-1 runs smriprep

    production = dict(harness.calls)["production"]
    assert production["backend_evidence"].endswith("evidence.json")
    assert production["backend_evidence_key"].endswith("autopilot.key")
    assert production["initialize"] is True and production["submit"] is False

    assert report["outcome"] == "released"
    assert report["approval"] == "autonomous"
    assert report["stages"]["pilot_selection"]["subjects"] == ["sub-1"]
    key = qualification_root(nfs_tmp / "run") / "keys" / "autopilot.key"
    assert key.is_file()
    assert (nfs_tmp / "run" / "autopilot-report.json").is_file()


def test_failed_pilot_never_reaches_production(nfs_tmp, monkeypatch):
    harness = _Harness(nfs_tmp, monkeypatch, pilot_fails=True)
    with pytest.raises(AutopilotError, match="pilot_run"):
        harness.run()
    assert "evidence" not in harness.labels()
    assert "production" not in harness.labels()
    report = json.loads((qualification_root(nfs_tmp / "run") / "autopilot-report.json").read_text())
    assert report["outcome"] == "failed"
    assert report["failed_stage"] == "pilot_run"
    assert report["stages"]["pilot_run"]["failed_executors"] == ["smriprep"]


def test_no_smoke_tests_skips_the_pilot(nfs_tmp, monkeypatch):
    harness = _Harness(nfs_tmp, monkeypatch, smoke=False)
    report = harness.run()
    assert "pilot" not in harness.labels()
    production = dict(harness.calls)["production"]
    assert production["backend_evidence"] is None
    assert report["stages"]["pilot_selection"]["status"] == "skipped"


def test_slurm_mode_submits_instead_of_running_locally(nfs_tmp, monkeypatch):
    harness = _Harness(nfs_tmp, monkeypatch)
    report = harness.run(execute="slurm")
    production = dict(harness.calls)["production"]
    assert production["submit"] is True
    assert harness.labels().count("execute") == 1  # the pilot only
    assert report["outcome"] == "submitted"


def test_refuses_a_non_empty_run_root(nfs_tmp, monkeypatch):
    harness = _Harness(nfs_tmp, monkeypatch)
    (nfs_tmp / "run").mkdir()
    (nfs_tmp / "run" / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        harness.run()


def test_pilot_compares_alternatives_and_a_loser_does_not_stop_the_run(nfs_tmp, monkeypatch):
    harness = _Harness(nfs_tmp, monkeypatch, compare=True, losing="fastsurfer")
    report = harness.run()

    pilot = dict(harness.calls)["pilot"]
    assert pilot["qualify_alternatives"] is True
    assert report["stages"]["pilot_selection"]["subjects"] == ["sub-1"]  # covers routed backends
    assert report["stages"]["pilot_run"]["compared_backends"] == ["deepprep", "fastsurfer"]
    assert report["stages"]["pilot_run"]["losing_backends"] == ["fastsurfer"]
    decision = report["stages"]["backend_selection"]["decisions"]["structural_segment"]
    assert decision["selected"] == "deepprep" and decision["compared"] is True
    assert report["outcome"] == "released"


def test_a_failing_required_backend_still_stops_a_comparison_pilot(nfs_tmp, monkeypatch):
    harness = _Harness(nfs_tmp, monkeypatch, compare=True, losing="smriprep")
    with pytest.raises(AutopilotError, match="smriprep"):
        harness.run()
    assert "production" not in harness.labels()
