from pathlib import Path


def test_launcher_reports_partial_scheduler_submissions() -> None:
    launcher = Path("scripts/launch_neuroprep_run.sh").read_text()

    assert "trap report_incomplete_launch EXIT" in launcher
    assert 'echo "launch_complete=false"' in launcher
    assert 'echo "review_job=${REVIEW_JOB:-not-submitted}"' in launcher
    assert 'echo "worker_jobs=${WORKER_JOBS:-not-submitted}"' in launcher
    assert 'echo "finalize_job=${FINALIZE_JOB:-not-submitted}"' in launcher
    assert "trap - EXIT" in launcher
    assert 'echo "launch_complete=true"' in launcher
    assert "NEUROPREP_AGENTIC_PROJECT" in launcher
    assert "Execution plan agentic project differs from launcher repository" in launcher
    assert 'p.get("backend_qualification_tests") or p.get("planned_smoke_tests")' in launcher
    assert "NEUROPREP_BACKEND_EVIDENCE_KEY" in launcher
    assert "--export=ALL,NEUROPREP_EXECUTION_PLAN_PATH" in launcher
    assert "--backend-evidence-key" not in launcher
    assert 'echo "$NEUROPREP_BACKEND_EVIDENCE_KEY"' not in launcher


def test_prompt_run_launcher_is_gpu_and_nfs_fail_closed() -> None:
    launcher = Path("scripts/slurm/neuroprep_prompt_run_gpu.sbatch").read_text()

    assert "#SBATCH --chdir=/nfs-stor/salma.hassan" in launcher
    assert "#SBATCH --gres=gpu:1" in launcher
    assert "NEUROPREP_REQUEST_FILE is required" in launcher
    assert 'test -n "${CUDA_VISIBLE_DEVICES:-}"' in launcher
    assert 'nvidia-smi -i "$CUDA_VISIBLE_DEVICES"' in launcher
    assert 'timeout 20 stat "$APP_ROOT" "$OLLAMA_IMAGE"' in launcher
    assert "/home/salma.hassan" not in launcher
    assert "prompt-run" in launcher
    assert "optional_flag NEUROPREP_SUBMIT --submit" in launcher
    assert "optional_value NEUROPREP_SOURCE_ROOT --source-root" in launcher
    assert "optional_value NEUROPREP_OUTPUT_ROOT --output-root" in launcher
    assert "optional_value NEUROPREP_AGENTIC_PROJECT --agentic-project" in launcher
    assert "APP_ROOT=${NEUROPREP_AGENTIC_PROJECT:-" in launcher
    assert "must be the exact git repository root" in launcher


def test_execution_review_consumes_evidence_key_from_environment_without_argv() -> None:
    launcher = Path("scripts/slurm/neuroprep_execution_review_gpu.sbatch").read_text()

    assert "NEUROPREP_BACKEND_EVIDENCE_KEY" in launcher
    assert "Signed backend evidence key is required for run initialization" in launcher
    assert 'run-init "$NEUROPREP_EXECUTION_PLAN_PATH"' in launcher
    assert "--backend-evidence-key" not in launcher
