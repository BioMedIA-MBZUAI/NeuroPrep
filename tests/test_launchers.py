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
