from __future__ import annotations

import time
from pathlib import Path

from core.jobs.job_store import JobStore


def _wait_for_job(store: JobStore, job_id: str, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = store.get_status(job_id)
        if record and record.get("status") not in {"queued", "running"}:
            return record
        time.sleep(0.01)
    return store.get_status(job_id)


def test_job_store_queue_and_complete():
    store = JobStore(max_workers=1, ttl_seconds=60, max_entries=10, cleanup_interval_seconds=0)

    def runner():
        return {"ok": True}

    ticket = store.queue_job(
        "test",
        runner,
        run_metadata={"artifact": Path("artifacts/test.json")},
    )
    assert ticket["status"] == "queued"
    assert ticket["tool"] == "test"
    assert ticket["artifact"] == "artifacts/test.json"

    record = _wait_for_job(store, ticket["job_id"])
    assert record is not None
    assert record["status"] == "completed"
    assert record["result"] == {"ok": True}
    assert record["tool"] == "test"


def test_job_store_list_jobs_filter():
    store = JobStore(max_workers=1, ttl_seconds=60, max_entries=10, cleanup_interval_seconds=0)

    def runner():
        return "done"

    ticket_a = store.queue_job("tool_a", runner)
    ticket_b = store.queue_job("tool_b", runner)

    record_a = _wait_for_job(store, ticket_a["job_id"])
    record_b = _wait_for_job(store, ticket_b["job_id"])
    assert record_a is not None
    assert record_b is not None

    all_jobs = store.list_jobs()
    assert len(all_jobs) >= 2

    tool_a_jobs = store.list_jobs(tool="tool_a")
    assert tool_a_jobs
    assert all(job["tool"] == "tool_a" for job in tool_a_jobs)


def test_job_store_marks_error_status_when_runner_returns_error_payload():
    store = JobStore(max_workers=1, ttl_seconds=60, max_entries=10, cleanup_interval_seconds=0)

    def runner():
        return {"success": False, "error": "boom"}

    ticket = store.queue_job("tool_error", runner)
    record = _wait_for_job(store, ticket["job_id"])
    assert record is not None
    assert record["status"] == "error"
    assert isinstance(record.get("result"), dict)
    assert record["result"].get("error") == "boom"
