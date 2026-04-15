from pathlib import Path

from lightrag.base import DocStatus
from lightrag.sidecar.jobs import (
    SidecarJobQueue,
    map_job_status_to_doc_status,
    sidecar_job_to_doc_status_payload,
)


def test_sidecar_queue_claim_and_complete(tmp_path: Path):
    queue = SidecarJobQueue(tmp_path / "job_spool")
    job = queue.create_job(
        track_id="upload_123",
        workspace="default",
        input_path=tmp_path / "inputs" / "paper.pdf",
        original_filename="paper.pdf",
        parser="mineru",
        parse_method="auto",
        metadata={"content_length": 42},
    )

    queue.enqueue(job)
    claimed = queue.claim_next_job()

    assert claimed is not None
    assert claimed.job_id == job.job_id
    assert claimed.status == "processing"

    done_job = queue.mark_done(claimed, metadata={"chunks_count": 7})
    jobs = queue.list_jobs_by_track_id("upload_123")

    assert done_job.status == "done"
    assert len(jobs) == 1
    assert jobs[0].status == "done"
    assert jobs[0].metadata["chunks_count"] == 7


def test_sidecar_queue_retries_then_fails(tmp_path: Path):
    queue = SidecarJobQueue(tmp_path / "job_spool")
    job = queue.create_job(
        track_id="upload_retry",
        workspace="default",
        input_path=tmp_path / "inputs" / "broken.pdf",
        original_filename="broken.pdf",
        parser="mineru",
        parse_method="auto",
    )

    queue.enqueue(job)
    claimed = queue.claim_next_job()
    assert claimed is not None

    retried = queue.mark_retry_or_failed(
        claimed, error_message="first failure", max_retries=3
    )
    assert retried.status == "queued"
    assert retried.attempt_count == 1

    claimed_again = queue.claim_next_job()
    assert claimed_again is not None

    failed = queue.mark_retry_or_failed(
        claimed_again, error_message="terminal failure", max_retries=2
    )
    assert failed.status == "failed"
    assert failed.attempt_count == 2


def test_sidecar_job_payload_maps_to_doc_status(tmp_path: Path):
    queue = SidecarJobQueue(tmp_path / "job_spool")
    job = queue.create_job(
        track_id="scan_123",
        workspace="default",
        input_path=tmp_path / "inputs" / "deck.pptx",
        original_filename="deck.pptx",
        parser="mineru",
        parse_method="auto",
    )
    queue.enqueue(job)

    payload = sidecar_job_to_doc_status_payload(job)

    assert payload["file_path"] == "deck.pptx"
    assert payload["track_id"] == "scan_123"
    assert payload["status"] == DocStatus.PENDING
    assert payload["metadata"]["sidecar_job"] is True


def test_map_job_status_to_doc_status():
    assert map_job_status_to_doc_status("queued") == DocStatus.PENDING
    assert map_job_status_to_doc_status("processing") == DocStatus.PROCESSING
    assert map_job_status_to_doc_status("done") == DocStatus.PROCESSED
    assert map_job_status_to_doc_status("failed") == DocStatus.FAILED
