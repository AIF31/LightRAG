from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from lightrag.base import DocStatus
from lightrag.utils import compute_mdhash_id, logger


JOB_STATES = ("queued", "processing", "done", "failed")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SidecarJob:
    job_id: str
    track_id: str
    workspace: str
    input_path: str
    original_filename: str
    parser: str
    parse_method: str
    status: str
    attempt_count: int = 0
    error_message: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SidecarJob":
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def map_job_status_to_doc_status(status: str) -> DocStatus:
    if status == "queued":
        return DocStatus.PENDING
    if status == "processing":
        return DocStatus.PROCESSING
    if status == "done":
        return DocStatus.PROCESSED
    return DocStatus.FAILED


def sidecar_job_to_doc_status_payload(job: SidecarJob) -> dict[str, Any]:
    metadata = dict(job.metadata or {})
    metadata.update(
        {
            "sidecar_job": True,
            "sidecar_state": job.status,
            "parser": job.parser,
            "parse_method": job.parse_method,
            "effective_parser": metadata.get("effective_parser", job.parser),
            "effective_parse_method": metadata.get(
                "effective_parse_method", job.parse_method
            ),
            "ingestion_route": metadata.get("ingestion_route", "sidecar"),
            "attempt_count": job.attempt_count,
        }
    )
    if job.error_message:
        metadata["last_error"] = job.error_message

    return {
        "id": f"sidecar-{job.job_id}",
        "content_summary": metadata.get(
            "content_summary",
            f"[SIDECAR:{job.status.upper()}] {job.original_filename}",
        ),
        "content_length": int(metadata.get("content_length", 0)),
        "status": map_job_status_to_doc_status(job.status),
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "track_id": job.track_id,
        "chunks_count": metadata.get("chunks_count"),
        "error_msg": job.error_message,
        "metadata": metadata,
        "file_path": job.original_filename,
    }


class SidecarJobQueue:
    def __init__(self, spool_dir: str | Path):
        self.base_dir = Path(spool_dir)
        self.state_dirs = {
            state: self.base_dir / state for state in JOB_STATES
        }
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for state_dir in self.state_dirs.values():
            state_dir.mkdir(parents=True, exist_ok=True)

    def _job_path(self, state: str, job_id: str) -> Path:
        return self.state_dirs[state] / f"{job_id}.json"

    def _write_job(self, path: Path, job: SidecarJob) -> None:
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(job.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temp_path.replace(path)

    def _load_job(self, path: Path) -> SidecarJob:
        return SidecarJob.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def create_job(
        self,
        *,
        track_id: str,
        workspace: str,
        input_path: str | Path,
        original_filename: str,
        parser: str,
        parse_method: str,
        metadata: dict[str, Any] | None = None,
    ) -> SidecarJob:
        normalized_input_path = str(Path(input_path))
        job_id = compute_mdhash_id(
            f"{workspace}:{normalized_input_path}:{track_id}", prefix="sidecar-job-"
        )
        return SidecarJob(
            job_id=job_id,
            track_id=track_id,
            workspace=workspace,
            input_path=normalized_input_path,
            original_filename=original_filename,
            parser=parser,
            parse_method=parse_method,
            status="queued",
            metadata=metadata or {},
        )

    def enqueue(self, job: SidecarJob) -> SidecarJob:
        job.status = "queued"
        job.updated_at = _utc_now_iso()
        self._write_job(self._job_path("queued", job.job_id), job)
        return job

    def iter_jobs(self, states: Iterable[str] | None = None) -> list[SidecarJob]:
        selected_states = tuple(states or JOB_STATES)
        jobs: list[SidecarJob] = []
        for state in selected_states:
            state_dir = self.state_dirs[state]
            for path in sorted(state_dir.glob("*.json")):
                try:
                    jobs.append(self._load_job(path))
                except Exception as exc:
                    logger.error("Failed to load sidecar job %s: %s", path, exc)
        jobs.sort(key=lambda job: (job.created_at, job.job_id))
        return jobs

    def list_jobs_by_track_id(self, track_id: str) -> list[SidecarJob]:
        return [job for job in self.iter_jobs() if job.track_id == track_id]

    def find_active_job(
        self, *, input_path: str | Path, workspace: str = ""
    ) -> SidecarJob | None:
        normalized_input_path = str(Path(input_path))
        for job in self.iter_jobs(states=("queued", "processing")):
            if job.workspace == workspace and job.input_path == normalized_input_path:
                return job
        return None

    def claim_next_job(self) -> SidecarJob | None:
        queued_paths = sorted(self.state_dirs["queued"].glob("*.json"))
        for queued_path in queued_paths:
            try:
                job = self._load_job(queued_path)
            except Exception as exc:
                logger.error("Failed to parse queued sidecar job %s: %s", queued_path, exc)
                continue

            job.status = "processing"
            job.updated_at = _utc_now_iso()
            processing_path = self._job_path("processing", job.job_id)
            try:
                self._write_job(processing_path, job)
                queued_path.unlink()
                return job
            except FileNotFoundError:
                continue
        return None

    def requeue_processing_jobs(self) -> int:
        recovered = 0
        for path in sorted(self.state_dirs["processing"].glob("*.json")):
            job = self._load_job(path)
            job.status = "queued"
            job.updated_at = _utc_now_iso()
            self._write_job(self._job_path("queued", job.job_id), job)
            path.unlink(missing_ok=True)
            recovered += 1
        return recovered

    def mark_done(
        self, job: SidecarJob, metadata: dict[str, Any] | None = None
    ) -> SidecarJob:
        if metadata:
            job.metadata.update(metadata)
        job.status = "done"
        job.updated_at = _utc_now_iso()
        self._write_job(self._job_path("done", job.job_id), job)
        self._job_path("processing", job.job_id).unlink(missing_ok=True)
        return job

    def mark_retry_or_failed(
        self,
        job: SidecarJob,
        *,
        error_message: str,
        max_retries: int,
        metadata: dict[str, Any] | None = None,
    ) -> SidecarJob:
        if metadata:
            job.metadata.update(metadata)
        job.attempt_count += 1
        job.error_message = error_message
        job.updated_at = _utc_now_iso()
        self._job_path("processing", job.job_id).unlink(missing_ok=True)

        if job.attempt_count < max_retries:
            job.status = "queued"
            self._write_job(self._job_path("queued", job.job_id), job)
        else:
            job.status = "failed"
            self._write_job(self._job_path("failed", job.job_id), job)

        return job
