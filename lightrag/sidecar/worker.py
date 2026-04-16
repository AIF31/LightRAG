from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import traceback
from typing import Any

from lightrag import LightRAG
from lightrag.api.config import parse_args
from lightrag.base import DocStatus
from lightrag.llm.binding_options import OpenAILLMOptions
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.sidecar.jobs import SidecarJob, SidecarJobQueue
from lightrag.utils import EmbeddingFunc, compute_mdhash_id, logger


def _openai_llm_kwargs(args) -> dict[str, Any]:
    return OpenAILLMOptions.options_dict(args)


def _create_openai_llm(args):
    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        kwargs.update(_openai_llm_kwargs(args))
        return await openai_complete_if_cache(
            args.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=args.llm_binding_api_key,
            base_url=args.llm_binding_host,
            **kwargs,
        )

    return llm_model_func


def _create_openai_vision_llm(args):
    async def vision_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        image_data: str | None = None,
        **kwargs: Any,
    ) -> str:
        kwargs.update(_openai_llm_kwargs(args))
        if image_data:
            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        },
                    ],
                }
            )
            return await openai_complete_if_cache(
                args.raganything_vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=args.llm_binding_api_key,
                base_url=args.llm_binding_host,
                **kwargs,
            )

        return await openai_complete_if_cache(
            args.raganything_vision_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=args.llm_binding_api_key,
            base_url=args.llm_binding_host,
            **kwargs,
        )

    return vision_model_func


def _create_openai_embedding(args) -> EmbeddingFunc:
    async def embedding_func(texts, embedding_dim=None):
        kwargs: dict[str, Any] = {
            "texts": texts,
            "api_key": args.embedding_binding_api_key,
            "base_url": args.embedding_binding_host,
            "embedding_dim": embedding_dim,
        }
        if args.embedding_model:
            kwargs["model"] = args.embedding_model
        return await openai_embed(**kwargs)

    return EmbeddingFunc(
        embedding_dim=args.embedding_dim,
        max_token_size=args.embedding_token_limit,
        func=embedding_func,
        send_dimensions=bool(args.embedding_send_dim),
        model_name=args.embedding_model,
    )


def _create_lightrag(args) -> LightRAG:
    if args.llm_binding != "openai" or args.embedding_binding != "openai":
        raise RuntimeError(
            "RAG-Anything sidecar currently supports only OpenAI bindings. "
            f"Received llm={args.llm_binding} embedding={args.embedding_binding}."
        )

    return LightRAG(
        working_dir=args.working_dir,
        workspace=args.workspace,
        llm_model_func=_create_openai_llm(args),
        llm_model_name=args.llm_model,
        embedding_func=_create_openai_embedding(args),
        kv_storage=args.kv_storage,
        graph_storage=args.graph_storage,
        vector_storage=args.vector_storage,
        doc_status_storage=args.doc_status_storage,
        summary_max_tokens=args.summary_max_tokens,
        summary_context_size=args.summary_context_size,
        chunk_token_size=int(args.chunk_size),
        chunk_overlap_token_size=int(args.chunk_overlap_size),
        enable_llm_cache_for_entity_extract=args.enable_llm_cache_for_extract,
        enable_llm_cache=args.enable_llm_cache,
        vector_db_storage_cls_kwargs={
            "cosine_better_than_threshold": args.cosine_threshold
        },
    )


def _make_status_record(
    *,
    job: SidecarJob,
    status: DocStatus,
    error_msg: str | None = None,
) -> dict[str, dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    metadata = dict(job.metadata or {})
    metadata.update(
        {
            "sidecar_job": True,
            "parser": job.parser,
            "parse_method": job.parse_method,
            "attempt_count": job.attempt_count,
        }
    )
    if error_msg:
        metadata["last_error"] = error_msg

    record_id = compute_mdhash_id(f"{job.track_id}:{job.input_path}", prefix="sidecar-")
    return {
        record_id: {
            "status": status,
            "content_summary": f"[SIDECAR:{status.value.upper()}] {job.original_filename}",
            "content_length": int(metadata.get("content_length", 0)),
            "created_at": job.created_at or now,
            "updated_at": now,
            "file_path": job.original_filename,
            "track_id": job.track_id,
            "chunks_count": metadata.get("chunks_count", 0),
            "error_msg": error_msg,
            "metadata": metadata,
            "multimodal_processed": status == DocStatus.PROCESSED,
        }
    }


async def _update_track_id_for_file(
    rag: LightRAG, job: SidecarJob, *, error_message: str | None = None
) -> None:
    all_docs = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
    matched_updates: dict[str, dict[str, Any]] = {}

    for doc_id, doc_status in all_docs.items():
        if Path(doc_status.file_path).name != job.original_filename:
            continue

        metadata = dict(doc_status.metadata or {})
        metadata.update(
            {
                "sidecar_job": True,
                "sidecar_job_id": job.job_id,
                "parser": job.parser,
                "parse_method": job.parse_method,
                "attempt_count": job.attempt_count,
            }
        )
        matched_updates[doc_id] = {
            "status": DocStatus.PROCESSED,
            "content_summary": doc_status.content_summary,
            "content_length": doc_status.content_length,
            "created_at": doc_status.created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "file_path": doc_status.file_path,
            "track_id": job.track_id,
            "chunks_count": doc_status.chunks_count,
            "chunks_list": list(doc_status.chunks_list or []),
            "error_msg": error_message,
            "metadata": metadata,
            "multimodal_processed": True,
        }

    if matched_updates:
        await rag.doc_status.upsert(matched_updates)
    else:
        await rag.doc_status.upsert(
            _make_status_record(
                job=job,
                status=DocStatus.FAILED if error_message else DocStatus.PROCESSED,
                error_msg=error_message,
            )
        )

    await rag.doc_status.index_done_callback()


class SidecarWorker:
    def __init__(self, args):
        self.args = args
        self.queue = SidecarJobQueue(args.raganything_job_spool_dir)
        self.rag = _create_lightrag(args)
        self._rag_anything: dict[str, Any] = {}
        logger.info("Sidecar OpenAI LLM Options: %s", _openai_llm_kwargs(args))
        logger.info("Sidecar vision model: %s", args.raganything_vision_model)

    async def initialize(self) -> None:
        await self.rag.initialize_storages()
        await self.rag.check_and_migrate_data()
        recovered = self.queue.requeue_processing_jobs()
        if recovered:
            logger.warning("Recovered %s in-flight sidecar jobs back to queue", recovered)

    async def finalize(self) -> None:
        await self.rag.finalize_storages()

    async def get_raganything(self, parser_name: str):
        if parser_name not in self._rag_anything:
            try:
                from raganything import RAGAnything, RAGAnythingConfig
            except ImportError as exc:
                raise RuntimeError(
                    "RAGAnything is not installed in the sidecar image."
                ) from exc

            config = RAGAnythingConfig(
                working_dir=self.args.working_dir,
                parser=parser_name,
                parse_method=self.args.raganything_parse_method,
                enable_image_processing=self.args.raganything_enable_image_processing,
                enable_table_processing=self.args.raganything_enable_table_processing,
                enable_equation_processing=self.args.raganything_enable_equation_processing,
            )
            try:
                self._rag_anything[parser_name] = RAGAnything(
                    config=config,
                    lightrag=self.rag,
                    llm_model_func=_create_openai_llm(self.args),
                    vision_model_func=_create_openai_vision_llm(self.args),
                    embedding_func=_create_openai_embedding(self.args),
                )
            except TypeError:
                self._rag_anything[parser_name] = RAGAnything(
                    lightrag=self.rag,
                    vision_model_func=_create_openai_vision_llm(self.args),
                )
        return self._rag_anything[parser_name]

    async def process_job(self, job: SidecarJob) -> None:
        logger.info("Processing sidecar job %s for %s", job.job_id, job.original_filename)
        raganything = await self.get_raganything(job.parser)
        try:
            process_kwargs: dict[str, Any] = {}
            if job.parser == "mineru":
                for key in ("backend", "device", "table", "formula"):
                    if key in job.metadata:
                        process_kwargs[key] = job.metadata[key]
            await raganything.process_document_complete(
                file_path=job.input_path,
                output_dir=self.args.raganything_output_dir,
                parse_method=job.parse_method,
                **process_kwargs,
            )
            await _update_track_id_for_file(self.rag, job)
            self.queue.mark_done(job)
            logger.info("Completed sidecar job %s", job.job_id)
        except Exception as exc:
            error_message = str(exc)
            logger.error(
                "Sidecar job %s failed for %s: %s",
                job.job_id,
                job.original_filename,
                error_message,
            )
            logger.error(traceback.format_exc())
            final_job = self.queue.mark_retry_or_failed(
                job,
                error_message=error_message,
                max_retries=self.args.raganything_max_retries,
            )
            if final_job.status == "failed":
                await _update_track_id_for_file(self.rag, job, error_message=error_message)

    async def run(self) -> None:
        await self.initialize()
        try:
            while True:
                job = self.queue.claim_next_job()
                if job is None:
                    await asyncio.sleep(self.args.raganything_poll_interval_seconds)
                    continue
                await self.process_job(job)
        finally:
            await self.finalize()


async def _async_main() -> None:
    args = parse_args()
    worker = SidecarWorker(args)
    await worker.run()


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
