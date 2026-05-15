from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping

from pipeline.orchestration.artifacts import PipelinePaths, load_evidence
from pipeline.config import PipelineFeatures, ProviderConfig
from pipeline.io import write_json
from pipeline.retrieval.engine import save_index
from pipeline.schemas import EvidenceChunk, ProcessedDocument, to_jsonable


class EvidenceRetrievalComponent:
    name = "retrieve_evidence"

    def __init__(
        self,
        *,
        build_index: Callable[..., Any],
        retrieve: Callable[..., list[EvidenceChunk]],
    ) -> None:
        self._build_index = build_index
        self._retrieve = retrieve

    def run(
        self,
        *,
        processed: list[ProcessedDocument],
        task: str,
        paths: PipelinePaths,
        config: ProviderConfig,
        features: PipelineFeatures,
        retrieval_feedback: Mapping[str, Any] | None = None,
    ) -> list[EvidenceChunk]:
        index = self._build_index(
            processed,
            provider=config.retrieval_provider,
            config=config,
            include_field_chunks=features.field_chunks,
        )
        evidence = _call_with_supported_kwargs(
            self._retrieve,
            index,
            task,
            top_k=config.retrieval_top_k,
            provider=config.retrieval_provider,
            config=config,
            max_field_chunks=features.max_field_chunks,
            max_chunks_per_document=features.max_chunks_per_document,
            retrieval_feedback=retrieval_feedback if features.retrieval_feedback else None,
            field_chunk_score_penalty=features.field_chunk_score_penalty,
        )
        if hasattr(index, "chunks") and hasattr(index, "embeddings"):
            paths.retrieval_index.parent.mkdir(parents=True, exist_ok=True)
            save_index(index, paths.retrieval_index)
        write_json(paths.retrieved_evidence, to_jsonable(evidence))
        return evidence

    def load(self, paths: PipelinePaths) -> list[EvidenceChunk]:
        return load_evidence(paths.retrieved_evidence)


def _call_with_supported_kwargs(func: Callable[..., list[EvidenceChunk]], *args: Any, **kwargs: Any) -> list[EvidenceChunk]:
    signature = inspect.signature(func)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return func(*args, **kwargs)
    supported = {name for name in signature.parameters if name not in {"self"}}
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    return func(*args, **filtered)
