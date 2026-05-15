from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
import math
import os
import re
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from pipeline.config import ProviderConfig, is_cached_retrieval_provider, resolve_provider_name
from pipeline.io import read_json, write_json
from pipeline.providers import (
    ProviderUnavailable,
    embed_with_openai,
    responses_create,
)
from pipeline.schemas import EvidenceChunk, ProcessedDocument
from pipeline.schemas import now_iso


EMBEDDING_CACHE_FORMAT_VERSION = 2
RETRIEVAL_INDEX_FORMAT_VERSION = 1
_EMBEDDING_VECTOR_BYTES = 8
_FIELD_CHUNK_SCORE_PENALTY = 0.06
_RETRIEVAL_FEEDBACK_WEIGHT = 0.15
_RETRIEVAL_FEEDBACK_MAX_DELTA = 0.60
_DEFAULT_DENSE_WEIGHT = 0.9
_DEFAULT_LEXICAL_WEIGHT = 0.1
_BM25_K1 = 1.5
_BM25_B = 0.75
_SEARCH_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._'-][a-z0-9]+)*", re.IGNORECASE)
_QUOTE_NORMALIZATION_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
        " ": " ",
        " ": " ",
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
    }
)


@dataclass(slots=True)
class BM25Index:
    tokenized_documents: list[list[str]]
    term_frequencies: list[dict[str, int]]
    document_lengths: list[int]
    document_frequencies: dict[str, int]
    average_document_length: float
    k1: float = _BM25_K1
    b: float = _BM25_B


@dataclass(slots=True)
class RetrievalIndex:
    chunks: list[EvidenceChunk]
    embeddings: list[list[float]]
    provider: str
    embedding_model: str
    embedding_cache_dir: str = ""
    embedding_backend_id: str = "openai"
    bm25: BM25Index | None = None
    faiss_index: Any | None = None
    faiss_metric: str = ""


def build_index(
    processed_documents: list[ProcessedDocument],
    *,
    provider: str | None = None,
    config: ProviderConfig | None = None,
    chunk_words: int = 180,
    chunk_overlap: int = 40,
    include_field_chunks: bool = True,
    embedding_client: Any | Callable[[list[str]], list[list[float]]] | None = None,
    build_faiss: bool | None = None,
) -> RetrievalIndex:
    config = config or ProviderConfig.from_env()
    index_backend = (config.index_backend or "").strip().lower()
    retrieval_mode = (config.retrieval_mode or "").strip().lower()
    should_build_faiss = index_backend == "faiss" if build_faiss is None else build_faiss
    chunks = chunk_processed_documents(
        processed_documents,
        chunk_words=chunk_words,
        chunk_overlap=chunk_overlap,
        include_field_chunks=include_field_chunks,
    )
    provider_name = resolve_provider_name(provider, fallback=config.retrieval_provider)
    embedding_model = _embedding_model(config)
    embedding_backend_id = _embedding_backend_id(embedding_client)
    if not chunks:
        return RetrievalIndex(
            chunks=[],
            embeddings=[],
            provider=provider_name,
            embedding_model=embedding_model,
            embedding_cache_dir=str(getattr(config, "embedding_cache_dir", "") or ""),
            embedding_backend_id=embedding_backend_id,
            bm25=_build_bm25_index([]),
        )

    if retrieval_mode == "lexical":
        embeddings = [[0.0] for _chunk in chunks]
    elif embedding_client is not None:
        if is_cached_retrieval_provider(provider_name):
            embeddings = _embed_texts_with_cache(
                [chunk.text for chunk in chunks],
                embedding_model=embedding_model,
                embedding_backend_id=embedding_backend_id,
                cache_dir=_require_embedding_cache_dir(config),
                embed_missing=lambda texts: _embed_with_client(embedding_client, texts),
            )
        else:
            embeddings = _embed_with_client(embedding_client, [chunk.text for chunk in chunks])
    elif provider_name == "openai":
        embeddings = embed_with_openai([chunk.text for chunk in chunks], embedding_model)
    elif is_cached_retrieval_provider(provider_name):
        embeddings = _embed_texts_with_cache(
            [chunk.text for chunk in chunks],
            embedding_model=embedding_model,
            embedding_backend_id=embedding_backend_id,
            cache_dir=_require_embedding_cache_dir(config),
            embed_missing=lambda texts: embed_with_openai(texts, embedding_model),
        )
    elif provider_name == "fixture":
        embeddings = [_fixture_embedding(chunk.text) for chunk in chunks]
    else:
        raise ProviderUnavailable(f"Unsupported retrieval provider '{provider_name}'. This build only ships openai.")

    bm25 = _build_bm25_index(chunks)
    faiss_index, faiss_metric = _build_faiss_index(embeddings) if should_build_faiss else (None, "")
    return RetrievalIndex(
        chunks=chunks,
        embeddings=embeddings,
        provider=provider_name,
        embedding_model=embedding_model,
        embedding_cache_dir=str(getattr(config, "embedding_cache_dir", "") or ""),
        embedding_backend_id=embedding_backend_id,
        bm25=bm25,
        faiss_index=faiss_index,
        faiss_metric=faiss_metric,
    )


def retrieve(
    index: RetrievalIndex,
    task: str,
    *,
    top_k: int = 10,
    provider: str | None = None,
    config: ProviderConfig | None = None,
    embedding_client: Any | Callable[[list[str]], list[list[float]]] | None = None,
    reranker_provider: str | None = None,
    reranker_client: Any | None = None,
    rerank_pool: int | None = None,
    max_field_chunks: int = 1,
    max_chunks_per_document: int = 3,
    retrieval_feedback: Mapping[str, Any] | None = None,
    field_chunk_score_penalty: float = _FIELD_CHUNK_SCORE_PENALTY,
    dense_weight: float | None = None,
    lexical_weight: float | None = None,
    metadata_boosts: Mapping[str, Any] | None = None,
    use_faiss: bool | None = None,
) -> list[EvidenceChunk]:
    config = config or ProviderConfig.from_env()
    if top_k <= 0 or not index.chunks:
        return []
    if len(index.chunks) != len(index.embeddings):
        raise ProviderUnavailable("Retrieval index has mismatched chunks and embeddings")
    provider_name = resolve_provider_name(provider, fallback=index.provider)
    effective_dense_weight, effective_lexical_weight = _retrieval_weights(
        config,
        dense_weight=dense_weight,
        lexical_weight=lexical_weight,
    )
    effective_use_faiss = (config.index_backend or "").strip().lower() == "faiss" if use_faiss is None else use_faiss
    embedding_backend_id = (
        _embedding_backend_id(embedding_client)
        if embedding_client is not None
        else getattr(index, "embedding_backend_id", "openai")
    )

    if effective_dense_weight == 0:
        query_embedding = []
    elif embedding_client is not None:
        if is_cached_retrieval_provider(provider_name):
            query_embedding = _embed_texts_with_cache(
                [task],
                embedding_model=index.embedding_model,
                embedding_backend_id=embedding_backend_id,
                cache_dir=_require_embedding_cache_dir(config, fallback=index.embedding_cache_dir),
                embed_missing=lambda texts: _embed_with_client(embedding_client, texts),
            )[0]
        else:
            query_embedding = _embed_with_client(embedding_client, [task])[0]
    elif provider_name == "openai":
        query_embedding = embed_with_openai([task], index.embedding_model)[0]
    elif is_cached_retrieval_provider(provider_name):
        query_embedding = _embed_texts_with_cache(
            [task],
            embedding_model=index.embedding_model,
            embedding_backend_id=embedding_backend_id,
            cache_dir=_require_embedding_cache_dir(config, fallback=index.embedding_cache_dir),
            embed_missing=lambda texts: embed_with_openai(texts, index.embedding_model),
        )[0]
    elif provider_name == "fixture":
        query_embedding = _fixture_embedding(task)
    else:
        raise ProviderUnavailable(f"Unsupported retrieval provider '{provider_name}'. This build only ships openai.")

    dense_scores = (
        [0.0] * len(index.embeddings)
        if effective_dense_weight == 0
        else _dense_scores(index, query_embedding, use_faiss=effective_use_faiss)
    )
    lexical_scores = (
        [0.0] * len(index.chunks)
        if effective_lexical_weight == 0
        else _bm25_scores(_ensure_bm25_index(index), task)
    )
    hybrid_scores = _combine_hybrid_scores(
        dense_scores,
        lexical_scores,
        dense_weight=effective_dense_weight,
        lexical_weight=effective_lexical_weight,
    )

    scored: list[EvidenceChunk] = []
    for position, chunk in enumerate(index.chunks):
        base_score = hybrid_scores[position]
        score, feedback_metadata = _adjust_retrieval_score(
            base_score,
            chunk,
            retrieval_feedback=retrieval_feedback,
            field_chunk_score_penalty=field_chunk_score_penalty,
            metadata_boosts=metadata_boosts,
        )
        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "dense_score": dense_scores[position],
                "bm25_score": lexical_scores[position],
                "hybrid_score": base_score,
            }
        )
        if score != base_score:
            metadata["base_score"] = base_score
            metadata.update(feedback_metadata)
        scored.append(
            EvidenceChunk(
                evidence_id=chunk.evidence_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                text=chunk.text,
                score=score,
                metadata=metadata,
            )
        )

    scored.sort(key=lambda item: (-item.score, item.evidence_id))
    pool_size = rerank_pool or min(len(scored), max(top_k * 4, top_k))
    candidates = scored[:pool_size]
    candidates = _rerank_if_configured(
        task,
        candidates,
        top_k=top_k,
        reranker_provider=reranker_provider,
        reranker_client=reranker_client,
        config=config,
    )
    candidates = _cap_per_document(candidates, top_k=top_k, max_chunks_per_document=max_chunks_per_document)
    return _cap_field_chunks(candidates, top_k=top_k, max_field_chunks=max_field_chunks)


def save_index(index: RetrievalIndex, path: str | Path) -> None:
    """Persist a retrieval index as portable JSON.

    The optional FAISS object is intentionally excluded because it is a runtime
    acceleration structure. load_index() rebuilds it when faiss is available.
    """

    write_json(Path(path), _index_to_payload(index))


def load_index(path: str | Path, *, build_faiss: bool = True) -> RetrievalIndex:
    payload = read_json(Path(path))
    if not isinstance(payload, Mapping):
        raise ValueError("Retrieval index payload must be a JSON object")
    version = payload.get("retrieval_index_format_version")
    if version != RETRIEVAL_INDEX_FORMAT_VERSION:
        raise ValueError(f"Unsupported retrieval index format version: {version!r}")

    chunks_payload = payload.get("chunks")
    embeddings_payload = payload.get("embeddings")
    if not isinstance(chunks_payload, list) or not isinstance(embeddings_payload, list):
        raise ValueError("Retrieval index payload must include chunks and embeddings lists")

    chunks = [_chunk_from_payload(item) for item in chunks_payload]
    embeddings = [_embedding_from_payload(item) for item in embeddings_payload]
    if len(chunks) != len(embeddings):
        raise ValueError("Retrieval index payload has mismatched chunks and embeddings")

    bm25 = _bm25_from_payload(payload.get("bm25"))
    if bm25 is None or len(bm25.tokenized_documents) != len(chunks):
        bm25 = _build_bm25_index(chunks)
    faiss_index, faiss_metric = _build_faiss_index(embeddings) if build_faiss else (None, "")

    return RetrievalIndex(
        chunks=chunks,
        embeddings=embeddings,
        provider=str(payload.get("provider") or "fixture"),
        embedding_model=str(payload.get("embedding_model") or ""),
        embedding_cache_dir=str(payload.get("embedding_cache_dir") or ""),
        embedding_backend_id=str(payload.get("embedding_backend_id") or "openai"),
        bm25=bm25,
        faiss_index=faiss_index,
        faiss_metric=faiss_metric,
    )


def _index_to_payload(index: RetrievalIndex) -> dict[str, Any]:
    bm25 = index.bm25 if index.bm25 is not None else _build_bm25_index(index.chunks)
    return {
        "retrieval_index_format_version": RETRIEVAL_INDEX_FORMAT_VERSION,
        "provider": index.provider,
        "embedding_model": index.embedding_model,
        "embedding_cache_dir": index.embedding_cache_dir,
        "embedding_backend_id": getattr(index, "embedding_backend_id", "openai"),
        "embedding_dim": len(index.embeddings[0]) if index.embeddings else 0,
        "chunks": [_chunk_to_payload(chunk) for chunk in index.chunks],
        "embeddings": [[float(value) for value in embedding] for embedding in index.embeddings],
        "bm25": _bm25_to_payload(bm25),
    }


def _chunk_to_payload(chunk: EvidenceChunk) -> dict[str, Any]:
    return {
        "evidence_id": chunk.evidence_id,
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "page_number": chunk.page_number,
        "text": chunk.text,
        "score": chunk.score,
        "metadata": chunk.metadata,
    }


def _chunk_from_payload(value: Any) -> EvidenceChunk:
    if not isinstance(value, Mapping):
        raise ValueError("Retrieval index chunk must be a JSON object")
    metadata = value.get("metadata")
    return EvidenceChunk(
        evidence_id=str(value.get("evidence_id") or ""),
        document_id=str(value.get("document_id") or ""),
        filename=str(value.get("filename") or ""),
        page_number=int(value.get("page_number") or 0),
        text=str(value.get("text") or ""),
        score=float(value.get("score") or 0.0),
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )


def _embedding_from_payload(value: Any) -> list[float]:
    if not isinstance(value, list):
        raise ValueError("Retrieval index embedding must be a list")
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("Retrieval index embedding contains a non-numeric value") from exc
    if not vector:
        raise ValueError("Retrieval index embedding must not be empty")
    return vector


def _bm25_to_payload(index: BM25Index) -> dict[str, Any]:
    return {
        "tokenized_documents": index.tokenized_documents,
        "term_frequencies": index.term_frequencies,
        "document_lengths": index.document_lengths,
        "document_frequencies": index.document_frequencies,
        "average_document_length": index.average_document_length,
        "k1": index.k1,
        "b": index.b,
    }


def _bm25_from_payload(value: Any) -> BM25Index | None:
    if not isinstance(value, Mapping):
        return None
    try:
        tokenized_documents = [
            [str(token) for token in tokens]
            for tokens in value.get("tokenized_documents", [])
            if isinstance(tokens, list)
        ]
        term_frequencies = [
            {str(term): int(count) for term, count in frequencies.items()}
            for frequencies in value.get("term_frequencies", [])
            if isinstance(frequencies, Mapping)
        ]
        document_lengths = [int(item) for item in value.get("document_lengths", [])]
        document_frequencies = {
            str(term): int(count)
            for term, count in dict(value.get("document_frequencies", {})).items()
        }
        average_document_length = float(value.get("average_document_length") or 0.0)
        k1 = float(value.get("k1") or _BM25_K1)
        b = float(value.get("b") or _BM25_B)
    except (TypeError, ValueError):
        return None
    if not (
        len(tokenized_documents)
        == len(term_frequencies)
        == len(document_lengths)
    ):
        return None
    return BM25Index(
        tokenized_documents=tokenized_documents,
        term_frequencies=term_frequencies,
        document_lengths=document_lengths,
        document_frequencies=document_frequencies,
        average_document_length=average_document_length,
        k1=k1,
        b=b,
    )


def _ensure_bm25_index(index: RetrievalIndex) -> BM25Index:
    bm25 = getattr(index, "bm25", None)
    if bm25 is None or len(bm25.tokenized_documents) != len(index.chunks):
        bm25 = _build_bm25_index(index.chunks)
        index.bm25 = bm25
    return bm25


def _build_bm25_index(chunks: list[EvidenceChunk]) -> BM25Index:
    tokenized_documents = [_tokenize_search_text(chunk.text) for chunk in chunks]
    term_frequencies = [dict(Counter(tokens)) for tokens in tokenized_documents]
    document_lengths = [len(tokens) for tokens in tokenized_documents]
    document_frequencies: Counter[str] = Counter()
    for tokens in tokenized_documents:
        document_frequencies.update(set(tokens))
    average_document_length = (
        sum(document_lengths) / len(document_lengths)
        if document_lengths
        else 0.0
    )
    return BM25Index(
        tokenized_documents=tokenized_documents,
        term_frequencies=term_frequencies,
        document_lengths=document_lengths,
        document_frequencies=dict(document_frequencies),
        average_document_length=average_document_length,
    )


def _tokenize_search_text(text: str) -> list[str]:
    normalized = normalized_embedding_text(text)
    return [match.group(0) for match in _SEARCH_TOKEN_RE.finditer(normalized)]


def _bm25_scores(index: BM25Index, query: str) -> list[float]:
    terms = _tokenize_search_text(query)
    if not terms or not index.tokenized_documents:
        return [0.0] * len(index.tokenized_documents)
    query_counts = Counter(terms)
    document_count = len(index.tokenized_documents)
    average_length = index.average_document_length or 1.0
    scores: list[float] = []
    for frequencies, document_length in zip(index.term_frequencies, index.document_lengths):
        score = 0.0
        length_factor = 1.0 - index.b + index.b * (document_length / average_length)
        for term, query_count in query_counts.items():
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            document_frequency = index.document_frequencies.get(term, 0)
            idf = math.log(1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = frequency + index.k1 * length_factor
            if denominator:
                score += query_count * idf * (frequency * (index.k1 + 1.0) / denominator)
        scores.append(score)
    return _normalize_positive_scores(scores)


def _normalize_positive_scores(scores: list[float]) -> list[float]:
    max_score = max(scores, default=0.0)
    if max_score <= 0:
        return [0.0] * len(scores)
    return [score / max_score if score > 0 else 0.0 for score in scores]


def _dense_scores(
    index: RetrievalIndex,
    query_embedding: list[float],
    *,
    use_faiss: bool,
) -> list[float]:
    if use_faiss:
        faiss_scores = _faiss_dense_scores(index, query_embedding)
        if faiss_scores is not None:
            return faiss_scores
    return [_cosine(query_embedding, embedding) for embedding in index.embeddings]


def _combine_hybrid_scores(
    dense_scores: list[float],
    lexical_scores: list[float],
    *,
    dense_weight: float,
    lexical_weight: float,
) -> list[float]:
    dense_weight = _positive_float(dense_weight)
    lexical_weight = _positive_float(lexical_weight)
    total_weight = dense_weight + lexical_weight
    if total_weight == 0:
        dense_weight = 1.0
        total_weight = 1.0
    return [
        ((dense_scores[position] * dense_weight) + (lexical_scores[position] * lexical_weight)) / total_weight
        for position in range(min(len(dense_scores), len(lexical_scores)))
    ]


def _retrieval_weights(
    config: ProviderConfig,
    *,
    dense_weight: float | None,
    lexical_weight: float | None,
) -> tuple[float, float]:
    if dense_weight is None and lexical_weight is None:
        retrieval_mode = (config.retrieval_mode or "").strip().lower()
        if retrieval_mode == "dense":
            return 1.0, 0.0
        if retrieval_mode == "lexical":
            return 0.0, 1.0
        return _positive_float(config.hybrid_dense_weight), _positive_float(config.hybrid_bm25_weight)
    resolved_dense = config.hybrid_dense_weight if dense_weight is None else dense_weight
    resolved_lexical = config.hybrid_bm25_weight if lexical_weight is None else lexical_weight
    return _positive_float(resolved_dense), _positive_float(resolved_lexical)


def _positive_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number) or number < 0:
        return 0.0
    return number


def _build_faiss_index(embeddings: list[list[float]]) -> tuple[Any | None, str]:
    if not embeddings:
        return None, ""
    try:
        import faiss  # type: ignore[import-not-found]
        import numpy as np
    except ImportError:
        return None, ""
    try:
        matrix = np.asarray(embeddings, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[0] != len(embeddings) or matrix.shape[1] == 0:
            return None, ""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms
        index = faiss.IndexFlatIP(int(matrix.shape[1]))
        index.add(matrix)
    except Exception as exc:  # pragma: no cover - depends on optional faiss/numpy builds
        warnings.warn(f"FAISS vector index unavailable; falling back to cosine scan: {exc}", stacklevel=2)
        return None, ""
    return index, "inner_product_normalized"


def _faiss_dense_scores(index: RetrievalIndex, query_embedding: list[float]) -> list[float] | None:
    faiss_index = getattr(index, "faiss_index", None)
    if faiss_index is None:
        faiss_index, faiss_metric = _build_faiss_index(index.embeddings)
        if faiss_index is None:
            return None
        index.faiss_index = faiss_index
        index.faiss_metric = faiss_metric
    try:
        import numpy as np

        query = np.asarray([query_embedding], dtype="float32")
        if query.ndim != 2 or query.shape[1] == 0:
            return None
        norm = np.linalg.norm(query, axis=1, keepdims=True)
        norm[norm == 0.0] = 1.0
        query = query / norm
        distances, indices = faiss_index.search(query, len(index.embeddings))
    except Exception:
        return None

    scores = [0.0] * len(index.embeddings)
    for score, raw_index in zip(distances[0], indices[0]):
        position = int(raw_index)
        if 0 <= position < len(scores):
            scores[position] = float(score)
    return scores


def _cap_per_document(
    candidates: list[EvidenceChunk],
    *,
    top_k: int,
    max_chunks_per_document: int,
) -> list[EvidenceChunk]:
    """Diversity-pass primary slots, then refill remaining top_k from overflow.

    This is a **soft cap with overflow refill**, matching the contract of
    ``_cap_field_chunks`` so both diversity passes behave the same way:

    1. Walk candidates in score order. Each source contributes up to
       ``max_chunks_per_document`` chunks to the *primary* set; later chunks
       from a source already at the cap fall into the *overflow* set.
    2. If the primary set already has ``top_k`` items, return it (the cap
       was hard for this query).
    3. Otherwise concatenate ``primary + overflow`` and truncate to
       ``top_k`` (the cap was soft for this query — overflow refills so the
       drafter still sees a full top_k).

    The soft refill is deliberate: if only two sources have any retrieval
    signal but ``top_k=10``, returning four chunks would starve the drafter.
    Pass ``max_chunks_per_document=0`` to disable the cap entirely; values
    above the candidate pool's natural diversity reduce to a no-op.

    Apply this cap **after** scoring/rerank and **before**
    ``_cap_field_chunks``, so document diversity is established first and
    the structured-field slot is carved out of the diversified pool.
    """

    if max_chunks_per_document <= 0 or not candidates:
        return candidates[:top_k]
    primary: list[EvidenceChunk] = []
    overflow: list[EvidenceChunk] = []
    seen_per_doc: dict[str, int] = {}
    for chunk in candidates:
        key = chunk.document_id or chunk.filename or ""
        if seen_per_doc.get(key, 0) >= max_chunks_per_document:
            overflow.append(chunk)
            continue
        primary.append(chunk)
        seen_per_doc[key] = seen_per_doc.get(key, 0) + 1
        if len(primary) >= top_k:
            return primary
    return (primary + overflow)[:top_k]


def _cap_field_chunks(
    candidates: list[EvidenceChunk],
    *,
    top_k: int,
    max_field_chunks: int,
) -> list[EvidenceChunk]:
    """Return top_k candidates while keeping at most max_field_chunks :fields chunks.

    Synthetic field chunks (one per document, packing all extracted fields into
    one passage) tend to dominate cosine retrieval — they correlate strongly
    with most queries and crowd out raw page chunks. Cap the count so raw
    pages get real airtime in the evidence pack.
    """

    if max_field_chunks < 0 or not candidates:
        return candidates[:top_k]
    primary: list[EvidenceChunk] = []
    overflow: list[EvidenceChunk] = []
    field_count = 0
    for chunk in candidates:
        is_field = bool(chunk.metadata.get("is_field_chunk")) or chunk.evidence_id.endswith(":fields")
        if is_field and field_count >= max_field_chunks:
            overflow.append(chunk)
            continue
        primary.append(chunk)
        if is_field:
            field_count += 1
        if len(primary) >= top_k:
            return primary
    return (primary + overflow)[:top_k]


def _adjust_retrieval_score(
    base_score: float,
    chunk: EvidenceChunk,
    *,
    retrieval_feedback: Mapping[str, Any] | None,
    field_chunk_score_penalty: float,
    metadata_boosts: Mapping[str, Any] | None,
) -> tuple[float, dict[str, Any]]:
    score = base_score
    metadata: dict[str, Any] = {}
    if _is_field_chunk(chunk):
        penalty = max(0.0, float(field_chunk_score_penalty or 0.0))
        if penalty:
            score -= penalty
            metadata["derived_field_chunk_penalty"] = penalty

    metadata_delta, metadata_matches = _metadata_boost_delta(chunk, metadata_boosts)
    if metadata_delta:
        score += metadata_delta
        metadata["metadata_boost_delta"] = metadata_delta
        metadata["metadata_boosts"] = metadata_matches

    feedback_delta, boost_count, demote_count = _retrieval_feedback_delta(
        chunk.evidence_id,
        retrieval_feedback,
    )
    if feedback_delta:
        score += feedback_delta
        metadata["retrieval_feedback_delta"] = feedback_delta
        if boost_count:
            metadata["retrieval_feedback_boosts"] = boost_count
        if demote_count:
            metadata["retrieval_feedback_demotes"] = demote_count
    return score, metadata


def _metadata_boost_delta(
    chunk: EvidenceChunk,
    metadata_boosts: Mapping[str, Any] | None,
) -> tuple[float, dict[str, float]]:
    if not isinstance(metadata_boosts, Mapping):
        return 0.0, {}
    total = 0.0
    matches: dict[str, float] = {}
    for key, rule in metadata_boosts.items():
        field_name = str(key or "").strip()
        if not field_name:
            continue
        value = _metadata_value(chunk, field_name)
        delta = _metadata_rule_delta(value, rule)
        if not delta:
            continue
        total += delta
        matches[field_name] = matches.get(field_name, 0.0) + delta
    return total, matches


def _metadata_value(chunk: EvidenceChunk, key: str) -> Any:
    if key in chunk.metadata:
        return chunk.metadata.get(key)
    if key in {"evidence_id", "document_id", "filename", "page_number", "text", "score"}:
        return getattr(chunk, key)
    return None


def _metadata_rule_delta(value: Any, rule: Any) -> float:
    if isinstance(rule, Mapping):
        delta = 0.0
        for expected, raw_boost in rule.items():
            if _metadata_matches(value, expected):
                delta += _positive_float(raw_boost)
        return delta
    return _positive_float(rule) if _metadata_is_present(value) else 0.0


def _metadata_matches(value: Any, expected: Any) -> bool:
    if expected == "*":
        return _metadata_is_present(value)
    if isinstance(value, list | tuple | set):
        return any(_metadata_matches(item, expected) for item in value)
    return normalized_embedding_text(str(value)) == normalized_embedding_text(str(expected))


def _metadata_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | set | dict):
        return bool(value)
    return bool(value)


def _retrieval_feedback_delta(
    evidence_id: str,
    feedback: Mapping[str, Any] | None,
) -> tuple[float, int, int]:
    if not isinstance(feedback, Mapping):
        return 0.0, 0, 0
    boosts = _feedback_counter(feedback.get("evidence_boosts"))
    demotes = _feedback_counter(feedback.get("evidence_demotes"))
    boost_count = boosts.get(evidence_id, 0)
    demote_count = demotes.get(evidence_id, 0)
    raw_delta = (boost_count - demote_count) * _RETRIEVAL_FEEDBACK_WEIGHT
    raw_delta = max(-_RETRIEVAL_FEEDBACK_MAX_DELTA, min(_RETRIEVAL_FEEDBACK_MAX_DELTA, raw_delta))
    return raw_delta, boost_count, demote_count


def _feedback_counter(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    counters: dict[str, int] = {}
    for key, raw_count in value.items():
        evidence_id = str(key or "").strip()
        if not evidence_id:
            continue
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count > 0:
            counters[evidence_id] = count
    return counters


def _is_field_chunk(chunk: EvidenceChunk) -> bool:
    return bool(chunk.metadata.get("is_field_chunk")) or chunk.evidence_id.endswith(":fields")


def chunk_processed_documents(
    processed_documents: list[ProcessedDocument],
    *,
    chunk_words: int = 180,
    chunk_overlap: int = 40,
    include_field_chunks: bool = True,
) -> list[EvidenceChunk]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_words:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_words")

    chunks: list[EvidenceChunk] = []
    for document in processed_documents:
        for page in document.pages:
            words = page.text.split()
            if not words:
                continue
            start = 0
            local_index = 0
            step = chunk_words - chunk_overlap
            while start < len(words):
                end = min(start + chunk_words, len(words))
                text = " ".join(words[start:end])
                evidence_id = f"{document.source.document_id}:p{page.page_number}:c{local_index}"
                chunk_hash = content_hash(text)
                chunks.append(
                    EvidenceChunk(
                        evidence_id=evidence_id,
                        document_id=document.source.document_id,
                        filename=document.source.filename,
                        page_number=page.page_number,
                        text=text,
                        metadata={
                            "evidence_kind": "source-page",
                            "confidence": page.confidence,
                            "extraction_method": page.extraction_method,
                            "warnings": list(page.warnings),
                            "word_start": start,
                            "word_end": end,
                            "chunk_hash": chunk_hash,
                        },
                    )
                )
                if end == len(words):
                    break
                start += step
                local_index += 1
        if include_field_chunks:
            fields_chunk = _build_fields_chunk(document)
            if fields_chunk is not None:
                chunks.append(fields_chunk)
    return chunks


def _build_fields_chunk(document: ProcessedDocument) -> EvidenceChunk | None:
    if not document.fields:
        return None
    lines: list[str] = []
    confidences: list[float] = []
    source_evidence_ids: set[str] = set()
    for name, field in sorted(document.fields.items()):
        value = (field.value or "").strip()
        if not value:
            continue
        lines.append(f"{name}: {value}")
        confidences.append(field.confidence)
        source_evidence_ids.update(str(item) for item in field.evidence_ids if str(item).strip())
    if not lines:
        return None
    text = f"Extracted fields for {document.source.filename}:\n" + "\n".join(lines)
    average_confidence = sum(confidences) / len(confidences)
    return EvidenceChunk(
        evidence_id=f"{document.source.document_id}:fields",
        document_id=document.source.document_id,
        filename=document.source.filename,
        page_number=0,
        text=text,
        metadata={
            "evidence_kind": "derived-fields",
            "is_field_chunk": True,
            "confidence": average_confidence,
            "extraction_method": "structured-fields",
            "derived_from": "structured_extraction",
            "source_evidence_ids": sorted(source_evidence_ids),
            "chunk_hash": content_hash(text),
        },
    )


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ProviderUnavailable("Embedding vectors must have matching dimensions")
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _embedding_model(config: ProviderConfig) -> str:
    return getattr(config, "openai_embedding_model", None) or os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-large"


def normalized_embedding_text(text: str) -> str:
    translated = (text or "").translate(_QUOTE_NORMALIZATION_MAP)
    return " ".join(translated.split()).casefold()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalized_embedding_text(text).encode("utf-8")).hexdigest()


def raw_embedding_input_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def embedding_cache_key(
    text: str,
    *,
    embedding_model: str,
    embedding_backend_id: str = "openai",
) -> str:
    # Bump EMBEDDING_CACHE_FORMAT_VERSION if this payload changes.
    payload = {
        "embedding_backend_id": embedding_backend_id,
        "embedding_model": embedding_model,
        "input_hash": raw_embedding_input_hash(text),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _embedding_backend_id(embedding_client: Any | None) -> str:
    """Return the cache namespace for the embedding backend.

    Injected clients whose runtime state changes embeddings should set
    cache_identity to keep their persistent caches isolated.
    """
    if embedding_client is None:
        return "openai"
    identity = getattr(embedding_client, "cache_identity", None)
    if not identity:
        identity = f"{embedding_client.__class__.__module__}.{embedding_client.__class__.__qualname__}"
    return f"injected:{str(identity).strip() or 'anonymous'}"


def _require_embedding_cache_dir(config: ProviderConfig, fallback: str = "") -> Path:
    cache_dir = str(getattr(config, "embedding_cache_dir", "") or fallback or "").strip()
    if not cache_dir:
        raise ProviderUnavailable(
            "Cached OpenAI retrieval requires embedding_cache_dir. "
            "Set PIPELINE_EMBEDDING_CACHE_DIR or run through run_case(), which defaults to <output>/embedding_cache."
        )
    return Path(cache_dir)


def _embedding_cache_path(cache_dir: Path, embedding_model: str, embedding_backend_id: str) -> Path:
    model_key = hashlib.sha256(embedding_model.encode("utf-8")).hexdigest()[:12]
    if embedding_backend_id == "openai":
        return cache_dir / f"openai_embeddings_{model_key}.json"
    backend_key = hashlib.sha256(embedding_backend_id.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"openai_embeddings_{model_key}_{backend_key}.json"


def _empty_embedding_cache(embedding_model: str, embedding_backend_id: str) -> dict[str, Any]:
    return {
        "cache_format_version": EMBEDDING_CACHE_FORMAT_VERSION,
        "embedding_provider": "openai" if embedding_backend_id == "openai" else "injected",
        "embedding_backend_id": embedding_backend_id,
        "embedding_model": embedding_model,
        "embedding_dim": 0,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "entries": {},
    }


def _load_embedding_cache(cache_dir: Path, embedding_model: str, embedding_backend_id: str) -> dict[str, Any]:
    path = _embedding_cache_path(cache_dir, embedding_model, embedding_backend_id)
    if not path.exists():
        return _empty_embedding_cache(embedding_model, embedding_backend_id)
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        warnings.warn(f"Ignoring unreadable embedding cache at {path}: {exc}", stacklevel=2)
        return _empty_embedding_cache(embedding_model, embedding_backend_id)
    if not isinstance(payload, dict):
        warnings.warn(f"Ignoring embedding cache at {path}: cache root is not a JSON object", stacklevel=2)
        return _empty_embedding_cache(embedding_model, embedding_backend_id)
    if not _cache_payload_matches(payload, embedding_model, embedding_backend_id):
        return _empty_embedding_cache(embedding_model, embedding_backend_id)
    if not isinstance(payload.get("entries"), dict):
        return _empty_embedding_cache(embedding_model, embedding_backend_id)
    try:
        embedding_dim = int(payload.get("embedding_dim") or 0)
    except (TypeError, ValueError):
        warnings.warn(f"Ignoring embedding cache at {path}: embedding_dim is invalid", stacklevel=2)
        return _empty_embedding_cache(embedding_model, embedding_backend_id)
    payload["embedding_dim"] = embedding_dim
    return payload


def _cache_payload_matches(payload: dict[str, Any], embedding_model: str, embedding_backend_id: str) -> bool:
    expected_provider = "openai" if embedding_backend_id == "openai" else "injected"
    return (
        payload.get("cache_format_version") == EMBEDDING_CACHE_FORMAT_VERSION
        and payload.get("embedding_provider") == expected_provider
        and payload.get("embedding_backend_id", "openai") == embedding_backend_id
        and payload.get("embedding_model") == embedding_model
    )


def _save_embedding_cache(
    cache_dir: Path,
    embedding_model: str,
    embedding_backend_id: str,
    payload: dict[str, Any],
) -> None:
    payload["updated_at"] = now_iso()
    path = _embedding_cache_path(cache_dir, embedding_model, embedding_backend_id)
    try:
        if path.exists():
            prior = read_json(path)
            if isinstance(prior, dict) and _cache_payload_matches(prior, embedding_model, embedding_backend_id):
                prior_entries = prior.get("entries")
                if isinstance(prior_entries, dict):
                    merged_entries = dict(prior_entries)
                    merged_entries.update(payload.get("entries") or {})
                    payload["entries"] = merged_entries
        write_json(path, payload)
    except OSError as exc:
        warnings.warn(f"Could not write embedding cache at {path}: {exc}", stacklevel=2)


def _embed_texts_with_cache(
    texts: list[str],
    *,
    embedding_model: str,
    embedding_backend_id: str,
    cache_dir: Path,
    embed_missing: Callable[[list[str]], list[list[float]]],
) -> list[list[float]]:
    payload = _load_embedding_cache(cache_dir, embedding_model, embedding_backend_id)
    entries = payload.setdefault("entries", {})
    embedding_dim = int(payload.get("embedding_dim") or 0)

    missing_texts: list[str] = []
    missing_keys: list[str] = []
    seen_missing: set[str] = set()
    for text in texts:
        key = embedding_cache_key(
            text,
            embedding_model=embedding_model,
            embedding_backend_id=embedding_backend_id,
        )
        cached = entries.get(key)
        if _cached_embedding_is_valid(
            cached,
            embedding_dim=embedding_dim,
            input_hash=raw_embedding_input_hash(text),
        ):
            continue
        if key not in seen_missing:
            seen_missing.add(key)
            missing_keys.append(key)
            missing_texts.append(text)

    if missing_texts:
        new_embeddings = embed_missing(missing_texts)
        if len(new_embeddings) != len(missing_texts):
            raise ProviderUnavailable("Embedding provider returned the wrong number of vectors")
        for key, text, embedding in zip(missing_keys, missing_texts, new_embeddings):
            try:
                vector = [float(value) for value in embedding]
            except (TypeError, ValueError) as exc:
                raise ProviderUnavailable("Embedding provider returned a non-numeric vector") from exc
            if not vector:
                raise ProviderUnavailable("Embedding provider returned an empty vector")
            if embedding_dim and len(vector) != embedding_dim:
                raise ProviderUnavailable(
                    f"Embedding vector dimension changed for {embedding_model}: "
                    f"cached={embedding_dim}, new={len(vector)}"
                )
            embedding_dim = embedding_dim or len(vector)
            entries[key] = {
                "input_hash": raw_embedding_input_hash(text),
                "content_hash": content_hash(text),
                "embedding_b64": _encode_embedding(vector),
            }
        payload["embedding_dim"] = embedding_dim
        _save_embedding_cache(cache_dir, embedding_model, embedding_backend_id, payload)

    embeddings: list[list[float]] = []
    for text in texts:
        key = embedding_cache_key(
            text,
            embedding_model=embedding_model,
            embedding_backend_id=embedding_backend_id,
        )
        cached = entries.get(key)
        if not _cached_embedding_is_valid(
            cached,
            embedding_dim=embedding_dim,
            input_hash=raw_embedding_input_hash(text),
        ):
            raise ProviderUnavailable("Embedding cache did not contain an expected vector after update")
        embeddings.append(_decode_embedding(cached["embedding_b64"], embedding_dim=embedding_dim))
    return embeddings


def _cached_embedding_is_valid(value: Any, *, embedding_dim: int, input_hash: str) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("input_hash") != input_hash:
        return False
    encoded = value.get("embedding_b64")
    if not isinstance(encoded, str) or not encoded:
        return False
    if not embedding_dim:
        return False
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, TypeError):
        return False
    if len(raw) != embedding_dim * _EMBEDDING_VECTOR_BYTES:
        return False
    return True


def _encode_embedding(vector: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(vector)}d", *vector)).decode("ascii")


def _decode_embedding(encoded: str, *, embedding_dim: int) -> list[float]:
    raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    if embedding_dim <= 0 or len(raw) != embedding_dim * _EMBEDDING_VECTOR_BYTES:
        raise ProviderUnavailable("Cached embedding vector has the wrong byte length")
    return [float(value) for value in struct.unpack(f"<{embedding_dim}d", raw)]


def _embed_with_client(client: Any | Callable[[list[str]], list[list[float]]], texts: list[str]) -> list[list[float]]:
    if hasattr(client, "embed"):
        embeddings = client.embed(texts)
    elif callable(client):
        embeddings = client(texts)
    else:
        raise ProviderUnavailable("Injected embedding_client must be callable or expose embed(texts).")
    if len(embeddings) != len(texts):
        raise ProviderUnavailable("Embedding client returned the wrong number of vectors")
    try:
        return [[float(value) for value in vector] for vector in embeddings]
    except (TypeError, ValueError) as exc:
        raise ProviderUnavailable("Embedding client returned a non-numeric vector") from exc


def _resolve_reranker_provider(reranker_provider: str | None, config: ProviderConfig) -> str | None:
    raw = reranker_provider if reranker_provider is not None else getattr(config, "reranker_provider", "")
    provider_name = raw.strip().lower()
    if not provider_name or provider_name in {"none", "off", "false", "0"}:
        return None
    if provider_name != "openai":
        raise ProviderUnavailable(
            f"Unsupported reranker provider '{provider_name}'. This build only ships openai."
        )
    return provider_name


def _api_rerank(
    task: str,
    candidates: list[EvidenceChunk],
    *,
    top_k: int,
    provider: str,
    model: str,
    reasoning_effort: str = "low",
) -> list[str]:
    prompt = _rerank_prompt(task, candidates, top_k=top_k)
    if provider != "openai":
        raise ProviderUnavailable(
            f"Unsupported reranker provider '{provider}'. This build only ships openai."
        )
    if model != "gpt-5.5":
        raise ProviderUnavailable(f"OpenAI reranking is pinned to gpt-5.5, got {model!r}.")
    text = responses_create(
        prompt,
        model,
        reasoning_effort=reasoning_effort,
    )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderUnavailable("Reranker response was not valid JSON") from exc
    ids = payload.get("evidence_ids")
    if not isinstance(ids, list):
        raise ProviderUnavailable("Reranker response did not include evidence_ids")
    return [str(item) for item in ids if isinstance(item, str)]


def _rerank_prompt(task: str, candidates: list[EvidenceChunk], *, top_k: int) -> str:
    payload = [
        {
            "evidence_id": candidate.evidence_id,
            "score": candidate.score,
            "filename": candidate.filename,
            "page_number": candidate.page_number,
            "text": _truncate(candidate.text, 900),
        }
        for candidate in candidates
    ]
    return (
        "Rerank evidence for an internal legal memo drafting task. "
        "Use only the candidate evidence ids. Return strict JSON with one key, "
        '"evidence_ids", ordered from most to least relevant. '
        f"Return at most {top_k} ids.\n\n"
        f"Task:\n{task}\n\nCandidates:\n{json.dumps(payload, ensure_ascii=True)}"
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _rerank_if_configured(
    task: str,
    candidates: list[EvidenceChunk],
    *,
    top_k: int,
    reranker_provider: str | None,
    reranker_client: Any | None,
    config: ProviderConfig,
) -> list[EvidenceChunk]:
    provider_name = _resolve_reranker_provider(reranker_provider, config)
    if reranker_client is None and provider_name is None:
        return candidates
    if reranker_client is not None:
        ordered_ids = reranker_client.rerank(task, candidates, top_k=top_k)
    else:
        ordered_ids = _api_rerank(
            task,
            candidates,
            top_k=top_k,
            provider=provider_name or "",
            model="gpt-5.5",
            reasoning_effort=getattr(config, "openai_reasoning_effort", "low"),
        )

    by_id = {candidate.evidence_id: candidate for candidate in candidates}
    ordered: list[EvidenceChunk] = []
    seen: set[str] = set()
    for evidence_id in ordered_ids:
        if evidence_id in by_id and evidence_id not in seen:
            ordered.append(by_id[evidence_id])
            seen.add(evidence_id)
    ordered.extend(candidate for candidate in candidates if candidate.evidence_id not in seen)
    return ordered


def _fixture_embedding(text: str, dimensions: int = 128) -> list[float]:
    """Deterministic test embedding; not used by production defaults."""
    vector = [0.0] * dimensions
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        sign = -1.0 if digest[2] % 2 else 1.0
        vector[index] += sign
    return vector
