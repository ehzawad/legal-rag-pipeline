from __future__ import annotations

import json
import urllib.error

import pytest

from pipeline.config import ConfigError, ProviderConfig
from pipeline.providers import ProviderUnavailable
from pipeline.retrieval import engine
from pipeline.retrieval.engine import _api_rerank, build_index, retrieve
from pipeline.schemas import EvidenceChunk, PageText, ProcessedDocument, SourceDocument


_PROVIDER_ENV = [
    "PIPELINE_EXTRACTION_PROVIDER",
    "PIPELINE_RETRIEVAL_PROVIDER",
    "PIPELINE_GENERATION_PROVIDER",
    "OPENAI_MODEL_DRAFT",
    "OPENAI_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_REASONING_EFFORT",
    "PIPELINE_RERANK_PROVIDER",
    "PIPELINE_RERANK_MODEL",
    "COHERE_RERANK_MODEL",
    "COHERE_API_KEY",
    "CO_API_KEY",
    "PIPELINE_RETRIEVAL_MODE",
    "PIPELINE_INDEX_BACKEND",
    "PIPELINE_HYBRID_DENSE_WEIGHT",
    "PIPELINE_HYBRID_BM25_WEIGHT",
    "RETRIEVAL_TOP_K",
    "PIPELINE_EXTRACTION_CONFIDENCE_THRESHOLD",
    "OCR_CONFIDENCE_THRESHOLD",
    "EXTRACTION_CONCURRENCY",
    "PIPELINE_PDF_MAX_PAGES",
    "PIPELINE_PDF_RENDER_DPI",
    "PIPELINE_EMBEDDING_CACHE_DIR",
]


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)


def _doc(document_id: str, text: str) -> ProcessedDocument:
    return ProcessedDocument(
        source=SourceDocument(
            path=f"/fake/{document_id}.txt",
            document_id=document_id,
            filename=f"{document_id}.txt",
            mime_type="text/plain",
            sha256=document_id,
        ),
        pages=[
            PageText(
                document_id=document_id,
                page_number=1,
                text=text,
                extraction_method="fixture",
                confidence=0.99,
            )
        ],
        fields={},
    )


def _chunk(evidence_id: str, text: str) -> EvidenceChunk:
    return EvidenceChunk(
        evidence_id=evidence_id,
        document_id=evidence_id.split(":", 1)[0],
        filename=f"{evidence_id.split(':', 1)[0]}.pdf",
        page_number=1,
        text=text,
        score=1.0,
    )


def _patch_cohere_payload(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    *,
    captured: dict[str, object] | None = None,
) -> None:
    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(request: object, *, timeout: int) -> FakeResponse:
        if captured is not None:
            captured["timeout"] = timeout
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["method"] = request.get_method()
        return FakeResponse()

    monkeypatch.setattr(engine, "_cohere_api_key", lambda: "test-cohere-key")
    monkeypatch.setattr(engine.urllib.request, "urlopen", fake_urlopen)


def test_provider_config_accepts_cohere_reranker_and_model_env(monkeypatch: pytest.MonkeyPatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("PIPELINE_RERANK_PROVIDER", "cohere")
    monkeypatch.setenv("PIPELINE_RERANK_MODEL", "rerank-v4.0-fast")

    config = ProviderConfig.from_env()

    assert config.reranker_provider == "cohere"
    assert config.cohere_rerank_model == "rerank-v4.0-fast"


def test_provider_config_prefers_pipeline_rerank_model(monkeypatch: pytest.MonkeyPatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("PIPELINE_RERANK_PROVIDER", "cohere")
    monkeypatch.setenv("COHERE_RERANK_MODEL", "rerank-v4.0-fast")
    monkeypatch.setenv("PIPELINE_RERANK_MODEL", "rerank-v4.0-fast")

    config = ProviderConfig.from_env()

    assert config.cohere_rerank_model == "rerank-v4.0-fast"


def test_provider_config_rejects_openai_reranker_env(monkeypatch: pytest.MonkeyPatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("PIPELINE_RERANK_PROVIDER", "openai")

    with pytest.raises(ConfigError, match="Cohere reranker"):
        ProviderConfig.from_env()


def test_cohere_api_key_accepts_co_api_key_alias(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("CO_API_KEY", "alias-key")

    assert engine._cohere_api_key() == "alias-key"


def test_cohere_api_key_strips_and_prefers_primary_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COHERE_API_KEY", " primary-key \n")
    monkeypatch.setenv("CO_API_KEY", "alias-key")

    assert engine._cohere_api_key() == "primary-key"


def test_cohere_api_key_falls_back_when_primary_key_is_blank(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COHERE_API_KEY", "   ")
    monkeypatch.setenv("CO_API_KEY", "alias-key")

    assert engine._cohere_api_key() == "alias-key"


def test_cohere_api_key_missing_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)

    with pytest.raises(ProviderUnavailable, match="COHERE_API_KEY"):
        engine._cohere_api_key()


def test_api_rerank_rejects_openai_provider():
    with pytest.raises(ProviderUnavailable, match="Cohere"):
        _api_rerank(
            "task",
            [_chunk("doc:p1:c0", "candidate")],
            top_k=1,
            provider="openai",
            model="gpt-5.5",
        )


def test_retrieve_with_injected_reranker_client_still_orders_candidates():
    class RecordingReranker:
        calls: list[tuple[str, list[str], int]]

        def __init__(self) -> None:
            self.calls = []

        def rerank(self, task: str, candidates: list[EvidenceChunk], *, top_k: int) -> list[str]:
            self.calls.append((task, [candidate.evidence_id for candidate in candidates], top_k))
            return [
                candidate.evidence_id
                for candidate in sorted(candidates, key=lambda candidate: candidate.document_id, reverse=True)
            ]

    documents = [
        _doc("alpha", "Termination notice and cure period appear in the supplier agreement."),
        _doc("bravo", "Termination notice must be delivered before the cure period expires."),
        _doc("charlie", "Termination for convenience includes notice and transition assistance."),
    ]
    config = ProviderConfig()
    index = build_index(documents, provider="fixture", config=config, include_field_chunks=False)
    reranker = RecordingReranker()

    results = retrieve(
        index,
        "termination notice cure period",
        top_k=3,
        provider="fixture",
        config=config,
        reranker_client=reranker,
    )

    assert [chunk.document_id for chunk in results] == ["charlie", "bravo", "alpha"]
    assert reranker.calls
    assert reranker.calls[0][0] == "termination notice cure period"
    assert reranker.calls[0][2] == 3


def test_cohere_api_response_indexes_map_to_candidate_evidence_ids(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}
    _patch_cohere_payload(
        monkeypatch,
        {
            "results": [
                {"index": 2, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.74},
            ]
        },
        captured=captured,
    )
    candidates = [
        _chunk("first:p1:c0", "First candidate text."),
        _chunk("second:p1:c0", "Second candidate text."),
        _chunk("third:p1:c0", "Third candidate text."),
    ]

    ordered_ids = engine._cohere_rerank(
        "find the most relevant candidate",
        candidates,
        top_k=2,
        model="rerank-v4.0-pro",
    )

    assert ordered_ids == ["third:p1:c0", "first:p1:c0"]
    assert captured["method"] == "POST"
    assert captured["timeout"] == engine._COHERE_RERANK_TIMEOUT_SECONDS
    assert captured["body"] == {
        "model": "rerank-v4.0-pro",
        "query": "find the most relevant candidate",
        "documents": ["First candidate text.", "Second candidate text.", "Third candidate text."],
        "top_n": 2,
    }


def test_cohere_rerank_metadata_preserves_scores(monkeypatch: pytest.MonkeyPatch):
    _patch_cohere_payload(
        monkeypatch,
        {
            "results": [
                {"index": 0, "relevance_score": 0.96},
                {"index": 1, "relevance_score": 0.42},
            ]
        },
    )
    documents = [
        _doc("alpha", "Termination notice and cure period appear in the supplier agreement."),
        _doc("bravo", "Termination for convenience includes notice and transition assistance."),
    ]
    config = ProviderConfig(reranker_provider="cohere", cohere_rerank_model="rerank-v4.0-pro")
    index = build_index(documents, provider="fixture", config=config, include_field_chunks=False)

    results = retrieve(
        index,
        "termination rights",
        top_k=2,
        provider="fixture",
        config=config,
        max_chunks_per_document=0,
    )

    assert results[0].metadata["rerank_provider"] == "cohere"
    assert results[0].metadata["rerank_model"] == "rerank-v4.0-pro"
    assert results[0].metadata["rerank_rank"] == 1
    assert results[0].metadata["rerank_score"] == 0.96
    assert "pre_rerank_rank" in results[0].metadata
    assert "pre_rerank_score" in results[0].metadata


def test_cohere_rerank_requests_full_pool_before_local_caps(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}
    _patch_cohere_payload(
        monkeypatch,
        {
            "results": [
                {"index": 3, "relevance_score": 0.99},
                {"index": 2, "relevance_score": 0.82},
                {"index": 1, "relevance_score": 0.61},
                {"index": 0, "relevance_score": 0.40},
            ]
        },
        captured=captured,
    )
    candidates = [
        _chunk("alpha:p1:c0", "Alpha notice text."),
        _chunk("bravo:p1:c0", "Bravo notice text."),
        _chunk("charlie:p1:c0", "Charlie notice text."),
        _chunk("delta:p1:c0", "Delta notice text."),
    ]
    config = ProviderConfig(reranker_provider="cohere", cohere_rerank_model="rerank-v4.0-pro")

    results = engine._rerank_if_configured(
        "notice",
        candidates,
        top_k=2,
        reranker_provider=None,
        reranker_client=None,
        config=config,
    )

    assert captured["body"]["top_n"] == len(candidates)
    assert [chunk.evidence_id for chunk in results] == [
        "delta:p1:c0",
        "charlie:p1:c0",
        "bravo:p1:c0",
        "alpha:p1:c0",
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"results": [{"index": 0}]}, "relevance_score"),
        ({"results": [{"index": 0, "relevance_score": 1.2}]}, "relevance_score"),
        ({"results": [{"index": 0, "relevance_score": float("nan")}]}, "relevance_score"),
        ({"results": [{"index": True, "relevance_score": 0.9}]}, "index"),
        ({"results": [{"index": 9, "relevance_score": 0.9}]}, "out-of-range"),
        ({"results": [{"index": 0, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.8}]}, "duplicate"),
        ({"results": ["not-an-object"]}, "non-object"),
        ({"items": []}, "results"),
    ],
)
def test_cohere_rerank_rejects_malformed_results(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    message: str,
):
    _patch_cohere_payload(monkeypatch, payload)
    candidates = [_chunk("first:p1:c0", "First candidate text.")]

    with pytest.raises(ProviderUnavailable, match=message):
        engine._cohere_rerank("query", candidates, top_k=1, model="rerank-v4.0-pro")


def test_cohere_rerank_wraps_network_errors(monkeypatch: pytest.MonkeyPatch):
    def fake_urlopen(request: object, *, timeout: int):
        raise urllib.error.URLError("dns failed")

    monkeypatch.setattr(engine, "_cohere_api_key", lambda: "test-cohere-key")
    monkeypatch.setattr(engine.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ProviderUnavailable, match="Cohere rerank request failed"):
        engine._cohere_rerank("query", [_chunk("first:p1:c0", "text")], top_k=1, model="rerank-v4.0-pro")


def test_cohere_rerank_clamps_top_n(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}
    _patch_cohere_payload(
        monkeypatch,
        {"results": [{"index": 0, "relevance_score": 0.9}]},
        captured=captured,
    )

    engine._cohere_rerank(
        "query",
        [_chunk("first:p1:c0", "First candidate text.")],
        top_k=0,
        model="rerank-v4.0-pro",
    )

    assert captured["body"]["top_n"] == 1


def test_cohere_rerank_leaves_unreturned_tail_in_original_order(monkeypatch: pytest.MonkeyPatch):
    _patch_cohere_payload(monkeypatch, {"results": [{"index": 1, "relevance_score": 0.9}]})
    candidates = [
        _chunk("alpha:p1:c0", "Termination notice and cure period appear in the supplier agreement."),
        _chunk("bravo:p1:c0", "Termination for convenience includes notice and transition assistance."),
        _chunk("charlie:p1:c0", "Governing law is Delaware."),
    ]
    config = ProviderConfig(reranker_provider="cohere", cohere_rerank_model="rerank-v4.0-pro")

    results = engine._rerank_if_configured(
        "termination rights",
        candidates,
        top_k=3,
        reranker_provider=None,
        reranker_client=None,
        config=config,
    )

    assert [chunk.evidence_id for chunk in results] == ["bravo:p1:c0", "alpha:p1:c0", "charlie:p1:c0"]
    assert results[0].metadata["rerank_rank"] == 1
    assert all("rerank_rank" not in chunk.metadata for chunk in results[1:])
