from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.api import IndexQueryRequest, _index_query_payload
from pipeline.cli import _query_index
from pipeline.config import ProviderConfig
from pipeline.retrieval import build_index, save_index
from pipeline.retrieval import engine
from pipeline.schemas import PageText, ProcessedDocument, SourceDocument


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


def _write_lexical_index(path: Path) -> Path:
    config = ProviderConfig(retrieval_provider="openai", retrieval_mode="lexical")
    index = build_index(
        [
            _doc("alpha", "Termination notice and cure period appear in the agreement."),
            _doc("bravo", "Governing law is Delaware."),
        ],
        provider="openai",
        config=config,
        include_field_chunks=False,
    )
    save_index(index, path)
    return path


def _set_lexical_query_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_RETRIEVAL_PROVIDER", "openai")
    monkeypatch.setenv("PIPELINE_RETRIEVAL_MODE", "lexical")
    monkeypatch.delenv("PIPELINE_RERANK_PROVIDER", raising=False)

    def fail_dense_embedding(*args: object, **kwargs: object) -> list[list[float]]:
        raise AssertionError("dense embedding path should not run for lexical index queries")

    monkeypatch.setattr(engine, "embed_with_openai", fail_dense_embedding)


def test_api_index_query_honors_lexical_mode_for_lexical_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _set_lexical_query_env(monkeypatch)
    index_path = _write_lexical_index(tmp_path / "retrieval_index.json")

    payload = _index_query_payload(
        IndexQueryRequest(index_path=str(index_path), task="termination notice", top_k=1)
    )

    assert payload["results"][0]["document_id"] == "alpha"


def test_cli_index_query_honors_lexical_mode_for_lexical_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _set_lexical_query_env(monkeypatch)
    index_path = _write_lexical_index(tmp_path / "retrieval_index.json")

    payload = _query_index(index_path, "termination notice", 1)

    assert payload["results"][0]["document_id"] == "alpha"
