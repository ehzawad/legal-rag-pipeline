"""max_chunks_per_document propagates through config, API model, and CLI.

The diversity cap is exposed on three surfaces: PipelineFeatures (env +
explicit overrides), the FastAPI request schema, and the CLI argument
parser. A regression on any of them would silently disable the cap for
that entry point, so each surface gets one focused test.
"""

from __future__ import annotations

import pytest

from pipeline.api import IndexSettingsRequest, _index_settings_payload, PipelineFeaturesRequest
from pipeline.cli import _index_settings, build_parser
from pipeline.config import ConfigError, PipelineFeatures, ProviderConfig


def test_pipeline_features_from_mapping_accepts_int_value():
    features = PipelineFeatures.from_mapping({"max_chunks_per_document": 5})
    assert features.max_chunks_per_document == 5


def test_pipeline_features_from_mapping_rejects_non_int_value():
    with pytest.raises(ConfigError):
        PipelineFeatures.from_mapping({"max_chunks_per_document": "not-an-int"})


def test_pipeline_features_request_accepts_max_chunks_per_document():
    model = PipelineFeaturesRequest(**{"max_chunks_per_document": 4})
    assert model.max_chunks_per_document == 4


def test_cli_parser_registers_max_chunks_per_document_argument():
    parser = build_parser()
    namespace = parser.parse_args(
        [
            "run",
            "--input",
            "/tmp/in",
            "--output",
            "/tmp/out",
            "--max-chunks-per-document",
            "7",
        ]
    )
    assert namespace.max_chunks_per_document == 7


def test_api_index_settings_exposes_cohere_reranker(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("PIPELINE_RERANK_PROVIDER", "cohere")
    monkeypatch.setenv("PIPELINE_RERANK_MODEL", "rerank-v4.0-fast")

    payload = _index_settings_payload(
        IndexSettingsRequest(
            output_dir=str(tmp_path / "out"),
            state_dir=str(tmp_path / "state"),
        )
    )

    assert payload["settings"]["reranker_provider"] == "cohere"
    assert payload["settings"]["reranker_model"] == "rerank-v4.0-fast"


def test_cli_index_settings_exposes_cohere_reranker(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("PIPELINE_RERANK_PROVIDER", "cohere")
    monkeypatch.setenv("COHERE_RERANK_MODEL", "rerank-v4.0-pro")
    monkeypatch.delenv("PIPELINE_RERANK_MODEL", raising=False)

    payload = _index_settings(tmp_path / "out", tmp_path / "state")

    assert payload["settings"]["reranker_provider"] == "cohere"
    assert payload["settings"]["reranker_model"] == "rerank-v4.0-pro"


def test_provider_config_default_pdf_page_limit_is_100(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PIPELINE_PDF_MAX_PAGES", raising=False)

    config = ProviderConfig.from_env()

    assert config.pdf_max_pages == 100


def test_provider_config_rejects_non_positive_pdf_page_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PIPELINE_PDF_MAX_PAGES", "0")

    with pytest.raises(ConfigError, match="PIPELINE_PDF_MAX_PAGES"):
        ProviderConfig.from_env()
