"""max_chunks_per_document propagates through config, API model, and CLI.

The diversity cap is exposed on three surfaces: PipelineFeatures (env +
explicit overrides), the FastAPI request schema, and the CLI argument
parser. A regression on any of them would silently disable the cap for
that entry point, so each surface gets one focused test.
"""

from __future__ import annotations

import pytest

from pipeline.api import PipelineFeaturesRequest
from pipeline.cli import build_parser
from pipeline.config import ConfigError, PipelineFeatures


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
