"""Run fingerprint reacts to max_chunks_per_document changes.

The resume-checkpoint logic in run_case keys off the digest from
_compute_run_fingerprint. If the diversity cap stopped contributing to
that digest, a `--resume` run would silently reuse a checkpoint computed
with a different cap and quietly produce stale retrieval output.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pipeline.config import PipelineFeatures, ProviderConfig
from pipeline.orchestration.run import _compute_run_fingerprint


def _fingerprint(
    input_dir: Path,
    features: PipelineFeatures,
    *,
    config: ProviderConfig | None = None,
):
    return _compute_run_fingerprint(
        input_dir=input_dir,
        task="first-pass internal memo",
        profile_path=input_dir / "missing-profile.json",
        config=config or ProviderConfig(),
        features=features,
    )


def test_changing_max_chunks_per_document_changes_run_digest(tmp_path: Path):
    base = PipelineFeatures.from_env()
    a = replace(base, max_chunks_per_document=3)
    b = replace(base, max_chunks_per_document=5)

    digest_a = _fingerprint(tmp_path, a).digest
    digest_b = _fingerprint(tmp_path, b).digest

    assert digest_a != digest_b, (
        "max_chunks_per_document must contribute to the run fingerprint; "
        "otherwise --resume would reuse stale retrieval checkpoints."
    )


def test_identical_features_produce_identical_digest(tmp_path: Path):
    features = PipelineFeatures.from_env()
    digest_a = _fingerprint(tmp_path, features).digest
    digest_b = _fingerprint(tmp_path, features).digest
    assert digest_a == digest_b


def test_changing_cohere_rerank_model_changes_run_digest(tmp_path: Path):
    features = PipelineFeatures.from_env()
    base = ProviderConfig(reranker_provider="cohere", cohere_rerank_model="rerank-v4.0-pro")
    changed = ProviderConfig(reranker_provider="cohere", cohere_rerank_model="rerank-v4.0-fast")

    digest_a = _fingerprint(tmp_path, features, config=base).digest
    digest_b = _fingerprint(tmp_path, features, config=changed).digest

    assert digest_a != digest_b, (
        "the Cohere rerank model must contribute to the run fingerprint; "
        "otherwise --resume could reuse stale reranked evidence."
    )


def test_enabling_cohere_reranker_changes_run_digest(tmp_path: Path):
    features = PipelineFeatures.from_env()
    off = ProviderConfig(reranker_provider="")
    on = ProviderConfig(reranker_provider="cohere", cohere_rerank_model="rerank-v4.0-pro")

    digest_a = _fingerprint(tmp_path, features, config=off).digest
    digest_b = _fingerprint(tmp_path, features, config=on).digest

    assert digest_a != digest_b, (
        "the reranker provider must contribute to the run fingerprint; "
        "otherwise --resume could reuse evidence ranked without Cohere."
    )
