"""Run fingerprint reacts to max_chunks_per_document changes.

The resume-checkpoint logic in run_case keys off the digest from
_compute_run_fingerprint. If the diversity cap stopped contributing to
that digest, a `--resume` run would silently reuse a checkpoint computed
with a different cap and quietly produce stale retrieval output.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pipeline.drafting import specs
from pipeline.io import write_json
from pipeline.config import PipelineFeatures, ProviderConfig
from pipeline.orchestration.run import _compute_run_fingerprint, _detect_invalidation


def _fingerprint(
    input_dir: Path,
    features: PipelineFeatures,
    *,
    config: ProviderConfig | None = None,
):
    return _compute_run_fingerprint(
        input_dir=input_dir,
        task="first-pass case fact summary",
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


def test_changing_qdrant_index_identity_changes_run_digest(tmp_path: Path):
    features = PipelineFeatures.from_env()
    base = ProviderConfig(
        index_backend="qdrant",
        qdrant_path=str(tmp_path / "qdrant-a"),
        qdrant_collection="legal_rag",
    )
    changed = ProviderConfig(
        index_backend="qdrant",
        qdrant_path=str(tmp_path / "qdrant-b"),
        qdrant_collection="legal_rag",
    )

    digest_a = _fingerprint(tmp_path, features, config=base).digest
    digest_b = _fingerprint(tmp_path, features, config=changed).digest

    assert digest_a != digest_b, (
        "Qdrant backend identity must contribute to the run fingerprint; "
        "otherwise --resume could reuse evidence against a stale vector store."
    )


def test_changing_draft_spec_content_changes_run_digest(tmp_path: Path, monkeypatch):
    features = PipelineFeatures.from_env()
    base_digest = _fingerprint(tmp_path, features).digest
    spec = specs.resolve_draft_spec("case_fact_summary")
    changed_spec = replace(spec, version="2")
    monkeypatch.setitem(specs._SPECS, "case_fact_summary", changed_spec)

    changed_digest = _fingerprint(tmp_path, features).digest

    assert base_digest != changed_digest, (
        "the registered draft spec must contribute to the run fingerprint; "
        "otherwise --resume could reuse stale drafts after spec edits."
    )


def test_changing_draft_adapter_contract_changes_run_digest(tmp_path: Path, monkeypatch):
    features = PipelineFeatures.from_env()
    base_digest = _fingerprint(tmp_path, features).digest
    adapter = specs.resolve_draft_adapter(specs.resolve_draft_spec("case_fact_summary"))
    changed_adapter = replace(adapter, version="2")
    monkeypatch.setitem(specs._ADAPTERS, adapter.id, changed_adapter)

    changed_digest = _fingerprint(tmp_path, features).digest

    assert base_digest != changed_digest, (
        "the draft adapter contract must contribute to the run fingerprint; "
        "otherwise --resume could reuse stale drafts after adapter/schema edits."
    )


def test_resume_reuses_matching_fingerprint_only_after_completed_manifest(tmp_path: Path):
    features = PipelineFeatures.from_env()
    fingerprint = _fingerprint(tmp_path, features)
    manifest = tmp_path / "workflow_manifest.json"
    write_json(
        manifest,
        {
            "status": "completed",
            "metadata": {"run_fingerprint": fingerprint.to_jsonable()},
            "stages": [],
        },
    )

    invalidation = _detect_invalidation(
        manifest_path=manifest,
        new_fingerprint=fingerprint,
        resume=True,
        force=False,
    )

    assert invalidation.invalidate_draft is False


def test_resume_invalidates_matching_fingerprint_when_manifest_not_completed(tmp_path: Path):
    features = PipelineFeatures.from_env()
    fingerprint = _fingerprint(tmp_path, features)
    manifest = tmp_path / "workflow_manifest.json"
    write_json(
        manifest,
        {
            "status": "failed",
            "metadata": {"run_fingerprint": fingerprint.to_jsonable()},
            "stages": [{"name": "generate_draft", "status": "failed"}],
        },
    )

    invalidation = _detect_invalidation(
        manifest_path=manifest,
        new_fingerprint=fingerprint,
        resume=True,
        force=False,
    )

    assert invalidation.invalidate_process is True
    assert invalidation.invalidate_retrieve is True
    assert invalidation.invalidate_draft is True
    assert "not marked completed" in invalidation.warnings[0]
