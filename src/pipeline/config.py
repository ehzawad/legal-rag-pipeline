from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping


_TEST_PROVIDER_NAMES = {"", "local", "mock", "fixture"}
_OPENAI_RETRIEVAL_PROVIDER_NAMES = {"openai", "openai-cached"}
_RERANK_OFF_NAMES = {"", "none", "off", "false", "0"}
_COHERE_RERANK_PROVIDER_NAMES = {"cohere"}
_VALID_REASONING_EFFORTS = {"low", "medium", "high"}
_VALID_RETRIEVAL_MODES = {"dense", "hybrid", "lexical"}
_VALID_INDEX_BACKENDS = {"memory", "qdrant"}


class ConfigError(ValueError):
    pass


def resolve_provider_name(
    value: str | Any | None,
    *,
    fallback: str,
    reject_test_names: bool = False,
) -> str:
    if value is not None and not isinstance(value, str):
        name = getattr(value, "name", value.__class__.__name__)
        resolved = str(name or "").lower().strip()
    elif isinstance(value, str) and value.strip():
        resolved = value.lower().strip()
    else:
        resolved = (fallback or "").lower().strip()
    if reject_test_names and resolved in _TEST_PROVIDER_NAMES:
        return "openai"
    return resolved


def is_cached_retrieval_provider(value: str | None) -> bool:
    return (value or "").strip().lower() == "openai-cached"


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_secret(name: str, file_name: str, default: str = "") -> str:
    value = _env(name, "")
    if value:
        return value
    path = _env(file_name, "")
    if not path:
        return default
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError as exc:
        raise ConfigError(f"{file_name} is set but could not be read: {path}") from exc


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").casefold()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean value, got {raw!r}")


def _parse_bool_value(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        raw = value.strip().casefold()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    raise ConfigError(f"{name} must be a boolean value, got {value!r}")


def _validate_reasoning_effort(value: str) -> str:
    resolved = (value or "").lower().strip()
    if resolved not in _VALID_REASONING_EFFORTS:
        raise ConfigError(
            f"OPENAI_REASONING_EFFORT must be one of {sorted(_VALID_REASONING_EFFORTS)}, got {value!r}"
        )
    return resolved


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    extraction_provider: str = "openai"
    retrieval_provider: str = "openai"
    generation_provider: str = "openai"
    openai_model: str = "gpt-5.5"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_reasoning_effort: str = "low"
    reranker_provider: str = ""
    cohere_rerank_model: str = "rerank-v4.0-pro"
    retrieval_mode: str = "hybrid"
    index_backend: str = "memory"
    hybrid_dense_weight: float = 0.65
    hybrid_bm25_weight: float = 0.35
    retrieval_top_k: int = 10
    extraction_confidence_threshold: float = 0.55
    extraction_concurrency: int = 4
    pdf_max_pages: int = 100
    pdf_render_dpi: int = 180
    embedding_cache_dir: str = ""
    qdrant_url: str = ""
    qdrant_path: str = "state/qdrant"
    qdrant_collection: str = ""
    qdrant_api_key: str = ""
    qdrant_prefer_grpc: bool = False

    @classmethod
    def from_env(cls) -> "ProviderConfig":
        config = cls(
            extraction_provider=_env("PIPELINE_EXTRACTION_PROVIDER", "openai"),
            retrieval_provider=_env("PIPELINE_RETRIEVAL_PROVIDER", "openai"),
            generation_provider=_env("PIPELINE_GENERATION_PROVIDER", "openai"),
            openai_model=_env("OPENAI_MODEL_DRAFT", _env("OPENAI_MODEL", "gpt-5.5")),
            openai_embedding_model=_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            openai_reasoning_effort=_validate_reasoning_effort(_env("OPENAI_REASONING_EFFORT", "low")),
            reranker_provider=_env("PIPELINE_RERANK_PROVIDER", ""),
            cohere_rerank_model=_env("PIPELINE_RERANK_MODEL", _env("COHERE_RERANK_MODEL", "rerank-v4.0-pro")),
            retrieval_mode=_env("PIPELINE_RETRIEVAL_MODE", "hybrid").lower(),
            index_backend=_env("PIPELINE_INDEX_BACKEND", "memory").lower(),
            hybrid_dense_weight=_env_float("PIPELINE_HYBRID_DENSE_WEIGHT", 0.65),
            hybrid_bm25_weight=_env_float("PIPELINE_HYBRID_BM25_WEIGHT", 0.35),
            retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 10),
            extraction_confidence_threshold=_env_float(
                "PIPELINE_EXTRACTION_CONFIDENCE_THRESHOLD",
                _env("OCR_CONFIDENCE_THRESHOLD", "0.55"),
            ),
            extraction_concurrency=_env_int("EXTRACTION_CONCURRENCY", 4),
            pdf_max_pages=_env_int("PIPELINE_PDF_MAX_PAGES", 100),
            pdf_render_dpi=_env_int("PIPELINE_PDF_RENDER_DPI", 180),
            embedding_cache_dir=_env("PIPELINE_EMBEDDING_CACHE_DIR", ""),
            qdrant_url=_env("QDRANT_URL", ""),
            qdrant_path=_env("QDRANT_PATH", "state/qdrant"),
            qdrant_collection=_env("QDRANT_COLLECTION", ""),
            qdrant_api_key=_env_secret("QDRANT_API_KEY", "QDRANT_API_KEY_FILE", ""),
            qdrant_prefer_grpc=_env_bool("QDRANT_PREFER_GRPC", False),
        )
        config.validate_runtime()
        return config

    def validate_runtime(self) -> None:
        _reject_non_openai("PIPELINE_EXTRACTION_PROVIDER", self.extraction_provider)
        _reject_retrieval_provider("PIPELINE_RETRIEVAL_PROVIDER", self.retrieval_provider)
        _reject_non_openai("PIPELINE_GENERATION_PROVIDER", self.generation_provider)
        _reject_reranker_provider("PIPELINE_RERANK_PROVIDER", self.reranker_provider)
        if self.reranker_provider.strip().lower() in _COHERE_RERANK_PROVIDER_NAMES and not self.cohere_rerank_model.strip():
            raise ConfigError("COHERE_RERANK_MODEL / PIPELINE_RERANK_MODEL must not be empty when Cohere rerank is enabled")
        _validate_choice("PIPELINE_RETRIEVAL_MODE", self.retrieval_mode, _VALID_RETRIEVAL_MODES)
        _validate_choice("PIPELINE_INDEX_BACKEND", self.index_backend, _VALID_INDEX_BACKENDS)
        _validate_non_negative("PIPELINE_HYBRID_DENSE_WEIGHT", self.hybrid_dense_weight)
        _validate_non_negative("PIPELINE_HYBRID_BM25_WEIGHT", self.hybrid_bm25_weight)
        if self.pdf_max_pages <= 0:
            raise ConfigError("PIPELINE_PDF_MAX_PAGES must be a positive integer")
        if self.index_backend.strip().lower() == "qdrant" and not (self.qdrant_url.strip() or self.qdrant_path.strip()):
            raise ConfigError("QDRANT_URL or QDRANT_PATH must be set when PIPELINE_INDEX_BACKEND=qdrant")
        if self.index_backend.strip().lower() == "qdrant" and self.qdrant_path.strip() == ":memory:":
            raise ConfigError("QDRANT_PATH=:memory: is not supported for pipeline retrieval; use a persistent local path")


@dataclass(frozen=True, slots=True)
class PipelineFeatures:
    """Runtime feature switches for pluggable pipeline components.

    Disabling a core stage (`process_documents`, `retrieve_evidence`, or
    `generate_draft`) means the pipeline reuses that stage's checkpoint
    artifact. If the artifact is missing, the run fails loudly instead of
    silently producing a partial result.
    """

    process_documents: bool = True
    retrieve_evidence: bool = True
    generate_draft: bool = True
    learning_guidance: bool = True
    operator_profile_guidance: bool = True
    knowledge_layer_guidance: bool = True
    exemplar_guidance: bool = True
    retrieval_feedback: bool = True
    field_chunks: bool = True
    evidence_pack: bool = True
    claim_first_drafting: bool = True
    claim_support_check: bool = True
    claim_entailment_judge: bool = False
    playbook_risk: bool = True
    max_field_chunks: int = 1
    max_chunks_per_document: int = 3
    field_chunk_score_penalty: float = 0.06

    @classmethod
    def from_env(cls) -> "PipelineFeatures":
        config = cls(
            process_documents=_env_bool("PIPELINE_PROCESSING", True),
            retrieve_evidence=_env_bool("PIPELINE_RETRIEVAL", True),
            generate_draft=_env_bool("PIPELINE_DRAFTING", True),
            learning_guidance=_env_bool("PIPELINE_LEARNING_GUIDANCE", True),
            operator_profile_guidance=_env_bool("PIPELINE_OPERATOR_PROFILE_GUIDANCE", True),
            knowledge_layer_guidance=_env_bool("PIPELINE_KNOWLEDGE_LAYER_GUIDANCE", True),
            exemplar_guidance=_env_bool("PIPELINE_EXEMPLAR_GUIDANCE", True),
            retrieval_feedback=_env_bool("PIPELINE_RETRIEVAL_FEEDBACK", True),
            field_chunks=_env_bool("PIPELINE_FIELD_CHUNKS", True),
            evidence_pack=_env_bool("PIPELINE_EVIDENCE_PACK", True),
            claim_first_drafting=_env_bool("PIPELINE_CLAIM_FIRST_DRAFTING", True),
            claim_support_check=_env_bool("PIPELINE_CLAIM_SUPPORT_CHECK", True),
            claim_entailment_judge=_env_bool("PIPELINE_CLAIM_ENTAILMENT_JUDGE", False),
            playbook_risk=_env_bool("PIPELINE_PLAYBOOK_RISK", True),
            max_field_chunks=_env_int("PIPELINE_MAX_FIELD_CHUNKS", 1),
            max_chunks_per_document=_env_int("PIPELINE_MAX_CHUNKS_PER_DOCUMENT", 3),
            field_chunk_score_penalty=_env_float("PIPELINE_FIELD_CHUNK_SCORE_PENALTY", 0.06),
        )
        config.validate_runtime()
        return config

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
        *,
        base: "PipelineFeatures | None" = None,
    ) -> "PipelineFeatures":
        config = base or cls.from_env()
        if not values:
            return config
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ConfigError(f"Unknown pipeline feature option(s): {', '.join(unknown)}")
        updates: dict[str, Any] = {}
        for key, value in values.items():
            if value is None:
                continue
            updates[key] = _coerce_feature_value(key, value)
        updated = replace(config, **updates)
        updated.validate_runtime()
        return updated

    def validate_runtime(self) -> None:
        if self.max_field_chunks < 0:
            raise ConfigError("PIPELINE_MAX_FIELD_CHUNKS must be non-negative")
        if self.max_chunks_per_document < 0:
            raise ConfigError("PIPELINE_MAX_CHUNKS_PER_DOCUMENT must be non-negative")
        if self.field_chunk_score_penalty < 0:
            raise ConfigError("PIPELINE_FIELD_CHUNK_SCORE_PENALTY must be non-negative")

    def to_jsonable(self) -> dict[str, bool | int | float]:
        return asdict(self)


def _coerce_feature_value(name: str, value: Any) -> bool | int | float:
    bool_fields = {
        "process_documents",
        "retrieve_evidence",
        "generate_draft",
        "learning_guidance",
        "operator_profile_guidance",
        "knowledge_layer_guidance",
        "exemplar_guidance",
        "retrieval_feedback",
        "field_chunks",
        "evidence_pack",
        "claim_first_drafting",
        "claim_support_check",
        "claim_entailment_judge",
        "playbook_risk",
    }
    if name in bool_fields:
        return _parse_bool_value(name, value)
    if name in {"max_field_chunks", "max_chunks_per_document"}:
        if isinstance(value, bool):
            raise ConfigError(f"{name} must be an integer, got {value!r}")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{name} must be an integer, got {value!r}") from exc
    if name == "field_chunk_score_penalty":
        if isinstance(value, bool):
            raise ConfigError(f"{name} must be a number, got {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{name} must be a number, got {value!r}") from exc
    raise ConfigError(f"Unknown pipeline feature option: {name}")


def _reject_non_openai(name: str, value: str) -> None:
    resolved = (value or "").strip().lower()
    if resolved and resolved in _TEST_PROVIDER_NAMES:
        raise ConfigError(f"{name}={value!r} is test-only and cannot be used by runtime configuration")
    if resolved and resolved != "openai":
        raise ConfigError(
            f"{name}={value!r} is no longer supported; this build only ships the openai provider"
        )


def _reject_retrieval_provider(name: str, value: str) -> None:
    resolved = (value or "").strip().lower()
    if resolved and resolved in _TEST_PROVIDER_NAMES:
        raise ConfigError(f"{name}={value!r} is test-only and cannot be used by runtime configuration")
    if resolved and resolved not in _OPENAI_RETRIEVAL_PROVIDER_NAMES:
        raise ConfigError(
            f"{name}={value!r} is no longer supported; this build only ships openai retrieval providers"
        )


def _reject_reranker_provider(name: str, value: str) -> None:
    resolved = (value or "").strip().lower()
    if resolved in _RERANK_OFF_NAMES:
        return
    if resolved and resolved in _TEST_PROVIDER_NAMES:
        raise ConfigError(f"{name}={value!r} is test-only and cannot be used by runtime configuration")
    if resolved not in _COHERE_RERANK_PROVIDER_NAMES:
        raise ConfigError(
            f"{name}={value!r} is no longer supported; this build only ships the Cohere reranker"
        )


def _validate_choice(name: str, value: str, choices: set[str]) -> None:
    resolved = (value or "").strip().lower()
    if resolved not in choices:
        raise ConfigError(f"{name} must be one of {sorted(choices)}, got {value!r}")


def _validate_non_negative(name: str, value: float) -> None:
    if float(value) < 0:
        raise ConfigError(f"{name} must be non-negative")
