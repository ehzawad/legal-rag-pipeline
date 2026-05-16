"""FastAPI HTTP surface for the grounded legal-document pipeline.

This module is a thin routing layer. Pipeline business logic lives in
:mod:`pipeline.orchestration.run`, :mod:`pipeline.learning`,
:mod:`pipeline.evaluation`, and :mod:`pipeline.playbooks`; the handlers
here only translate requests/responses and map known exceptions to HTTP
status codes.

The API also serves the React operator console from
``pipeline/ui/static/dist`` under ``/ui``. The console source lives in
``frontend/`` and is rebuilt with ``npm run build``.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline.config import ConfigError, PipelineFeatures, ProviderConfig, is_cached_retrieval_provider
from pipeline.drafting.memo import _operator_claim_key, _operator_language_text, visible_review_warnings
from pipeline.drafting.rendering import render_draft_markdown
from pipeline.io import read_json
from pipeline.orchestration.workflow import WorkflowStageError
from pipeline.orchestration.artifacts import draft_from_json
from pipeline.providers import ProviderUnavailable
from pipeline.schemas import to_jsonable


app = FastAPI(
    title="Pipeline API",
    description="HTTP surface over the grounded legal-document pipeline.",
    version="0.2.0",
)


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "PIPELINE_API_CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    ).strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


_origins = _cors_origins()
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


_UI_DIR = Path(__file__).resolve().parent / "ui" / "static"
_UI_DIST_DIR = _UI_DIR / "dist"
_UI_DIST_ASSETS_DIR = _UI_DIST_DIR / "assets"
_FAVICON_PATH = _UI_DIR / "favicon.ico"
if _UI_DIST_ASSETS_DIR.exists():
    app.mount(
        "/ui/assets",
        StaticFiles(directory=_UI_DIST_ASSETS_DIR),
        name="ui-assets",
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    if _FAVICON_PATH.exists():
        return FileResponse(_FAVICON_PATH, media_type="image/x-icon")
    return Response(status_code=204)


# ---------- request models ---------------------------------------------------


class PipelineFeaturesRequest(BaseModel):
    process_documents: bool | None = None
    retrieve_evidence: bool | None = None
    generate_draft: bool | None = None
    learning_guidance: bool | None = None
    operator_profile_guidance: bool | None = None
    knowledge_layer_guidance: bool | None = None
    exemplar_guidance: bool | None = None
    retrieval_feedback: bool | None = None
    field_chunks: bool | None = None
    evidence_pack: bool | None = None
    claim_first_drafting: bool | None = None
    claim_support_check: bool | None = None
    claim_entailment_judge: bool | None = None
    playbook_risk: bool | None = None
    max_field_chunks: int | None = None
    max_chunks_per_document: int | None = None
    field_chunk_score_penalty: float | None = None


class RunRequest(BaseModel):
    input_dir: str = Field(..., description="Directory of source documents.")
    output_dir: str = Field(..., description="Directory for processed artifacts.")
    case_id: str = Field(default="sample-case")
    task: str = Field(default="first-pass internal memo")
    profile_path: str | None = Field(default=None)
    state_dir: str | None = Field(default=None)
    resume: bool = Field(default=False)
    force: bool = Field(default=False)
    features: PipelineFeaturesRequest | None = Field(default=None)
    playbook_path: str | None = Field(default=None)


class CorpusSettingsRequest(BaseModel):
    input_dir: str = Field(default="datasets")
    output_dir: str = Field(default="outputs/corpus")
    features: PipelineFeaturesRequest | None = Field(default=None)


class CorpusBuildRequest(BaseModel):
    input_dir: str
    corpus_dir: str
    processed_path: str | None = None


class IndexSettingsRequest(BaseModel):
    output_dir: str = Field(default="outputs/index")
    state_dir: str = Field(default="state")
    features: PipelineFeaturesRequest | None = Field(default=None)


class IndexBuildRequest(BaseModel):
    corpus_dir: str
    index_path: str


class IndexQueryRequest(BaseModel):
    index_path: str
    task: str
    top_k: int = 5


class EditMemorySettingsRequest(BaseModel):
    state_dir: str = Field(default="state")
    profile_path: str | None = Field(default=None)
    category: str = Field(default="")
    features: PipelineFeaturesRequest | None = Field(default=None)


class EditMemoryQueryRequest(BaseModel):
    state_dir: str = Field(default="state")
    task: str = ""
    category: str = ""
    text: str = ""
    limit: int = 5


class RunResponse(BaseModel):
    case_id: str
    output_dir: str
    artifacts: dict[str, str]
    run_fingerprint: str


class LearnRequest(BaseModel):
    draft_path: str
    edited_path: str
    profile_path: str | None = None
    state_dir: str | None = None
    category: str | None = None
    note: str = ""
    intent: str = Field(
        default="preference",
        pattern="^(auto|evidence_dispute|field_correction|knowledge_rule|preference|rejection)$",
    )
    actor_type: str = Field(default="human", pattern="^(human|agent)$")
    actor_id: str = Field(default="default-operator")
    event_log_path: str | None = None
    suggestions_path: str | None = None


class ABEvalRequest(BaseModel):
    case_dir: str
    profile_path: str
    task: str | None = None
    edited_reference_path: str | None = None
    draft_output_dir: str | None = None
    state_dir: str | None = None


class EvalSuiteRequest(BaseModel):
    run_dir: str
    manifest_path: str | None = None
    gold_dir: str | None = None
    eval_points_path: str | None = None
    simulate_edits: bool = False
    simulated_edits_output_dir: str | None = None


class RiskRequest(BaseModel):
    run_dir: str
    playbook_path: str | None = None
    output_path: str | None = None
    markdown_path: str | None = None


class HarnessRunRequest(BaseModel):
    manifest_path: str
    output_dir: str


class PublicDataRequest(BaseModel):
    manifest_path: str
    dataset_dir: str
    force: bool = False
    dry_run: bool = False
    include_optional: bool = False


class ExportDPORequest(BaseModel):
    edits_log: str
    output: str
    include_non_preference: bool = False


class EditSubmitRequest(BaseModel):
    edited_text: str = Field(..., description="Operator-edited markdown body for the draft.")
    note: str = Field(default="", description="Operator note explaining the edit.")
    category: str | None = Field(default=None)
    intent: str = Field(
        default="preference",
        pattern="^(auto|evidence_dispute|field_correction|knowledge_rule|preference|rejection)$",
    )
    actor_type: str = Field(default="human", pattern="^(human|agent)$")
    actor_id: str = Field(default="default-operator")
    state_dir: str | None = Field(
        default=None,
        description="State dir (defaults to ./state). Path is checked against PIPELINE_API_ALLOWED_ROOTS.",
    )
    output_dir: str | None = Field(
        default=None,
        description="Run output directory (defaults to outputs/<case_id>).",
    )


class HealthResponse(BaseModel):
    status: str


# ---------- artifact registry -------------------------------------------------


_ARTIFACT_FILES: dict[str, tuple[str, str]] = {
    # key → (relative path inside run dir, media_type). Keys are stable
    # identifiers, never filenames or paths supplied by the client.
    "draft_md": ("draft.md", "text/markdown"),
    "draft_json": ("draft.json", "application/json"),
    "edited_md": ("edited.md", "text/markdown"),
    "case_run": ("case_run.json", "application/json"),
    "processed_documents": ("processed_documents.json", "application/json"),
    "retrieved_evidence": ("retrieved_evidence.json", "application/json"),
    "evidence_pack": ("evidence_pack.json", "application/json"),
    "case_fact_summary": ("case_fact_summary.json", "application/json"),
    "grounding_report": ("grounding_report.json", "application/json"),
    "risk_report_json": ("risk_report.json", "application/json"),
    "risk_report_md": ("risk_report.md", "text/markdown"),
    "evaluation_json": ("evaluation.json", "application/json"),
    "evaluation_md": ("evaluation.md", "text/markdown"),
    "ab_eval_md": ("ab_eval.md", "text/markdown"),
    "audit_log": ("audit.jsonl", "application/x-ndjson"),
    "edits_log": ("edits.jsonl", "application/x-ndjson"),
    "workflow_manifest": ("workflow_manifest.json", "application/json"),
    "learned_guidance": ("learned_guidance.json", "application/json"),
    "corpus_manifest": ("corpus/manifest.json", "application/json"),
    "retrieval_index": ("index/retrieval_index.json", "application/json"),
}

_STATE_FILES: dict[str, tuple[str, str]] = {
    "operator_profile": ("operator_profile.json", "application/json"),
    "knowledge_layer": ("knowledge_layer.json", "application/json"),
    "retrieval_feedback": ("retrieval_feedback.json", "application/json"),
    "edit_memory": ("edit_memory.json", "application/json"),
    "edits": ("edits.jsonl", "application/x-ndjson"),
    "edit_classifications": ("edit_classifications.jsonl", "application/x-ndjson"),
    "learn_suggestions": ("learn_suggestions.json", "application/json"),
    "dpo_pairs": ("dpo_pairs.jsonl", "application/x-ndjson"),
    "dpo_pairs_candidates": ("dpo_pairs.candidates.jsonl", "application/x-ndjson"),
}


_CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.]{0,127}$")

# Mirrors pipeline.ingestion.documents._SUPPORTED_DOC_EXTENSIONS so the UI
# can't sneak a file type into the run that the ingestor would just skip.
_UPLOAD_ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt"}
_UPLOAD_MAX_BYTES = 30 * 1024 * 1024  # 30 MB / file — matches our PDF cap shape


def _validate_case_id(case_id: str) -> str:
    """Reject path fragments, leading dots, traversal, and oversized ids.

    ``case_id`` is the directory name on disk; allowing arbitrary characters
    would let URL parsing leak into filesystem reads. The pattern keeps the
    set close to ``[case_id]`` style identifiers actually written by
    ``run_case``.
    """

    if not _CASE_ID_PATTERN.match(case_id):
        raise _http_error(400, f"invalid case_id: {case_id!r}")
    return case_id


# ---------- error mapping ----------------------------------------------------


def _http_error(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail=message)


def _api_allowed_roots() -> list[Path]:
    raw = os.getenv("PIPELINE_API_ALLOWED_ROOTS", "").strip()
    if raw:
        candidates = [Path(item).expanduser() for item in raw.split(os.pathsep) if item.strip()]
    else:
        candidates = [Path.cwd(), Path(tempfile.gettempdir())]
    return [path.resolve(strict=False) for path in candidates]


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _checked_path(value: str | Path, label: str, *, must_exist: bool = False) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    if not any(_path_is_relative_to(path, root) for root in _api_allowed_roots()):
        allowed = ", ".join(str(root) for root in _api_allowed_roots())
        raise _http_error(403, f"{label} is outside PIPELINE_API_ALLOWED_ROOTS: {path} (allowed: {allowed})")
    if must_exist and not path.exists():
        raise _http_error(404, f"{label} not found: {path}")
    return path


def _map_pipeline_call(func, *args: Any, **kwargs: Any) -> Any:
    """Invoke ``func`` and translate known exceptions into HTTPExceptions."""

    try:
        return func(*args, **kwargs)
    except FileNotFoundError as exc:
        raise _http_error(404, str(exc)) from exc
    except ConfigError as exc:
        raise _http_error(400, str(exc)) from exc
    except ValueError as exc:
        raise _http_error(400, str(exc)) from exc
    except ProviderUnavailable as exc:
        raise _http_error(502, str(exc)) from exc
    except WorkflowStageError as exc:
        cause = exc.cause
        if isinstance(cause, FileNotFoundError):
            raise _http_error(404, str(cause)) from exc
        if isinstance(cause, ConfigError):
            raise _http_error(400, str(cause)) from exc
        if isinstance(cause, ValueError):
            raise _http_error(400, str(cause)) from exc
        if isinstance(cause, ProviderUnavailable):
            raise _http_error(502, str(cause)) from exc
        raise _http_error(500, str(exc)) from exc


def _dataclass_to_jsonable(value: Any) -> Any:
    """Best-effort JSON-ready conversion that preserves nested dataclasses."""

    if is_dataclass(value):
        return asdict(value)
    return to_jsonable(value)


def _features_from_request(value: PipelineFeaturesRequest | None) -> PipelineFeatures:
    if value is None:
        return PipelineFeatures.from_env()
    if hasattr(value, "model_dump"):
        payload = value.model_dump(exclude_none=True)
    else:
        payload = value.dict(exclude_none=True)
    return PipelineFeatures.from_mapping(payload)


def _corpus_settings_payload(body: CorpusSettingsRequest) -> dict[str, Any]:
    config = ProviderConfig.from_env()
    features = _features_from_request(body.features)
    input_dir = _checked_path(body.input_dir, "input_dir")
    output_dir = _checked_path(body.output_dir, "output_dir")
    return {
        "surface": "corpus",
        "mode": "settings",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "artifacts": {
            "processed_documents": str(output_dir / "processed_documents.json"),
            "corpus_manifest": str(output_dir / "corpus" / "manifest.json"),
            "workflow_manifest": str(output_dir / "workflow_manifest.json"),
        },
        "settings": {
            "process_documents": features.process_documents,
            "extraction_provider": config.extraction_provider,
            "pdf_max_pages": config.pdf_max_pages,
            "pdf_render_dpi": config.pdf_render_dpi,
            "extraction_concurrency": config.extraction_concurrency,
        },
        "commands": {
            "ingest_with_run": (
                f"pipeline run --input {input_dir} --output {output_dir} --disable-retrieval --disable-drafting"
            ),
            "full_run": f"pipeline run --input {input_dir} --output {output_dir}",
        },
        "notes": [
            "Use /corpus/build to persist originals plus parsed documents.",
            "A normal /runs call also writes output/corpus/ automatically.",
        ],
    }


def _corpus_build_payload(body: CorpusBuildRequest) -> dict[str, Any]:
    from pipeline.corpus import write_corpus_store
    from pipeline.ingestion.documents import process_directory
    from pipeline.orchestration.artifacts import load_processed_documents

    config = ProviderConfig.from_env()
    input_dir = _checked_path(body.input_dir, "input_dir", must_exist=True)
    corpus_dir = _checked_path(body.corpus_dir, "corpus_dir")
    processed = (
        load_processed_documents(_checked_path(body.processed_path, "processed_path", must_exist=True))
        if body.processed_path
        else process_directory(input_dir, provider=config.extraction_provider, config=config)
    )
    manifest = write_corpus_store(input_dir=input_dir, processed_documents=processed, corpus_dir=corpus_dir)
    return {
        "surface": "corpus",
        "mode": "built",
        "input_dir": str(input_dir),
        "corpus_dir": str(corpus_dir),
        "manifest": manifest.to_jsonable(),
    }


def _index_settings_payload(body: IndexSettingsRequest) -> dict[str, Any]:
    config = ProviderConfig.from_env()
    features = _features_from_request(body.features)
    output_dir = _checked_path(body.output_dir, "output_dir")
    state_dir = _checked_path(body.state_dir, "state_dir")
    effective_cache_dir = ""
    if is_cached_retrieval_provider(config.retrieval_provider):
        effective_cache_dir = config.embedding_cache_dir or str(output_dir / "embedding_cache")
    return {
        "surface": "index",
        "mode": "settings",
        "output_dir": str(output_dir),
        "artifacts": {
            "retrieved_evidence": str(output_dir / "retrieved_evidence.json"),
            "evidence_pack": str(output_dir / "evidence_pack.json"),
            "retrieval_index": str(output_dir / "index" / "retrieval_index.json"),
            "embedding_cache_dir": effective_cache_dir,
            "workflow_manifest": str(output_dir / "workflow_manifest.json"),
        },
        "settings": {
            "retrieve_evidence": features.retrieve_evidence,
            "retrieval_provider": config.retrieval_provider,
            "retrieval_mode": config.retrieval_mode,
            "index_backend": config.index_backend,
            "embedding_model": config.openai_embedding_model,
            "hybrid_dense_weight": config.hybrid_dense_weight,
            "hybrid_bm25_weight": config.hybrid_bm25_weight,
            "retrieval_top_k": config.retrieval_top_k,
            "reranker_provider": config.reranker_provider,
            "reranker_model": config.cohere_rerank_model,
            "field_chunks": features.field_chunks,
            "max_field_chunks": features.max_field_chunks,
            "max_chunks_per_document": features.max_chunks_per_document,
            "field_chunk_score_penalty": features.field_chunk_score_penalty,
            "retrieval_feedback": features.retrieval_feedback,
            "retrieval_feedback_path": str(state_dir / "retrieval_feedback.json"),
        },
        "commands": {
            "retrieve_with_run": (
                f"pipeline run --input <corpus-dir> --output {output_dir} --disable-processing --disable-drafting"
            ),
            "full_run": f"pipeline run --input <corpus-dir> --output {output_dir}",
        },
        "notes": [
            "Use /index/build to persist a queryable BM25+dense index.",
            "A normal /runs call also writes output/index/retrieval_index.json automatically.",
        ],
    }


def _index_build_payload(body: IndexBuildRequest) -> dict[str, Any]:
    from pipeline.corpus import load_corpus_documents
    from pipeline.retrieval import build_index, save_index

    config = ProviderConfig.from_env()
    corpus_dir = _checked_path(body.corpus_dir, "corpus_dir", must_exist=True)
    documents = load_corpus_documents(corpus_dir)
    index = build_index(documents, provider=config.retrieval_provider, config=config)
    index_path = _checked_path(body.index_path, "index_path")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    save_index(index, index_path)
    return {
        "surface": "index",
        "mode": "built",
        "corpus_dir": str(corpus_dir),
        "index": str(index_path),
        "chunk_count": len(index.chunks),
        "embedding_model": index.embedding_model,
    }


def _index_query_payload(body: IndexQueryRequest) -> dict[str, Any]:
    from pipeline.retrieval import load_index, retrieve

    config = ProviderConfig.from_env()
    index_path = _checked_path(body.index_path, "index_path", must_exist=True)
    index = load_index(index_path, build_faiss=config.index_backend.strip().lower() == "faiss")
    # Let retrieve() resolve PIPELINE_RETRIEVAL_MODE. Persisted lexical indexes
    # store placeholder embeddings, so forcing hybrid weights here can make the
    # query path call dense embeddings and then fail on vector dimensions.
    evidence = retrieve(
        index,
        body.task,
        top_k=body.top_k,
        provider=config.retrieval_provider,
        config=config,
    )
    return {
        "surface": "index",
        "mode": "query",
        "index": str(index_path),
        "task": body.task,
        "settings": {
            "retrieval_provider": config.retrieval_provider,
            "reranker_provider": config.reranker_provider,
            "reranker_model": config.cohere_rerank_model,
        },
        "results": to_jsonable(evidence),
    }


def _edit_memory_settings_payload(body: EditMemorySettingsRequest) -> dict[str, Any]:
    from pipeline.learning import state_paths

    features = _features_from_request(body.features)
    state_dir = _checked_path(body.state_dir, "state_dir")
    paths = state_paths(state_dir)
    resolved_profile = (
        _checked_path(body.profile_path, "profile_path")
        if body.profile_path
        else paths["operator_profile"]
    )
    artifacts = {
        "operator_profile": str(resolved_profile),
        "knowledge_layer": str(paths["knowledge_layer"]),
        "retrieval_feedback": str(paths["retrieval_feedback"]),
        "edit_memory": str(paths["edit_memory"]),
        "edits_log": str(paths["state_dir"] / "edits.jsonl"),
        "learn_suggestions": str(paths["state_dir"] / "learn_suggestions.json"),
        "exemplars_dir": str(paths["exemplars_dir"]),
    }
    if body.category:
        artifacts["category_exemplars"] = str(paths["exemplars_dir"] / f"{body.category}.jsonl")
    return {
        "surface": "edit-memory",
        "mode": "settings",
        "state_dir": str(paths["state_dir"]),
        "artifacts": artifacts,
        "settings": {
            "learning_guidance": features.learning_guidance,
            "operator_profile_guidance": features.operator_profile_guidance,
            "knowledge_layer_guidance": features.knowledge_layer_guidance,
            "exemplar_guidance": features.exemplar_guidance,
            "retrieval_feedback": features.retrieval_feedback,
            "playbook_risk": features.playbook_risk,
        },
        "commands": {
            "capture_edit": (
                f"pipeline learn --draft <draft.md> --edited <edited.md> --state-dir {paths['state_dir']}"
            ),
            "export_dpo": f"pipeline export-dpo --edits-log {paths['state_dir'] / 'edits.jsonl'} --output {paths['state_dir'] / 'dpo_pairs.jsonl'}",
        },
        "notes": [
            "Use /edit-memory/query to retrieve operator preference suggestions.",
            "AI critique remains advisory unless captured as a human operator edit.",
        ],
    }


def _edit_memory_query_payload(body: EditMemoryQueryRequest) -> dict[str, Any]:
    from pipeline.learning import load_edit_memory, render_edit_memory_guidance, retrieve_edit_memory, state_paths

    paths = state_paths(_checked_path(body.state_dir, "state_dir"))
    matches = retrieve_edit_memory(
        load_edit_memory(paths["edit_memory"]),
        task=body.task,
        category=body.category,
        text=body.text,
        limit=body.limit,
    )
    return {
        "surface": "edit-memory",
        "mode": "query",
        "state_dir": body.state_dir,
        "matches": matches,
        "guidance": render_edit_memory_guidance(matches),
    }


def _harness_run_payload(body: HarnessRunRequest) -> dict[str, Any]:
    from pipeline.evaluation import HarnessCase, evaluate_run, run_harness
    from pipeline.orchestration.run import run_case

    manifest_path = _checked_path(body.manifest_path, "manifest_path", must_exist=True)
    output_dir = _checked_path(body.output_dir, "output_dir")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list):
        raise ValueError("harness manifest must be a list or an object with a cases list")
    cases = [
        HarnessCase(
            case_id=str(item["case_id"]),
            input_dir=str(item["input_dir"]),
            task=str(item["task"]),
            deterministic_checks=dict(item.get("deterministic_checks") or {}),
            metadata=dict(item.get("metadata") or {}),
        )
        for item in raw_cases
    ]

    def runner(case: HarnessCase, case_dir: Path) -> dict[str, Any]:
        run_dir = case_dir / "run"
        run_case(
            _checked_path(case.input_dir, "input_dir", must_exist=True),
            run_dir,
            case_id=case.case_id,
            task=case.task,
            force=True,
        )
        static = evaluate_run(run_dir)
        return {
            "case_id": case.case_id,
            "output_dir": str(run_dir),
            "static": asdict(static),
            "artifacts": {
                "case_run": str(run_dir / "case_run.json"),
                "draft": str(run_dir / "draft.md"),
                "retrieval_index": str(run_dir / "index" / "retrieval_index.json"),
                "corpus_manifest": str(run_dir / "corpus" / "manifest.json"),
            },
        }

    return asdict(run_harness(cases, output_dir, runner=runner))


# ---------- routes -----------------------------------------------------------


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/", include_in_schema=False)
def index_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui")


def _ui_index_path() -> Path:
    return _UI_DIST_DIR / "index.html"


def _ui_missing_response() -> Response:
    return HTMLResponse(
        "<!doctype html><body><h1>Pipeline UI assets are missing.</h1>"
        "<p>Build the frontend before serving:</p>"
        "<pre>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</pre>"
        "<p>The build emits its bundle into <code>src/pipeline/ui/static/dist/</code>, "
        "which the API serves under <code>/ui</code>.</p></body>",
        status_code=503,
    )


@app.get("/ui", include_in_schema=False)
def ui() -> Response:
    index_path = _ui_index_path()
    if not index_path.exists():
        return _ui_missing_response()
    return FileResponse(index_path)


@app.get("/ui/{rest:path}", include_in_schema=False)
def ui_spa_fallback(rest: str) -> Response:
    """SPA fallback. Serve real files from dist when present; otherwise hand
    back index.html so client-side routing under /ui/* keeps working."""

    if _UI_DIST_DIR.exists():
        candidate = (_UI_DIST_DIR / rest).resolve()
        try:
            candidate.relative_to(_UI_DIST_DIR.resolve())
        except ValueError:
            raise _http_error(404, "ui asset not found")
        if candidate.is_file():
            return FileResponse(candidate)
    index_path = _ui_index_path()
    if not index_path.exists():
        return _ui_missing_response()
    return FileResponse(index_path)


@app.post("/corpus/settings")
def post_corpus_settings(body: CorpusSettingsRequest) -> JSONResponse:
    payload = _map_pipeline_call(_corpus_settings_payload, body)
    return JSONResponse(content=payload)


@app.post("/corpus/build")
def post_corpus_build(body: CorpusBuildRequest) -> JSONResponse:
    payload = _map_pipeline_call(_corpus_build_payload, body)
    return JSONResponse(content=payload)


@app.post("/index/settings")
def post_index_settings(body: IndexSettingsRequest) -> JSONResponse:
    payload = _map_pipeline_call(_index_settings_payload, body)
    return JSONResponse(content=payload)


@app.post("/index/build")
def post_index_build(body: IndexBuildRequest) -> JSONResponse:
    payload = _map_pipeline_call(_index_build_payload, body)
    return JSONResponse(content=payload)


@app.post("/index/query")
def post_index_query(body: IndexQueryRequest) -> JSONResponse:
    payload = _map_pipeline_call(_index_query_payload, body)
    return JSONResponse(content=payload)


@app.post("/edit-memory/settings")
def post_edit_memory_settings(body: EditMemorySettingsRequest) -> JSONResponse:
    payload = _map_pipeline_call(_edit_memory_settings_payload, body)
    return JSONResponse(content=payload)


@app.post("/edit-memory/query")
def post_edit_memory_query(body: EditMemoryQueryRequest) -> JSONResponse:
    payload = _map_pipeline_call(_edit_memory_query_payload, body)
    return JSONResponse(content=payload)


@app.post("/harness")
def post_harness(body: HarnessRunRequest) -> JSONResponse:
    payload = _map_pipeline_call(_harness_run_payload, body)
    return JSONResponse(content=payload)


@app.post("/runs", response_model=RunResponse)
def post_runs(body: RunRequest) -> RunResponse:
    from pipeline.orchestration.run import run_case

    input_dir = _checked_path(body.input_dir, "input_dir")
    output_dir = _checked_path(body.output_dir, "output_dir")
    profile_path = _checked_path(body.profile_path, "profile_path") if body.profile_path else None
    state_dir = _checked_path(body.state_dir, "state_dir") if body.state_dir else None
    features = _features_from_request(body.features)

    if features.process_documents and not input_dir.exists():
        raise _http_error(404, f"input_dir not found: {input_dir}")

    run = _map_pipeline_call(
        run_case,
        input_dir,
        output_dir,
        case_id=body.case_id,
        task=body.task,
        profile_path=profile_path,
        state_dir=state_dir,
        resume=body.resume,
        force=body.force,
        features=features,
        playbook_path=_checked_path(body.playbook_path, "playbook_path") if body.playbook_path else None,
    )

    # Use the canonical artifact-key namespace defined in `_ARTIFACT_FILES` so
    # the POST /runs response and the GET /runs/{id}/artifacts listing use one
    # set of identifiers — the UI never has to map between two naming
    # conventions.
    artifacts = {
        key: str(output_dir / relative)
        for key, (relative, _media) in _ARTIFACT_FILES.items()
    }
    return RunResponse(
        case_id=getattr(run, "case_id", body.case_id),
        output_dir=str(output_dir),
        artifacts=artifacts,
        run_fingerprint=getattr(run, "run_fingerprint", ""),
    )


def _default_runs_root() -> Path:
    raw = os.getenv("PIPELINE_API_RUNS_ROOT", "").strip()
    return Path(raw).expanduser().resolve(strict=False) if raw else (Path.cwd() / "outputs").resolve(strict=False)


def _summarize_run_dir(run_dir: Path) -> dict[str, Any] | None:
    case_run_path = run_dir / "case_run.json"
    workflow_path = run_dir / "workflow_manifest.json"
    draft_md = run_dir / "draft.md"
    case_run_data: dict[str, Any] = {}
    if case_run_path.exists():
        try:
            case_run_data = read_json(case_run_path)
        except (OSError, ValueError):
            case_run_data = {}
    workflow_data: dict[str, Any] = {}
    if workflow_path.exists():
        try:
            workflow_data = read_json(workflow_path)
        except (OSError, ValueError):
            workflow_data = {}
    if not case_run_data and not workflow_data and not draft_md.exists():
        return None
    artifacts: dict[str, str] = {}
    for name, (relative, _media) in _ARTIFACT_FILES.items():
        path = run_dir / relative
        if path.exists():
            artifacts[name] = str(path)
    draft = case_run_data.get("draft") if isinstance(case_run_data, dict) else None
    draft_warnings = (
        visible_review_warnings(draft.get("warnings", []), evidence=draft.get("evidence", []))
        if isinstance(draft, dict) and isinstance(draft.get("warnings"), list)
        else []
    )
    return {
        "case_id": str(case_run_data.get("case_id") or workflow_data.get("metadata", {}).get("case_id") or run_dir.name),
        "name": run_dir.name,
        "output_dir": str(run_dir),
        "created_at": str(case_run_data.get("created_at") or ""),
        "task": str(case_run_data.get("task") or workflow_data.get("metadata", {}).get("task") or ""),
        "draft_type": str(draft.get("draft_type") if isinstance(draft, dict) else ""),
        "draft_title": str(draft.get("title") if isinstance(draft, dict) else ""),
        "section_count": (
            len(draft.get("sections", []))
            if isinstance(draft, dict) and isinstance(draft.get("sections"), list)
            else 0
        ),
        "warning_count": len(draft_warnings),
        "run_fingerprint": str(case_run_data.get("run_fingerprint") or ""),
        "artifacts": artifacts,
        "has_draft": draft_md.exists(),
        "has_evaluation": (run_dir / "evaluation.json").exists(),
        "has_risk_report": (run_dir / "risk_report.json").exists(),
    }


def _sanitize_upload_filename(raw: str) -> str:
    """Reject filenames containing path separators, traversal segments, or
    disallowed characters. We intentionally do NOT silently strip ``..`` or
    directory components — a client that sent us one is either buggy or
    hostile and should hear about it.
    """

    if not raw:
        raise _http_error(400, "upload filename is empty")
    if any(sep in raw for sep in ("/", "\\", "\x00")):
        raise _http_error(400, f"upload filename contains a path separator: {raw!r}")
    if raw in {"..", "."} or raw.startswith("."):
        raise _http_error(400, f"invalid upload filename: {raw!r}")
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.\- ()]{0,254}$", raw):
        raise _http_error(400, f"upload filename has unsupported characters: {raw!r}")
    suffix = Path(raw).suffix.lower()
    if suffix not in _UPLOAD_ALLOWED_EXTENSIONS:
        raise _http_error(
            400,
            f"unsupported file type {suffix!r}; allowed: {sorted(_UPLOAD_ALLOWED_EXTENSIONS)}",
        )
    return raw


@app.post("/uploads/{case_id}")
async def upload_documents(case_id: str, files: list[UploadFile] = File(...)) -> JSONResponse:
    """Persist a batch of uploaded source documents under a per-case input dir.

    The endpoint exists so the React console can offer a real file picker
    instead of asking operators to type a server-side path. Files are
    written to ``outputs/<case_id>/_inputs/`` (an allowlisted root) with
    sanitized names and an extension allowlist; oversized uploads are
    rejected mid-stream so we never buffer untrusted megabytes in memory.

    Trust model: same as the rest of the API surface — local/operator
    control plane behind ``PIPELINE_API_ALLOWED_ROOTS``. The returned
    ``input_dir`` is the exact path the caller should pass to
    ``POST /runs``.
    """

    _validate_case_id(case_id)
    if not files:
        raise _http_error(400, "no files supplied")

    runs_root = _default_runs_root()
    input_dir = _checked_path(str(runs_root / case_id / "_inputs"), "input_dir")
    input_dir.mkdir(parents=True, exist_ok=True)

    saved: list[dict[str, Any]] = []
    for upload in files:
        if upload.filename is None:
            raise _http_error(400, "upload missing filename")
        safe_name = _sanitize_upload_filename(upload.filename)
        destination = (input_dir / safe_name).resolve()
        try:
            destination.relative_to(input_dir.resolve())
        except ValueError:
            raise _http_error(403, f"upload destination escapes input dir: {destination}")

        bytes_written = 0
        with destination.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > _UPLOAD_MAX_BYTES:
                    handle.close()
                    destination.unlink(missing_ok=True)
                    raise _http_error(
                        413,
                        f"{safe_name} exceeds {_UPLOAD_MAX_BYTES // (1024 * 1024)} MB upload cap",
                    )
                handle.write(chunk)
        saved.append({"filename": safe_name, "size": bytes_written, "path": str(destination)})

    return JSONResponse(
        content={
            "case_id": case_id,
            "input_dir": str(input_dir),
            "files": saved,
            "count": len(saved),
        }
    )


@app.get("/runs", include_in_schema=True)
def list_runs(root: str | None = Query(default=None)) -> JSONResponse:
    """Enumerate run output directories beneath ``root`` (default ``./outputs``).

    Each entry is the minimal information the UI needs to render a list (case id,
    task, draft summary, generated artifact paths). The endpoint is read-only and
    re-uses :func:`_checked_path` to enforce the same allowlist as every other
    path-handling endpoint.
    """

    runs_root = _checked_path(root, "root") if root else _default_runs_root()
    if not runs_root.exists():
        return JSONResponse(content={"root": str(runs_root), "runs": []})
    entries: list[dict[str, Any]] = []
    for child in sorted(runs_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        summary = _summarize_run_dir(child)
        if summary is None:
            continue
        entries.append(summary)
    return JSONResponse(content={"root": str(runs_root), "runs": entries})


def _resolve_run_dir(case_id: str, output_dir: str | None) -> Path:
    _validate_case_id(case_id)
    if output_dir:
        candidate = _checked_path(output_dir, "output_dir", must_exist=True)
        return candidate
    candidate = (_default_runs_root() / case_id)
    return _checked_path(str(candidate), "output_dir", must_exist=True)


@app.get("/runs/{case_id}/summary")
def get_run_summary(case_id: str, output_dir: str | None = Query(default=None)) -> JSONResponse:
    """Compact summary used to render the dashboard row for a single run."""

    run_dir = _resolve_run_dir(case_id, output_dir)
    summary = _summarize_run_dir(run_dir)
    if summary is None:
        raise _http_error(404, f"no run artifacts present under {run_dir}")
    return JSONResponse(content=summary)


@app.get("/runs/{case_id}/artifacts")
def list_run_artifacts(case_id: str, output_dir: str | None = Query(default=None)) -> JSONResponse:
    """List available artifacts for a run.

    The response shape is ``{ run_dir, artifacts: [ { key, filename, path,
    media_type, size, exists } ] }`` so the UI never has to construct
    filenames; it always references the stable artifact ``key``.
    """

    run_dir = _resolve_run_dir(case_id, output_dir)
    entries: list[dict[str, Any]] = []
    for key, (relative, media) in _ARTIFACT_FILES.items():
        path = run_dir / relative
        exists = path.exists()
        entries.append(
            {
                "key": key,
                "filename": relative,
                "path": str(path),
                "media_type": media,
                "exists": exists,
                "size": (path.stat().st_size if exists else 0),
            }
        )
    return JSONResponse(content={"run_dir": str(run_dir), "artifacts": entries})


@app.get("/runs/{case_id}/artifacts/{artifact_key}")
def get_run_artifact(
    case_id: str,
    artifact_key: str,
    output_dir: str | None = Query(default=None),
    raw: bool = Query(default=False, description="If true, return the file as-is rather than as JSON."),
    limit: int | None = Query(default=None, ge=1, le=5000),
) -> Response:
    """Fetch a named artifact from a run directory.

    ``artifact_key`` must be one of the keys in :data:`_ARTIFACT_FILES`. JSON
    files are returned as ``application/json`` unless ``raw=true``; markdown
    is returned with ``text/markdown``; NDJSON files are parsed into a
    structured list with optional tail ``limit`` so the UI can render audit
    rows without re-parsing.
    """

    if artifact_key not in _ARTIFACT_FILES:
        raise _http_error(404, f"unknown artifact: {artifact_key}")
    relative, media = _ARTIFACT_FILES[artifact_key]
    run_dir = _resolve_run_dir(case_id, output_dir)
    artifact_path = (run_dir / relative).resolve()
    try:
        artifact_path.relative_to(run_dir.resolve())
    except ValueError:
        raise _http_error(403, f"artifact escapes run dir: {artifact_path}")
    if not artifact_path.exists():
        raise _http_error(404, f"artifact not found: {relative}")
    if raw:
        return FileResponse(artifact_path, media_type=media, filename=artifact_path.name)
    if media == "application/json":
        try:
            payload = read_json(artifact_path)
        except (OSError, ValueError) as exc:
            raise _http_error(400, f"failed to read {artifact_path}: {exc}") from exc
        if artifact_key == "draft_json" and isinstance(payload, dict):
            payload = _operator_view_draft_payload(payload)
        return JSONResponse(content=payload)
    if media == "application/x-ndjson":
        rows: list[dict[str, Any]] = []
        try:
            with artifact_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        rows.append({"raw": line})
        except OSError as exc:
            raise _http_error(400, f"failed to read {artifact_path}: {exc}") from exc
        if limit is not None and len(rows) > limit:
            rows = rows[-limit:]
        return JSONResponse(content={"rows": rows, "row_count": len(rows)})
    # Markdown: return text
    if artifact_key == "draft_md":
        operator_markdown = _operator_view_draft_markdown(run_dir)
        if operator_markdown is not None:
            return PlainTextResponse(content=operator_markdown, media_type=media)
    try:
        text = artifact_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _http_error(400, f"failed to read {artifact_path}: {exc}") from exc
    return PlainTextResponse(content=text, media_type=media)


def _operator_view_draft_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    evidence = result.get("evidence")
    evidence_items = evidence if isinstance(evidence, list) else ()
    _filter_operator_warnings_in_place(result, evidence_items)
    sections = result.get("sections")
    if isinstance(sections, list):
        result["sections"] = _operator_view_sections(sections)
    case_summary = result.get("case_summary")
    if isinstance(case_summary, dict):
        summary = dict(case_summary)
        summary_evidence = summary.get("evidence")
        _filter_operator_warnings_in_place(
            summary,
            summary_evidence if isinstance(summary_evidence, list) else evidence_items,
        )
        result["case_summary"] = summary
    return result


def _operator_view_sections(sections: Sequence[Any]) -> list[Any]:
    cleaned: list[Any] = []
    for section in sections:
        if not isinstance(section, dict):
            cleaned.append(section)
            continue
        copy = dict(section)
        body = copy.get("body")
        if isinstance(body, str):
            copy["body"] = _operator_view_section_body(
                body,
                heading=str(copy.get("heading") or ""),
            )
        cleaned.append(copy)
    return cleaned


def _operator_view_section_body(body: str, *, heading: str) -> str:
    seen_bullets: set[str] = set()
    output: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if re.match(r"^-\s*(claim_id|grounding)\s*:", stripped, flags=re.I):
            continue
        bullet = re.match(r"^(\s*-\s+)(.+)$", line)
        if bullet:
            text = _operator_view_bullet_text(bullet.group(2), heading=heading)
            key = _operator_claim_key(text)
            if key in seen_bullets:
                continue
            seen_bullets.add(key)
            output.append(f"{bullet.group(1)}{text}")
        else:
            output.append(line)
    return "\n".join(output).strip()


def _operator_view_bullet_text(text: str, *, heading: str) -> str:
    cleaned = " ".join(text.split())
    if heading.lower() == "open questions":
        for prefix in (
            "Items Requiring Verification:",
            "Open Questions:",
            "Operator Directive:",
        ):
            if cleaned.startswith(prefix):
                return _operator_language_text(cleaned[len(prefix) :].strip())
    return _operator_language_text(cleaned)


def _filter_operator_warnings_in_place(payload: dict[str, Any], evidence: Sequence[Any]) -> None:
    warnings = payload.get("warnings")
    if isinstance(warnings, list):
        payload["warnings"] = [
            _operator_language_text(warning)
            for warning in visible_review_warnings(warnings, evidence=evidence)
        ]


def _operator_view_draft_markdown(run_dir: Path) -> str | None:
    draft_json_path = run_dir / "draft.json"
    if not draft_json_path.exists():
        return None
    try:
        payload = read_json(draft_json_path)
        if not isinstance(payload, dict):
            return None
        return render_draft_markdown(draft_from_json(_operator_view_draft_payload(payload)))
    except (OSError, KeyError, TypeError, ValueError):
        return None


@app.post("/runs/{case_id}/edits")
def post_run_edit(case_id: str, body: EditSubmitRequest) -> JSONResponse:
    """Persist an operator-edited markdown body and feed it back through learning.

    The endpoint writes ``edited.md`` into the run directory next to the draft
    (so the artifact-fetch endpoint exposes it for download), then calls the
    same :func:`pipeline.learning.learn_from_files` routine the CLI uses. The
    response surfaces the updated profile snippet, the routed channel, and the
    edit classification so the UI can show the operator exactly what their
    edit changed.
    """

    from pipeline.learning import capture_operator_edit_from_files, state_paths

    run_dir = _resolve_run_dir(case_id, body.output_dir)
    draft_path = run_dir / "draft.md"
    if not draft_path.exists():
        raise _http_error(404, f"draft.md not found in {run_dir}; run the pipeline first")
    edited_text = body.edited_text
    if not isinstance(edited_text, str) or not edited_text.strip():
        raise _http_error(400, "edited_text must be a non-empty markdown body")
    edited_path = run_dir / "edited.md"
    edited_path.write_text(edited_text, encoding="utf-8")

    state_dir = (
        _checked_path(body.state_dir, "state_dir")
        if body.state_dir
        else _checked_path(str(Path.cwd() / "state"), "state_dir")
    )
    paths = state_paths(state_dir)
    profile_path = paths["operator_profile"]

    # Use the structured capture so the response carries the event +
    # classification + edit-memory delta produced by this exact call —
    # re-reading edits.jsonl/edit_classifications.jsonl would race under
    # concurrent edits and misattribute the result.
    result = _map_pipeline_call(
        capture_operator_edit_from_files,
        draft_path,
        edited_path,
        profile_path,
        operator_note=body.note,
        intent=body.intent,
        actor_type=body.actor_type,
        actor_id=body.actor_id,
        state_dir=state_dir,
        category=body.category,
    )

    latest_classification = {
        "timestamp": result.event.get("timestamp", ""),
        "edit_signature": result.edit_signature,
        "classification": result.classification,
        "draft_path": str(draft_path),
        "edited_path": str(edited_path),
    }

    return JSONResponse(
        content={
            "case_id": case_id,
            "run_dir": str(run_dir),
            "edited_path": str(edited_path),
            "profile_path": str(profile_path),
            "state_dir": str(state_dir),
            "profile": result.profile,
            "latest_event": result.event,
            "latest_classification": latest_classification,
            "edit_signature": result.edit_signature,
            "intent": body.intent,
            "category": body.category,
            "edit_memory_delta": {
                "patterns_before": result.edit_memory_patterns_before,
                "patterns_after": result.edit_memory_patterns_after,
                "patterns_added": max(
                    0,
                    result.edit_memory_patterns_after - result.edit_memory_patterns_before,
                ),
            },
        }
    )


def _state_dir_from_query(state_dir: str | None) -> Path:
    if state_dir:
        return _checked_path(state_dir, "state_dir")
    return _checked_path(str(Path.cwd() / "state"), "state_dir")


@app.get("/state/{name}")
def get_state_file(
    name: str,
    state_dir: str | None = Query(default=None),
    limit: int | None = Query(default=None, ge=1, le=2000),
) -> Response:
    """Read a typed-learning state file from ``state_dir`` (default ``./state``).

    ``name`` is one of the keys in :data:`_STATE_FILES`. JSONL files are parsed
    line-by-line and returned as a JSON array (newest-first when ``limit`` is
    set) so the UI can paginate without re-implementing a JSONL parser.
    """

    if name not in _STATE_FILES:
        raise _http_error(404, f"unknown state file: {name}")
    relative, media = _STATE_FILES[name]
    root = _state_dir_from_query(state_dir)
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise _http_error(403, f"state path escapes state_dir: {path}")
    if not path.exists():
        return JSONResponse(
            content={"state_dir": str(root), "name": name, "path": str(path), "exists": False, "data": None}
        )
    if media == "application/json":
        try:
            payload = read_json(path)
        except (OSError, ValueError) as exc:
            raise _http_error(400, f"failed to read {path}: {exc}") from exc
        return JSONResponse(
            content={"state_dir": str(root), "name": name, "path": str(path), "exists": True, "data": payload}
        )
    # JSONL: parse into structured rows so the UI can render without re-parsing.
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    rows.append({"raw": line})
    except OSError as exc:
        raise _http_error(400, f"failed to read {path}: {exc}") from exc
    if limit is not None and len(rows) > limit:
        rows = rows[-limit:]
    return JSONResponse(
        content={
            "state_dir": str(root),
            "name": name,
            "path": str(path),
            "exists": True,
            "rows": rows,
            "row_count": len(rows),
        }
    )


@app.get("/runs/{case_id}")
def get_run(case_id: str, output_dir: str | None = Query(default=None)) -> JSONResponse:
    """Return the persisted ``case_run.json`` for a case.

    By default this looks under ``outputs/<case_id>/case_run.json`` (matching
    the convention used by the sample commands in the README). Pass
    ``?output_dir=...`` to override the directory; the ``case_id`` is still
    validated against the file's ``case_id`` field when present.
    """

    _validate_case_id(case_id)
    if output_dir:
        candidate = _checked_path(output_dir, "output_dir") / "case_run.json"
    else:
        candidate = _checked_path(Path("outputs") / case_id, "output_dir") / "case_run.json"

    if not candidate.exists():
        raise _http_error(404, f"case_run.json not found at {candidate}")

    try:
        payload = read_json(candidate)
    except (OSError, ValueError) as exc:
        raise _http_error(400, f"failed to read {candidate}: {exc}") from exc

    # Defense-in-depth: when the caller overrides output_dir, refuse if the
    # persisted case_id disagrees with the URL. Mismatch likely means the
    # operator is reading the wrong run; the legacy endpoint did not enforce
    # this, but the new identity-resolution helpers do, and consistency is
    # cheaper than a confusing audit trail.
    persisted = payload.get("case_id") if isinstance(payload, dict) else None
    if output_dir and isinstance(persisted, str) and persisted and persisted != case_id:
        raise _http_error(
            409,
            f"case_run.json at {candidate} has case_id={persisted!r}, which does not match URL case_id={case_id!r}",
        )

    return JSONResponse(content=payload)


@app.post("/learn")
def post_learn(body: LearnRequest) -> JSONResponse:
    from pipeline.learning import learn_from_files, state_paths

    draft_path = _checked_path(body.draft_path, "draft_path")
    edited_path = _checked_path(body.edited_path, "edited_path")
    state_dir = _checked_path(body.state_dir, "state_dir") if body.state_dir else None
    if body.profile_path:
        profile_path = _checked_path(body.profile_path, "profile_path")
    elif state_dir is not None:
        profile_path = state_paths(state_dir)["operator_profile"]
    else:
        raise _http_error(400, "profile_path or state_dir is required")

    if not draft_path.exists():
        raise _http_error(404, f"draft_path not found: {draft_path}")
    if not edited_path.exists():
        raise _http_error(404, f"edited_path not found: {edited_path}")

    learn_kwargs: dict[str, Any] = {
        "operator_note": body.note,
        "intent": body.intent,
        "actor_type": body.actor_type,
        "actor_id": body.actor_id,
        "event_log_path": (
            _checked_path(body.event_log_path, "event_log_path") if body.event_log_path else None
        ),
        "suggestions_path": (
            _checked_path(body.suggestions_path, "suggestions_path") if body.suggestions_path else None
        ),
    }
    if state_dir is not None:
        learn_kwargs["state_dir"] = state_dir
    if body.category is not None:
        learn_kwargs["category"] = body.category
    profile = _map_pipeline_call(
        learn_from_files,
        draft_path,
        edited_path,
        profile_path,
        **learn_kwargs,
    )
    return JSONResponse(content=profile)


@app.get("/evaluate")
def get_evaluate(run_dir: str = Query(..., description="Directory containing draft.json.")) -> JSONResponse:
    from pipeline.evaluation import evaluate_run

    target = _checked_path(run_dir, "run_dir")
    if not target.exists():
        raise _http_error(404, f"run_dir not found: {target}")

    result = _map_pipeline_call(evaluate_run, target)
    return JSONResponse(content=_dataclass_to_jsonable(result))


@app.post("/eval-suite")
def post_eval_suite(body: EvalSuiteRequest) -> JSONResponse:
    from pipeline.evaluation import evaluate_suite

    run_dir = _checked_path(body.run_dir, "run_dir")
    if not run_dir.exists():
        raise _http_error(404, f"run_dir not found: {run_dir}")
    manifest_path = _checked_path(body.manifest_path, "manifest_path") if body.manifest_path else None
    gold_dir = _checked_path(body.gold_dir, "gold_dir") if body.gold_dir else None
    eval_points_path = _checked_path(body.eval_points_path, "eval_points_path") if body.eval_points_path else None
    simulated_edits_output_dir = (
        _checked_path(body.simulated_edits_output_dir, "simulated_edits_output_dir")
        if body.simulated_edits_output_dir
        else None
    )
    result = _map_pipeline_call(
        evaluate_suite,
        run_dir,
        manifest_path=manifest_path,
        gold_dir=gold_dir,
        eval_points_path=eval_points_path,
        simulate_edits=body.simulate_edits,
        simulated_edits_output_dir=simulated_edits_output_dir,
    )
    return JSONResponse(content=_dataclass_to_jsonable(result))


@app.post("/risk")
def post_risk(body: RiskRequest) -> JSONResponse:
    from pipeline.playbooks import score_playbook_run_dir

    result = _map_pipeline_call(
        score_playbook_run_dir,
        _checked_path(body.run_dir, "run_dir", must_exist=True),
        playbook_path=_checked_path(body.playbook_path, "playbook_path") if body.playbook_path else None,
        output_path=_checked_path(body.output_path, "output_path") if body.output_path else None,
        markdown_path=_checked_path(body.markdown_path, "markdown_path") if body.markdown_path else None,
    )
    return JSONResponse(content=result)


@app.post("/ab-eval")
def post_ab_eval(body: ABEvalRequest) -> JSONResponse:
    from pipeline.evaluation import evaluate_ab

    case_dir = _checked_path(body.case_dir, "case_dir")
    profile_path = _checked_path(body.profile_path, "profile_path")
    if not case_dir.exists():
        raise _http_error(404, f"case_dir not found: {case_dir}")
    if not profile_path.exists():
        raise _http_error(404, f"profile_path not found: {profile_path}")

    if body.task is None:
        from pipeline.evaluation import resolve_ab_task

        kwargs: dict[str, Any] = {"task": resolve_ab_task(case_dir)}
    else:
        kwargs = {"task": body.task}
    if body.edited_reference_path:
        kwargs["edited_reference_path"] = _checked_path(body.edited_reference_path, "edited_reference_path")
    if body.draft_output_dir:
        kwargs["draft_output_dir"] = _checked_path(body.draft_output_dir, "draft_output_dir")
    if body.state_dir:
        kwargs["state_dir"] = _checked_path(body.state_dir, "state_dir")

    result = _map_pipeline_call(evaluate_ab, case_dir, profile_path, **kwargs)
    return JSONResponse(content=_dataclass_to_jsonable(result))


@app.post("/public-data/download")
def post_public_data_download(body: PublicDataRequest) -> JSONResponse:
    from pipeline.public_data import download_public_eval_set

    result = _map_pipeline_call(
        download_public_eval_set,
        _checked_path(body.manifest_path, "manifest_path"),
        _checked_path(body.dataset_dir, "dataset_dir"),
        force=body.force,
        dry_run=body.dry_run,
        include_optional=body.include_optional,
    )
    return JSONResponse(content=result.to_jsonable())


@app.post("/public-data/verify")
def post_public_data_verify(body: PublicDataRequest) -> JSONResponse:
    from pipeline.public_data import verify_public_eval_set

    result = _map_pipeline_call(
        verify_public_eval_set,
        _checked_path(body.manifest_path, "manifest_path"),
        _checked_path(body.dataset_dir, "dataset_dir"),
        include_optional=body.include_optional,
    )
    return JSONResponse(content=result.to_jsonable(), status_code=400 if result.failed else 200)


@app.post("/export-dpo")
def post_export_dpo(body: ExportDPORequest) -> JSONResponse:
    from pipeline.learning import export_dpo_pairs

    output = _checked_path(body.output, "output")
    counts = _map_pipeline_call(
        export_dpo_pairs,
        _checked_path(body.edits_log, "edits_log"),
        output,
        include_non_preference=body.include_non_preference,
    )
    candidates_path = output.with_name(output.stem + ".candidates.jsonl")
    return JSONResponse(
        content={
            "strict_pairs": counts.get("strict", 0),
            "candidate_pairs": counts.get("candidates", 0),
            "output": str(output),
            "candidates_output": str(candidates_path),
        }
    )


# ---------- server entrypoint ------------------------------------------------


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch the FastAPI app under uvicorn. Used by the console script."""

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
