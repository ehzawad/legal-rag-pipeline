from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

from pipeline.drafting.component import MemoDraftingComponent
from pipeline.drafting.rendering import render_draft_markdown as _render_draft_markdown
from pipeline.evidence_pack import build_evidence_pack
from pipeline.ingestion.component import DocumentProcessingComponent
from pipeline.learning.guidance import LearningGuidanceComponent, dominant_category
from pipeline.orchestration.components import PipelineComponents
from pipeline.orchestration.artifacts import (
    PipelinePaths,
    evidence_from_json,
    evidence_pack_from_json,
    processed_documents_from_json,
)
from pipeline.retrieval.component import EvidenceRetrievalComponent
from pipeline.config import PipelineFeatures, ProviderConfig, is_cached_retrieval_provider
from pipeline.io import read_json, write_json
from pipeline.orchestration.audit import append_audit_event
from pipeline.playbooks import DEFAULT_PLAYBOOK_PATH, load_playbook, score_playbook, write_risk_report
from pipeline.schemas import (
    CaseRun,
    Draft,
    EvidenceChunk,
    EvidencePack,
    ProcessedDocument,
    now_iso,
    to_jsonable,
)
from pipeline.orchestration.workflow import RetryPolicy, WorkflowRecorder


@dataclass(frozen=True, slots=True)
class RunFingerprint:
    digest: str
    inputs: dict[str, str]
    task: str
    profile_digest: str
    playbook_digest: str
    providers: dict[str, str | int | float]
    features: dict[str, bool | int | float]

    def to_jsonable(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "inputs": dict(self.inputs),
            "task": self.task,
            "profile_digest": self.profile_digest,
            "playbook_digest": self.playbook_digest,
            "providers": dict(self.providers),
            "features": dict(self.features),
        }


@dataclass(frozen=True, slots=True)
class _Invalidation:
    invalidate_process: bool
    invalidate_retrieve: bool
    invalidate_draft: bool
    warnings: tuple[str, ...]


def _load_processor():
    from pipeline.ingestion.documents import process_directory

    return process_directory


def _load_retrieval_and_drafting():
    from pipeline.drafting import generate_internal_memo
    from pipeline.retrieval import build_index, retrieve

    return build_index, retrieve, generate_internal_memo


def _load_learning():
    from pipeline.learning import (
        apply_profile_to_prompt_or_policy,
        load_knowledge_layer,
        load_profile,
        load_retrieval_feedback,
        load_exemplars_for_category,
        render_exemplars_guidance,
        render_knowledge_layer_guidance,
        state_paths,
    )

    return {
        "load_profile": load_profile,
        "profile_guidance": apply_profile_to_prompt_or_policy,
        "load_knowledge_layer": load_knowledge_layer,
        "load_retrieval_feedback": load_retrieval_feedback,
        "load_exemplars_for_category": load_exemplars_for_category,
        "render_exemplars_guidance": render_exemplars_guidance,
        "render_knowledge_layer_guidance": render_knowledge_layer_guidance,
        "state_paths": state_paths,
    }


def _default_components() -> PipelineComponents:
    process_directory = _load_processor()
    build_index, retrieve, generate_internal_memo = _load_retrieval_and_drafting()
    return PipelineComponents(
        processing=DocumentProcessingComponent(process_directory),
        retrieval=EvidenceRetrievalComponent(build_index=build_index, retrieve=retrieve),
        guidance=LearningGuidanceComponent(),
        drafting=MemoDraftingComponent(
            generate_internal_memo=generate_internal_memo,
            render_draft_markdown=_render_draft_markdown,
        ),
    )


def run_case(
    input_dir: Path,
    output_dir: Path,
    *,
    case_id: str = "sample-case",
    task: str = "first-pass internal memo",
    profile_path: Path | None = None,
    state_dir: Path | None = None,
    config: ProviderConfig | None = None,
    resume: bool = False,
    retry_policy: RetryPolicy | None = None,
    force: bool = False,
    features: PipelineFeatures | None = None,
    components: PipelineComponents | None = None,
    playbook_path: Path | None = None,
) -> CaseRun:
    config = config or ProviderConfig.from_env()
    config.validate_runtime()
    features = features or PipelineFeatures.from_env()
    features.validate_runtime()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = PipelinePaths.for_output_dir(output_dir)
    config = _with_default_embedding_cache_dir(config, output_dir)

    resolved_profile_path = profile_path or (
        Path(state_dir) / "operator_profile.json"
        if state_dir is not None
        else output_dir / "operator_profile.json"
    )
    fingerprint = _compute_run_fingerprint(
        input_dir=input_dir,
        task=task,
        profile_path=resolved_profile_path,
        config=config,
        state_dir=state_dir,
        features=features,
        playbook_path=playbook_path,
    )

    invalidation = _detect_invalidation(
        manifest_path=paths.workflow_manifest,
        new_fingerprint=fingerprint,
        resume=resume,
        force=force,
    )

    components = components or _default_components()
    learning_api = _load_learning()
    resolved_state_paths = learning_api["state_paths"](state_dir)
    recorder = WorkflowRecorder(
        paths.workflow_manifest,
        metadata={
            "case_id": case_id,
            "input_dir": str(input_dir),
            "task": task,
            "providers": {
                "extraction": config.extraction_provider,
                "retrieval": config.retrieval_provider,
                "generation": config.generation_provider,
            },
            "extraction_concurrency": config.extraction_concurrency,
            "models": {
                "openai": config.openai_model,
                "openai_embedding": config.openai_embedding_model,
                "openai_reasoning_effort": config.openai_reasoning_effort,
            },
            "embedding_cache_dir": config.embedding_cache_dir,
            "corpus_dir": str(paths.corpus_dir),
            "retrieval_index": str(paths.retrieval_index),
            "evidence_pack": str(paths.evidence_pack),
            "features": features.to_jsonable(),
            "playbook_path": str(playbook_path or DEFAULT_PLAYBOOK_PATH),
            "run_fingerprint": fingerprint.to_jsonable(),
        },
        retry_policy=retry_policy,
    )

    invalidation_warnings = list(invalidation.warnings)

    if not features.process_documents:
        _require_checkpoint(paths.processed_documents, "process_documents")
        processed = components.processing.load(paths)
        recorder.record_skipped(
            "process_documents",
            artifacts=[paths.processed_documents],
            reason="component disabled; reused checkpoint",
            warnings=invalidation_warnings if invalidation.invalidate_process else None,
        )
    elif resume and not force and not invalidation.invalidate_process and paths.processed_documents.exists():
        processed = components.processing.load(paths)
        recorder.record_skipped(
            "process_documents",
            artifacts=[paths.processed_documents],
            reason="resume reused checkpoint",
        )
    else:
        def process_stage() -> list[ProcessedDocument]:
            return components.processing.run(
                input_dir=input_dir,
                paths=paths,
                config=config,
            )

        processed = recorder.run_stage(
            "process_documents",
            process_stage,
            artifacts=[paths.processed_documents, paths.corpus_dir / "manifest.json"],
            warnings=invalidation_warnings if invalidation.invalidate_process else None,
        )

    if not features.retrieve_evidence:
        _require_checkpoint(paths.retrieved_evidence, "retrieve_evidence")
        evidence = components.retrieval.load(paths)
        recorder.record_skipped(
            "retrieve_evidence",
            artifacts=[paths.retrieved_evidence],
            reason="component disabled; reused checkpoint",
            warnings=invalidation_warnings if invalidation.invalidate_retrieve else None,
        )
    elif resume and not force and not invalidation.invalidate_retrieve and paths.retrieved_evidence.exists():
        evidence = components.retrieval.load(paths)
        recorder.record_skipped(
            "retrieve_evidence",
            artifacts=[paths.retrieved_evidence],
            reason="resume reused checkpoint",
        )
    else:
        def retrieval_stage() -> list[EvidenceChunk]:
            retrieval_feedback = None
            retrieval_feedback_path = resolved_state_paths["retrieval_feedback"]
            if features.retrieval_feedback and retrieval_feedback_path.exists():
                retrieval_feedback = learning_api["load_retrieval_feedback"](retrieval_feedback_path)
            return components.retrieval.run(
                processed=processed,
                task=task,
                paths=paths,
                config=config,
                features=features,
                retrieval_feedback=retrieval_feedback,
            )

        evidence = recorder.run_stage(
            "retrieve_evidence",
            retrieval_stage,
            artifacts=[paths.retrieved_evidence, paths.retrieval_index],
            warnings=invalidation_warnings if invalidation.invalidate_retrieve else None,
        )

    evidence_pack: EvidencePack | None = None
    if features.evidence_pack:
        if resume and not force and not invalidation.invalidate_retrieve and paths.evidence_pack.exists():
            evidence_pack = evidence_pack_from_json(read_json(paths.evidence_pack))
            recorder.record_skipped(
                "build_evidence_pack",
                artifacts=[paths.evidence_pack],
                reason="resume reused checkpoint",
            )
        else:
            def evidence_pack_stage() -> EvidencePack:
                pack = build_evidence_pack(
                    case_id=case_id,
                    task=task,
                    evidence=evidence,
                    processed_documents=processed,
                )
                write_json(paths.evidence_pack, to_jsonable(pack))
                return pack

            evidence_pack = recorder.run_stage(
                "build_evidence_pack",
                evidence_pack_stage,
                artifacts=[paths.evidence_pack],
                warnings=invalidation_warnings if invalidation.invalidate_retrieve else None,
            )
    else:
        recorder.record_skipped(
            "build_evidence_pack",
            artifacts=[],
            reason="component disabled",
        )

    guidance, guidance_sources = components.guidance.build(
        processed=processed,
        evidence=evidence,
        paths=paths,
        profile_path=resolved_profile_path,
        state_paths=resolved_state_paths,
        learning_api=learning_api,
        features=features,
    )
    append_audit_event(
        paths.audit_log,
        action="learning_guidance.composed",
        object_type="case",
        object_id=case_id,
        payload={"sources": guidance_sources, "guidance_chars": len(guidance)},
    )

    if not features.generate_draft:
        _require_checkpoint(paths.draft_json, "generate_draft")
        _require_checkpoint(paths.draft_markdown, "generate_draft")
        draft = components.drafting.load(paths)
        recorder.record_skipped(
            "generate_draft",
            artifacts=[paths.draft_json, paths.draft_markdown],
            reason="component disabled; reused checkpoint",
            warnings=invalidation_warnings if invalidation.invalidate_draft else None,
        )
    elif (
        resume
        and not force
        and not invalidation.invalidate_draft
        and paths.draft_json.exists()
        and paths.draft_markdown.exists()
    ):
        draft = components.drafting.load(paths)
        recorder.record_skipped(
            "generate_draft",
            artifacts=[paths.draft_json, paths.draft_markdown],
            reason="resume reused checkpoint",
        )
    else:
        def draft_stage() -> Draft:
            return components.drafting.run(
                processed=processed,
                evidence=evidence,
                evidence_pack=evidence_pack,
                task=task,
                guidance=guidance,
                paths=paths,
                config=config,
                features=features,
                case_id=case_id,
            )

        draft = recorder.run_stage(
            "generate_draft",
            draft_stage,
            artifacts=[paths.draft_json, paths.draft_markdown, paths.case_fact_summary, paths.grounding_report],
            warnings=invalidation_warnings if invalidation.invalidate_draft else None,
        )

    if paths.grounding_report.exists():
        try:
            grounding_report = read_json(paths.grounding_report)
        except (OSError, ValueError):
            grounding_report = {}
        append_audit_event(
            paths.audit_log,
            action="grounding.scored",
            object_type="case",
            object_id=case_id,
            payload={
                "grounding_report": str(paths.grounding_report),
                "total_claims": grounding_report.get("total_claims", 0),
                "grounding_pass_rate": grounding_report.get("grounding_pass_rate"),
                "unsupported_claim_count": grounding_report.get("unsupported_claim_count", 0),
            },
        )

    if not features.playbook_risk:
        risk_report = {"summary": {}}
        append_audit_event(
            paths.audit_log,
            action="playbook_risk.skipped",
            object_type="case",
            object_id=case_id,
            payload={"reason": "feature disabled"},
        )
        recorder.record_skipped(
            "score_playbook_risk",
            artifacts=[paths.audit_log],
            reason="component disabled",
        )
    elif resume and not force and not invalidation.invalidate_draft and paths.risk_report.exists():
        risk_report = read_json(paths.risk_report)
        recorder.record_skipped(
            "score_playbook_risk",
            artifacts=[paths.risk_report, paths.risk_report_markdown, paths.audit_log],
            reason="resume reused checkpoint",
        )
    else:
        def risk_stage() -> dict[str, object]:
            report = score_playbook(
                processed=processed,
                evidence=evidence,
                draft=draft,
                playbook=load_playbook(playbook_path),
            )
            write_risk_report(report, paths.risk_report, paths.risk_report_markdown)
            append_audit_event(
                paths.audit_log,
                action="playbook_risk.scored",
                object_type="case",
                object_id=case_id,
                payload={
                    "risk_report": str(paths.risk_report),
                    "finding_count": report.get("summary", {}).get("finding_count", 0),
                    "high": report.get("summary", {}).get("high", 0),
                    "critical": report.get("summary", {}).get("critical", 0),
                },
            )
            return report

        risk_report = recorder.run_stage(
            "score_playbook_risk",
            risk_stage,
            artifacts=[paths.risk_report, paths.risk_report_markdown, paths.audit_log],
            warnings=invalidation_warnings if invalidation.invalidate_draft else None,
        )

    profile_consumed = (
        features.learning_guidance
        and features.operator_profile_guidance
        and resolved_profile_path.exists()
    )
    run = CaseRun(
        case_id=case_id,
        created_at=now_iso(),
        input_dir=str(input_dir),
        processed_documents=processed,
        draft=draft,
        task=task,
        operator_profile_path=str(resolved_profile_path) if profile_consumed else "",
        learned_guidance=guidance,
        run_fingerprint=fingerprint.digest,
    )
    run_record = to_jsonable(run)
    run_record["run_fingerprint_detail"] = fingerprint.to_jsonable()
    run_record["learning_guidance_sources"] = guidance_sources
    run_record["evidence_pack_path"] = str(paths.evidence_pack)
    run_record["case_fact_summary_path"] = str(paths.case_fact_summary)
    run_record["grounding_report_path"] = str(paths.grounding_report)
    run_record["risk_report_path"] = str(paths.risk_report)
    run_record["risk_summary"] = risk_report.get("summary", {})
    run_record["audit_log_path"] = str(paths.audit_log)
    write_json(paths.case_run, run_record)
    append_audit_event(
        paths.audit_log,
        action="case.completed",
        object_type="case",
        object_id=case_id,
        payload={
            "case_run": str(paths.case_run),
            "run_fingerprint": fingerprint.digest,
        },
    )
    return run


def _require_checkpoint(path: Path, stage_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{stage_name} is disabled but checkpoint artifact is missing: {path}")


def render_draft_markdown(draft) -> str:
    return _render_draft_markdown(draft)


def _dominant_category(
    processed: list[ProcessedDocument],
    evidence: list[EvidenceChunk],
) -> str:
    return dominant_category(processed, evidence)


def _compute_run_fingerprint(
    *,
    input_dir: Path,
    task: str,
    profile_path: Path,
    config: ProviderConfig,
    features: PipelineFeatures,
    state_dir: Path | None = None,
    playbook_path: Path | None = None,
) -> RunFingerprint:
    inputs = _hash_input_directory(input_dir)
    profile_digest = _hash_state_contents(profile_path, state_dir, features=features)
    playbook_digest = _hash_playbook(playbook_path or DEFAULT_PLAYBOOK_PATH, features=features)
    providers = {
        "extraction_provider": config.extraction_provider,
        "retrieval_provider": config.retrieval_provider,
        "generation_provider": config.generation_provider,
        "openai_model": config.openai_model,
        "openai_embedding_model": config.openai_embedding_model,
        "openai_reasoning_effort": config.openai_reasoning_effort,
        "retrieval_mode": config.retrieval_mode,
        "index_backend": config.index_backend,
        "hybrid_dense_weight": config.hybrid_dense_weight,
        "hybrid_bm25_weight": config.hybrid_bm25_weight,
        "retrieval_top_k": config.retrieval_top_k,
        "reranker_provider": config.reranker_provider,
        "pdf_max_pages": config.pdf_max_pages,
        "pdf_render_dpi": config.pdf_render_dpi,
        "embedding_cache": "enabled" if is_cached_retrieval_provider(config.retrieval_provider) else "disabled",
    }
    feature_payload = features.to_jsonable()
    payload = {
        "inputs": inputs,
        "task": task,
        "profile_digest": profile_digest,
        "playbook_digest": playbook_digest,
        "providers": providers,
        "features": feature_payload,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return RunFingerprint(
        digest=digest,
        inputs=inputs,
        task=task,
        profile_digest=profile_digest,
        playbook_digest=playbook_digest,
        providers=providers,
        features=feature_payload,
    )


def _hash_state_contents(
    profile_path: Path,
    state_dir: Path | None,
    *,
    features: PipelineFeatures,
) -> str:
    """Hash every state-backed input that influences the drafter prompt.

    The hash covers ``profile_path``, ``knowledge_layer.json``,
    ``retrieval_feedback.json``, and the per-category exemplar store, so
    ``--resume`` will invalidate the cached draft whenever an operator
    edit updates any of those — keeping the resume path coherent with
    state-driven guidance changes.
    """

    hasher = hashlib.sha256()
    if features.learning_guidance and features.operator_profile_guidance:
        hasher.update(_hash_profile(profile_path).encode("utf-8"))
    else:
        hasher.update(b"profile-guidance-disabled")
    if state_dir is None:
        return hasher.hexdigest()
    state_root = Path(state_dir)
    if not state_root.exists():
        return hasher.hexdigest()
    state_files: list[Path] = []
    state_names: list[str] = []
    if features.learning_guidance and features.knowledge_layer_guidance:
        state_names.append("knowledge_layer.json")
    if features.retrieval_feedback:
        state_names.append("retrieval_feedback.json")
    for name in state_names:
        candidate = state_root / name
        if candidate.exists():
            state_files.append(candidate)
    exemplars_dir = state_root / "exemplars"
    if features.learning_guidance and features.exemplar_guidance and exemplars_dir.exists():
        state_files.extend(sorted(exemplars_dir.glob("*.jsonl")))
    for path in sorted(state_files):
        hasher.update(b"\n--FILE:" + str(path.name).encode("utf-8") + b"--\n")
        try:
            hasher.update(path.read_bytes())
        except OSError:
            continue
    return hasher.hexdigest()


def _with_default_embedding_cache_dir(config: ProviderConfig, output_dir: Path) -> ProviderConfig:
    if not is_cached_retrieval_provider(config.retrieval_provider):
        return config
    if str(config.embedding_cache_dir or "").strip():
        return config
    return replace(config, embedding_cache_dir=str(output_dir / "embedding_cache"))


def _hash_input_directory(input_dir: Path) -> dict[str, str]:
    if not input_dir.exists():
        return {}
    from pipeline.ingestion.documents import _SUPPORTED_DOC_EXTENSIONS  # local to avoid cycle

    entries: dict[str, str] = {}
    for path in sorted(p for p in input_dir.rglob("*") if p.is_file()):
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in _SUPPORTED_DOC_EXTENSIONS:
            continue
        rel = path.relative_to(input_dir).as_posix()
        entries[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return entries


def _hash_profile(profile_path: Path) -> str:
    if not profile_path.exists():
        return "no-profile"
    return hashlib.sha256(profile_path.read_bytes()).hexdigest()


def _hash_playbook(playbook_path: Path, *, features: PipelineFeatures) -> str:
    if not features.playbook_risk:
        return "playbook-risk-disabled"
    if not playbook_path.exists():
        return "missing-playbook"
    return hashlib.sha256(playbook_path.read_bytes()).hexdigest()


def _detect_invalidation(
    *,
    manifest_path: Path,
    new_fingerprint: RunFingerprint,
    resume: bool,
    force: bool,
) -> _Invalidation:
    if force:
        return _Invalidation(
            invalidate_process=True,
            invalidate_retrieve=True,
            invalidate_draft=True,
            warnings=("force=True: ignoring any cached checkpoints",),
        )
    if not resume:
        return _Invalidation(False, False, False, ())

    if not manifest_path.exists():
        return _Invalidation(
            invalidate_process=True,
            invalidate_retrieve=True,
            invalidate_draft=True,
            warnings=("resume requested but no prior workflow_manifest.json was found; re-running all stages",),
        )

    try:
        prior = read_json(manifest_path)
    except (OSError, ValueError):
        return _Invalidation(
            invalidate_process=True,
            invalidate_retrieve=True,
            invalidate_draft=True,
            warnings=("prior workflow_manifest.json is unreadable; re-running all stages",),
        )

    prior_meta = prior.get("metadata") or {}
    prior_fp = prior_meta.get("run_fingerprint")
    if not isinstance(prior_fp, dict):
        return _Invalidation(
            invalidate_process=True,
            invalidate_retrieve=True,
            invalidate_draft=True,
            warnings=(
                "prior workflow_manifest.json has no run_fingerprint; treating as drift and re-running all stages",
            ),
        )

    if prior_fp.get("digest") == new_fingerprint.digest:
        return _Invalidation(False, False, False, ())

    changes = _describe_fingerprint_changes(prior_fp, new_fingerprint)
    warnings = (
        "run_fingerprint mismatch detected; invalidating cached checkpoints. Changed: "
        + "; ".join(changes),
    )
    # Today every recorded input feeds the earliest stage, so any mismatch
    # invalidates the entire pipeline. We still surface the specific change so
    # reviewers see what shifted.
    return _Invalidation(
        invalidate_process=True,
        invalidate_retrieve=True,
        invalidate_draft=True,
        warnings=warnings,
    )


def _describe_fingerprint_changes(prior: dict, new: RunFingerprint) -> list[str]:
    changes: list[str] = []
    prior_inputs = prior.get("inputs") or {}
    if prior_inputs != new.inputs:
        added = sorted(set(new.inputs) - set(prior_inputs))
        removed = sorted(set(prior_inputs) - set(new.inputs))
        modified = sorted(
            name
            for name in set(prior_inputs) & set(new.inputs)
            if prior_inputs.get(name) != new.inputs.get(name)
        )
        if added:
            changes.append(f"inputs added={added}")
        if removed:
            changes.append(f"inputs removed={removed}")
        if modified:
            changes.append(f"inputs modified={modified}")
    if prior.get("task") != new.task:
        changes.append("task changed")
    if prior.get("profile_digest") != new.profile_digest:
        changes.append("profile changed")
    if prior.get("playbook_digest") != new.playbook_digest:
        changes.append("playbook changed")
    prior_providers = prior.get("providers") or {}
    if prior_providers != new.providers:
        diffs = sorted(
            key
            for key in set(prior_providers) | set(new.providers)
            if prior_providers.get(key) != new.providers.get(key)
        )
        if diffs:
            changes.append(f"providers changed={diffs}")
    prior_features = prior.get("features") or {}
    if prior_features != new.features:
        diffs = sorted(
            key
            for key in set(prior_features) | set(new.features)
            if prior_features.get(key) != new.features.get(key)
        )
        if diffs:
            changes.append(f"features changed={diffs}")
    if not changes:
        changes.append("fingerprint digest changed but no individual field diffed (treating as drift)")
    return changes


def _processed_documents_from_json(items) -> list[ProcessedDocument]:
    return processed_documents_from_json(items)


def _evidence_from_json(items) -> list[EvidenceChunk]:
    return evidence_from_json(items)
