from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from pipeline.config import ProviderConfig
from pipeline.drafting.grounding import apply_claim_grounding
from pipeline.drafting import sentence_units
from pipeline.drafting.memo import sections_from_case_summary, visible_review_warnings
from pipeline.drafting.validation import validate_draft_contract
from pipeline.evaluation.eval_points import score_eval_points, simulate_operator_edit_improvement
from pipeline.io import read_json, write_json
from pipeline.learning import (
    CAUTION_MARKERS,
    REQUIRED_SECTION_LABELS,
    apply_profile_to_prompt_or_policy,
    load_profile,
)


_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    citation_id_validity: float
    unsupported_section_count: int
    evidence_count: int
    warning_count: int
    sentence_grounding_score: float | None
    notes: list[str]
    claim_count: int | None = None
    unsupported_claim_count: int | None = None
    claim_grounding_pass_rate: float | None = None
    claim_citation_coverage: float | None = None

    def to_markdown(self) -> str:
        lines = [
            "# Evaluation",
            "",
            f"- Citation id validity: {self.citation_id_validity:.2%}",
            f"- Evidence chunks retrieved: {self.evidence_count}",
            f"- Unsupported/review sections: {self.unsupported_section_count}",
            f"- Processing/draft warnings: {self.warning_count}",
        ]
        if self.sentence_grounding_score is None:
            lines.append("- Sentence grounding score: not available")
        else:
            lines.append(f"- Sentence grounding score: {self.sentence_grounding_score:.2%}")
        if self.claim_count is not None:
            lines.append(f"- Claim count: {self.claim_count}")
            lines.append(f"- Unsupported claims: {self.unsupported_claim_count or 0}")
            lines.append(f"- Claim grounding pass rate: {_format_score(self.claim_grounding_pass_rate)}")
            lines.append(f"- Claim citation coverage: {_format_score(self.claim_citation_coverage)}")
        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class EvalSuiteResult:
    run_dir: str
    static: EvaluationResult
    processing: dict[str, Any]
    retrieval: dict[str, Any]
    draft_quality: dict[str, Any]
    public_manifest: dict[str, Any]
    gold: dict[str, Any]
    eval_points: dict[str, Any] = field(default_factory=dict)
    edit_improvement: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, sort_keys=True)

    def to_markdown(self) -> str:
        lines = [
            "# Evaluation Suite",
            "",
            f"- Run dir: {self.run_dir}",
            f"- Citation id validity: {self.static.citation_id_validity:.2%}",
            f"- Sentence grounding score: {_format_score(self.static.sentence_grounding_score)}",
            f"- Claim grounding pass rate: {_format_score(self.static.claim_grounding_pass_rate)}",
            f"- Unsupported claims: {self.static.unsupported_claim_count if self.static.unsupported_claim_count is not None else 'not available'}",
            f"- Eval-points required pass rate: {_format_optional_percent(self.eval_points.get('required_pass_rate') if self.eval_points else None)}",
            f"- Simulated edit-improvement pass rate: {_format_optional_percent(self.edit_improvement.get('pass_rate') if self.edit_improvement else None)}",
            f"- Supported section ratio: {_format_optional_percent(self.draft_quality.get('supported_section_ratio'))}",
            f"- Claim citation rate: {_format_optional_percent(self.draft_quality.get('claim_citation_rate'))}",
            f"- Cited field-chunk share: {_format_optional_percent(self.retrieval.get('cited_field_chunk_share'))}",
            f"- Cited raw-page share: {_format_optional_percent(self.retrieval.get('cited_raw_page_chunk_share'))}",
            "",
            "## Processing",
            "",
            f"- Documents: {self.processing.get('document_count', 0)}",
            f"- Pages: {self.processing.get('page_count', 0)}",
            f"- Non-empty page rate: {_format_optional_percent(self.processing.get('nonempty_page_rate'))}",
            f"- Average page confidence: {_format_optional_float(self.processing.get('average_page_confidence'))}",
            f"- Low-confidence pages (<0.55): {self.processing.get('low_confidence_page_count', 0)}",
            f"- Extraction methods: {_format_mapping(self.processing.get('extraction_methods', {}))}",
            "",
            "## Retrieval",
            "",
            f"- Retrieved chunks: {self.retrieval.get('retrieved_chunk_count', 0)}",
            f"- Draft-conditioned chunks: {self.retrieval.get('draft_evidence_count', 0)}",
            f"- Retrieved field chunks: {self.retrieval.get('retrieved_field_chunk_count', 0)}",
            f"- Retrieved raw-page chunks: {self.retrieval.get('retrieved_raw_page_chunk_count', 0)}",
            f"- Distinct cited files: {self.retrieval.get('distinct_cited_file_count', 0)}",
            "",
            "## Category Coverage",
            "",
        ]
        categories = self.processing.get("categories", {})
        if categories:
            lines.append("| Category | Docs | Pages | Rendered pages | Text-layer pages | Avg conf |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for category, summary in sorted(categories.items()):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(category),
                            str(summary.get("document_count", 0)),
                            str(summary.get("page_count", 0)),
                            str(summary.get("rendered_page_count", 0)),
                            str(summary.get("text_layer_page_count", 0)),
                            _format_optional_float(summary.get("average_page_confidence")),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("- No category information found.")

        if self.public_manifest:
            lines.extend(
                [
                    "",
                    "## Public Manifest",
                    "",
                    f"- Manifest: {self.public_manifest.get('manifest_path', '')}",
                    f"- Items: {self.public_manifest.get('item_count', 0)}",
                    f"- Matched processed files: {self.public_manifest.get('matched_processed_file_count', 0)}",
                    f"- Missing processed files: {self.public_manifest.get('missing_processed_file_count', 0)}",
                    f"- Required categories present: {self.public_manifest.get('required_categories_present', False)}",
                ]
            )

        if self.eval_points:
            lines.extend(["", "## Eval Points", ""])
            lines.extend(
                [
                    f"- File: {self.eval_points.get('path', '')}",
                    f"- Required pass rate: {_format_optional_percent(self.eval_points.get('required_pass_rate'))}",
                    f"- Overall pass rate: {_format_optional_percent(self.eval_points.get('pass_rate'))}",
                ]
            )
            criteria = self.eval_points.get("criteria", {})
            if isinstance(criteria, Mapping):
                lines.extend(["", "| Criterion | Required pass | Overall pass | Failed required |"])
                lines.append("|---|---:|---:|---|")
                for name in ("document_understanding", "grounded_drafting", "edit_improvement"):
                    item = criteria.get(name, {})
                    if not isinstance(item, Mapping):
                        continue
                    failed = ", ".join(str(value) for value in item.get("failed_required", []) or []) or "-"
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                name,
                                _format_optional_percent(item.get("required_pass_rate")),
                                _format_optional_percent(item.get("pass_rate")),
                                failed,
                            ]
                        )
                        + " |"
                    )
            categories = self.eval_points.get("category_summary", {})
            if isinstance(categories, Mapping) and categories:
                lines.extend(["", "Category point coverage:"])
                for category, item in sorted(categories.items()):
                    if isinstance(item, Mapping):
                        lines.append(
                            f"- {category}: {_format_optional_percent(item.get('pass_rate'))} "
                            f"({item.get('passed_count', 0)}/{item.get('point_count', 0)})"
                        )

        if self.edit_improvement:
            lines.extend(["", "## Simulated Edit Improvement", ""])
            lines.extend(
                [
                    f"- Passed: {self.edit_improvement.get('passed', False)}",
                    f"- Pass rate: {_format_optional_percent(self.edit_improvement.get('pass_rate'))}",
                    f"- State dir: {self.edit_improvement.get('state_dir', '')}",
                ]
            )
            points = self.edit_improvement.get("points", [])
            if isinstance(points, list):
                for point in points:
                    if isinstance(point, Mapping):
                        status = "pass" if point.get("passed") else "fail"
                        lines.append(f"- {status}: {point.get('id', '')}")

        if self.gold:
            lines.extend(["", "## Gold Checks", ""])
            document_gold = self.gold.get("documents", {})
            if document_gold:
                lines.extend(
                    [
                        f"- Page-text term recall: {_format_optional_percent(document_gold.get('page_term_recall'))}",
                        f"- Field-value term recall: {_format_optional_percent(document_gold.get('field_term_recall'))}",
                        f"- Combined term recall: {_format_optional_percent(document_gold.get('combined_term_recall'))}",
                        f"- Documents checked: {document_gold.get('document_count', 0)}",
                        f"- Missing gold documents: {document_gold.get('missing_document_count', 0)}",
                    ]
                )
            retrieval_gold = self.gold.get("retrieval", {})
            if retrieval_gold:
                lines.extend(
                    [
                        f"- Retrieval recall@k: {_format_optional_percent(retrieval_gold.get('recall_at_k'))}",
                        f"- Retrieval precision@k: {_format_optional_percent(retrieval_gold.get('precision_at_k'))}",
                        f"- Retrieval MRR: {_format_optional_float(retrieval_gold.get('mrr'))}",
                    ]
                )

        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ABResult:
    case_dir: str
    profile_path: str
    section_label_adoption_delta: int
    section_label_suppression_delta: int
    caution_marker_delta: int
    preferred_phrase_adoption: float
    preferred_phrases_checked: list[str]
    citation_id_validity_without: float
    citation_id_validity_with: float
    sentence_grounding_score_without: float | None
    sentence_grounding_score_with: float | None
    claim_citation_rate_without: float | None
    claim_citation_rate_with: float | None
    raw_page_citation_share_without: float | None
    raw_page_citation_share_with: float | None
    unsupported_claim_count_without_profile: int | None
    unsupported_claim_count_with_profile: int | None
    claim_grounding_pass_rate_without: float | None
    claim_grounding_pass_rate_with: float | None
    without_profile_summary: dict[str, Any]
    with_profile_summary: dict[str, Any]
    edited_reference_similarity_without: float | None = None
    edited_reference_similarity_with: float | None = None
    edited_reference_section_f1_without: float | None = None
    edited_reference_section_f1_with: float | None = None
    profile_guidance: str = ""
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# A/B Improvement-Loop Evaluation",
            "",
            f"- Case dir: {self.case_dir}",
            f"- Profile: {self.profile_path}",
            "",
            "## Methodology",
            "",
            "- Both drafts use the same processed documents and retrieved evidence.",
            "- The with-profile draft receives prompt-time guidance learned from human operator edits.",
            "- The profile should steer structure, tone, and phrasing; grounding/factuality is checked separately.",
            "- AI critique or deterministic warnings are advisory unless accepted as human operator annotations.",
        ]
        if self.profile_guidance:
            lines.extend(["", "Profile guidance:", self.profile_guidance])
        lines.extend(
            [
                "",
                "## Deltas (with-profile vs. without-profile)",
                "",
                f"- Section-label adoption delta: {self.section_label_adoption_delta}",
                f"- Section-label suppression delta: {self.section_label_suppression_delta}",
                f"- Caution-marker delta (body, signed): {self.caution_marker_delta}",
                f"- Preferred-phrase adoption: {self.preferred_phrase_adoption:.2%}"
                f" of top {len(self.preferred_phrases_checked)} preferred phrases",
            ]
        )
        if self.preferred_phrases_checked:
            lines.append(
                "  - Phrases checked: " + ", ".join(repr(p) for p in self.preferred_phrases_checked)
            )
        lines.append(
            "- Preferred phrases are one signal; section labels and caution tone are scored separately."
        )
        lines.extend(
            [
                "",
                "## Citation id validity",
                "",
                f"- Without profile: {self.citation_id_validity_without:.2%}",
                f"- With profile:    {self.citation_id_validity_with:.2%}",
                "",
                "## Sentence grounding score (verbatim quote match)",
                "",
                f"- Without profile: {_format_score(self.sentence_grounding_score_without)}",
                f"- With profile:    {_format_score(self.sentence_grounding_score_with)}",
                "",
                "## Non-regression checks",
                "",
                f"- Claim citation rate without profile: {_format_optional_percent(self.claim_citation_rate_without)}",
                f"- Claim citation rate with profile:    {_format_optional_percent(self.claim_citation_rate_with)}",
                f"- Unsupported claims without profile: {self.unsupported_claim_count_without_profile}",
                f"- Unsupported claims with profile:    {self.unsupported_claim_count_with_profile}",
                f"- Claim grounding pass rate without profile: {_format_optional_percent(self.claim_grounding_pass_rate_without)}",
                f"- Claim grounding pass rate with profile:    {_format_optional_percent(self.claim_grounding_pass_rate_with)}",
                f"- Raw-page citation share without profile: {_format_optional_percent(self.raw_page_citation_share_without)}",
                f"- Raw-page citation share with profile:    {_format_optional_percent(self.raw_page_citation_share_with)}",
            ]
        )
        if self.edited_reference_similarity_without is not None or self.edited_reference_similarity_with is not None:
            lines.extend(
                [
                    "",
                    "## Edited-reference similarity",
                    "",
                    f"- Body token similarity without profile: {_format_optional_percent(self.edited_reference_similarity_without)}",
                    f"- Body token similarity with profile:    {_format_optional_percent(self.edited_reference_similarity_with)}",
                    f"- Section F1 without profile:            {_format_optional_percent(self.edited_reference_section_f1_without)}",
                    f"- Section F1 with profile:               {_format_optional_percent(self.edited_reference_section_f1_with)}",
                ]
            )
        lines.extend(
            [
                "",
                "## Per-draft summary",
                "",
                "Without profile:",
                f"- Sections: {self.without_profile_summary.get('section_count', 0)}",
                f"- Cited sections: {self.without_profile_summary.get('cited_section_count', 0)}",
                f"- Body length (chars): {self.without_profile_summary.get('body_chars', 0)}",
                "",
                "With profile:",
                f"- Sections: {self.with_profile_summary.get('section_count', 0)}",
                f"- Cited sections: {self.with_profile_summary.get('cited_section_count', 0)}",
                f"- Body length (chars): {self.with_profile_summary.get('body_chars', 0)}",
            ]
        )
        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, sort_keys=True)


def evaluate_run(run_dir: Path) -> EvaluationResult:
    draft = read_json(run_dir / "draft.json")
    grounding_report = _read_optional_json(run_dir / "grounding_report.json", default={})
    return _evaluate_draft_payload(draft, grounding_report=grounding_report)


def evaluate_suite(
    run_dir: Path,
    *,
    manifest_path: Path | None = None,
    gold_dir: Path | None = None,
    eval_points_path: Path | None = None,
    simulate_edits: bool = False,
    simulated_edits_output_dir: Path | None = None,
) -> EvalSuiteResult:
    draft = read_json(run_dir / "draft.json")
    grounding_report = _read_optional_json(run_dir / "grounding_report.json", default={})
    static = _evaluate_draft_payload(draft, grounding_report=grounding_report)
    processed = _read_optional_json(run_dir / "processed_documents.json", default=[])
    retrieved = _read_optional_json(run_dir / "retrieved_evidence.json", default=draft.get("evidence", []))
    manifest_summary = _public_manifest_summary(manifest_path, processed) if manifest_path else {}
    gold_summary = _gold_summary(gold_dir, processed, retrieved) if gold_dir else {}
    processing_summary = _processing_summary(processed)
    retrieval_summary = _retrieval_summary(draft, retrieved)
    draft_quality_summary = _draft_quality_summary(draft)
    edit_improvement_summary = (
        simulate_operator_edit_improvement(simulated_edits_output_dir or (run_dir / "eval" / "simulated_edits"))
        if simulate_edits
        else {}
    )
    eval_points_summary = (
        score_eval_points(
            eval_points_path,
            processed=processed,
            retrieved=retrieved,
            draft=draft,
            static=asdict(static),
            processing=processing_summary,
            retrieval=retrieval_summary,
            draft_quality=draft_quality_summary,
            edit_improvement=edit_improvement_summary,
        ).to_jsonable()
        if eval_points_path
        else {}
    )

    notes = list(static.notes)
    if retrieval_summary.get("cited_field_chunk_share", 0.0) > 0.5:
        notes.append(
            "More than half of cited evidence ids point at structured-field chunks; "
            "review raw-page citation coverage before treating this as source-text recall."
        )
    claim_rate = draft_quality_summary.get("claim_citation_rate")
    if claim_rate is not None and claim_rate < 0.9:
        notes.append(
            f"Claim citation rate is {claim_rate:.2%}; quote-substring validity only covers cited evidence pairs."
        )
    raw_share = retrieval_summary.get("cited_raw_page_chunk_share")
    if raw_share is not None and raw_share < 0.25:
        notes.append(
            f"Only {raw_share:.2%} of cited evidence ids are raw page chunks; most grounding is through derived fields."
        )
    if manifest_summary:
        notes.extend(_manifest_threshold_notes(manifest_summary, static, retrieval_summary))
    if eval_points_summary:
        for name, summary in (eval_points_summary.get("criteria", {}) or {}).items():
            if isinstance(summary, Mapping) and summary.get("failed_required"):
                notes.append(
                    f"Eval-points criterion {name} failed required checks: "
                    + ", ".join(str(item) for item in summary.get("failed_required", []))
                )
    if edit_improvement_summary and not edit_improvement_summary.get("passed"):
        notes.append("Simulated edit-improvement eval failed one or more structural loop checks.")
    if not gold_summary:
        notes.append("No gold_dir supplied; suite reports artifact integrity and coverage only.")
    return EvalSuiteResult(
        run_dir=str(run_dir),
        static=static,
        processing=processing_summary,
        retrieval=retrieval_summary,
        draft_quality=draft_quality_summary,
        public_manifest=manifest_summary,
        gold=gold_summary,
        eval_points=eval_points_summary,
        edit_improvement=edit_improvement_summary,
        notes=notes,
    )


def evaluate_ab(
    case_dir: Path,
    profile_path: Path,
    *,
    generator_without: Callable[[str], str] | None = None,
    generator_with: Callable[[str], str] | None = None,
    task: str = "first-pass case fact summary",
    preferred_phrase_limit: int = 5,
    edited_reference_path: Path | None = None,
    draft_output_dir: Path | None = None,
    state_dir: Path | None = None,
) -> ABResult:
    """Run a two-draft A/B improvement-loop evaluation.

    Loads the case checkpoints, generates one draft without operator guidance
    and one with, then computes deltas that should correlate with operator
    preferences. The two ``generator`` callables let tests avoid live providers.

    When ``state_dir`` is supplied, the with-profile draft also sees the
    knowledge layer (rules + field defaults) and per-category exemplars
    derived from the dominant retrieved-document category. Without
    ``state_dir`` the with-profile draft only sees the operator profile.
    """

    from pipeline.drafting import generate_internal_memo
    from pipeline.orchestration.run import (
        _dominant_category,
        _evidence_from_json,
        _processed_documents_from_json,
        render_draft_markdown,
    )
    from pipeline.learning import (
        load_exemplars_for_category,
        load_knowledge_layer,
        render_exemplars_guidance,
        render_knowledge_layer_guidance,
        state_paths,
    )

    processed_path = case_dir / "processed_documents.json"
    evidence_path = case_dir / "retrieved_evidence.json"
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed_documents.json under {case_dir}")
    if not evidence_path.exists():
        raise FileNotFoundError(f"Missing retrieved_evidence.json under {case_dir}")

    processed_documents = _processed_documents_from_json(read_json(processed_path))
    evidence = _evidence_from_json(read_json(evidence_path))

    profile = load_profile(profile_path)
    profile_text = apply_profile_to_prompt_or_policy(profile)
    guidance_parts = [profile_text] if profile_text else []

    if state_dir is not None:
        paths = state_paths(state_dir)
        if paths["knowledge_layer"].exists():
            knowledge_text = render_knowledge_layer_guidance(
                load_knowledge_layer(paths["knowledge_layer"])
            )
            if knowledge_text:
                guidance_parts.append(knowledge_text)
        category = _dominant_category(processed_documents, evidence)
        if category:
            exemplars = load_exemplars_for_category(
                paths["exemplars_dir"], category, limit=1
            )
            exemplars_text = render_exemplars_guidance(exemplars)
            if exemplars_text:
                guidance_parts.append(exemplars_text)

    guidance = "\n\n".join(guidance_parts)
    config = ProviderConfig.from_env()

    without_kwargs: dict[str, Any] = {
        "task": task,
        "evidence": evidence,
        "processed_documents": processed_documents,
        "learned_guidance": "",
        "config": config,
        "draft_type": config.draft_type,
    }
    with_kwargs: dict[str, Any] = {
        "task": task,
        "evidence": evidence,
        "processed_documents": processed_documents,
        "learned_guidance": guidance,
        "config": config,
        "draft_type": config.draft_type,
    }
    if generator_without is not None:
        without_kwargs["generator"] = generator_without
    if generator_with is not None:
        with_kwargs["generator"] = generator_with

    draft_without = _generate_grounded_eval_draft(without_kwargs)
    draft_with = _generate_grounded_eval_draft(with_kwargs)
    if draft_output_dir is not None:
        draft_output_dir.mkdir(parents=True, exist_ok=True)
        write_json(draft_output_dir / "without_profile_draft.json", _draft_to_payload(draft_without))
        write_json(draft_output_dir / "with_profile_draft.json", _draft_to_payload(draft_with))
        (draft_output_dir / "without_profile_draft.md").write_text(
            render_draft_markdown(draft_without),
            encoding="utf-8",
        )
        (draft_output_dir / "with_profile_draft.md").write_text(
            render_draft_markdown(draft_with),
            encoding="utf-8",
        )

    return _compare_drafts(
        draft_without=draft_without,
        draft_with=draft_with,
        profile=profile,
        profile_guidance=guidance,
        case_dir=case_dir,
        profile_path=profile_path,
        preferred_phrase_limit=preferred_phrase_limit,
        edited_reference_path=edited_reference_path,
    )


def _generate_grounded_eval_draft(kwargs: dict[str, Any]):
    from pipeline.drafting import generate_internal_memo

    attempt_kwargs = dict(kwargs)
    base_guidance = str(attempt_kwargs.get("learned_guidance") or "")
    last_error: ValueError | None = None
    for attempt in range(2):
        try:
            return _ground_and_validate_eval_draft(generate_internal_memo(**attempt_kwargs))
        except ValueError as exc:
            last_error = exc
            if attempt:
                raise
            repair = (
                "Mandatory validation repair: the previous draft was discarded because "
                f"{exc}. Regenerate every factual claim with citation quotes copied exactly from the supplied "
                "evidence JSON. If an exact quote is not available, move that item to open_questions."
            )
            attempt_kwargs["learned_guidance"] = f"{base_guidance.rstrip()}\n\n{repair}" if base_guidance.strip() else repair
    raise last_error or ValueError("A/B draft validation failed")


def _ground_and_validate_eval_draft(draft):
    if draft.case_summary is None:
        raise ValueError("A/B evaluation draft is missing the required CaseFactSummary")
    summary, grounding_report = apply_claim_grounding(draft.case_summary, draft_type=draft.draft_type)
    grounded = replace(draft, case_summary=summary, sections=sections_from_case_summary(summary))
    validate_draft_contract(grounded, grounding_report)
    return grounded


def resolve_ab_task(case_dir: Path) -> str:
    case_run = case_dir / "case_run.json"
    if case_run.exists():
        payload = read_json(case_run)
        draft = payload.get("draft") if isinstance(payload, dict) else None
        task_hint = ""
        if isinstance(payload, dict):
            task_hint = str(payload.get("task") or "").strip()
        if not task_hint and isinstance(draft, dict):
            task_hint = str(draft.get("title") or "").strip()
        if task_hint:
            return task_hint
    workflow_manifest = case_dir / "workflow_manifest.json"
    if workflow_manifest.exists():
        manifest = read_json(workflow_manifest)
        metadata = manifest.get("metadata", {}) if isinstance(manifest, dict) else {}
        if isinstance(metadata, dict):
            task_hint = str(metadata.get("task") or "").strip()
            if task_hint:
                return task_hint
    return "first-pass case fact summary"


def _read_optional_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return read_json(path)


def _compare_drafts(
    *,
    draft_without: Any,
    draft_with: Any,
    profile: Mapping[str, Any],
    case_dir: Path,
    profile_path: Path,
    preferred_phrase_limit: int,
    profile_guidance: str = "",
    edited_reference_path: Path | None = None,
) -> ABResult:
    signals = profile.get("signals", {}) if isinstance(profile, Mapping) else {}
    section_signals = signals.get("section_labels", {}) if isinstance(signals, Mapping) else {}
    preferred_signal = signals.get("preferred_phrases", {}) if isinstance(signals, Mapping) else {}

    preferred_labels = _ranked_keys(section_signals.get("added", {}))
    removed_labels = [
        label
        for label in _ranked_keys(section_signals.get("removed", {}))
        if label not in REQUIRED_SECTION_LABELS
    ]
    preferred_phrases = _ranked_keys(preferred_signal)[:preferred_phrase_limit]

    headings_without = {_normalize_label(s.heading) for s in draft_without.sections}
    headings_with = {_normalize_label(s.heading) for s in draft_with.sections}

    adoption_delta = sum(
        1
        for label in preferred_labels
        if _normalize_label(label) in headings_with
        and _normalize_label(label) not in headings_without
    )
    suppression_delta = sum(
        1
        for label in removed_labels
        if _normalize_label(label) in headings_without
        and _normalize_label(label) not in headings_with
    )

    body_without = _draft_body_text(draft_without)
    body_with = _draft_body_text(draft_with)
    caution_delta = _caution_marker_count(body_with) - _caution_marker_count(body_without)

    if preferred_phrases:
        hits = sum(1 for phrase in preferred_phrases if _phrase_in(body_with, phrase))
        phrase_adoption = hits / len(preferred_phrases)
    else:
        phrase_adoption = 0.0

    payload_without = _draft_to_payload(draft_without)
    payload_with = _draft_to_payload(draft_with)
    eval_without = _evaluate_draft_payload(payload_without)
    eval_with = _evaluate_draft_payload(payload_with)
    quality_without = _draft_quality_summary(payload_without)
    quality_with = _draft_quality_summary(payload_with)
    retrieval_without = _retrieval_summary(payload_without, payload_without.get("evidence", []))
    retrieval_with = _retrieval_summary(payload_with, payload_with.get("evidence", []))
    reference_similarity_without = None
    reference_similarity_with = None
    reference_section_f1_without = None
    reference_section_f1_with = None
    if edited_reference_path is not None:
        reference_text = edited_reference_path.read_text(encoding="utf-8")
        reference_similarity_without = _text_jaccard_similarity(_draft_body_text(draft_without), reference_text)
        reference_similarity_with = _text_jaccard_similarity(_draft_body_text(draft_with), reference_text)
        reference_headings = _markdown_headings(reference_text)
        reference_section_f1_without = _heading_f1(_draft_headings(draft_without), reference_headings)
        reference_section_f1_with = _heading_f1(_draft_headings(draft_with), reference_headings)

    notes: list[str] = []
    if not preferred_labels:
        notes.append("Profile has no learned section_labels.added; adoption delta is structurally zero.")
    if not removed_labels:
        notes.append(
            "Profile has no learned section_labels.removed outside required labels; suppression delta is structurally zero."
        )
    if not preferred_phrases:
        notes.append("Profile has no learned preferred_phrases; phrase adoption is 0/0.")
    if caution_delta < 0:
        notes.append("With-profile draft uses fewer caution markers than baseline; review profile guidance.")
    cited_without = int(_draft_summary(draft_without).get("cited_section_count", 0))
    cited_with = int(_draft_summary(draft_with).get("cited_section_count", 0))
    if cited_with < cited_without:
        notes.append("With-profile draft has fewer cited sections than baseline; review for coverage regression.")
    claim_without = quality_without.get("claim_citation_rate")
    claim_with = quality_with.get("claim_citation_rate")
    if claim_without is not None and claim_with is not None and claim_with < claim_without:
        notes.append("With-profile draft has lower claim citation rate than baseline.")
    if (
        reference_similarity_without is not None
        and reference_similarity_with is not None
        and reference_similarity_with < reference_similarity_without
    ):
        notes.append("With-profile draft is less similar to the edited reference than baseline.")

    return ABResult(
        case_dir=str(case_dir),
        profile_path=str(profile_path),
        section_label_adoption_delta=adoption_delta,
        section_label_suppression_delta=suppression_delta,
        caution_marker_delta=caution_delta,
        preferred_phrase_adoption=phrase_adoption,
        preferred_phrases_checked=list(preferred_phrases),
        citation_id_validity_without=eval_without.citation_id_validity,
        citation_id_validity_with=eval_with.citation_id_validity,
        sentence_grounding_score_without=eval_without.sentence_grounding_score,
        sentence_grounding_score_with=eval_with.sentence_grounding_score,
        claim_citation_rate_without=quality_without.get("claim_citation_rate"),
        claim_citation_rate_with=quality_with.get("claim_citation_rate"),
        raw_page_citation_share_without=retrieval_without.get("cited_raw_page_chunk_share"),
        raw_page_citation_share_with=retrieval_with.get("cited_raw_page_chunk_share"),
        unsupported_claim_count_without_profile=eval_without.unsupported_claim_count,
        unsupported_claim_count_with_profile=eval_with.unsupported_claim_count,
        claim_grounding_pass_rate_without=eval_without.claim_grounding_pass_rate,
        claim_grounding_pass_rate_with=eval_with.claim_grounding_pass_rate,
        without_profile_summary=_draft_summary(draft_without),
        with_profile_summary=_draft_summary(draft_with),
        edited_reference_similarity_without=reference_similarity_without,
        edited_reference_similarity_with=reference_similarity_with,
        edited_reference_section_f1_without=reference_section_f1_without,
        edited_reference_section_f1_with=reference_section_f1_with,
        profile_guidance=profile_guidance,
        notes=notes,
    )


def _format_score(value: float | None) -> str:
    return "not available" if value is None else f"{value:.2%}"


def _evaluate_draft_payload(
    draft: Mapping[str, Any],
    *,
    grounding_report: Mapping[str, Any] | None = None,
) -> EvaluationResult:
    evidence = draft.get("evidence", []) or []
    evidence_ids = {item.get("evidence_id") for item in evidence if isinstance(item, Mapping)}
    evidence_by_id = {
        item.get("evidence_id"): item
        for item in evidence
        if isinstance(item, Mapping) and item.get("evidence_id")
    }
    sections = draft.get("sections", []) or []
    # A section is eligible for citation-validity scoring iff drafting did not
    # flag it as unsupported/scaffolding. Sections marked ``unsupported=True``
    # (e.g., the auto-injected "Unsupported or Unclear Facts" review hook, or
    # framing scaffolds the drafting layer marks as such) are review hooks, not
    # factual claims, so they belong outside the denominator. Any other section
    # IS a factual claim that must carry valid retrieved-set citations; a
    # missing/empty evidence_ids list on such a section is a real failure and
    # MUST deflate the metric.
    eligible = [
        section
        for section in sections
        if isinstance(section, Mapping) and not section.get("unsupported")
    ]
    supported = [
        section
        for section in eligible
        if section.get("evidence_ids")
        and set(section["evidence_ids"]).issubset(evidence_ids)
    ]
    citation_id_validity = len(supported) / max(len(eligible), 1)
    unsupported_count = sum(
        1 for section in sections if isinstance(section, Mapping) and section.get("unsupported")
    )
    warning_count = len(visible_review_warnings(draft.get("warnings", []) or [], evidence=evidence))

    grounding_score, grounding_note = _sentence_grounding(sections, evidence_by_id)

    notes: list[str] = []
    if eligible and citation_id_validity < 0.75:
        notes.append(
            "Citation id validity below target; review retrieval query and section evidence mapping."
        )
    if not eligible:
        notes.append("No supported sections found; citation id validity is undefined (reported as 0%).")
    if unsupported_count:
        notes.append("Unsupported sections are intentional review hooks, not generation failures.")
    if not evidence:
        notes.append("No evidence was retrieved; generated draft should be treated as incomplete.")
    if grounding_note:
        notes.append(grounding_note)
    if grounding_score is not None:
        notes.append("Sentence grounding score is quote-substring validity, not semantic entailment.")
    claim_summary = _claim_grounding_summary(draft, grounding_report or {})
    if claim_summary.get("claim_count") is not None:
        notes.append("Claim grounding pass rate is claim-graph level; section quote metrics remain separate.")

    return EvaluationResult(
        citation_id_validity=citation_id_validity,
        unsupported_section_count=unsupported_count,
        evidence_count=len(evidence),
        warning_count=warning_count,
        sentence_grounding_score=grounding_score,
        notes=notes,
        claim_count=claim_summary.get("claim_count"),
        unsupported_claim_count=claim_summary.get("unsupported_claim_count"),
        claim_grounding_pass_rate=claim_summary.get("claim_grounding_pass_rate"),
        claim_citation_coverage=claim_summary.get("claim_citation_coverage"),
    )


def _processing_summary(processed: Sequence[Any]) -> dict[str, Any]:
    docs = [doc for doc in processed if isinstance(doc, Mapping)]
    pages: list[Mapping[str, Any]] = []
    warnings_count = 0
    categories: dict[str, dict[str, Any]] = {}
    for doc in docs:
        source = doc.get("source", {}) if isinstance(doc.get("source"), Mapping) else {}
        doc_pages = [p for p in (doc.get("pages") or []) if isinstance(p, Mapping)]
        warnings_count += len(doc.get("warnings", []) or [])
        pages.extend(doc_pages)
        category = _category_from_source_path(str(source.get("path") or source.get("filename") or ""))
        bucket = categories.setdefault(
            category,
            {
                "document_count": 0,
                "page_count": 0,
                "rendered_page_count": 0,
                "text_layer_page_count": 0,
                "_confidence_total": 0.0,
            },
        )
        bucket["document_count"] += 1
        bucket["page_count"] += len(doc_pages)
        for page in doc_pages:
            method = str(page.get("extraction_method") or "")
            if method == "pdf-rendered-png":
                bucket["rendered_page_count"] += 1
            if method == "pdf-text-layer":
                bucket["text_layer_page_count"] += 1
            bucket["_confidence_total"] += _as_float(page.get("confidence"))

    methods: dict[str, int] = {}
    confidence_total = 0.0
    low_confidence = 0
    nonempty = 0
    page_warning_count = 0
    for page in pages:
        method = str(page.get("extraction_method") or "unknown")
        methods[method] = methods.get(method, 0) + 1
        confidence = _as_float(page.get("confidence"))
        confidence_total += confidence
        if confidence < 0.55:
            low_confidence += 1
        if str(page.get("text") or "").strip():
            nonempty += 1
        page_warning_count += len(page.get("warnings", []) or [])

    for bucket in categories.values():
        page_count = int(bucket.get("page_count") or 0)
        total = float(bucket.pop("_confidence_total", 0.0))
        bucket["average_page_confidence"] = total / page_count if page_count else None

    page_count = len(pages)
    return {
        "document_count": len(docs),
        "page_count": page_count,
        "nonempty_page_count": nonempty,
        "nonempty_page_rate": nonempty / page_count if page_count else None,
        "average_page_confidence": confidence_total / page_count if page_count else None,
        "low_confidence_page_count": low_confidence,
        "extraction_methods": methods,
        "document_warning_count": warnings_count,
        "page_warning_count": page_warning_count,
        "categories": categories,
    }


def _claim_grounding_summary(
    draft: Mapping[str, Any],
    grounding_report: Mapping[str, Any],
) -> dict[str, Any]:
    if grounding_report:
        return {
            "claim_count": _optional_int(grounding_report.get("total_claims")),
            "unsupported_claim_count": _optional_int(grounding_report.get("unsupported_claim_count")),
            "claim_grounding_pass_rate": _optional_float_value(grounding_report.get("grounding_pass_rate")),
            "claim_citation_coverage": _optional_float_value(grounding_report.get("citation_coverage")),
        }
    summary = draft.get("case_summary") if isinstance(draft.get("case_summary"), Mapping) else {}
    claims = [claim for claim in (summary.get("claims", []) if isinstance(summary, Mapping) else []) if isinstance(claim, Mapping)]
    if not claims:
        return {
            "claim_count": None,
            "unsupported_claim_count": None,
            "claim_grounding_pass_rate": None,
            "claim_citation_coverage": None,
        }
    factual = [claim for claim in claims if str(claim.get("claim_type") or "") != "scaffolding"]
    cited = [claim for claim in factual if claim.get("citations")]
    supported = [
        claim
        for claim in factual
        if isinstance(claim.get("grounding"), Mapping)
        and str(claim["grounding"].get("status") or "") in {"grounded", "entailed", "supported"}
    ]
    unsupported = len(factual) - len(supported)
    return {
        "claim_count": len(claims),
        "unsupported_claim_count": unsupported,
        "claim_grounding_pass_rate": len(supported) / len(factual) if factual else None,
        "claim_citation_coverage": len(cited) / len(factual) if factual else None,
    }


def _retrieval_summary(draft: Mapping[str, Any], retrieved: Sequence[Any]) -> dict[str, Any]:
    retrieved_items = [item for item in retrieved if isinstance(item, Mapping)]
    draft_evidence = [item for item in (draft.get("evidence", []) or []) if isinstance(item, Mapping)]
    sections = [section for section in (draft.get("sections", []) or []) if isinstance(section, Mapping)]
    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in draft_evidence or retrieved_items
        if item.get("evidence_id")
    }
    cited_ids: list[str] = []
    for section in sections:
        if section.get("unsupported"):
            continue
        cited_ids.extend(str(item) for item in section.get("evidence_ids", []) or [] if str(item).strip())

    cited_field = sum(1 for evidence_id in cited_ids if _is_field_evidence(evidence_by_id.get(evidence_id), evidence_id))
    cited_raw = sum(1 for evidence_id in cited_ids if _is_raw_page_evidence(evidence_by_id.get(evidence_id), evidence_id))
    cited_total = len(cited_ids)
    retrieved_field = sum(1 for item in retrieved_items if _is_field_evidence(item, str(item.get("evidence_id") or "")))
    retrieved_raw = sum(1 for item in retrieved_items if _is_raw_page_evidence(item, str(item.get("evidence_id") or "")))
    cited_files = {
        str(evidence_by_id[evidence_id].get("filename") or "")
        for evidence_id in cited_ids
        if evidence_id in evidence_by_id and str(evidence_by_id[evidence_id].get("filename") or "").strip()
    }
    return {
        "retrieved_chunk_count": len(retrieved_items),
        "draft_evidence_count": len(draft_evidence),
        "retrieved_field_chunk_count": retrieved_field,
        "retrieved_raw_page_chunk_count": retrieved_raw,
        "cited_evidence_id_count": cited_total,
        "cited_field_chunk_count": cited_field,
        "cited_raw_page_chunk_count": cited_raw,
        "cited_field_chunk_share": cited_field / cited_total if cited_total else None,
        "cited_raw_page_chunk_share": cited_raw / cited_total if cited_total else None,
        "distinct_cited_file_count": len(cited_files),
        "distinct_cited_files": sorted(cited_files),
    }


def _draft_quality_summary(draft: Mapping[str, Any]) -> dict[str, Any]:
    sections = [section for section in (draft.get("sections", []) or []) if isinstance(section, Mapping)]
    supported_sections = [section for section in sections if not section.get("unsupported")]
    case_summary = draft.get("case_summary") if isinstance(draft.get("case_summary"), Mapping) else {}
    graph_claims = [
        claim
        for claim in (case_summary.get("claims", []) if isinstance(case_summary, Mapping) else [])
        if isinstance(claim, Mapping) and str(claim.get("claim_type") or "") != "scaffolding"
    ]
    if graph_claims:
        claim_total = len(graph_claims)
        claim_cited = sum(1 for claim in graph_claims if claim.get("citations"))
    else:
        claim_total = 0
        claim_cited = 0
        for section in supported_sections:
            for unit in sentence_units(str(section.get("body") or "")):
                if len(re.sub(r"\[[^\]]+\]", "", unit).strip()) < 18:
                    continue
                claim_total += 1
                if _CITATION_RE.search(unit):
                    claim_cited += 1
    warnings = [str(item) for item in draft.get("warnings", []) or []]
    uncited_section_warnings = [warning for warning in warnings if warning.startswith("Marked section ")]
    advisory_support_warnings = [
        warning for warning in warnings if warning.startswith("Advisory claim-support check")
    ]
    return {
        "section_count": len(sections),
        "supported_section_count": len(supported_sections),
        "unsupported_section_count": len(sections) - len(supported_sections),
        "supported_section_ratio": len(supported_sections) / len(sections) if sections else None,
        "claim_sentence_count": claim_total,
        "claim_sentence_with_inline_citation_count": claim_cited,
        "claim_citation_rate": claim_cited / claim_total if claim_total else None,
        "uncited_factual_sentence_section_warnings": len(uncited_section_warnings),
        "advisory_claim_support_warnings": len(advisory_support_warnings),
        "warning_count": len(warnings),
    }


def _public_manifest_summary(manifest_path: Path, processed: Sequence[Any]) -> dict[str, Any]:
    from pipeline.public_data import load_public_eval_manifest

    manifest = load_public_eval_manifest(manifest_path)
    processed_filenames = {
        str(doc.get("source", {}).get("filename") or Path(str(doc.get("source", {}).get("path") or "")).name)
        for doc in processed
        if isinstance(doc, Mapping) and isinstance(doc.get("source"), Mapping)
    }
    item_basenames = {Path(item.filename).name: item for item in manifest.items}
    matched = sorted(name for name in item_basenames if name in processed_filenames)
    missing = sorted(name for name in item_basenames if name not in processed_filenames)
    page_counts = {
        str(doc.get("source", {}).get("filename") or Path(str(doc.get("source", {}).get("path") or "")).name): len(
            doc.get("pages", []) or []
        )
        for doc in processed
        if isinstance(doc, Mapping) and isinstance(doc.get("source"), Mapping)
    }
    required_categories = set(manifest.golden_checks.get("required_categories", []) or [])
    matched_categories = {item_basenames[name].category for name in matched}
    return {
        "manifest_path": str(manifest_path),
        "name": manifest.name,
        "item_count": len(manifest.items),
        "matched_processed_file_count": len(matched),
        "missing_processed_file_count": len(missing),
        "matched_processed_files": matched,
        "missing_processed_files": missing,
        "required_categories": sorted(required_categories),
        "matched_categories": sorted(matched_categories),
        "required_categories_present": required_categories.issubset(matched_categories),
        "max_processed_pages_per_document": max((page_counts.get(name, 0) for name in matched), default=0),
        "golden_checks": dict(manifest.golden_checks),
    }


def _manifest_threshold_notes(
    manifest_summary: Mapping[str, Any],
    static: EvaluationResult,
    retrieval_summary: Mapping[str, Any],
) -> list[str]:
    checks = manifest_summary.get("golden_checks", {})
    if not isinstance(checks, Mapping):
        return []
    notes: list[str] = []
    min_docs = checks.get("min_documents")
    if min_docs is not None and int(manifest_summary.get("matched_processed_file_count", 0)) < int(min_docs):
        notes.append(f"Public manifest check failed: fewer than {min_docs} documents were processed.")
    required_categories = set(checks.get("required_categories", []) or [])
    matched_categories = set(manifest_summary.get("matched_categories", []) or [])
    missing_categories = sorted(required_categories - matched_categories)
    if missing_categories:
        notes.append(
            "Public manifest check failed: missing required categories "
            + ", ".join(missing_categories)
            + "."
        )
    max_pages = checks.get("max_pages_per_document")
    observed_max_pages = int(manifest_summary.get("max_processed_pages_per_document", 0) or 0)
    if max_pages is not None and observed_max_pages > int(max_pages):
        notes.append(
            f"Public manifest check failed: processed document page count {observed_max_pages} "
            f"exceeds max_pages_per_document={int(max_pages)}."
        )
    min_citation = checks.get("min_citation_id_validity")
    if min_citation is not None and static.citation_id_validity < float(min_citation):
        notes.append(
            f"Public manifest check failed: citation id validity {static.citation_id_validity:.2%} "
            f"is below {float(min_citation):.2%}."
        )
    min_grounding = checks.get("min_sentence_grounding_score")
    if (
        min_grounding is not None
        and static.sentence_grounding_score is not None
        and static.sentence_grounding_score < float(min_grounding)
    ):
        notes.append(
            f"Public manifest check failed: quote-substring validity {static.sentence_grounding_score:.2%} "
            f"is below {float(min_grounding):.2%}."
        )
    max_field_share = checks.get("max_cited_field_chunk_share")
    cited_field_share = retrieval_summary.get("cited_field_chunk_share")
    if max_field_share is not None and cited_field_share is not None and float(cited_field_share) > float(max_field_share):
        notes.append(
            f"Public manifest check failed: cited field-chunk share {float(cited_field_share):.2%} "
            f"is above {float(max_field_share):.2%}."
        )
    return notes


def _gold_summary(gold_dir: Path, processed: Sequence[Any], retrieved: Sequence[Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    doc_gold = gold_dir / "gold_documents.json"
    if doc_gold.exists():
        summary["documents"] = _gold_document_summary(doc_gold, processed)
    retrieval_gold = gold_dir / "gold_retrieval.json"
    if retrieval_gold.exists():
        summary["retrieval"] = _gold_retrieval_summary(retrieval_gold, processed, retrieved)
    return summary


def _gold_document_summary(gold_path: Path, processed: Sequence[Any]) -> dict[str, Any]:
    payload = read_json(gold_path)
    documents = payload.get("documents", []) if isinstance(payload, Mapping) else []
    processed_by_filename = {}
    for doc in processed:
        if not isinstance(doc, Mapping) or not isinstance(doc.get("source"), Mapping):
            continue
        source = doc["source"]
        filename = str(source.get("filename") or Path(str(source.get("path") or "")).name)
        processed_by_filename[filename] = doc

    total_terms = 0
    page_hits_total = 0
    field_hits_total = 0
    combined_hits_total = 0
    missing_docs = 0
    per_document = []
    for expected in documents:
        if not isinstance(expected, Mapping):
            continue
        filename = str(expected.get("filename") or "")
        category = str(expected.get("category") or "")
        doc = processed_by_filename.get(filename)
        terms = [str(term) for term in expected.get("must_extract_terms", []) or [] if str(term).strip()]
        if doc is None:
            missing_docs += 1
            per_document.append(
                {
                    "filename": filename,
                    "category": category,
                    "page_term_recall": None,
                    "field_term_recall": None,
                    "combined_term_recall": None,
                    "missing": True,
                }
            )
            continue
        page_haystack = _processed_page_text(doc)
        field_haystack = _processed_field_text(doc)
        page_hits = 0
        field_hits = 0
        combined_hits = 0
        term_results = []
        for term in terms:
            in_page = _normalized_contains(page_haystack, term)
            in_field = _normalized_contains(field_haystack, term)
            if in_page:
                page_hits += 1
            if in_field:
                field_hits += 1
            if in_page or in_field:
                combined_hits += 1
            term_results.append({"term": term, "hit_in": _term_hit_location(in_page, in_field)})
        total_terms += len(terms)
        page_hits_total += page_hits
        field_hits_total += field_hits
        combined_hits_total += combined_hits
        per_document.append(
            {
                "filename": filename,
                "category": category,
                "terms": len(terms),
                "page_hits": page_hits,
                "field_hits": field_hits,
                "combined_hits": combined_hits,
                "page_term_recall": page_hits / len(terms) if terms else None,
                "field_term_recall": field_hits / len(terms) if terms else None,
                "combined_term_recall": combined_hits / len(terms) if terms else None,
                "term_results": term_results,
                "missing": False,
            }
        )
    return {
        "document_count": len([item for item in documents if isinstance(item, Mapping)]),
        "missing_document_count": missing_docs,
        "term_count": total_terms,
        "page_term_hits": page_hits_total,
        "field_term_hits": field_hits_total,
        "combined_term_hits": combined_hits_total,
        "page_term_recall": page_hits_total / total_terms if total_terms else None,
        "field_term_recall": field_hits_total / total_terms if total_terms else None,
        "combined_term_recall": combined_hits_total / total_terms if total_terms else None,
        "per_document": per_document,
    }


def _gold_retrieval_summary(
    gold_path: Path,
    processed: Sequence[Any],
    retrieved: Sequence[Any],
) -> dict[str, Any]:
    # WHY: the gold gate must score the same retrieval the drafter actually
    # saw, so we read the persisted retrieved_evidence.json rather than
    # re-running a parallel ranking with a different scorer.
    payload = read_json(gold_path)
    queries = payload.get("queries", []) if isinstance(payload, Mapping) else []
    saved_retrieved_items = [item for item in retrieved if isinstance(item, Mapping)]
    retrieval_note = (
        "Scored against the persisted production retrieved_evidence.json "
        "(BM25 + text-embedding-3-large hybrid retrieval, plus Cohere rerank "
        "when PIPELINE_RERANK_PROVIDER=cohere is enabled; gpt-5.5 is only "
        "used for extraction and drafting)."
    )
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    per_query = []
    for query in queries:
        if not isinstance(query, Mapping):
            continue
        query_id = str(query.get("id") or "")
        top = list(saved_retrieved_items)
        k = int(query.get("k") or len(top) or 1)
        relevant_ids = {
            str(item) for item in query.get("relevant_evidence_ids", []) or [] if str(item).strip()
        }
        relevant = {str(item) for item in query.get("relevant_filenames", []) or [] if str(item).strip()}
        top = top[:k]
        top_filenames = [str(item.get("filename") or "") for item in top]
        top_ids = [str(item.get("evidence_id") or "") for item in top]
        if relevant_ids:
            hit_positions = [index + 1 for index, evidence_id in enumerate(top_ids) if evidence_id in relevant_ids]
            unique_hits = set(evidence_id for evidence_id in top_ids if evidence_id in relevant_ids)
            recall = len(unique_hits) / len(relevant_ids)
            precision = len([evidence_id for evidence_id in top_ids if evidence_id in relevant_ids]) / len(top) if top else None
            hit_values = sorted(unique_hits)
        else:
            hit_positions = [index + 1 for index, name in enumerate(top_filenames) if name in relevant]
            unique_hits = set(name for name in top_filenames if name in relevant)
            recall = len(unique_hits) / len(relevant) if relevant else None
            precision = len([name for name in top_filenames if name in relevant]) / len(top) if top else None
            hit_values = sorted(unique_hits)
        mrr = 1 / hit_positions[0] if hit_positions else 0.0
        if recall is not None:
            recalls.append(recall)
        if precision is not None:
            precisions.append(precision)
        reciprocal_ranks.append(mrr)
        per_query.append(
            {
                "id": str(query.get("id") or ""),
                "k": k,
                "task_used": str(query.get("task") or ""),
                "scored_against": "relevant_evidence_ids" if relevant_ids else "relevant_filenames",
                "relevant_filenames": sorted(relevant),
                "relevant_evidence_ids": sorted(relevant_ids),
                "top_evidence_ids": top_ids,
                "top_filenames": top_filenames,
                "hits": hit_values,
                "recall_at_k": recall,
                "precision_at_k": precision,
                "mrr": mrr,
            }
        )
    return {
        "query_count": len(per_query),
        "recall_at_k": sum(recalls) / len(recalls) if recalls else None,
        "precision_at_k": sum(precisions) / len(precisions) if precisions else None,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None,
        "per_query": per_query,
        "retrieval_mode": "production-retrieval",
        "note": retrieval_note,
    }


def _processed_page_text(doc: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for page in doc.get("pages", []) or []:
        if isinstance(page, Mapping):
            parts.append(str(page.get("text") or ""))
    return "\n".join(parts)


def _processed_field_text(doc: Mapping[str, Any]) -> str:
    parts: list[str] = []
    fields = doc.get("fields", {})
    if isinstance(fields, Mapping):
        for field in fields.values():
            if isinstance(field, Mapping):
                parts.append(str(field.get("value") or ""))
    return "\n".join(parts)


def _term_hit_location(in_page: bool, in_field: bool) -> str:
    if in_page and in_field:
        return "both"
    if in_page:
        return "page"
    if in_field:
        return "fields"
    return "neither"


def _normalized_contains(haystack: str, needle: str) -> bool:
    return re.sub(r"\s+", " ", needle).strip().casefold() in re.sub(r"\s+", " ", haystack).casefold()


def _category_from_source_path(path: str) -> str:
    parts = Path(path).parts
    if "datasets" in parts:
        index = parts.index("datasets")
        if index + 1 < len(parts) and parts[index + 1] in {"public-eval", "public"} and index + 2 < len(parts):
            return parts[index + 2]
        if index + 1 < len(parts):
            return parts[index + 1]
    if len(parts) > 1:
        return parts[0]
    return "uncategorized"


def _is_field_evidence(item: Mapping[str, Any] | None, evidence_id: str) -> bool:
    metadata = item.get("metadata", {}) if isinstance(item, Mapping) else {}
    return bool(isinstance(metadata, Mapping) and metadata.get("is_field_chunk")) or evidence_id.endswith(":fields")


def _is_raw_page_evidence(item: Mapping[str, Any] | None, evidence_id: str) -> bool:
    return not _is_field_evidence(item, evidence_id) and ":p" in evidence_id and ":c" in evidence_id


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_optional_percent(value: Any) -> str:
    if value is None:
        return "not available"
    return f"{float(value):.2%}"


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "not available"
    return f"{float(value):.2f}"


def _format_mapping(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return "none"
    return ", ".join(f"{key}={value[key]}" for key in sorted(value))


def _sentence_grounding(
    sections: Sequence[Any],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[float | None, str | None]:
    total = 0
    grounded = 0
    missing_quotes = 0
    for section in sections:
        if not isinstance(section, Mapping) or section.get("unsupported"):
            continue
        evidence_ids = [str(item) for item in section.get("evidence_ids", []) or [] if str(item).strip()]
        if not evidence_ids:
            continue
        quotes = section.get("citation_quotes")
        quote_map = quotes if isinstance(quotes, Mapping) else {}
        for evidence_id in evidence_ids:
            total += 1
            quote_text = str(quote_map.get(evidence_id) or "").strip()
            if not quote_text:
                missing_quotes += 1
                continue
            chunk = evidence_by_id.get(evidence_id)
            if not isinstance(chunk, Mapping):
                continue
            source_text = str(chunk.get("text") or "")
            if _is_substring(quote_text, source_text):
                grounded += 1
    if total == 0:
        return None, "verbatim grounding data not available in this draft."
    if missing_quotes:
        return grounded / total, f"{missing_quotes} cited evidence id(s) lacked verbatim citation_quotes."
    return grounded / total, None


def _is_substring(quote: str, source: str) -> bool:
    # Reuse drafting's quote normalization so eval scores match the validator's decisions.
    from pipeline.drafting import normalize_quote_text

    needle = normalize_quote_text(quote)
    haystack = normalize_quote_text(source)
    return bool(needle) and needle in haystack


def _draft_to_payload(draft: Any) -> dict[str, Any]:
    from pipeline.schemas import to_jsonable

    return to_jsonable(draft)


def _draft_summary(draft: Any) -> dict[str, Any]:
    sections = getattr(draft, "sections", []) or []
    cited = sum(1 for section in sections if getattr(section, "evidence_ids", None))
    body_chars = sum(len(getattr(section, "body", "")) for section in sections)
    return {
        "section_count": len(sections),
        "cited_section_count": cited,
        "body_chars": body_chars,
        "headings": [getattr(section, "heading", "") for section in sections],
    }


def _draft_headings(draft: Any) -> list[str]:
    return [getattr(section, "heading", "") for section in getattr(draft, "sections", []) or []]


def _draft_body_text(draft: Any) -> str:
    return "\n\n".join(getattr(section, "body", "") or "" for section in getattr(draft, "sections", []) or [])


def _markdown_headings(text: str) -> list[str]:
    headings: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(?P<label>.+?)\s*$", line)
        if match:
            headings.append(match.group("label").strip())
    return headings


def _heading_f1(candidate: Sequence[str], reference: Sequence[str]) -> float | None:
    candidate_set = {_normalize_label(item) for item in candidate if _normalize_label(item)}
    reference_set = {_normalize_label(item) for item in reference if _normalize_label(item)}
    if not reference_set:
        return None
    hits = len(candidate_set & reference_set)
    precision = hits / len(candidate_set) if candidate_set else 0.0
    recall = hits / len(reference_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _text_jaccard_similarity(left: str, right: str) -> float:
    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9$][A-Za-z0-9$_.-]{2,}", text)
        if not token.startswith("[")
    }


def _caution_marker_count(text: str) -> int:
    if not text:
        return 0
    lowered = text.casefold()
    total = 0
    for marker in CAUTION_MARKERS:
        if " " in marker:
            pattern = re.escape(marker)
        else:
            pattern = rf"\b{re.escape(marker)}\b"
        total += len(re.findall(pattern, lowered))
    return total


def _phrase_in(text: str, phrase: str) -> bool:
    needle = re.sub(r"\s+", " ", phrase).strip().casefold()
    haystack = re.sub(r"\s+", " ", text).casefold()
    return bool(needle) and needle in haystack


def _ranked_keys(counter: Any) -> list[str]:
    if not isinstance(counter, Mapping):
        return []
    return [
        key
        for key, _count in sorted(
            counter.items(), key=lambda item: (-int(item[1] or 0), str(item[0]).lower())
        )
        if str(key).strip()
    ]


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", str(label or "")).strip().casefold()
