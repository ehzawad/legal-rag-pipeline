from __future__ import annotations

import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pipeline.io import read_json, write_json


CRITERIA = ("document_understanding", "grounded_drafting", "edit_improvement")


@dataclass(frozen=True, slots=True)
class EvalPointResult:
    id: str
    criterion: str
    kind: str
    passed: bool
    required: bool
    observed: Any = None
    expected: Any = None
    category: str = ""
    message: str = ""


@dataclass(frozen=True, slots=True)
class EvalPointsReport:
    path: str
    name: str
    point_count: int
    required_count: int
    passed_count: int
    required_passed_count: int
    pass_rate: float | None
    required_pass_rate: float | None
    criteria: dict[str, dict[str, Any]]
    category_summary: dict[str, dict[str, Any]]
    results: list[EvalPointResult] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def load_eval_points(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Eval-points file must be a JSON object: {path}")
    version = int(payload.get("version", 0) or 0)
    if version != 1:
        raise ValueError(f"Unsupported eval-points version: {version}")
    points = payload.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("Eval-points file must include a non-empty points list")
    return dict(payload)


def score_eval_points(
    eval_points_path: Path,
    *,
    processed: Sequence[Any],
    retrieved: Sequence[Any],
    draft: Mapping[str, Any],
    static: Mapping[str, Any],
    processing: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    draft_quality: Mapping[str, Any],
    edit_improvement: Mapping[str, Any] | None = None,
) -> EvalPointsReport:
    payload = load_eval_points(eval_points_path)
    context = _Context(
        processed=[item for item in processed if isinstance(item, Mapping)],
        retrieved=[item for item in retrieved if isinstance(item, Mapping)],
        draft=draft,
        static=static,
        processing=processing,
        retrieval=retrieval,
        draft_quality=draft_quality,
        edit_improvement=edit_improvement or {},
    )
    results = [_score_point(point, context) for point in payload["points"] if isinstance(point, Mapping)]
    return _summarize_results(eval_points_path, str(payload.get("name") or ""), results)


def simulate_operator_edit_improvement(output_dir: Path) -> dict[str, Any]:
    """Run deterministic simulated edits through the public learning API.

    The goal is not to prove legal correctness. It proves that operator edits
    travel through capture/classify/route/mine/apply and create measurable
    downstream effects: profile guidance, edit-memory recall, knowledge rules,
    retrieval feedback, and non-regression gates.
    """

    from pipeline.config import ProviderConfig
    from pipeline.evaluation.report import _evaluate_draft_payload
    from pipeline.learning import (
        apply_profile_to_prompt_or_policy,
        learn_from_files,
        load_edit_memory,
        load_knowledge_layer,
        load_profile,
        load_retrieval_feedback,
        render_knowledge_layer_guidance,
        retrieve_edit_memory,
        state_paths,
    )
    from pipeline.retrieval.engine import RetrievalIndex, _build_bm25_index, retrieve
    from pipeline.schemas import EvidenceChunk

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures_dir = output_dir / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "state"
    paths = state_paths(state_dir)
    profile_path = paths["operator_profile"]

    preference_before = """# Issue
The notice was sent on March 12, 2025. [doc:p1:c0]

# Deadlines and Dates
The amount due is $2,400. [doc:p1:c0]
"""
    preference_after = """# Evidence Reviewed
Provided records include a notice and payment record. [doc:p1:c0]

# Facts That Appear Supported
Based on provided records, the notice appears to have been sent on March 12, 2025. [doc:p1:c0]

# Items Requiring Verification
Verify the $2,400 balance before relying on it as nonpayment. [doc:p1:c0]
"""
    field_before = "Jurisdiction: NY\nCourt: Supreme Court\n"
    field_after = "Jurisdiction: New York\nCourt: Supreme Court\n"
    knowledge_before = "# Risk Notes\nAmounts were copied into the draft. [doc:p1:c0]\n"
    knowledge_after = "# Risk Notes\nAmounts with decimal punctuation should be checked against the source. [doc:p1:c0]\n"
    dispute_before = "# Material Facts\nThe tenant failed to pay rent. [bad:p1:c0]\n"
    dispute_after = "# Material Facts\nThe payment record lists a $2,400 balance; it does not state breach. [good:p1:c0]\n"

    edits = [
        (
            "preference",
            preference_before,
            preference_after,
            "Prefer Evidence Reviewed, Facts That Appear Supported, and Items Requiring Verification sections; use cautious wording from provided records.",
            "public-eval",
        ),
        (
            "field_correction",
            field_before,
            field_after,
            "Reusable field correction: expand jurisdiction abbreviations.",
            None,
        ),
        (
            "knowledge_rule",
            knowledge_before,
            knowledge_after,
            "Always flag decimal-punctuated financial values as transcription-risk pending source reconciliation.",
            None,
        ),
        (
            "evidence_dispute",
            dispute_before,
            dispute_after,
            "Wrong citation and unsupported claim: do not infer breach or nonpayment from balance records.",
            None,
        ),
    ]

    before_memory = retrieve_edit_memory(
        load_edit_memory(paths["edit_memory"]),
        task="case fact summary",
        category="public-eval",
        text="provided records facts appear supported verification",
        limit=5,
    )

    for index, (intent, before, after, note, category) in enumerate(edits, start=1):
        draft_path = fixtures_dir / f"edit_{index:02d}_draft.md"
        edited_path = fixtures_dir / f"edit_{index:02d}_edited.md"
        draft_path.write_text(before, encoding="utf-8")
        edited_path.write_text(after, encoding="utf-8")
        learn_from_files(
            draft_path,
            edited_path,
            profile_path,
            operator_note=note,
            intent=intent,
            actor_type="human",
            actor_id="eval-operator",
            state_dir=state_dir,
            category=category,
        )

    profile = load_profile(profile_path)
    profile_guidance = apply_profile_to_prompt_or_policy(profile)
    knowledge_guidance = render_knowledge_layer_guidance(load_knowledge_layer(paths["knowledge_layer"]))
    feedback = load_retrieval_feedback(paths["retrieval_feedback"])
    after_memory = retrieve_edit_memory(
        load_edit_memory(paths["edit_memory"]),
        task="case fact summary",
        category="public-eval",
        text="provided records facts appear supported verification",
        limit=5,
    )

    chunks = [
        EvidenceChunk(
            evidence_id="bad:p1:c0",
            document_id="bad",
            filename="bad_notice.pdf",
            page_number=1,
            text="notice notice notice balance record",
        ),
        EvidenceChunk(
            evidence_id="good:p1:c0",
            document_id="good",
            filename="good_payment_record.pdf",
            page_number=1,
            text="payment record lists a $2,400 balance and a property address at 1 Main Street",
        ),
    ]
    index = RetrievalIndex(
        chunks=chunks,
        embeddings=[[0.0], [0.0]],
        provider="fixture",
        embedding_model="fixture",
        bm25=_build_bm25_index(chunks),
    )
    config = ProviderConfig(
        retrieval_mode="lexical",
        hybrid_dense_weight=0.0,
        hybrid_bm25_weight=1.0,
    )
    rank_query = "notice payment"
    before_rank = [
        item.evidence_id
        for item in retrieve(index, rank_query, top_k=2, provider="fixture", config=config, dense_weight=0, lexical_weight=1)
    ]
    boosted_feedback = {
        "version": 1,
        "evidence_boosts": {"good:p1:c0": 5},
        "evidence_demotes": {"bad:p1:c0": 3, **dict(feedback.get("evidence_demotes", {}))},
        "disputes": list(feedback.get("disputes", [])),
    }
    after_results = retrieve(
        index,
        rank_query,
        top_k=2,
        provider="fixture",
        config=config,
        dense_weight=0,
        lexical_weight=1,
        retrieval_feedback=boosted_feedback,
    )
    after_rank = [item.evidence_id for item in after_results]
    after_metadata = {
        item.evidence_id: dict(item.metadata)
        for item in after_results
    }

    grounded_draft = {
        "sections": [
            {
                "heading": "Facts That Appear Supported",
                "body": "The payment record lists a $2,400 balance. [good:p1:c0]",
                "evidence_ids": ["good:p1:c0"],
                "citation_quotes": {"good:p1:c0": "payment record lists a $2,400 balance"},
                "unsupported": False,
            }
        ],
        "evidence": [
            {
                "evidence_id": "good:p1:c0",
                "text": "payment record lists a $2,400 balance and a property address at 1 Main Street",
            }
        ],
        "warnings": [],
    }
    non_regression = _evaluate_draft_payload(grounded_draft)

    point_results = [
        _simple_point(
            "edit-memory-before-empty",
            "edit_improvement",
            "memory_empty_before",
            not before_memory,
            len(before_memory),
            0,
        ),
        _simple_point(
            "edit-memory-retrieves-learned-section",
            "edit_improvement",
            "memory_recall",
            any("Facts That Appear Supported" in str(item.get("suggestion") or "") for item in after_memory),
            [item.get("suggestion") for item in after_memory],
            "Facts That Appear Supported",
        ),
        _simple_point(
            "profile-guidance-adopts-operator-section",
            "edit_improvement",
            "guidance_adoption",
            "Facts That Appear Supported" in profile_guidance,
            profile_guidance,
            "Facts That Appear Supported",
        ),
        _simple_point(
            "knowledge-rule-rendered",
            "edit_improvement",
            "knowledge_guidance",
            "decimal-punctuated financial values" in knowledge_guidance,
            knowledge_guidance,
            "decimal-punctuated financial values",
        ),
        _simple_point(
            "retrieval-feedback-moves-good-evidence-to-top",
            "edit_improvement",
            "retrieval_rank_movement",
            before_rank[:1] == ["bad:p1:c0"] and after_rank[:1] == ["good:p1:c0"],
            {"before_rank": before_rank, "after_rank": after_rank, "metadata": after_metadata},
            "good:p1:c0 top-1 after feedback",
        ),
        _simple_point(
            "grounding-non-regression",
            "edit_improvement",
            "grounding_non_regression",
            non_regression.citation_id_validity >= 1.0
            and (non_regression.sentence_grounding_score or 0.0) >= 1.0
            and non_regression.unsupported_section_count == 0,
            asdict(non_regression),
            "citation and quote grounding stay perfect",
        ),
    ]
    summary = _summarize_improvement_points(point_results)
    report = {
        "criterion": "edit_improvement",
        "state_dir": str(state_dir),
        "profile_path": str(profile_path),
        "edit_memory_path": str(paths["edit_memory"]),
        "retrieval_feedback_path": str(paths["retrieval_feedback"]),
        "knowledge_layer_path": str(paths["knowledge_layer"]),
        "point_count": len(point_results),
        "passed_count": sum(1 for point in point_results if point["passed"]),
        "pass_rate": summary["pass_rate"],
        "passed": summary["passed"],
        "points": point_results,
        "notes": [
            "Simulated edits are deterministic and routed through learn_from_files.",
            "Improvement is structural: memory recall, guidance adoption, retrieval rank movement, and grounding non-regression.",
        ],
    }
    write_json(output_dir / "simulated_edit_improvement.json", report)
    return report


@dataclass(slots=True)
class _Context:
    processed: list[Mapping[str, Any]]
    retrieved: list[Mapping[str, Any]]
    draft: Mapping[str, Any]
    static: Mapping[str, Any]
    processing: Mapping[str, Any]
    retrieval: Mapping[str, Any]
    draft_quality: Mapping[str, Any]
    edit_improvement: Mapping[str, Any]


def _score_point(point: Mapping[str, Any], context: _Context) -> EvalPointResult:
    kind = str(point.get("kind") or "").strip()
    criterion = str(point.get("criterion") or "document_understanding").strip()
    if criterion not in CRITERIA:
        criterion = "document_understanding"
    point_id = str(point.get("id") or kind or "eval-point")
    required = bool(point.get("required", True))
    category = str(point.get("category") or "")
    try:
        if kind == "document_count_min":
            observed = int(context.processing.get("document_count") or 0)
            expected = int(point.get("min") or point.get("value") or 0)
            passed = observed >= expected
        elif kind == "page_count_max":
            observed = _max_doc_pages(context.processed, filename=str(point.get("filename") or ""))
            expected = int(point.get("max") or point.get("value") or 0)
            passed = observed is not None and observed <= expected
        elif kind == "category_count_min":
            expected = int(point.get("min") or point.get("value") or 0)
            observed = int(
                (context.processing.get("categories", {}) or {})
                .get(category, {})
                .get("document_count", 0)
            )
            passed = observed >= expected
        elif kind == "extraction_contains_term":
            term = str(point.get("term") or "")
            haystack = _processed_text(context.processed, filename=str(point.get("filename") or ""))
            observed = _contains(haystack, term)
            expected = term
            passed = bool(observed)
        elif kind == "term_absent":
            term = str(point.get("term") or "")
            haystack = _artifact_text(point, context)
            observed = _contains(haystack, term)
            expected = f"absent: {term}"
            passed = not observed
        elif kind == "metric_min":
            metric = str(point.get("metric") or "")
            observed = _metric_value(metric, context)
            expected = float(point.get("min") if point.get("min") is not None else point.get("value"))
            passed = _number(observed) >= expected
        elif kind == "edit_improvement_metric_min":
            metric = str(point.get("metric") or "")
            observed = _nested_metric(context.edit_improvement, metric)
            expected = float(point.get("min") if point.get("min") is not None else point.get("value"))
            passed = _number(observed) >= expected
        else:
            observed = None
            expected = None
            passed = False
            return EvalPointResult(
                id=point_id,
                criterion=criterion,
                kind=kind or "unknown",
                passed=False,
                required=required,
                observed=observed,
                expected=expected,
                category=category,
                message=f"Unsupported eval-point kind: {kind}",
            )
    except Exception as exc:  # noqa: BLE001 - reports should capture point failures.
        return EvalPointResult(
            id=point_id,
            criterion=criterion,
            kind=kind or "unknown",
            passed=False,
            required=required,
            category=category,
            message=str(exc),
        )
    return EvalPointResult(
        id=point_id,
        criterion=criterion,
        kind=kind,
        passed=bool(passed),
        required=required,
        observed=observed,
        expected=expected,
        category=category,
        message="" if passed else "eval point failed",
    )


def _summarize_results(path: Path, name: str, results: list[EvalPointResult]) -> EvalPointsReport:
    required = [item for item in results if item.required]
    criteria: dict[str, dict[str, Any]] = {}
    for criterion in CRITERIA:
        selected = [item for item in results if item.criterion == criterion]
        selected_required = [item for item in selected if item.required]
        criteria[criterion] = {
            "point_count": len(selected),
            "passed_count": sum(1 for item in selected if item.passed),
            "required_count": len(selected_required),
            "required_passed_count": sum(1 for item in selected_required if item.passed),
            "pass_rate": _rate(sum(1 for item in selected if item.passed), len(selected)),
            "required_pass_rate": _rate(sum(1 for item in selected_required if item.passed), len(selected_required)),
            "failed_required": [item.id for item in selected_required if not item.passed],
        }
    by_category: dict[str, list[EvalPointResult]] = {}
    for item in results:
        if item.category:
            by_category.setdefault(item.category, []).append(item)
    category_summary = {
        category: {
            "point_count": len(items),
            "passed_count": sum(1 for item in items if item.passed),
            "pass_rate": _rate(sum(1 for item in items if item.passed), len(items)),
            "failed": [item.id for item in items if not item.passed],
        }
        for category, items in sorted(by_category.items())
    }
    return EvalPointsReport(
        path=str(path),
        name=name,
        point_count=len(results),
        required_count=len(required),
        passed_count=sum(1 for item in results if item.passed),
        required_passed_count=sum(1 for item in required if item.passed),
        pass_rate=_rate(sum(1 for item in results if item.passed), len(results)),
        required_pass_rate=_rate(sum(1 for item in required if item.passed), len(required)),
        criteria=criteria,
        category_summary=category_summary,
        results=results,
    )


def _simple_point(
    point_id: str,
    criterion: str,
    kind: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> dict[str, Any]:
    return {
        "id": point_id,
        "criterion": criterion,
        "kind": kind,
        "passed": bool(passed),
        "required": True,
        "observed": observed,
        "expected": expected,
    }


def _summarize_improvement_points(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(points)
    passed = sum(1 for point in points if point.get("passed"))
    return {"passed": passed == total if total else False, "pass_rate": _rate(passed, total)}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _max_doc_pages(processed: Sequence[Mapping[str, Any]], *, filename: str = "") -> int | None:
    counts = []
    for doc in processed:
        source = doc.get("source", {}) if isinstance(doc.get("source"), Mapping) else {}
        doc_name = Path(str(source.get("filename") or source.get("path") or "")).name
        if filename and doc_name != Path(filename).name:
            continue
        pages = doc.get("pages", []) or []
        if isinstance(pages, list):
            counts.append(len(pages))
    return max(counts) if counts else None


def _processed_text(processed: Sequence[Mapping[str, Any]], *, filename: str = "") -> str:
    parts: list[str] = []
    target = Path(filename).name if filename else ""
    for doc in processed:
        source = doc.get("source", {}) if isinstance(doc.get("source"), Mapping) else {}
        doc_name = Path(str(source.get("filename") or source.get("path") or "")).name
        if target and doc_name != target:
            continue
        for page in doc.get("pages", []) or []:
            if isinstance(page, Mapping):
                parts.append(str(page.get("text") or ""))
        fields = doc.get("fields", {})
        if isinstance(fields, Mapping):
            for field in fields.values():
                if isinstance(field, Mapping):
                    parts.append(str(field.get("value") or ""))
    return "\n".join(parts)


def _artifact_text(point: Mapping[str, Any], context: _Context) -> str:
    target = str(point.get("target") or "draft")
    if target == "processed":
        return _processed_text(context.processed, filename=str(point.get("filename") or ""))
    if target == "retrieved":
        return "\n".join(str(item.get("text") or "") for item in context.retrieved)
    if target == "all":
        return "\n".join(
            [
                _processed_text(context.processed, filename=str(point.get("filename") or "")),
                "\n".join(str(item.get("text") or "") for item in context.retrieved),
                _draft_text(context.draft),
            ]
        )
    return _draft_text(context.draft)


def _draft_text(draft: Mapping[str, Any]) -> str:
    parts = [str(draft.get("title") or "")]
    for section in draft.get("sections", []) or []:
        if isinstance(section, Mapping):
            parts.append(str(section.get("heading") or ""))
            parts.append(str(section.get("body") or ""))
    summary = draft.get("case_summary") if isinstance(draft.get("case_summary"), Mapping) else {}
    for claim in summary.get("claims", []) if isinstance(summary, Mapping) else []:
        if isinstance(claim, Mapping):
            parts.append(str(claim.get("section") or ""))
            parts.append(str(claim.get("text") or ""))
    return "\n".join(parts)


def _contains(haystack: str, needle: str) -> bool:
    return _normalize(needle) in _normalize(haystack) if needle else False


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _metric_value(metric: str, context: _Context) -> Any:
    if metric in context.static:
        return context.static.get(metric)
    if metric in context.processing:
        return context.processing.get(metric)
    if metric in context.retrieval:
        return context.retrieval.get(metric)
    if metric in context.draft_quality:
        return context.draft_quality.get(metric)
    return _nested_metric(
        {
            "static": context.static,
            "processing": context.processing,
            "retrieval": context.retrieval,
            "draft_quality": context.draft_quality,
            "edit_improvement": context.edit_improvement,
        },
        metric,
    )


def _nested_metric(root: Mapping[str, Any], dotted: str) -> Any:
    current: Any = root
    for part in dotted.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _number(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
