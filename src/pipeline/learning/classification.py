from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import re
from typing import Any

from pipeline.providers import parse_json_response
from pipeline.schemas import EditClassification, to_jsonable


_LEGAL_CONCLUSION_RE = re.compile(
    r"\b(breach|breached|liable|liability|default|fault|legally sufficient|failed to comply)\b",
    flags=re.IGNORECASE,
)
_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")


def classify_edit(
    *,
    section: str = "",
    draft_type: str = "case_fact_summary",
    original_text: str,
    edited_text: str,
    operator_note: str = "",
    evidence_ids: list[str] | None = None,
    min_extraction_confidence: float | None = None,
    citation_in_retrieval_context: bool | None = None,
    generator: Callable[[str], str] | None = None,
) -> EditClassification:
    """Classify one operator edit and extract a reusable routing lesson.

    When ``generator`` is supplied it is expected to return JSON matching
    ``EditClassification``. Without it, deterministic rules provide the same
    schema so the mining path remains testable and local-first.
    """

    if generator is not None:
        payload = parse_json_response(generator(_classification_prompt(
            section=section,
            draft_type=draft_type,
            original_text=original_text,
            edited_text=edited_text,
            operator_note=operator_note,
            evidence_ids=evidence_ids or [],
            min_extraction_confidence=min_extraction_confidence,
            citation_in_retrieval_context=citation_in_retrieval_context,
        )))
        if isinstance(payload, Mapping):
            return _classification_from_mapping(payload)
    return _deterministic_classification(
        section=section,
        draft_type=draft_type,
        original_text=original_text,
        edited_text=edited_text,
        operator_note=operator_note,
        min_extraction_confidence=min_extraction_confidence,
        citation_in_retrieval_context=citation_in_retrieval_context,
    )


def classification_to_jsonable(classification: EditClassification) -> dict[str, Any]:
    return to_jsonable(classification)


def _classification_prompt(**payload: Any) -> str:
    return (
        "Classify a single operator edit on a grounded case fact summary and extract one reusable lesson. "
        "Return strict JSON with fields edit_type, reusable_rule, target_mechanisms, rule_scope, "
        "diagnostic, confidence, rationale. target_mechanisms values may include field_override, "
        "style_preference, retrieval_feedback, grounding_rule, template_preference.\n\n"
        f"Edit JSON:\n{json.dumps(payload, ensure_ascii=True)}"
    )


def _classification_from_mapping(payload: Mapping[str, Any]) -> EditClassification:
    return EditClassification(
        edit_type=_choice(payload.get("edit_type"), default="style_rewrite"),
        reusable_rule=str(payload.get("reusable_rule") or payload.get("reusable_lesson") or "").strip(),
        target_mechanisms=[
            str(item)
            for item in payload.get("target_mechanisms", payload.get("memory_targets", [])) or []
            if str(item).strip()
        ],
        rule_scope=_choice(payload.get("rule_scope"), default="this_draft_type"),
        diagnostic=_choice(payload.get("diagnostic"), default="n_a"),
        confidence=_choice(payload.get("confidence"), default="medium"),
        rationale=str(payload.get("rationale") or payload.get("reason") or "").strip(),
    )


def _deterministic_classification(
    *,
    section: str,
    draft_type: str,
    original_text: str,
    edited_text: str,
    operator_note: str,
    min_extraction_confidence: float | None,
    citation_in_retrieval_context: bool | None,
) -> EditClassification:
    before = original_text.strip()
    after = edited_text.strip()
    note = operator_note.casefold()
    if not after and before:
        if _LEGAL_CONCLUSION_RE.search(before):
            return EditClassification(
                edit_type="legal_conclusion_removed",
                reusable_rule="Do not state legal conclusions unless the cited source directly says them.",
                target_mechanisms=["style_preference", "grounding_rule"],
                rule_scope="global",
                diagnostic="drafter_ignored",
                confidence="high",
                rationale="Operator removed a legal conclusion from the draft.",
            )
        return EditClassification(
            edit_type="unsupported_removed",
            reusable_rule="Move unsupported factual claims to open questions instead of presenting them as facts.",
            target_mechanisms=["retrieval_feedback", "grounding_rule"],
            rule_scope="this_draft_type",
            diagnostic="drafter_ignored",
            confidence="medium",
            rationale="Operator deleted a drafted claim.",
        )
    if after and not before:
        diagnostic = "retrieval_missed" if citation_in_retrieval_context is False else "drafter_ignored"
        return EditClassification(
            edit_type="missing_fact_added",
            reusable_rule=f"When drafting {section or draft_type}, surface operator-added cited facts as positive evidence.",
            target_mechanisms=["retrieval_feedback"],
            rule_scope="this_draft_type",
            diagnostic=diagnostic,
            confidence="medium",
            rationale="Operator inserted a new fact.",
        )
    if _citations(before) != _citations(after) and _strip_citations(before) == _strip_citations(after):
        return EditClassification(
            edit_type="citation_correction",
            reusable_rule="Prefer the operator-corrected citation when the claim text is unchanged.",
            target_mechanisms=["retrieval_feedback"],
            rule_scope="this_draft_type",
            diagnostic="drafter_ignored",
            confidence="high",
            rationale="Only citation pointers changed.",
        )
    if (
        min_extraction_confidence is not None
        and min_extraction_confidence < 0.85
        and _is_certainty_downgrade(before, after)
    ):
        return EditClassification(
            edit_type="certainty_downgrade",
            reusable_rule="When cited source confidence is below 0.85, use cautious language rather than bare assertions.",
            target_mechanisms=["style_preference", "grounding_rule"],
            rule_scope="this_draft_type",
            diagnostic="n_a",
            confidence="high",
            rationale="Operator softened certainty while preserving the fact.",
        )
    if "citation" in note or "evidence" in note:
        target = ["retrieval_feedback"]
        edit_type = "citation_correction"
    elif _LEGAL_CONCLUSION_RE.search(before) and not _LEGAL_CONCLUSION_RE.search(after):
        target = ["style_preference", "grounding_rule"]
        edit_type = "legal_conclusion_removed"
    else:
        target = ["style_preference"]
        edit_type = "style_rewrite"
    return EditClassification(
        edit_type=edit_type,
        reusable_rule=_fallback_rule(edit_type, section=section),
        target_mechanisms=target,
        rule_scope="this_draft_type",
        diagnostic="n_a",
        confidence="medium",
        rationale="Deterministic classifier routed the edit from text and operator note.",
    )


def _fallback_rule(edit_type: str, *, section: str) -> str:
    if edit_type == "legal_conclusion_removed":
        return "Avoid legal conclusions unless the cited evidence states the conclusion directly."
    if edit_type == "citation_correction":
        return "Use operator-corrected citations as preferred evidence for similar future claims."
    return f"Follow operator wording preferences for {section or 'this draft type'} when grounding is preserved."


def _citations(text: str) -> list[str]:
    return _CITATION_RE.findall(text or "")


def _strip_citations(text: str) -> str:
    return " ".join(_CITATION_RE.sub("", text or "").split())


def _is_certainty_downgrade(before: str, after: str) -> bool:
    hedges = ("appears", "appears to", "may", "seems", "indicates", "suggests", "alleged")
    return not any(hedge in before.casefold() for hedge in hedges) and any(
        hedge in after.casefold() for hedge in hedges
    )


def _choice(value: Any, *, default: str) -> str:
    text = str(value or "").strip()
    return text or default
