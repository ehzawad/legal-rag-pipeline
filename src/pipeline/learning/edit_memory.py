"""Persistent operator edit-memory for future preference suggestions."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


EDIT_MEMORY_VERSION = 1
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")


def create_edit_memory_store() -> dict[str, Any]:
    """Return an empty edit-memory store."""

    return {
        "version": EDIT_MEMORY_VERSION,
        "patterns": [],
    }


def create_edit_memory() -> dict[str, Any]:
    """Alias for callers that treat edit memory as the public store."""

    return create_edit_memory_store()


def load_edit_memory(path: str | Path) -> dict[str, Any]:
    """Load edit-memory JSON, returning an empty store when absent."""

    file = Path(path)
    if not file.exists():
        return create_edit_memory_store()
    with file.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return normalized_edit_memory(payload)


def save_edit_memory(store: Mapping[str, Any], path: str | Path) -> None:
    """Persist edit memory as stable, human-readable JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(normalized_edit_memory(store), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def update_edit_memory(
    store: Mapping[str, Any] | None,
    edit_record: Mapping[str, Any],
    *,
    analysis: Mapping[str, Any] | None = None,
    category: str | None = None,
    task: str = "",
    draft_path: str | Path | None = None,
    edited_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return ``store`` updated with retrievable patterns from one operator edit.

    Only human, in-envelope edits are persisted. Evidence disputes, knowledge
    rules, rejections, and agent suggestions remain in their existing channels.
    """

    state = normalized_edit_memory(store)
    actor = edit_record.get("actor") if isinstance(edit_record.get("actor"), Mapping) else {}
    actor_type = str(actor.get("type") or edit_record.get("actor_type") or "human")
    intent = str(edit_record.get("intent") or "preference")
    if actor_type != "human" or intent not in {"preference", "field_correction"}:
        return state

    if analysis is None:
        from pipeline.learning.state import analyze_edit

        analysis_data = analyze_edit(edit_record)
    else:
        analysis_data = analysis
    patterns = _patterns_from_edit(
        edit_record,
        analysis_data,
        category=category,
        task=task,
        draft_path=draft_path,
        edited_path=edited_path,
    )
    if not patterns:
        return state

    existing_by_id = {str(pattern.get("pattern_id")): pattern for pattern in state["patterns"]}
    for pattern in patterns:
        existing = existing_by_id.get(pattern["pattern_id"])
        if existing is None:
            state["patterns"].append(pattern)
            existing_by_id[pattern["pattern_id"]] = pattern
            continue
        existing["last_seen_at"] = max(str(existing.get("last_seen_at") or ""), pattern["last_seen_at"])
        existing["count"] = max(int(existing.get("count", 1)), int(pattern.get("count", 1)))
        sources = _dedupe_strings(list(existing.get("source_edit_signatures") or []) + pattern["source_edit_signatures"])
        existing["source_edit_signatures"] = sources
        existing["trigger_terms"] = _dedupe_strings(list(existing.get("trigger_terms") or []) + pattern["trigger_terms"])
    state["patterns"].sort(
        key=lambda item: (
            -int(item.get("count", 1)),
            str(item.get("last_seen_at") or ""),
            str(item.get("pattern_id") or ""),
        ),
        reverse=False,
    )
    return normalized_edit_memory(state)


def retrieve_edit_memory(
    store: Mapping[str, Any] | str | Path | None,
    *,
    task: str = "",
    draft_text: str = "",
    text: str = "",
    category: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve relevant operator edit patterns for a future task or draft."""

    if isinstance(store, (str, Path)):
        state = load_edit_memory(store)
    else:
        state = normalized_edit_memory(store)
    if limit <= 0:
        return []

    draft_text = draft_text or text
    query_terms = _token_set(" ".join([task, draft_text, category or ""]))
    query_terms.update(_section_terms(draft_text))
    normalized_category = _normalize_category(category)

    scored: list[tuple[float, dict[str, Any]]] = []
    for pattern in state["patterns"]:
        trigger_terms = set(str(item).casefold() for item in pattern.get("trigger_terms", []))
        overlap = len(query_terms & trigger_terms)
        score = float(overlap)
        if normalized_category and normalized_category == pattern.get("category"):
            score += 2.0
        if not query_terms:
            score += 0.25
        score += min(int(pattern.get("count", 1)), 5) * 0.05
        if score <= 0:
            continue
        matched = dict(pattern)
        matched["score"] = round(score, 4)
        matched["matched_terms"] = sorted(query_terms & trigger_terms)[:8]
        scored.append((score, matched))

    scored.sort(
        key=lambda item: (
            -item[0],
            -int(item[1].get("count", 1)),
            str(item[1].get("last_seen_at") or ""),
            str(item[1].get("suggestion") or ""),
        )
    )
    return [item for _score, item in scored[:limit]]


def render_edit_memory_suggestions(
    patterns: Sequence[Mapping[str, Any]],
    *,
    heading: str = "Operator Preference Suggestions",
    limit: int = 5,
) -> str:
    """Render retrieved edit-memory patterns for a prompt or review UI."""

    selected = [pattern for pattern in patterns if str(pattern.get("suggestion") or "").strip()][:limit]
    if not selected:
        return ""
    lines = [heading + ":"]
    for pattern in selected:
        suggestion = _normalize_space(str(pattern.get("suggestion") or ""))
        rationale = _normalize_space(str(pattern.get("rationale") or ""))
        if rationale:
            lines.append(f"- {suggestion} ({rationale})")
        else:
            lines.append(f"- {suggestion}")
    lines.append("Apply these as suggestions only; preserve all citation and grounding requirements.")
    return "\n".join(lines)


def render_edit_memory_guidance(
    patterns: Sequence[Mapping[str, Any]],
    *,
    heading: str = "Operator Preference Suggestions",
    limit: int = 5,
) -> str:
    """Alias for prompt-guidance consumers."""

    return render_edit_memory_suggestions(patterns, heading=heading, limit=limit)


def normalized_edit_memory(store: Mapping[str, Any] | None) -> dict[str, Any]:
    base = create_edit_memory_store()
    if not isinstance(store, Mapping):
        return base
    try:
        base["version"] = int(store.get("version", EDIT_MEMORY_VERSION))
    except (TypeError, ValueError):
        base["version"] = EDIT_MEMORY_VERSION
    patterns = []
    seen: set[str] = set()
    for raw in store.get("patterns", []) or []:
        if not isinstance(raw, Mapping):
            continue
        pattern = _normalize_pattern(raw)
        if not pattern or pattern["pattern_id"] in seen:
            continue
        seen.add(pattern["pattern_id"])
        patterns.append(pattern)
    base["patterns"] = patterns
    return base


def _patterns_from_edit(
    edit_record: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    category: str | None,
    task: str,
    draft_path: str | Path | None,
    edited_path: str | Path | None,
) -> list[dict[str, Any]]:
    timestamp = str(edit_record.get("timestamp") or "")
    actor = edit_record.get("actor") if isinstance(edit_record.get("actor"), Mapping) else {}
    actor_id = str(actor.get("id") or edit_record.get("actor_id") or "default-operator")
    intent = str(edit_record.get("intent") or "preference")
    edit_signature = _edit_signature(edit_record)
    normalized_category = _normalize_category(category)
    base_terms = _context_terms(edit_record, analysis, normalized_category, task)
    source = {
        "draft_path": str(draft_path or ""),
        "edited_path": str(edited_path or ""),
    }

    patterns: list[dict[str, Any]] = []
    section_labels = analysis.get("section_labels") if isinstance(analysis.get("section_labels"), Mapping) else {}
    for label in section_labels.get("added", []) or []:
        cleaned = _clean_label(str(label))
        if cleaned:
            patterns.append(
                _pattern(
                    kind="section_add",
                    suggestion=f"Consider adding a '{cleaned}' section when this task needs the same review shape.",
                    rationale="learned from an operator-added section",
                    trigger_terms=base_terms | _token_set(cleaned),
                    timestamp=timestamp,
                    actor_id=actor_id,
                    intent=intent,
                    category=normalized_category,
                    edit_signature=edit_signature,
                    source=source,
                )
            )
    for label in section_labels.get("removed", []) or []:
        cleaned = _clean_label(str(label))
        if cleaned:
            patterns.append(
                _pattern(
                    kind="section_avoid",
                    suggestion=f"Avoid a '{cleaned}' section unless the task specifically asks for it.",
                    rationale="learned from an operator-removed section",
                    trigger_terms=base_terms | _token_set(cleaned),
                    timestamp=timestamp,
                    actor_id=actor_id,
                    intent=intent,
                    category=normalized_category,
                    edit_signature=edit_signature,
                    source=source,
                )
            )

    for phrase in analysis.get("preferred_phrases", []) or []:
        cleaned = _normalize_space(str(phrase))
        if cleaned:
            patterns.append(
                _pattern(
                    kind="preferred_phrase",
                    suggestion=f"Prefer phrasing like '{cleaned}'.",
                    rationale="learned from operator wording",
                    trigger_terms=base_terms | _token_set(cleaned),
                    timestamp=timestamp,
                    actor_id=actor_id,
                    intent=intent,
                    category=normalized_category,
                    edit_signature=edit_signature,
                    source=source,
                )
            )

    caution = analysis.get("caution_uncertainty_tone")
    if isinstance(caution, Mapping) and int(caution.get("score_delta", 0) or 0) > 0:
        markers = [str(item) for item in caution.get("added_markers", []) or [] if str(item).strip()]
        marker_text = ", ".join(markers[:5]) if markers else "evidence-bound qualifiers"
        patterns.append(
            _pattern(
                kind="caution_tone",
                suggestion=f"Use cautious, evidence-bound language with markers such as {marker_text}.",
                rationale="operator added uncertainty qualifiers",
                trigger_terms=base_terms | _token_set(" ".join(markers)) | {"risk", "unclear", "evidence"},
                timestamp=timestamp,
                actor_id=actor_id,
                intent=intent,
                category=normalized_category,
                edit_signature=edit_signature,
                source=source,
            )
        )

    for correction in analysis.get("field_corrections", []) or []:
        if not isinstance(correction, Mapping):
            continue
        field = _clean_label(str(correction.get("field") or ""))
        old_value = _normalize_space(str(correction.get("from") or ""))
        new_value = _normalize_space(str(correction.get("to") or ""))
        if field and old_value and new_value:
            patterns.append(
                _pattern(
                    kind="field_rewrite",
                    suggestion=f"For {field}, prefer '{new_value}' over '{old_value}' when the context matches.",
                    rationale="learned from an operator field correction",
                    trigger_terms=base_terms | _token_set(" ".join([field, old_value, new_value])),
                    timestamp=timestamp,
                    actor_id=actor_id,
                    intent=intent,
                    category=normalized_category,
                    edit_signature=edit_signature,
                    source=source,
                )
            )
    return patterns


def _pattern(
    *,
    kind: str,
    suggestion: str,
    rationale: str,
    trigger_terms: set[str],
    timestamp: str,
    actor_id: str,
    intent: str,
    category: str,
    edit_signature: str,
    source: Mapping[str, str],
) -> dict[str, Any]:
    normalized_suggestion = _normalize_space(suggestion)
    pattern_id = _pattern_id(kind, normalized_suggestion, category)
    return {
        "pattern_id": pattern_id,
        "kind": kind,
        "suggestion": normalized_suggestion,
        "rationale": _normalize_space(rationale),
        "trigger_terms": sorted(term for term in trigger_terms if term),
        "category": category,
        "intent": intent,
        "count": 1,
        "first_seen_at": timestamp,
        "last_seen_at": timestamp,
        "actor_ids": [actor_id] if actor_id else [],
        "source_edit_signatures": [edit_signature],
        "source": dict(source),
    }


def _normalize_pattern(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    suggestion = _normalize_space(str(raw.get("suggestion") or ""))
    kind = _normalize_space(str(raw.get("kind") or "preference"))
    category = _normalize_category(raw.get("category"))
    if not suggestion:
        return None
    pattern_id = str(raw.get("pattern_id") or _pattern_id(kind, suggestion, category))
    try:
        count = max(1, int(raw.get("count", 1)))
    except (TypeError, ValueError):
        count = 1
    actor_ids = _dedupe_strings(str(item) for item in raw.get("actor_ids", []) or [] if str(item).strip())
    source_signatures = _dedupe_strings(
        str(item) for item in raw.get("source_edit_signatures", []) or [] if str(item).strip()
    )
    trigger_terms = _dedupe_strings(
        str(item).casefold()
        for item in raw.get("trigger_terms", []) or []
        if str(item).strip()
    )
    source = raw.get("source") if isinstance(raw.get("source"), Mapping) else {}
    return {
        "pattern_id": pattern_id,
        "kind": kind,
        "suggestion": suggestion,
        "rationale": _normalize_space(str(raw.get("rationale") or "")),
        "trigger_terms": trigger_terms,
        "category": category,
        "intent": str(raw.get("intent") or "preference"),
        "count": count,
        "first_seen_at": str(raw.get("first_seen_at") or raw.get("last_seen_at") or ""),
        "last_seen_at": str(raw.get("last_seen_at") or raw.get("first_seen_at") or ""),
        "actor_ids": actor_ids,
        "source_edit_signatures": source_signatures,
        "source": {
            "draft_path": str(source.get("draft_path") or ""),
            "edited_path": str(source.get("edited_path") or ""),
        },
    }


def _context_terms(
    edit_record: Mapping[str, Any],
    analysis: Mapping[str, Any],
    category: str,
    task: str = "",
) -> set[str]:
    terms = _token_set(
        " ".join(
            [
                str(edit_record.get("operator_note") or ""),
                task,
                str(edit_record.get("original_draft") or "")[:1200],
                str(edit_record.get("edited_draft") or "")[:1200],
                category,
            ]
        )
    )
    section_labels = analysis.get("section_labels") if isinstance(analysis.get("section_labels"), Mapping) else {}
    terms.update(_token_set(" ".join(str(item) for item in section_labels.get("added", []) or [])))
    terms.update(_token_set(" ".join(str(item) for item in section_labels.get("removed", []) or [])))
    return terms


def _section_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip(" #:-*\t")
        if stripped and len(stripped) <= 80:
            terms.update(_token_set(stripped))
    return terms


def _token_set(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(text or "") if len(token) > 2}


def _pattern_id(kind: str, suggestion: str, category: str) -> str:
    payload = {
        "kind": kind,
        "suggestion": suggestion.casefold(),
        "category": category,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _edit_signature(edit_record: Mapping[str, Any]) -> str:
    existing = str(edit_record.get("edit_signature") or "")
    if existing:
        return existing
    payload = {
        "draft_type": str(edit_record.get("draft_type") or "case_fact_summary"),
        "original_draft": str(edit_record.get("original_draft") or ""),
        "edited_draft": str(edit_record.get("edited_draft") or ""),
        "evidence_ids": sorted(str(item) for item in edit_record.get("evidence_ids", []) or []),
        "operator_note": str(edit_record.get("operator_note") or ""),
        "intent": str(edit_record.get("intent") or "preference"),
        "actor": edit_record.get("actor", {}),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_category(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def _clean_label(value: str) -> str:
    cleaned = _normalize_space(value.strip(" #*-"))
    return re.sub(r"^\d+[\).]\s*", "", cleaned)


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _dedupe_strings(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _normalize_space(str(value))
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result
