"""Learn reusable drafting preferences from operator-edited legal drafts."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.learning.edit_memory import (
    load_edit_memory,
    save_edit_memory,
    update_edit_memory,
)
from pipeline.learning.classification import classify_edit, classification_to_jsonable
from pipeline.schemas import EditRecord


@dataclass(frozen=True, slots=True)
class CaptureResult:
    """Structured return value for :func:`capture_operator_edit_from_files`.

    Returning the captured event + classification + edit-memory delta inline
    lets HTTP callers report exactly what happened without re-reading the
    append-only JSONL logs, which is unsafe under concurrent edits.
    """

    profile: dict[str, Any]
    event: dict[str, Any]
    classification: dict[str, Any]
    edit_signature: str
    edit_memory_patterns_before: int
    edit_memory_patterns_after: int
    suggestions: list[dict[str, Any]] = field(default_factory=list)


PROFILE_VERSION = 1
KNOWLEDGE_LAYER_VERSION = 1
RETRIEVAL_FEEDBACK_VERSION = 1
EXEMPLAR_STORE_VERSION = 1

# Five substantive edit kinds plus the `auto` router label. The five split
# along an in-envelope vs out-of-envelope axis the drafter respects:
#
# in-envelope (changes the drafter's persona, not facts):
#   - preference        : section labels, tone, prose style, phrasing
#   - field_correction  : operator-corrected extracted fields
#                         (jurisdiction-class labels persist as knowledge rules;
#                          case-specific corrections stay advisory)
#
# out-of-envelope (changes facts/retrieval, not persona):
#   - knowledge_rule    : extrinsic hard constraints
#                         ("this client's contracts default to NY law")
#   - evidence_dispute  : "wrong evidence cited" or "missing evidence" →
#                         retrieval feedback (boosts / demotes / gold flags)
#
# audit only:
#   - rejection         : wholesale rejection — too coarse to auto-learn safely
EDIT_INTENTS = {
    "preference",
    "field_correction",
    "knowledge_rule",
    "evidence_dispute",
    "rejection",
}
INTENT_CHOICES = tuple(sorted(EDIT_INTENTS | {"auto"}))
IN_ENVELOPE_INTENTS = {"preference", "field_correction"}
OUT_OF_ENVELOPE_INTENTS = {"knowledge_rule", "evidence_dispute"}
ACTOR_TYPES = {"human", "agent"}
DEFAULT_ACTOR_ID = "default-operator"
DEFAULT_STATE_DIR = Path("state")
REQUIRED_SECTION_LABELS = {
    "Evidence Index",
    "Evidence Reviewed",
    "Issue",
    "Recommended Next Steps",
    "Review Flags",
    "Risk Notes",
    "Unsupported or Unclear Facts",
    "Items Requiring Verification",
}

CAUTION_MARKERS = (
    "appears",
    "may",
    "might",
    "could",
    "likely",
    "unlikely",
    "possible",
    "potential",
    "alleged",
    "allegedly",
    "unclear",
    "unless",
    "assuming",
    "subject to",
    "based on",
    "available evidence",
    "provided evidence",
    "provided records",
    "cannot confirm",
    "should verify",
    "verify",
)

SECTION_LABEL_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<label>.+?)\s*$")
FIELD_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?P<label>[A-Za-z][A-Za-z0-9 /&().'-]{1,60})\s*:\s*(?P<value>.+?)\s*$"
)
QUOTED_PHRASE_RE = re.compile(r"'([^']{3,100})'|\"([^\"]{3,100})\"")
REUSABLE_FIELD_CORRECTION_LABELS = {
    "court",
    "forum",
    "governing law",
    "jurisdiction",
    "state",
    "venue",
}
EVIDENCE_DISPUTE_HINT_RE = re.compile(
    r"\b("
    r"wrong citation|bad citation|missing evidence|wrong evidence|evidence dispute|"
    r"unsupported claim|unsupported fact|wrong page|citation mismatch|retrieval miss"
    r")\b",
    re.IGNORECASE,
)
KNOWLEDGE_RULE_HINT_RE = re.compile(
    r"\b("
    r"always|never|default to|standing rule|house style|firm policy|"
    r"client preference|matter type|jurisdiction default|governing law default"
    r")\b",
    re.IGNORECASE,
)
REJECTION_HINT_RE = re.compile(
    r"\b(reject|rejected|redo|rewrite|start over|not usable|unusable)\b",
    re.IGNORECASE,
)


def create_profile() -> dict[str, Any]:
    """Return an empty learning profile."""

    return {
        "version": PROFILE_VERSION,
        "examples": [],
        "signals": {
            "section_labels": {"added": {}, "removed": {}},
            "preferred_phrases": {},
            "caution_uncertainty_tone": {
                "markers": {},
                "edits_with_added_caution": 0,
            },
            "field_corrections": [],
        },
    }


def capture_operator_edit(
    original_draft: str,
    edited_draft: str,
    evidence_ids: Iterable[str],
    operator_note: str = "",
    timestamp: str | datetime | None = None,
    *,
    intent: str = "preference",
    actor_type: str = "human",
    actor_id: str = DEFAULT_ACTOR_ID,
) -> dict[str, Any]:
    """Capture one operator edit event as JSON-serializable data."""

    original = _require_text(original_draft, "original_draft")
    edited = _require_text(edited_draft, "edited_draft")
    note = str(operator_note or "")
    record = EditRecord(
        draft_id="",
        draft_type="internal_memo",
        original_text=original,
        edited_text=edited,
        evidence_ids=[str(evidence_id) for evidence_id in evidence_ids],
        operator_note=note,
        intent=_resolve_intent(intent, original, edited, note),
        actor_type=_normalize_actor_type(actor_type),
        actor_id=_normalize_actor_id(actor_id),
        created_at=_format_timestamp(timestamp),
    )
    return record.to_learning_dict()


def analyze_edit(edit_record: Mapping[str, Any]) -> dict[str, Any]:
    """Compute reusable signals from a captured edit record."""

    original = _require_text(edit_record.get("original_draft"), "original_draft")
    edited = _require_text(edit_record.get("edited_draft"), "edited_draft")
    operator_note = str(edit_record.get("operator_note") or "")

    original_sections = _section_labels(original)
    edited_sections = _section_labels(edited)
    field_corrections = _field_corrections(original, edited)
    caution = _caution_signal(original, edited)

    return {
        "section_labels": {
            "added": sorted(edited_sections - original_sections),
            "removed": sorted(original_sections - edited_sections),
        },
        "preferred_phrases": _preferred_phrases(original, edited, operator_note),
        "caution_uncertainty_tone": caution,
        "field_corrections": field_corrections,
    }


def learn_from_edit(
    original_draft: str,
    edited_draft: str,
    evidence_ids: Iterable[str],
    operator_note: str = "",
    timestamp: str | datetime | None = None,
    profile: Mapping[str, Any] | None = None,
    *,
    intent: str = "preference",
    actor_type: str = "human",
    actor_id: str = DEFAULT_ACTOR_ID,
) -> dict[str, Any]:
    """Capture an edit and return an updated learning profile."""

    edit_record = capture_operator_edit(
        original_draft=original_draft,
        edited_draft=edited_draft,
        evidence_ids=evidence_ids,
        operator_note=operator_note,
        timestamp=timestamp,
        intent=intent,
        actor_type=actor_type,
        actor_id=actor_id,
    )
    return update_profile(profile or create_profile(), edit_record)


def update_profile(
    profile: Mapping[str, Any] | None,
    edit_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copy of ``profile`` updated with one captured edit record."""

    next_profile = _normalized_profile(profile)
    captured = capture_operator_edit(
        original_draft=str(edit_record.get("original_draft", "")),
        edited_draft=str(edit_record.get("edited_draft", "")),
        evidence_ids=edit_record.get("evidence_ids", []),
        operator_note=str(edit_record.get("operator_note") or ""),
        timestamp=edit_record.get("timestamp"),
        intent=str(edit_record.get("intent") or "preference"),
        actor_type=_actor_type_from_record(edit_record),
        actor_id=_actor_id_from_record(edit_record),
    )
    captured["edit_signature"] = _edit_signature(captured)
    if captured["intent"] != "preference" or captured["actor"]["type"] != "human":
        return next_profile
    for existing in next_profile["examples"]:
        if _edit_signature(existing) == captured["edit_signature"]:
            return next_profile

    analysis = analyze_edit(captured)

    next_profile["examples"].append(captured)
    section_labels = next_profile["signals"]["section_labels"]
    for label in analysis["section_labels"]["added"]:
        _increment(section_labels["added"], label)
    for label in analysis["section_labels"]["removed"]:
        _increment(section_labels["removed"], label)

    for phrase in analysis["preferred_phrases"]:
        _increment(next_profile["signals"]["preferred_phrases"], phrase)

    caution = analysis["caution_uncertainty_tone"]
    if caution["score_delta"] > 0:
        next_profile["signals"]["caution_uncertainty_tone"][
            "edits_with_added_caution"
        ] += 1
    for marker in caution["added_markers"]:
        _increment(next_profile["signals"]["caution_uncertainty_tone"]["markers"], marker)

    for correction in analysis["field_corrections"]:
        _merge_field_correction(
            next_profile["signals"]["field_corrections"],
            correction["field"],
            correction["from"],
            correction["to"],
        )

    return next_profile


def save_profile(profile: Mapping[str, Any], path: str | Path) -> None:
    """Persist a learning profile as stable, human-readable JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_normalized_profile(profile), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_profile(path: str | Path) -> dict[str, Any]:
    """Load a learning profile from JSON."""

    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    return _normalized_profile(data)


# ---------------------------------------------------------------------------
# Channel state: typed-edit routing persists to four files under state/.
# `state_paths()` returns the canonical layout; callers can override per
# channel (for example, outputs/<case>/operator_profile.json artifacts
# still work — the channel I/O is path-driven, not directory-magic).
# ---------------------------------------------------------------------------


def state_paths(state_dir: str | Path | None = None) -> dict[str, Path]:
    """Resolve typed-routing channel file paths under state_dir.

    Channels:
      - operator_profile   in-env: drafter persona (style/section/tone)
      - knowledge_layer    out-env: extrinsic hard constraints
      - retrieval_feedback out-env: evidence boosts / demotes / gold flags
      - exemplars_dir      in-env: per-category few-shot exemplar store
      - edit_memory        in-env: retrievable operator edit patterns
    """

    root = Path(state_dir or DEFAULT_STATE_DIR)
    return {
        "state_dir": root,
        "operator_profile": root / "operator_profile.json",
        "knowledge_layer": root / "knowledge_layer.json",
        "retrieval_feedback": root / "retrieval_feedback.json",
        "exemplars_dir": root / "exemplars",
        "edit_memory": root / "edit_memory.json",
        "edit_classifications": root / "edit_classifications.jsonl",
    }


def create_knowledge_layer() -> dict[str, Any]:
    """Return an empty knowledge-layer state."""

    return {
        "version": KNOWLEDGE_LAYER_VERSION,
        "rules": [],
        "field_defaults": {},
    }


def load_knowledge_layer(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return create_knowledge_layer()
    with file.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return _normalized_knowledge_layer(data)


def save_knowledge_layer(layer: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_normalized_knowledge_layer(layer), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def update_knowledge_layer(
    layer: Mapping[str, Any] | None,
    edit_record: Mapping[str, Any],
    *,
    analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one captured edit to the knowledge layer.

    Two write paths:
      1. ``knowledge_rule`` intent: the operator note becomes a hard rule.
      2. ``field_correction`` intent with reusable label (jurisdiction etc.):
         the corrected value becomes a field default under the same label.
    """

    state = _normalized_knowledge_layer(layer)
    intent = str(edit_record.get("intent") or "")
    timestamp = str(edit_record.get("timestamp") or "")
    actor = edit_record.get("actor") if isinstance(edit_record.get("actor"), Mapping) else {}
    actor_id = str(actor.get("id") or DEFAULT_ACTOR_ID)
    note = str(edit_record.get("operator_note") or "").strip()

    if intent == "knowledge_rule" and note:
        signature = hashlib.sha256(note.casefold().encode("utf-8")).hexdigest()[:16]
        existing = next(
            (rule for rule in state["rules"] if rule.get("rule_signature") == signature),
            None,
        )
        if existing is None:
            state["rules"].append(
                {
                    "rule_signature": signature,
                    "rule": note,
                    "added_at": timestamp,
                    "actor_id": actor_id,
                    "evidence_ids": [str(eid) for eid in edit_record.get("evidence_ids", [])],
                }
            )

    if intent == "field_correction":
        analysis_data = analysis or analyze_edit(edit_record)
        for correction in analysis_data.get("field_corrections", []):
            field = _clean_label(str(correction.get("field", "")))
            value = str(correction.get("to", "")).strip()
            if not field or not value:
                continue
            if field.casefold() not in REUSABLE_FIELD_CORRECTION_LABELS:
                continue
            state["field_defaults"][field] = {
                "value": value,
                "updated_at": timestamp,
                "actor_id": actor_id,
            }

    return state


def create_retrieval_feedback() -> dict[str, Any]:
    """Return an empty retrieval-feedback state."""

    return {
        "version": RETRIEVAL_FEEDBACK_VERSION,
        "disputes": [],
        "evidence_boosts": {},
        "evidence_demotes": {},
    }


def load_retrieval_feedback(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return create_retrieval_feedback()
    with file.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return _normalized_retrieval_feedback(data)


def save_retrieval_feedback(feedback: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_normalized_retrieval_feedback(feedback), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def update_retrieval_feedback(
    feedback: Mapping[str, Any] | None,
    edit_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply one captured evidence_dispute edit to retrieval feedback.

    The cited evidence ids in the original draft are recorded as ids the
    operator considered wrong (demote candidates). The dispute is also kept
    as a queryable record so the operator's note becomes a candidate gold
    query for retrieval evaluation.
    """

    state = _normalized_retrieval_feedback(feedback)
    if str(edit_record.get("intent") or "") != "evidence_dispute":
        return state
    note = str(edit_record.get("operator_note") or "").strip()
    timestamp = str(edit_record.get("timestamp") or "")
    actor = edit_record.get("actor") if isinstance(edit_record.get("actor"), Mapping) else {}
    actor_id = str(actor.get("id") or DEFAULT_ACTOR_ID)
    evidence_ids = [str(eid).strip() for eid in edit_record.get("evidence_ids", []) if str(eid).strip()]

    dispute_signature = hashlib.sha256(
        (note.casefold() + "|" + ",".join(sorted(evidence_ids))).encode("utf-8")
    ).hexdigest()[:16]
    existing = next(
        (dispute for dispute in state["disputes"] if dispute.get("dispute_signature") == dispute_signature),
        None,
    )
    if existing is None:
        state["disputes"].append(
            {
                "dispute_signature": dispute_signature,
                "query": note or "evidence dispute (no operator note)",
                "disputed_evidence_ids": evidence_ids,
                "added_at": timestamp,
                "actor_id": actor_id,
            }
        )
    for eid in evidence_ids:
        state["evidence_demotes"][eid] = state["evidence_demotes"].get(eid, 0) + 1
    return state


def mine_classified_retrieval_feedback(
    feedback: Mapping[str, Any] | None,
    edit_record: Mapping[str, Any],
    classification: Any,
) -> dict[str, Any]:
    """Apply classifier-routed retrieval feedback without reclassifying intent."""

    state = _normalized_retrieval_feedback(feedback)
    evidence_ids = [str(eid).strip() for eid in edit_record.get("evidence_ids", []) if str(eid).strip()]
    if not evidence_ids:
        return state
    edit_type = str(getattr(classification, "edit_type", "") or "").strip()
    target = state["evidence_boosts"] if edit_type in {"missing_fact_added", "citation_correction"} else state["evidence_demotes"]
    for evidence_id in evidence_ids:
        target[evidence_id] = target.get(evidence_id, 0) + 1
    note = str(edit_record.get("operator_note") or "").strip()
    signature = hashlib.sha256(
        ("classification|" + edit_type + "|" + ",".join(sorted(evidence_ids)) + "|" + note.casefold()).encode("utf-8")
    ).hexdigest()[:16]
    if not any(dispute.get("dispute_signature") == signature for dispute in state["disputes"]):
        state["disputes"].append(
            {
                "dispute_signature": signature,
                "query": note or str(getattr(classification, "reusable_rule", "") or "classified retrieval feedback"),
                "disputed_evidence_ids": evidence_ids,
                "added_at": str(edit_record.get("timestamp") or ""),
                "actor_id": _actor_id_from_record(edit_record),
            }
        )
    return state


def append_exemplar(
    exemplars_dir: str | Path,
    *,
    category: str,
    edit_record: Mapping[str, Any],
    analysis: Mapping[str, Any] | None = None,
) -> Path | None:
    """Persist a per-category few-shot exemplar derived from a preference edit.

    The exemplar is a small JSON object — not the full edit — designed to be
    cheap to retrieve and prepend at draft time. At most one exemplar per
    edit signature per category is stored (idempotent on re-runs).
    """

    if str(edit_record.get("intent") or "") != "preference":
        return None
    normalized_category = _normalize_category(category)
    if not normalized_category:
        return None
    file = Path(exemplars_dir) / f"{normalized_category}.jsonl"
    file.parent.mkdir(parents=True, exist_ok=True)
    signature = _edit_signature(edit_record)
    if file.exists():
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if existing.get("edit_signature") == signature:
                    return file

    analysis_data = analysis or analyze_edit(edit_record)
    diff_stats = _diff_stats(
        str(edit_record.get("original_draft", "")),
        str(edit_record.get("edited_draft", "")),
    )
    entry = {
        "version": EXEMPLAR_STORE_VERSION,
        "edit_signature": signature,
        "category": normalized_category,
        "added_at": str(edit_record.get("timestamp") or ""),
        "operator_note": str(edit_record.get("operator_note") or ""),
        "added_section_labels": list(analysis_data.get("section_labels", {}).get("added", [])),
        "removed_section_labels": list(analysis_data.get("section_labels", {}).get("removed", [])),
        "preferred_phrases": [phrase for phrase in analysis_data.get("preferred_phrases", [])][:3],
        "diff_stats": diff_stats,
        # Short snippet of the edited draft head so a downstream consumer can
        # see the operator's preferred shape without re-reading the full file.
        "edited_head": _truncate_text(
            str(edit_record.get("edited_draft", "")).strip(), limit=400
        ),
    }
    with file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
    return file


def load_exemplars_for_category(
    exemplars_dir: str | Path, category: str, *, limit: int = 1
) -> list[dict[str, Any]]:
    normalized_category = _normalize_category(category)
    file = Path(exemplars_dir) / f"{normalized_category}.jsonl"
    if not normalized_category or not file.exists():
        return []
    entries: list[dict[str, Any]] = []
    with file.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    entries.sort(key=lambda item: str(item.get("added_at") or ""), reverse=True)
    return entries[:limit]


def render_exemplars_guidance(exemplars: list[Mapping[str, Any]]) -> str:
    """Render exemplar entries as a short few-shot block for the drafting prompt."""

    if not exemplars:
        return ""
    lines = [
        "Past operator-edited memos for similar documents (for tone and structure reference):"
    ]
    for index, exemplar in enumerate(exemplars, start=1):
        category = str(exemplar.get("category") or "unknown")
        added = list(exemplar.get("added_section_labels") or [])
        removed = list(exemplar.get("removed_section_labels") or [])
        head = str(exemplar.get("edited_head") or "").strip()
        lines.append(f"Example {index} (category: {category}):")
        if added:
            lines.append(f"  Operator added sections: {', '.join(added[:4])}")
        if removed:
            lines.append(f"  Operator removed sections: {', '.join(removed[:4])}")
        if head:
            lines.append(f"  Opening of operator-edited memo: {head[:300]}")
    lines.append(
        "Adapt structure and tone to match these prior operator edits where "
        "appropriate, but never relax the citation and grounding requirements."
    )
    return "\n".join(lines)


def render_knowledge_layer_guidance(layer: Mapping[str, Any]) -> str:
    """Render knowledge-layer state as a short hard-constraint block."""

    state = _normalized_knowledge_layer(layer)
    rules = [str(rule.get("rule") or "").strip() for rule in state.get("rules", [])]
    rules = [rule for rule in rules if rule]
    defaults = state.get("field_defaults", {})
    if not rules and not defaults:
        return ""
    lines = ["Operator-defined knowledge rules (treat as hard constraints):"]
    for rule in rules[:6]:
        lines.append(f"- {rule}")
    if defaults:
        for field, payload in sorted(defaults.items()):
            value = payload.get("value", "") if isinstance(payload, Mapping) else ""
            if value:
                lines.append(f"- Default for {field}: {value}")
    return "\n".join(lines)


def capture_operator_edit_from_files(
    draft_path: str | Path,
    edited_path: str | Path,
    profile_path: str | Path,
    *,
    operator_note: str = "",
    intent: str = "preference",
    actor_type: str = "human",
    actor_id: str = DEFAULT_ACTOR_ID,
    event_log_path: str | Path | None = None,
    suggestions_path: str | Path | None = None,
    state_dir: str | Path | None = None,
    category: str | None = None,
) -> CaptureResult:
    """Capture an operator edit and route it through the five-intent channel set.

    Returns a :class:`CaptureResult` with the updated profile plus the
    exact event + classification this call captured and the
    edit-memory pattern count before/after. Callers that need the data
    they wrote (e.g., the HTTP endpoint surfacing what changed) should
    use these values directly — never re-read ``edits.jsonl`` or
    ``edit_classifications.jsonl`` to attribute the event, because
    concurrent edits would race.

    Channels (selected by the resolved intent at capture time):

      preference        → operator_profile.json + per-category exemplar store
      field_correction  → knowledge_layer.json (reusable labels) or advisory
      knowledge_rule    → knowledge_layer.json
      evidence_dispute  → retrieval_feedback.json + learn_suggestions.json
      rejection         → audit log only (edits.jsonl)

    `edits.jsonl` is always appended — it is the source-of-truth event log
    regardless of which downstream channels updated.

    Compatibility: if ``state_dir`` is omitted, channel paths default to the
    directory containing ``profile_path`` so existing callers (and tests
    that pass only ``profile_path``) continue to work. When ``state_dir`` is
    given, the knowledge layer / retrieval feedback / exemplars are written
    under it regardless of profile_path; this is the recommended layout.
    """

    profile_file = Path(profile_path)
    draft_file = Path(draft_path)
    edited_file = Path(edited_path)
    original = draft_file.read_text(encoding="utf-8")
    edited = edited_file.read_text(encoding="utf-8")
    evidence_ids = _citation_ids_from_text(original)
    edit_record = capture_operator_edit(
        original_draft=original,
        edited_draft=edited,
        evidence_ids=evidence_ids,
        operator_note=operator_note,
        intent=intent,
        actor_type=actor_type,
        actor_id=actor_id,
    )
    analysis = analyze_edit(edit_record)
    classification = classify_edit(
        draft_type="case_fact_summary",
        original_text=original,
        edited_text=edited,
        operator_note=operator_note,
        evidence_ids=evidence_ids,
    )
    classification_payload = classification_to_jsonable(classification)
    edit_record["classification"] = classification_payload

    paths = state_paths(state_dir) if state_dir is not None else None
    knowledge_path = paths["knowledge_layer"] if paths else profile_file.parent / "knowledge_layer.json"
    retrieval_path = paths["retrieval_feedback"] if paths else profile_file.parent / "retrieval_feedback.json"
    exemplars_root = paths["exemplars_dir"] if paths else profile_file.parent / "exemplars"
    edit_memory_path = paths["edit_memory"] if paths else profile_file.parent / "edit_memory.json"
    classifications_path = paths["edit_classifications"] if paths else profile_file.parent / "edit_classifications.jsonl"

    edit_signature = _edit_signature(edit_record)

    event = _edit_event_log_entry(
        edit_record,
        analysis=analysis,
        draft_path=draft_file,
        edited_path=edited_file,
        profile_path=profile_file,
    )
    _append_jsonl(
        Path(event_log_path) if event_log_path else profile_file.parent / "edits.jsonl",
        event,
    )
    _append_jsonl(
        classifications_path,
        {
            "timestamp": edit_record["timestamp"],
            "edit_signature": edit_signature,
            "classification": classification_payload,
            "draft_path": str(draft_file),
            "edited_path": str(edited_file),
        },
    )

    routing = event.get("routing", {})
    channel = routing.get("channel")

    # Edit-memory delta is computed inline so the return value carries it
    # without forcing the caller to read edit_memory.json before/after.
    memory_before = 0
    if edit_memory_path.exists():
        try:
            memory_before = len((load_edit_memory(edit_memory_path).get("patterns") or []))
        except (OSError, ValueError):
            memory_before = 0

    suggestions_list: list[dict[str, Any]] = []

    # Audit-only intents (rejection, agent edits) skip all channel writes.
    if channel == "audit_log" or routing.get("envelope") == "advisory":
        profile = load_profile(profile_file) if profile_file.exists() else create_profile()
        return CaptureResult(
            profile=profile,
            event=event,
            classification=classification_payload,
            edit_signature=edit_signature,
            edit_memory_patterns_before=memory_before,
            edit_memory_patterns_after=memory_before,
            suggestions=suggestions_list,
        )

    if channel == "knowledge_layer":
        layer = load_knowledge_layer(knowledge_path)
        layer = update_knowledge_layer(layer, edit_record, analysis=analysis)
        save_knowledge_layer(layer, knowledge_path)

    if channel == "retrieval_feedback":
        feedback = load_retrieval_feedback(retrieval_path)
        feedback = update_retrieval_feedback(feedback, edit_record)
        save_retrieval_feedback(feedback, retrieval_path)

    if "retrieval_feedback" in classification.target_mechanisms and channel != "retrieval_feedback":
        feedback = load_retrieval_feedback(retrieval_path)
        feedback = mine_classified_retrieval_feedback(feedback, edit_record, classification)
        save_retrieval_feedback(feedback, retrieval_path)

    if str(edit_record.get("intent") or "") in {"preference", "field_correction"}:
        memory = load_edit_memory(edit_memory_path)
        memory = update_edit_memory(
            memory,
            edit_record,
            analysis=analysis,
            category=category,
            draft_path=draft_file,
            edited_path=edited_file,
        )
        if memory.get("patterns"):
            save_edit_memory(memory, edit_memory_path)

    suggestions_list = _learning_suggestions(
        edit_record,
        analysis=analysis,
        draft_path=draft_file,
        edited_path=edited_file,
    )
    if suggestions_list:
        _write_suggestions(
            Path(suggestions_path) if suggestions_path else profile_file.parent / "learn_suggestions.json",
            suggestions_list,
        )

    memory_after = memory_before
    if edit_memory_path.exists():
        try:
            memory_after = len((load_edit_memory(edit_memory_path).get("patterns") or []))
        except (OSError, ValueError):
            memory_after = memory_before

    profile = load_profile(profile_file) if profile_file.exists() else create_profile()
    if channel != "operator_profile":
        return CaptureResult(
            profile=profile,
            event=event,
            classification=classification_payload,
            edit_signature=edit_signature,
            edit_memory_patterns_before=memory_before,
            edit_memory_patterns_after=memory_after,
            suggestions=list(suggestions_list),
        )

    updated = update_profile(profile, edit_record)
    save_profile(updated, profile_file)

    if category:
        append_exemplar(
            exemplars_root,
            category=category,
            edit_record=edit_record,
            analysis=analysis,
        )

    return CaptureResult(
        profile=updated,
        event=event,
        classification=classification_payload,
        edit_signature=edit_signature,
        edit_memory_patterns_before=memory_before,
        edit_memory_patterns_after=memory_after,
        suggestions=list(suggestions_list),
    )


def learn_from_files(
    draft_path: str | Path,
    edited_path: str | Path,
    profile_path: str | Path,
    *,
    operator_note: str = "",
    intent: str = "preference",
    actor_type: str = "human",
    actor_id: str = DEFAULT_ACTOR_ID,
    event_log_path: str | Path | None = None,
    suggestions_path: str | Path | None = None,
    state_dir: str | Path | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """Backwards-compatible wrapper that returns only the updated profile.

    New code (HTTP endpoint, anything that needs to report what happened
    in this exact call) should call :func:`capture_operator_edit_from_files`
    directly.
    """

    return capture_operator_edit_from_files(
        draft_path,
        edited_path,
        profile_path,
        operator_note=operator_note,
        intent=intent,
        actor_type=actor_type,
        actor_id=actor_id,
        event_log_path=event_log_path,
        suggestions_path=suggestions_path,
        state_dir=state_dir,
        category=category,
    ).profile


def build_learning_suggestions(
    edit_record: Mapping[str, Any],
    *,
    draft_path: str | Path = "",
    edited_path: str | Path = "",
) -> dict[str, Any]:
    """Return routed non-profile suggestions derived from an edit record."""

    return _learning_suggestions(
        edit_record,
        analysis=analyze_edit(edit_record),
        draft_path=Path(draft_path) if draft_path else None,
        edited_path=Path(edited_path) if edited_path else None,
    )


def apply_profile_to_prompt_or_policy(profile: Mapping[str, Any]) -> str:
    """Return concise drafting guidance derived from a learning profile."""

    profile_data = _normalized_profile(profile)
    signals = profile_data["signals"]
    guidance: list[str] = []

    added_sections = _top_items(signals["section_labels"]["added"], limit=4)
    removed_sections = [
        section
        for section in _top_items(signals["section_labels"]["removed"], limit=8)
        if section not in REQUIRED_SECTION_LABELS
    ][:4]
    if added_sections:
        guidance.append("Use these section labels when relevant: " + ", ".join(added_sections) + ".")
    if removed_sections:
        guidance.append("Avoid these section labels unless specifically requested: " + ", ".join(removed_sections) + ".")

    preferred_phrases = _top_items(signals["preferred_phrases"], limit=5)
    if preferred_phrases:
        phrase_text = "; ".join(phrase.rstrip(".") for phrase in preferred_phrases)
        guidance.append("Prefer concise phrasing such as: " + phrase_text + ".")

    caution = signals["caution_uncertainty_tone"]
    caution_markers = _top_items(caution["markers"], limit=5)
    if caution["edits_with_added_caution"] or caution_markers:
        marker_text = ", ".join(caution_markers) if caution_markers else "evidence-limited qualifiers"
        guidance.append(
            "Use cautious, evidence-bound language; qualify uncertainty with "
            + marker_text
            + "."
        )

    correction_guidance = _field_correction_guidance(
        signals["field_corrections"], limit=4
    )
    guidance.extend(correction_guidance)

    if not guidance:
        return "No reusable operator drafting preferences have been learned yet."

    # Non-negotiable counter-weight: a single terse-style edit can teach
    # "collapse + caution" so strongly that the next draft drops inline
    # citations from claims that were grounded in the baseline. The drafter
    # already enforces citation validity, but the guidance renderer is the
    # right place to remind the model that conciseness must not erode
    # per-claim grounding.
    guidance.append(
        "When consolidating or shortening sections, preserve or increase "
        "inline citation density: never drop a citation that supports a "
        "kept factual claim, and prefer adding citations to any new "
        "factual sentence introduced during consolidation."
    )
    return " ".join(guidance)


def _edit_event_log_entry(
    edit_record: Mapping[str, Any],
    *,
    analysis: Mapping[str, Any],
    draft_path: Path,
    edited_path: Path,
    profile_path: Path,
) -> dict[str, Any]:
    captured = capture_operator_edit(
        original_draft=str(edit_record.get("original_draft", "")),
        edited_draft=str(edit_record.get("edited_draft", "")),
        evidence_ids=edit_record.get("evidence_ids", []),
        operator_note=str(edit_record.get("operator_note") or ""),
        timestamp=edit_record.get("timestamp"),
        intent=str(edit_record.get("intent") or "preference"),
        actor_type=_actor_type_from_record(edit_record),
        actor_id=_actor_id_from_record(edit_record),
    )
    edit_signature = _edit_signature(captured)
    return {
        "event_id": _event_id(captured, edit_signature),
        "edit_signature": edit_signature,
        "timestamp": captured["timestamp"],
        "actor": captured["actor"],
        "intent": captured["intent"],
        "draft_path": str(draft_path),
        "edited_path": str(edited_path),
        "profile_path": str(profile_path),
        "evidence_ids": captured["evidence_ids"],
        "operator_note": captured["operator_note"],
        "diff_stats": _diff_stats(captured["original_draft"], captured["edited_draft"]),
        "analysis": {
            "section_labels": analysis.get("section_labels", {}),
            "preferred_phrase_count": len(analysis.get("preferred_phrases", [])),
            "caution_uncertainty_tone": analysis.get("caution_uncertainty_tone", {}),
            "field_correction_count": len(analysis.get("field_corrections", [])),
        },
        "classification": edit_record.get("classification", {}),
        "routing": _routing_for(captured),
    }


def _learning_suggestions(
    edit_record: Mapping[str, Any],
    *,
    analysis: Mapping[str, Any],
    draft_path: Path | None,
    edited_path: Path | None,
) -> dict[str, Any]:
    suggestions: dict[str, Any] = {}
    intent = str(edit_record.get("intent") or "preference")

    # Case-specific field corrections (not reusable labels) become gold-doc
    # suggestions regardless of intent — they're factual changes a reviewer
    # might want to promote into the evaluation gold set.
    factual_corrections = []
    for correction in analysis.get("field_corrections", []):
        if not isinstance(correction, Mapping):
            continue
        field = str(correction.get("field", "")).strip()
        if _clean_label(field).casefold() in REUSABLE_FIELD_CORRECTION_LABELS:
            continue
        factual_corrections.append(
            {
                "field": field,
                "previous_value": str(correction.get("from", "")),
                "corrected_value": str(correction.get("to", "")),
                "reason": "field correction is case-specific unless a human marks it reusable",
            }
        )
    if factual_corrections:
        suggestions["suggested_gold_documents"] = factual_corrections

    if intent == "evidence_dispute":
        suggestions["suggested_gold_retrieval"] = [
            {
                "query": str(edit_record.get("operator_note") or "operator disputed cited or missing evidence"),
                "evidence_ids_from_original_draft": list(edit_record.get("evidence_ids", [])),
                "reason": "evidence dispute should become retrieval gold or a case-level grounding annotation",
            }
        ]
    if intent == "knowledge_rule":
        suggestions["suggested_knowledge_rule"] = {
            "rule": str(edit_record.get("operator_note") or "").strip(),
            "reason": "operator-defined hard constraint; review for promotion or scoping before broad rollout",
        }
    if intent == "field_correction":
        suggestions["suggested_field_default_review"] = [
            {
                "field": str(correction.get("field", "")).strip(),
                "previous_value": str(correction.get("from", "")),
                "corrected_value": str(correction.get("to", "")),
                "reason": "reusable field correction stored as knowledge-layer default; confirm it should apply across cases",
            }
            for correction in analysis.get("field_corrections", [])
            if isinstance(correction, Mapping)
            and _clean_label(str(correction.get("field", ""))).casefold()
            in REUSABLE_FIELD_CORRECTION_LABELS
        ]
        if not suggestions["suggested_field_default_review"]:
            del suggestions["suggested_field_default_review"]
    if intent == "rejection":
        suggestions["triage"] = {
            "reason": str(edit_record.get("operator_note") or "operator rejected the draft"),
            "action": "review manually; do not auto-learn drafting preferences from this edit",
        }

    if not suggestions:
        return {}
    suggestions["generated_at"] = _format_timestamp(None)
    suggestions["draft_path"] = str(draft_path) if draft_path else ""
    suggestions["edited_path"] = str(edited_path) if edited_path else ""
    return suggestions


def _append_jsonl(path: Path, item: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")


def _write_suggestions(path: Path, suggestions: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(suggestions, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _routing_for(edit_record: Mapping[str, Any]) -> dict[str, Any]:
    actor = edit_record.get("actor", {})
    actor_type = actor.get("type") if isinstance(actor, Mapping) else ""
    intent = str(edit_record.get("intent") or "preference")
    if actor_type != "human":
        return {
            "envelope": "advisory",
            "channel": "advisory_review",
            "profile_update": False,
            "target": "advisory_review",
            "reason": "agent edits require human acceptance before learning",
        }
    if intent == "preference":
        return {
            "envelope": "in",
            "channel": "operator_profile",
            "profile_update": True,
            "target": "operator_profile",
            "reason": "prose/structure/tone preferences steer the drafter persona",
        }
    if intent == "field_correction":
        return {
            "envelope": "in",
            "channel": "knowledge_layer",
            "profile_update": False,
            "target": "knowledge_layer",
            "reason": "reusable field corrections become knowledge-layer defaults; case-specific ones become advisory gold suggestions",
        }
    if intent == "knowledge_rule":
        return {
            "envelope": "out",
            "channel": "knowledge_layer",
            "profile_update": False,
            "target": "knowledge_layer",
            "reason": "extrinsic hard constraints persist as knowledge-layer rules",
        }
    if intent == "evidence_dispute":
        return {
            "envelope": "out",
            "channel": "retrieval_feedback",
            "profile_update": False,
            "target": "retrieval_feedback",
            "reason": "evidence disputes feed retrieval feedback (demotes + gold queries)",
        }
    return {
        "envelope": "audit",
        "channel": "audit_log",
        "profile_update": False,
        "target": "human_triage",
        "reason": "draft rejections are too coarse to auto-learn safely",
    }


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _format_timestamp(timestamp: str | datetime | None) -> str:
    if timestamp is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    if isinstance(timestamp, datetime):
        value = timestamp
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat(timespec="seconds")
    if isinstance(timestamp, str) and timestamp.strip():
        return timestamp
    raise ValueError("timestamp must be a datetime, a non-empty string, or None")


def _resolve_intent(intent: str, original: str, edited: str, operator_note: str) -> str:
    value = _normalize_space(str(intent or "preference")).casefold()
    if value == "auto":
        return _infer_intent(original, edited, operator_note)
    if value not in EDIT_INTENTS:
        raise ValueError(
            "intent must be one of " + ", ".join(sorted(EDIT_INTENTS | {"auto"}))
        )
    return value


def _infer_intent(original: str, edited: str, operator_note: str) -> str:
    note = operator_note or ""
    if REJECTION_HINT_RE.search(note):
        return "rejection"
    if EVIDENCE_DISPUTE_HINT_RE.search(note):
        return "evidence_dispute"
    if KNOWLEDGE_RULE_HINT_RE.search(note):
        return "knowledge_rule"
    # Field-correction inference: if the only meaningful diff is in
    # `Label: value` lines for any of the persistable knowledge labels
    # (jurisdiction, governing law, court, etc.), route as field_correction
    # so the knowledge layer captures it rather than the prose profile.
    corrections = _field_corrections(original, edited)
    if corrections and any(
        _clean_label(correction.get("field", "")).casefold()
        in REUSABLE_FIELD_CORRECTION_LABELS
        for correction in corrections
    ):
        # Only label as field_correction if the prose body otherwise barely
        # changed; otherwise the edit is doing more than just correcting a
        # field and the prose changes should still feed the profile.
        stats = _diff_stats(original, edited)
        if float(stats["word_jaccard_similarity"]) >= 0.7:
            return "field_correction"
    stats = _diff_stats(original, edited)
    if (
        int(stats["original_word_count"]) >= 80
        and float(stats["word_jaccard_similarity"]) < 0.2
    ):
        return "rejection"
    return "preference"


def _normalize_actor_type(actor_type: str) -> str:
    value = _normalize_space(str(actor_type or "human")).casefold()
    if value not in ACTOR_TYPES:
        raise ValueError("actor_type must be one of " + ", ".join(sorted(ACTOR_TYPES)))
    return value


def _normalize_actor_id(actor_id: str) -> str:
    value = _normalize_space(str(actor_id or DEFAULT_ACTOR_ID))
    return value or DEFAULT_ACTOR_ID


def _actor_type_from_record(edit_record: Mapping[str, Any]) -> str:
    actor = edit_record.get("actor")
    if isinstance(actor, Mapping):
        return str(actor.get("type") or "human")
    return str(edit_record.get("actor_type") or "human")


def _actor_id_from_record(edit_record: Mapping[str, Any]) -> str:
    actor = edit_record.get("actor")
    if isinstance(actor, Mapping):
        return str(actor.get("id") or DEFAULT_ACTOR_ID)
    return str(edit_record.get("actor_id") or DEFAULT_ACTOR_ID)


def _diff_stats(original: str, edited: str) -> dict[str, int | float]:
    original_words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", original.casefold())
    edited_words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", edited.casefold())
    original_set = set(original_words)
    edited_set = set(edited_words)
    union = original_set | edited_set
    intersection = original_set & edited_set
    word_jaccard = len(intersection) / len(union) if union else 1.0
    return {
        "original_chars": len(original),
        "edited_chars": len(edited),
        "char_delta": len(edited) - len(original),
        "original_word_count": len(original_words),
        "edited_word_count": len(edited_words),
        "word_delta": len(edited_words) - len(original_words),
        "word_jaccard_similarity": round(word_jaccard, 4),
    }


def _event_id(edit_record: Mapping[str, Any], edit_signature: str) -> str:
    payload = {
        "edit_signature": edit_signature,
        "timestamp": str(edit_record.get("timestamp") or ""),
        "actor": edit_record.get("actor", {}),
        "intent": str(edit_record.get("intent") or ""),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_profile(profile: Mapping[str, Any] | None) -> dict[str, Any]:
    base = create_profile()
    if profile is None:
        return base

    incoming = copy.deepcopy(dict(profile))
    base["version"] = int(incoming.get("version", PROFILE_VERSION))
    examples = []
    for example in incoming.get("examples", []):
        if isinstance(example, Mapping):
            copied = dict(example)
            copied.setdefault("intent", "preference")
            copied.setdefault("actor", {"type": "human", "id": DEFAULT_ACTOR_ID})
            copied["edit_signature"] = _edit_signature(copied)
            examples.append(copied)
    base["examples"] = examples

    incoming_signals = incoming.get("signals", {})
    incoming_sections = incoming_signals.get("section_labels", {})
    base["signals"]["section_labels"]["added"].update(
        _clean_counter(incoming_sections.get("added", {}))
    )
    base["signals"]["section_labels"]["removed"].update(
        _clean_counter(incoming_sections.get("removed", {}))
    )
    base["signals"]["preferred_phrases"].update(
        _clean_counter(incoming_signals.get("preferred_phrases", {}))
    )

    incoming_caution = incoming_signals.get("caution_uncertainty_tone", {})
    base["signals"]["caution_uncertainty_tone"]["markers"].update(
        _clean_counter(incoming_caution.get("markers", {}))
    )
    base["signals"]["caution_uncertainty_tone"]["edits_with_added_caution"] = int(
        incoming_caution.get("edits_with_added_caution", 0)
    )

    corrections = incoming_signals.get("field_corrections", [])
    if isinstance(corrections, list):
        for correction in corrections:
            if not isinstance(correction, Mapping):
                continue
            field = str(correction.get("field", "")).strip()
            old_value = str(correction.get("from", "")).strip()
            new_value = str(correction.get("to", "")).strip()
            if field and old_value and new_value:
                _merge_field_correction(
                    base["signals"]["field_corrections"],
                    field,
                    old_value,
                    new_value,
                    int(correction.get("count", 1)),
                )
    return base


def export_dpo_pairs(
    edits_log: str | Path,
    output: str | Path,
    *,
    include_non_preference: bool = False,
) -> dict[str, int]:
    """Emit section-level (prompt, chosen, rejected) preference pairs from an edits log.

    Two output files:

      * ``<output>``                 — strict pairs only: section-level pairs
        with no dropped citations, plus whole-draft restructure fallbacks
        only when the operator dropped zero citations. Safe to hand to
        ``trl.DPOTrainer`` directly.
      * ``<output>.candidates.jsonl`` — restructure-style pairs where the
        edited draft retained at least one citation but dropped others.
        These are *candidate* training data: a human or downstream filter
        should approve them before training, because DPOTrainer does not
        read metadata and would happily reward the citation drop.

    Returns a dict with counts: ``{"strict": int, "candidates": int}``.
    """

    edits_path = Path(edits_log)
    output_path = Path(output)
    candidates_path = output_path.with_name(output_path.stem + ".candidates.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not edits_path.exists():
        output_path.write_text("", encoding="utf-8")
        candidates_path.write_text("", encoding="utf-8")
        return {"strict": 0, "candidates": 0}

    strict_count = 0
    candidate_count = 0
    with (
        edits_path.open(encoding="utf-8") as edits_handle,
        output_path.open("w", encoding="utf-8") as strict_handle,
        candidates_path.open("w", encoding="utf-8") as candidates_handle,
    ):
        for raw_line in edits_handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            intent = str(event.get("intent") or "")
            if not include_non_preference and intent != "preference":
                continue
            draft_path = event.get("draft_path")
            edited_path = event.get("edited_path")
            if not draft_path or not edited_path:
                continue
            try:
                original_text = Path(str(draft_path)).read_text(encoding="utf-8")
                edited_text = Path(str(edited_path)).read_text(encoding="utf-8")
            except OSError:
                continue
            for pair in _section_level_preference_pairs(
                original_text, edited_text, event
            ):
                line_out = json.dumps(pair, ensure_ascii=False) + "\n"
                if pair["metadata"].get("citations_dropped", 0) > 0:
                    candidates_handle.write(line_out)
                    candidate_count += 1
                else:
                    strict_handle.write(line_out)
                    strict_count += 1
    return {"strict": strict_count, "candidates": candidate_count}


def _section_level_preference_pairs(
    original: str, edited: str, event: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Split both drafts by H2 sections, pair by heading, filter citation drops.

    Falls back to a single whole-draft pair (still citation-filtered) when no
    H2 headings overlap, which is what restructure-style operator edits look
    like in practice — the operator replaces the section template wholesale
    so per-heading pairing yields nothing.
    """

    actor = event.get("actor") if isinstance(event.get("actor"), Mapping) else {}
    metadata = {
        "edit_signature": str(event.get("edit_signature") or ""),
        "intent": str(event.get("intent") or ""),
        "actor_id": str(actor.get("id") or DEFAULT_ACTOR_ID),
        "timestamp": str(event.get("timestamp") or ""),
        "operator_note": str(event.get("operator_note") or ""),
    }

    original_sections = _split_sections(original)
    edited_sections = _split_sections(edited)
    common_headings = set(original_sections) & set(edited_sections)
    pairs: list[dict[str, Any]] = []
    for heading in sorted(common_headings):
        original_body = original_sections[heading].strip()
        edited_body = edited_sections[heading].strip()
        if not original_body or not edited_body or original_body == edited_body:
            continue
        original_citations = set(_CITATION_ID_RE.findall(original_body))
        edited_citations = set(_CITATION_ID_RE.findall(edited_body))
        dropped = original_citations - edited_citations
        if dropped:
            continue
        pairs.append(
            {
                "prompt": f"Section heading: {heading}\nDraft this section grounded in the supplied evidence.",
                "chosen": edited_body,
                "rejected": original_body,
                "metadata": {
                    **metadata,
                    "section_heading": heading,
                    "scope": "section",
                    "citations_dropped": 0,
                    "edited_citation_count": len(edited_citations),
                    "original_citation_count": len(original_citations),
                },
            }
        )
    if pairs:
        return pairs

    # Whole-draft fallback for restructure-style edits. The section-level
    # filter requires *no* citations to be dropped because section pairs are
    # high-quality training signal and we should not contaminate them. The
    # whole-draft fallback is more permissive: it requires the edited draft
    # to retain at least one citation, and it tags every pair with the
    # dropped count so a downstream consumer (DPO trainer, reviewer) can
    # filter further by their own threshold.
    original_body = original.strip()
    edited_body = edited.strip()
    if not original_body or not edited_body or original_body == edited_body:
        return pairs
    original_citations = set(_CITATION_ID_RE.findall(original_body))
    edited_citations = set(_CITATION_ID_RE.findall(edited_body))
    if not edited_citations:
        # Pure prose rewrite with no inline citations — unsafe as a DPO pair.
        return pairs
    dropped = original_citations - edited_citations
    pairs.append(
        {
            "prompt": (
                "Draft a first-pass case fact summary grounded in the supplied "
                "evidence. Match the operator's preferred structure and tone."
            ),
            "chosen": edited_body,
            "rejected": original_body,
            "metadata": {
                **metadata,
                "section_heading": "(whole_draft)",
                "scope": "whole_draft",
                "citations_dropped": len(dropped),
                "edited_citation_count": len(edited_citations),
                "original_citation_count": len(original_citations),
            },
        }
    )
    return pairs


_CITATION_ID_RE = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")
_SECTION_SPLIT_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _split_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    matches = list(_SECTION_SPLIT_RE.finditer(text))
    for index, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[heading] = text[start:end].strip()
    return sections


def _normalized_knowledge_layer(layer: Mapping[str, Any] | None) -> dict[str, Any]:
    base = create_knowledge_layer()
    if not isinstance(layer, Mapping):
        return base
    base["version"] = int(layer.get("version", KNOWLEDGE_LAYER_VERSION))
    rules = []
    for rule in layer.get("rules", []) or []:
        if not isinstance(rule, Mapping):
            continue
        text = str(rule.get("rule") or "").strip()
        if not text:
            continue
        rules.append(
            {
                "rule": text,
                "rule_signature": str(
                    rule.get("rule_signature")
                    or hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()[:16]
                ),
                "added_at": str(rule.get("added_at") or ""),
                "actor_id": str(rule.get("actor_id") or DEFAULT_ACTOR_ID),
                "evidence_ids": [str(eid) for eid in rule.get("evidence_ids", []) or []],
            }
        )
    base["rules"] = rules
    defaults: dict[str, dict[str, str]] = {}
    incoming_defaults = layer.get("field_defaults", {}) or {}
    if isinstance(incoming_defaults, Mapping):
        for field, payload in incoming_defaults.items():
            label = _clean_label(str(field))
            if not label or not isinstance(payload, Mapping):
                continue
            value = str(payload.get("value") or "").strip()
            if not value:
                continue
            defaults[label] = {
                "value": value,
                "updated_at": str(payload.get("updated_at") or ""),
                "actor_id": str(payload.get("actor_id") or DEFAULT_ACTOR_ID),
            }
    base["field_defaults"] = defaults
    return base


def _normalized_retrieval_feedback(feedback: Mapping[str, Any] | None) -> dict[str, Any]:
    base = create_retrieval_feedback()
    if not isinstance(feedback, Mapping):
        return base
    base["version"] = int(feedback.get("version", RETRIEVAL_FEEDBACK_VERSION))
    disputes = []
    for dispute in feedback.get("disputes", []) or []:
        if not isinstance(dispute, Mapping):
            continue
        query = str(dispute.get("query") or "").strip()
        if not query:
            continue
        disputes.append(
            {
                "query": query,
                "dispute_signature": str(
                    dispute.get("dispute_signature")
                    or hashlib.sha256(query.casefold().encode("utf-8")).hexdigest()[:16]
                ),
                "disputed_evidence_ids": [
                    str(eid) for eid in dispute.get("disputed_evidence_ids", []) or []
                ],
                "added_at": str(dispute.get("added_at") or ""),
                "actor_id": str(dispute.get("actor_id") or DEFAULT_ACTOR_ID),
            }
        )
    base["disputes"] = disputes
    base["evidence_boosts"] = {
        str(key): int(value)
        for key, value in (feedback.get("evidence_boosts") or {}).items()
        if str(key).strip() and int(value) > 0
    }
    base["evidence_demotes"] = {
        str(key): int(value)
        for key, value in (feedback.get("evidence_demotes") or {}).items()
        if str(key).strip() and int(value) > 0
    }
    return base


_CATEGORY_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_category(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    return _CATEGORY_NORMALIZE_RE.sub("-", text).strip("-")


def _truncate_text(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut or text[:limit]).rstrip() + "…"


def _clean_counter(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): int(count)
        for key, count in value.items()
        if str(key).strip() and int(count) > 0
    }


def _increment(counter: dict[str, int], item: str, amount: int = 1) -> None:
    key = _normalize_space(item)
    if not key:
        return
    counter[key] = counter.get(key, 0) + amount


def _top_items(counter: Mapping[str, int], limit: int) -> list[str]:
    return [
        item
        for item, _count in sorted(
            counter.items(), key=lambda entry: (-int(entry[1]), entry[0].lower())
        )[:limit]
    ]


def _section_labels(text: str) -> set[str]:
    labels: set[str] = set()
    lines = text.splitlines()
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        markdown_heading = SECTION_LABEL_RE.match(line)
        if markdown_heading:
            labels.add(_clean_label(markdown_heading.group("label")))
            continue

        if line.endswith(":") and not FIELD_RE.match(line):
            labels.add(_clean_label(line[:-1]))
            continue

        if _looks_like_plain_section_label(line, _next_nonblank_line(lines, index)):
            labels.add(_clean_label(line))

    return {label for label in labels if label}


def _looks_like_plain_section_label(line: str, next_line: str | None) -> bool:
    if len(line) > 70 or line[-1:] in ".;," or ":" in line:
        return False
    words = line.split()
    if not 1 <= len(words) <= 7:
        return False
    if not next_line:
        return False
    return all(word[:1].isupper() or word.isupper() for word in words)


def _next_nonblank_line(lines: list[str], start_index: int) -> str | None:
    for line in lines[start_index + 1 :]:
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _clean_label(label: str) -> str:
    cleaned = _normalize_space(label.strip(" #*-"))
    cleaned = re.sub(r"^\d+[\).]\s*", "", cleaned)
    return cleaned


def _field_map(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        match = FIELD_RE.match(raw_line)
        if not match:
            continue
        label = _clean_label(match.group("label"))
        value = _normalize_space(match.group("value"))
        if label and value:
            fields[label] = value
    return fields


def _field_corrections(original: str, edited: str) -> list[dict[str, str]]:
    original_fields = _field_map(original)
    edited_fields = _field_map(edited)
    corrections: list[dict[str, str]] = []
    for field in sorted(set(original_fields) & set(edited_fields)):
        old_value = original_fields[field]
        new_value = edited_fields[field]
        if old_value != new_value:
            corrections.append({"field": field, "from": old_value, "to": new_value})
    return corrections


def _merge_field_correction(
    corrections: list[dict[str, Any]],
    field: str,
    old_value: str,
    new_value: str,
    count: int = 1,
) -> None:
    for correction in corrections:
        if (
            correction["field"] == field
            and correction["from"] == old_value
            and correction["to"] == new_value
        ):
            correction["count"] += count
            return
    corrections.append(
        {"field": field, "from": old_value, "to": new_value, "count": count}
    )


def _preferred_phrases(original: str, edited: str, operator_note: str) -> list[str]:
    original_units = {_casefold_phrase(unit) for unit in _phrase_units(original)}
    phrases: list[str] = []
    seen: set[str] = set()

    for unit in _phrase_units(edited):
        key = _casefold_phrase(unit)
        if key in original_units or key in seen or not _is_reusable_phrase(unit):
            continue
        seen.add(key)
        phrases.append(unit)

    for match in QUOTED_PHRASE_RE.finditer(operator_note):
        phrase = _normalize_space(match.group(1) or match.group(2))
        key = _casefold_phrase(phrase)
        if key not in seen and _is_reusable_phrase(phrase, min_words=2):
            seen.add(key)
            phrases.append(phrase)

    return phrases


def _phrase_units(text: str) -> list[str]:
    units: list[str] = []
    for raw_line in text.splitlines():
        line = _normalize_space(raw_line.strip(" -*\t"))
        if not line or _section_labels(line):
            continue

        # Strip inline citation tokens before phrase extraction. Without this,
        # a sentence whose only content after the period split is a citation
        # (e.g. `; [d61bba9a:p4:c0]`) gets captured as a "preferred phrase"
        # and contaminates the operator profile's reusable phrase memory.
        line = _CITATION_ID_RE.sub("", line)
        line = _normalize_space(line)
        if not line:
            continue

        field_match = FIELD_RE.match(line)
        if field_match:
            line = _normalize_space(field_match.group("value"))

        for sentence in re.split(r"(?<=[.!?])\s+|;\s+", line):
            cleaned = _normalize_space(sentence)
            if cleaned:
                units.append(cleaned)
    return units


def _is_reusable_phrase(phrase: str, min_words: int = 3) -> bool:
    # Defense in depth against citation-token leakage: refuse any phrase
    # that is structurally just a bracket-id, even if `_phrase_units` upstream
    # already strips citations.
    if _looks_like_citation_token(phrase):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", phrase)
    if not min_words <= len(words) <= 8:
        return False
    if FIELD_RE.match(phrase):
        return False
    if len(phrase) > 80:
        return False
    if re.search(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,}\b", phrase):
        return False
    if re.search(r"\b\d{2,}\b|\$[0-9]", phrase):
        return False
    return any(len(word) > 3 for word in words)


def _looks_like_citation_token(phrase: str) -> bool:
    """Return True when ``phrase`` is just citation-id brackets and separators.

    Anything inside ``[...]`` that matches our citation-id charset, optionally
    chained with semicolons or commas and whitespace, counts. The check is
    case-sensitive against the citation alphabet so prose tokens still pass.
    """

    stripped = phrase.strip()
    if not stripped:
        return True
    cleaned = _CITATION_ID_RE.sub("", stripped)
    cleaned = re.sub(r"[;\s,]+", "", cleaned).strip()
    return cleaned == ""


def _citation_ids_from_text(text: str) -> list[str]:
    ids = set()
    for evidence_id in re.findall(r"\[([A-Za-z0-9_.:-]+)\]", text):
        if ":p" in evidence_id and ":c" in evidence_id:
            ids.add(evidence_id)
    return sorted(ids)


def _casefold_phrase(phrase: str) -> str:
    return _normalize_space(phrase).casefold().strip(".")


def _caution_signal(original: str, edited: str) -> dict[str, Any]:
    original_counts = _marker_counts(original)
    edited_counts = _marker_counts(edited)
    added_markers: list[str] = []
    score_delta = 0
    for marker in CAUTION_MARKERS:
        delta = edited_counts.get(marker, 0) - original_counts.get(marker, 0)
        if delta > 0:
            added_markers.append(marker)
            score_delta += delta
    return {"added_markers": added_markers, "score_delta": score_delta}


def _marker_counts(text: str) -> dict[str, int]:
    lowered = text.casefold()
    counts: dict[str, int] = {}
    for marker in CAUTION_MARKERS:
        if " " in marker:
            pattern = re.escape(marker)
        else:
            pattern = rf"\b{re.escape(marker)}\b"
        matches = re.findall(pattern, lowered)
        if matches:
            counts[marker] = len(matches)
    return counts


def _field_correction_guidance(
    corrections: list[dict[str, Any]], limit: int
) -> list[str]:
    sorted_corrections = sorted(
        corrections,
        key=lambda correction: (
            -int(correction.get("count", 0)),
            str(correction.get("field", "")).lower(),
        ),
    )
    guidance: list[str] = []
    for correction in sorted_corrections:
        if _clean_label(str(correction.get("field", ""))).casefold() not in REUSABLE_FIELD_CORRECTION_LABELS:
            continue
        guidance.append(
            "For "
            + str(correction["field"])
            + ", prefer "
            + str(correction["to"])
            + " over "
            + str(correction["from"])
            + "."
        )
        if len(guidance) >= limit:
            break
    return guidance


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _edit_signature(edit_record: Mapping[str, Any]) -> str:
    payload = {
        "original_draft": str(edit_record.get("original_draft") or ""),
        "edited_draft": str(edit_record.get("edited_draft") or ""),
        "evidence_ids": sorted(str(item) for item in edit_record.get("evidence_ids", [])),
        "operator_note": str(edit_record.get("operator_note") or ""),
        "intent": str(edit_record.get("intent") or "preference"),
        "actor": {
            "type": _actor_type_from_record(edit_record),
            "id": _actor_id_from_record(edit_record),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
