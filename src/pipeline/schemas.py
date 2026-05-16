from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class SourceDocument:
    path: str
    document_id: str
    filename: str
    mime_type: str
    sha256: str


@dataclass(slots=True)
class PageText:
    document_id: str
    page_number: int
    text: str
    extraction_method: str
    confidence: float
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExtractedField:
    name: str
    value: str
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProcessedDocument:
    source: SourceDocument
    pages: list[PageText]
    fields: dict[str, ExtractedField]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceChunk:
    evidence_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DraftSection:
    heading: str
    body: str
    evidence_ids: list[str] = field(default_factory=list)
    citation_quotes: dict[str, str] = field(default_factory=dict)
    unsupported: bool = False


@dataclass(slots=True)
class ClaimCitation:
    evidence_id: str
    quote: str
    substring_grounded: bool = False
    entailed: bool | None = None


@dataclass(slots=True)
class GroundingVerdict:
    status: str
    reason: str = ""
    cited_evidence_ids: list[str] = field(default_factory=list)
    substring_grounded_count: int = 0
    entailed_count: int | None = None
    action: str = "keep"


@dataclass(slots=True)
class FactClaim:
    claim_id: str
    section_id: str
    section: str
    text: str
    claim_type: str = "fact"
    confidence: str = "medium"
    citations: list[ClaimCitation] = field(default_factory=list)
    grounding: GroundingVerdict | None = None


@dataclass(slots=True)
class CaseFactSummary:
    case_id: str
    generated_at: str
    title: str
    section_order: list[str]
    claims: list[FactClaim]
    evidence: list[EvidenceChunk]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvidencePack:
    case_id: str
    draft_type: str
    task: str
    chunks: list[EvidenceChunk]
    section_hints: dict[str, list[str]] = field(default_factory=dict)
    unavailable_facts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Draft:
    draft_type: str
    title: str
    generated_at: str
    sections: list[DraftSection]
    evidence: list[EvidenceChunk]
    warnings: list[str] = field(default_factory=list)
    case_summary: CaseFactSummary | None = None


@dataclass(slots=True)
class EditRecord:
    draft_id: str
    draft_type: str
    original_text: str
    edited_text: str
    evidence_ids: list[str]
    operator_note: str = ""
    intent: str = "preference"
    actor_type: str = "human"
    actor_id: str = "default-operator"
    created_at: str = field(default_factory=now_iso)

    def to_learning_dict(self) -> dict[str, Any]:
        return {
            "draft_type": self.draft_type,
            "original_draft": self.original_text,
            "edited_draft": self.edited_text,
            "evidence_ids": [str(evidence_id) for evidence_id in self.evidence_ids],
            "operator_note": self.operator_note,
            "timestamp": self.created_at,
            "intent": self.intent,
            "actor": {
                "type": self.actor_type,
                "id": self.actor_id,
            },
        }


@dataclass(slots=True)
class EditClassification:
    edit_type: str
    reusable_rule: str
    target_mechanisms: list[str]
    rule_scope: str
    diagnostic: str
    confidence: str
    rationale: str = ""


@dataclass(slots=True)
class CaseRun:
    case_id: str
    created_at: str
    input_dir: str
    processed_documents: list[ProcessedDocument]
    draft: Draft
    task: str = ""
    operator_profile_path: str = ""
    learned_guidance: str = ""
    run_fingerprint: str = ""


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value
