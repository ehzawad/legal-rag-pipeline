from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline.io import read_json
from pipeline.schemas import (
    CaseFactSummary,
    ClaimCitation,
    Draft,
    DraftSection,
    EvidenceChunk,
    EvidencePack,
    FactClaim,
    ExtractedField,
    GroundingVerdict,
    PageText,
    ProcessedDocument,
    SourceDocument,
)


@dataclass(frozen=True, slots=True)
class PipelinePaths:
    output_dir: Path
    corpus_dir: Path
    retrieval_index: Path
    evidence_pack: Path
    processed_documents: Path
    retrieved_evidence: Path
    case_fact_summary: Path
    grounding_report: Path
    draft_json: Path
    draft_markdown: Path
    workflow_manifest: Path
    learned_guidance: Path
    risk_report: Path
    risk_report_markdown: Path
    audit_log: Path
    case_run: Path

    @classmethod
    def for_output_dir(cls, output_dir: Path) -> "PipelinePaths":
        return cls(
            output_dir=output_dir,
            corpus_dir=output_dir / "corpus",
            retrieval_index=output_dir / "index" / "retrieval_index.json",
            evidence_pack=output_dir / "evidence_pack.json",
            processed_documents=output_dir / "processed_documents.json",
            retrieved_evidence=output_dir / "retrieved_evidence.json",
            case_fact_summary=output_dir / "case_fact_summary.json",
            grounding_report=output_dir / "grounding_report.json",
            draft_json=output_dir / "draft.json",
            draft_markdown=output_dir / "draft.md",
            workflow_manifest=output_dir / "workflow_manifest.json",
            learned_guidance=output_dir / "learned_guidance.json",
            risk_report=output_dir / "risk_report.json",
            risk_report_markdown=output_dir / "risk_report.md",
            audit_log=output_dir / "audit.jsonl",
            case_run=output_dir / "case_run.json",
        )


def load_processed_documents(path: Path) -> list[ProcessedDocument]:
    return processed_documents_from_json(read_json(path))


def processed_documents_from_json(items) -> list[ProcessedDocument]:
    documents: list[ProcessedDocument] = []
    for item in items:
        source = SourceDocument(**item["source"])
        pages = [PageText(**page) for page in item.get("pages", [])]
        fields = {
            name: ExtractedField(**field)
            for name, field in (item.get("fields") or {}).items()
        }
        documents.append(
            ProcessedDocument(
                source=source,
                pages=pages,
                fields=fields,
                warnings=list(item.get("warnings") or []),
            )
        )
    return documents


def load_evidence(path: Path) -> list[EvidenceChunk]:
    return evidence_from_json(read_json(path))


def evidence_from_json(items) -> list[EvidenceChunk]:
    return [EvidenceChunk(**item) for item in items]


def load_evidence_pack(path: Path) -> EvidencePack:
    return evidence_pack_from_json(read_json(path))


def evidence_pack_from_json(item) -> EvidencePack:
    return EvidencePack(
        case_id=str(item.get("case_id") or ""),
        draft_type=str(item.get("draft_type") or ""),
        task=str(item.get("task") or ""),
        chunks=evidence_from_json(item.get("chunks", [])),
        section_hints={
            str(key): [str(value) for value in values]
            for key, values in (item.get("section_hints") or {}).items()
            if isinstance(values, list)
        },
        unavailable_facts=[str(value) for value in item.get("unavailable_facts", []) or []],
        warnings=[str(value) for value in item.get("warnings", []) or []],
    )


def load_draft(path: Path) -> Draft:
    return draft_from_json(read_json(path))


def draft_from_json(item) -> Draft:
    case_summary = None
    if item.get("case_summary"):
        case_summary = case_fact_summary_from_json(item["case_summary"])
    return Draft(
        draft_type=item["draft_type"],
        title=item["title"],
        generated_at=item["generated_at"],
        sections=[DraftSection(**section) for section in item.get("sections", [])],
        evidence=evidence_from_json(item.get("evidence", [])),
        warnings=list(item.get("warnings") or []),
        case_summary=case_summary,
    )


def case_fact_summary_from_json(item) -> CaseFactSummary:
    return CaseFactSummary(
        case_id=str(item.get("case_id") or ""),
        generated_at=str(item.get("generated_at") or ""),
        title=str(item.get("title") or ""),
        section_order=[str(section) for section in item.get("section_order", []) or []],
        claims=[fact_claim_from_json(claim) for claim in item.get("claims", []) or []],
        evidence=evidence_from_json(item.get("evidence", [])),
        warnings=[str(warning) for warning in item.get("warnings", []) or []],
    )


def fact_claim_from_json(item) -> FactClaim:
    grounding = item.get("grounding")
    return FactClaim(
        claim_id=str(item.get("claim_id") or ""),
        section_id=str(item.get("section_id") or ""),
        section=str(item.get("section") or ""),
        text=str(item.get("text") or ""),
        claim_type=str(item.get("claim_type") or "fact"),
        confidence=str(item.get("confidence") or "medium"),
        citations=[
            ClaimCitation(
                evidence_id=str(citation.get("evidence_id") or ""),
                quote=str(citation.get("quote") or ""),
                substring_grounded=bool(citation.get("substring_grounded", False)),
                entailed=citation.get("entailed"),
            )
            for citation in item.get("citations", []) or []
            if isinstance(citation, dict)
        ],
        grounding=GroundingVerdict(**grounding) if isinstance(grounding, dict) else None,
    )
