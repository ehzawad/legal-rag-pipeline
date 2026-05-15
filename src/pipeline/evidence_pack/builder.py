from __future__ import annotations

from collections.abc import Sequence
import re

from pipeline.schemas import EvidenceChunk, EvidencePack, ProcessedDocument


_SECTION_KEYWORDS = {
    "case_caption": ("caption", "case", "court", "plaintiff", "defendant", "petitioner", "respondent"),
    "parties": ("party", "parties", "plaintiff", "defendant", "petitioner", "respondent", "tenant", "landlord"),
    "documents_reviewed": ("agreement", "notice", "motion", "order", "complaint", "exhibit", "filing"),
    "procedural_posture": ("filed", "filing", "motion", "order", "hearing", "summons", "complaint", "court"),
    "material_facts": ("fact", "dated", "notice", "payment", "deadline", "signed", "record", "property"),
    "relief_sought": ("relief", "request", "requested", "seek", "sought", "damages", "injunction", "dismiss"),
    "open_questions": ("unclear", "illegible", "unknown", "not provided", "missing", "verify"),
}


def build_evidence_pack(
    *,
    case_id: str,
    task: str,
    evidence: Sequence[EvidenceChunk],
    processed_documents: Sequence[ProcessedDocument],
    draft_type: str = "case_fact_summary",
) -> EvidencePack:
    """Group retrieved chunks into section-aware hints for the claim drafter."""

    chunks = list(evidence)
    section_hints = {
        section: _rank_chunk_ids_for_section(section, chunks)
        for section in _SECTION_KEYWORDS
    }
    unavailable = _unavailable_facts(processed_documents)
    warnings: list[str] = []
    if not chunks:
        warnings.append("No retrieved chunks were available for the evidence pack.")
    if unavailable:
        warnings.append("Input processing produced warnings that should appear as open questions.")
    return EvidencePack(
        case_id=case_id,
        draft_type=draft_type,
        task=task,
        chunks=chunks,
        section_hints=section_hints,
        unavailable_facts=unavailable,
        warnings=warnings,
    )


def _rank_chunk_ids_for_section(section: str, chunks: Sequence[EvidenceChunk]) -> list[str]:
    keywords = _SECTION_KEYWORDS[section]
    scored: list[tuple[float, str]] = []
    for chunk in chunks:
        text = _haystack(chunk)
        score = sum(text.count(keyword) for keyword in keywords)
        if chunk.metadata.get("is_field_chunk"):
            score += 0.5
        if score > 0:
            scored.append((float(score) + float(chunk.score or 0.0), chunk.evidence_id))
    if not scored:
        return [chunk.evidence_id for chunk in chunks[:3]]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [evidence_id for _, evidence_id in scored[:5]]


def _haystack(chunk: EvidenceChunk) -> str:
    metadata_text = " ".join(str(value) for value in chunk.metadata.values() if isinstance(value, str))
    return f"{chunk.filename} {chunk.text} {metadata_text}".casefold()


def _unavailable_facts(processed_documents: Sequence[ProcessedDocument]) -> list[str]:
    items: list[str] = []
    for document in processed_documents:
        for warning in document.warnings:
            if str(warning).strip():
                items.append(f"{document.source.filename}: {warning}")
        for page in document.pages:
            if page.confidence < 0.55:
                items.append(
                    f"{document.source.filename} p.{page.page_number}: low extraction confidence {page.confidence:.2f}"
                )
            for warning in page.warnings:
                if str(warning).strip():
                    items.append(f"{document.source.filename} p.{page.page_number}: {warning}")
    return _dedupe(items)


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        clean = re.sub(r"\s+", " ", str(item)).strip()
        key = clean.casefold()
        if clean and key not in seen:
            result.append(clean)
            seen.add(key)
    return result
