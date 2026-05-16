"""API-backed case fact summary drafting from retrieved evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
import os
import re
from typing import Any, Mapping, Sequence

from pipeline.config import ProviderConfig, resolve_provider_name
from pipeline.providers import (
    ProviderUnavailable,
    parse_json_response,
    responses_create,
)
from pipeline.drafting.specs import (
    CASE_FACT_SUMMARY_ADAPTER,
    DraftResponseAdapter,
    DraftSpec,
    resolve_draft_adapter,
    resolve_draft_spec,
)
from pipeline.schemas import (
    CaseFactSummary,
    ClaimCitation,
    Draft,
    DraftSection,
    EvidenceChunk,
    EvidencePack,
    FactClaim,
    now_iso,
)


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")
_CITATION_TOKEN_PATTERN = r"\[[A-Za-z0-9_.:-]+\]"
_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TRAILING_CITATIONS_RE = re.compile(r"([.!?])\s+((?:" + _CITATION_TOKEN_PATTERN + r"\s*)+)")
_SUPPORT_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "among",
    "and",
    "any",
    "are",
    "based",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "can",
    "cannot",
    "could",
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "into",
    "its",
    "may",
    "might",
    "must",
    "not",
    "only",
    "our",
    "out",
    "over",
    "per",
    "provided",
    "same",
    "should",
    "such",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "this",
    "through",
    "under",
    "upon",
    "was",
    "were",
    "when",
    "where",
    "with",
    "within",
    "would",
}

# Abbreviations whose terminal period must not be treated as a sentence boundary.
# Spurious splits on these inflate the uncited-factual-sentence count and cause
# whole sections to be marked unsupported when they're actually fine.
_ABBREVIATIONS: tuple[str, ...] = tuple(
    sorted(
        (
            "a.m.", "p.m.", "A.M.", "P.M.",
            "v.", "vs.", "No.", "no.", "Nos.",
            "pp.", "p.", "Pp.",
            "e.g.", "i.e.", "cf.", "viz.", "et al.", "etc.",
            "Inc.", "Corp.", "Co.", "Ltd.", "L.L.C.", "LLC.",
            "U.S.", "U.S.C.", "U.K.", "U.S.A.",
            "vol.", "Vol.", "ed.", "Ed.", "rev.", "Rev.",
            "Mr.", "Mrs.", "Ms.", "Dr.", "Jr.", "Sr.",
            "St.", "Ave.", "Blvd.", "Rd.", "Esq.", "et seq.",
        ),
        key=len,
        reverse=True,
    )
)
_ABBREV_PERIOD_PLACEHOLDER = "​"  # zero-width space; never appears in source text


def _protect_abbreviations(text: str) -> str:
    for abbrev in _ABBREVIATIONS:
        text = text.replace(abbrev, abbrev.replace(".", _ABBREV_PERIOD_PLACEHOLDER))
    return text


def _restore_abbreviations(text: str) -> str:
    return text.replace(_ABBREV_PERIOD_PLACEHOLDER, ".")


@dataclass(frozen=True)
class DraftingTask:
    """Inputs for a grounded case fact summary draft."""

    request: str
    matter_name: str | None = None
    audience: str | None = None
    known_facts: Mapping[str, str] = field(default_factory=dict)
    missing_facts: Sequence[str] = field(default_factory=tuple)
    preferred_sections: Sequence[str] = field(default_factory=tuple)


def generate_internal_memo(
    task: DraftingTask | Mapping[str, Any] | str | None = None,
    evidence: Sequence[EvidenceChunk | Mapping[str, Any] | Any] | None = None,
    *,
    processed_documents: Sequence[Any] | None = None,
    learned_guidance: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    config: ProviderConfig | None = None,
    generator: Callable[[str], str] | None = None,
    max_evidence_items: int = 8,
    claim_support_check: bool = True,
    draft_type: str | None = None,
    evidence_pack: EvidencePack | Mapping[str, Any] | None = None,
    case_id: str = "",
) -> Draft:
    """Generate a structured, citation-preserving case fact summary draft.

    Real runs use OpenAI. Tests can inject ``generator`` to avoid
    API calls; local deterministic generation is intentionally not the default.
    """

    drafting_task = _coerce_task(task)
    config = config or ProviderConfig.from_env()
    spec = resolve_draft_spec(draft_type or config.draft_type)
    adapter = resolve_draft_adapter(spec)
    selected_evidence = _select_evidence(evidence or (), max_evidence_items=max_evidence_items)
    guidance = _parse_guidance(learned_guidance)
    computed_unsupported = _unsupported_or_unclear_facts(
        drafting_task,
        selected_evidence,
        processed_documents or (),
    )
    prompt = _draft_prompt(
        drafting_task=drafting_task,
        evidence=selected_evidence,
        learned_guidance=learned_guidance,
        computed_unsupported=computed_unsupported,
        spec=spec,
        evidence_pack=evidence_pack,
    )
    resolved_provider = resolve_provider_name(
        provider,
        fallback=config.generation_provider,
        reject_test_names=provider is None,
    )
    response_text = _generate_with_provider(
        prompt,
        provider=resolved_provider,
        model=model or config.openai_model,
        generator=generator,
        reasoning_effort=config.openai_reasoning_effort,
    )
    payload = _parse_draft_payload(response_text)
    draft = _draft_from_payload(
        payload,
        task=drafting_task,
        evidence=selected_evidence,
        computed_unsupported=computed_unsupported,
        guidance=guidance,
        claim_support_check=claim_support_check,
        spec=spec,
        adapter=adapter,
        case_id=case_id,
    )
    _validate_adapter_output(draft)
    return draft


def _generate_with_provider(
    prompt: str,
    *,
    provider: str | None,
    model: str | None,
    generator: Callable[[str], str] | None,
    reasoning_effort: str = "low",
) -> str:
    if generator is not None:
        return generator(prompt)

    resolved_provider = resolve_provider_name(
        provider,
        fallback=os.getenv("PIPELINE_GENERATION_PROVIDER") or "openai",
    )
    if resolved_provider in {"", "local", "mock", "fixture"}:
        raise ProviderUnavailable(
            "Draft generation requires provider='openai' for real runs; "
            "tests must inject generator explicitly."
        )
    if resolved_provider != "openai":
        raise ProviderUnavailable(
            f"Unsupported draft generation provider: {resolved_provider!r}. "
            "This build only ships the openai provider."
        )
    return responses_create(
        prompt,
        model or os.getenv("OPENAI_MODEL_DRAFT", os.getenv("OPENAI_MODEL", "gpt-5.5")),
        reasoning_effort=reasoning_effort,
    )


def _draft_prompt(
    *,
    drafting_task: DraftingTask,
    evidence: Sequence[EvidenceChunk],
    learned_guidance: str | None,
    computed_unsupported: Sequence[str],
    spec: DraftSpec,
    evidence_pack: EvidencePack | Mapping[str, Any] | None = None,
) -> str:
    evidence_payload = [
        {
            "evidence_id": item.evidence_id,
            "document_id": item.document_id,
            "filename": item.filename,
            "page_number": item.page_number,
            "score": item.score,
            "evidence_kind": str(item.metadata.get("evidence_kind") or "source-page"),
            "is_derived": bool(item.metadata.get("is_field_chunk")),
            "text": _truncate(item.text, 1800),
        }
        for item in evidence
    ]
    task_payload = {
        "request": drafting_task.request,
        "matter_name": drafting_task.matter_name,
        "audience": drafting_task.audience,
        "known_facts": dict(drafting_task.known_facts),
        "missing_facts": list(drafting_task.missing_facts),
        "preferred_sections": list(drafting_task.preferred_sections),
    }
    guidance = learned_guidance.strip() if learned_guidance else ""
    pack_payload = _evidence_pack_payload(evidence_pack)
    return spec.build_prompt(
        task_payload=task_payload,
        evidence_payload=evidence_payload,
        learned_guidance=guidance,
        computed_unsupported=computed_unsupported,
        evidence_pack_payload=pack_payload,
    )


def _parse_draft_payload(text: str) -> Mapping[str, Any]:
    try:
        payload = parse_json_response(text)
    except ProviderUnavailable:
        payload = _json_from_markdown_fence(text)
    if not isinstance(payload, Mapping):
        raise ProviderUnavailable("Draft provider response must be a JSON object")
    return payload


def _evidence_pack_payload(value: EvidencePack | Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, EvidencePack):
        return {
            "case_id": value.case_id,
            "draft_type": value.draft_type,
            "task": value.task,
            "section_hints": value.section_hints,
            "unavailable_facts": value.unavailable_facts,
            "warnings": value.warnings,
        }
    if isinstance(value, Mapping):
        return {
            "case_id": value.get("case_id", ""),
            "draft_type": value.get("draft_type", ""),
            "task": value.get("task", ""),
            "section_hints": value.get("section_hints", {}),
            "unavailable_facts": value.get("unavailable_facts", []),
            "warnings": value.get("warnings", []),
        }
    return {}


def _json_from_markdown_fence(text: str) -> Mapping[str, Any]:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if not match:
        raise ProviderUnavailable("Draft provider response was not valid JSON")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ProviderUnavailable("Draft provider response was not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ProviderUnavailable("Draft provider response must be a JSON object")
    return payload


def _draft_from_payload(
    payload: Mapping[str, Any],
    *,
    task: DraftingTask,
    evidence: Sequence[EvidenceChunk],
    computed_unsupported: Sequence[str],
    guidance: set[str],
    claim_support_check: bool = True,
    spec: DraftSpec,
    adapter: DraftResponseAdapter,
    case_id: str = "",
) -> Draft:
    valid_ids = {item.evidence_id for item in evidence}
    if adapter.id == CASE_FACT_SUMMARY_ADAPTER:
        if not _has_claim_payload(payload):
            raise ProviderUnavailable(
                f"Draft adapter {adapter.id!r} requires a case-fact claims response with a claims list"
            )
        case_summary, claim_warnings = _case_summary_from_claim_payload(
            payload,
            evidence=evidence,
            valid_ids=valid_ids,
            case_id=case_id,
        )
        sections = _sections_from_case_summary(case_summary)
        citation_warnings = claim_warnings
        quote_warnings = []
    else:
        raise ProviderUnavailable(f"Unsupported draft response adapter: {adapter.id!r}")
    sections, sentence_warnings = _validate_sentence_citations(sections)
    support_warnings = _validate_claim_support(sections) if claim_support_check else []
    unsupported = _combine_unsupported(computed_unsupported, payload.get("unsupported_or_unclear_facts"))
    sections = _ensure_required_sections(
        sections,
        evidence=evidence,
        unsupported=unsupported,
        guidance=guidance,
        task=task,
    )
    warnings = visible_review_warnings(_warnings_from_payload(payload), evidence=evidence)
    warnings.extend(citation_warnings)
    warnings.extend(quote_warnings)
    warnings.extend(sentence_warnings)
    warnings.extend(support_warnings)
    if not evidence:
        warnings.append("No evidence was supplied, so supported memo sections should remain empty or preliminary.")

    return Draft(
        draft_type=spec.id,
        title=_safe_title(payload.get("title")),
        generated_at=now_iso(),
        sections=sections,
        evidence=list(evidence),
        warnings=_dedupe_strings(warnings),
        case_summary=case_summary,
    )


def _validate_adapter_output(draft: Draft) -> None:
    summary = draft.case_summary
    if summary is None:
        raise ProviderUnavailable("Draft adapter did not return a CaseFactSummary")
    if not summary.claims:
        raise ProviderUnavailable("Draft adapter returned an empty CaseFactSummary")
    evidence_ids = {chunk.evidence_id for chunk in draft.evidence}
    summary_evidence_ids = {chunk.evidence_id for chunk in summary.evidence}
    for claim in summary.claims:
        if claim.claim_type == "scaffolding":
            continue
        if not claim.citations:
            raise ProviderUnavailable(f"Draft adapter returned uncited factual claim {claim.claim_id}")
        for citation in claim.citations:
            if not citation.evidence_id or citation.evidence_id not in evidence_ids:
                raise ProviderUnavailable(f"Draft adapter returned invalid citation id for {claim.claim_id}")
            if citation.evidence_id not in summary_evidence_ids:
                raise ProviderUnavailable(f"Draft adapter summary evidence is missing {citation.evidence_id}")
            if not citation.quote.strip():
                raise ProviderUnavailable(f"Draft adapter returned an empty quote for {claim.claim_id}")


def _has_claim_payload(payload: Mapping[str, Any]) -> bool:
    if isinstance(payload.get("claims"), list):
        return True
    nested = payload.get("case_fact_summary")
    return isinstance(nested, Mapping) and isinstance(nested.get("claims"), list)


def _case_summary_from_claim_payload(
    payload: Mapping[str, Any],
    *,
    evidence: Sequence[EvidenceChunk],
    valid_ids: set[str],
    case_id: str,
) -> tuple[CaseFactSummary, list[str]]:
    source = payload.get("case_fact_summary") if isinstance(payload.get("case_fact_summary"), Mapping) else payload
    raw_claims = source.get("claims") if isinstance(source, Mapping) else []
    warnings: list[str] = []
    claims: list[FactClaim] = []
    section_order: list[str] = []
    seen_claim_ids: set[str] = set()
    if not isinstance(raw_claims, list):
        raw_claims = []
    for raw_claim in raw_claims:
        if not isinstance(raw_claim, Mapping):
            continue
        text = _clean_str(raw_claim.get("text"))
        if not text:
            continue
        section = _canonical_section(raw_claim.get("section"))
        if section not in section_order:
            section_order.append(section)
        citations, citation_warnings = _claim_citations_from_payload(raw_claim, valid_ids=valid_ids)
        warnings.extend(citation_warnings)
        claim_type = _claim_type(raw_claim.get("claim_type"), section=section, citations=citations)
        confidence = _claim_confidence(raw_claim.get("confidence"))
        claim_id = _stable_claim_id(section, text)
        if claim_id in seen_claim_ids:
            continue
        seen_claim_ids.add(claim_id)
        claims.append(
            FactClaim(
                claim_id=claim_id,
                section_id=section,
                section=_section_heading(section),
                text=text,
                claim_type=claim_type,
                confidence=confidence,
                citations=citations,
            )
        )
    open_questions = source.get("open_questions") if isinstance(source, Mapping) else []
    if isinstance(open_questions, list):
        for raw_question in open_questions:
            question = _clean_str(raw_question)
            if not question:
                continue
            section = "open_questions"
            if section not in section_order:
                section_order.append(section)
            claim_id = _stable_claim_id(section, question)
            if claim_id in seen_claim_ids:
                continue
            seen_claim_ids.add(claim_id)
            claims.append(
                FactClaim(
                    claim_id=claim_id,
                    section_id=section,
                    section=_section_heading(section),
                    text=question,
                    claim_type="scaffolding",
                    confidence="low",
                    citations=[],
                )
            )
    if not claims:
        warnings.append("Claim-first draft payload did not include usable claims.")
    return (
        CaseFactSummary(
            case_id=case_id,
            generated_at=now_iso(),
            title=_safe_title(source.get("title") if isinstance(source, Mapping) else payload.get("title")),
            section_order=section_order or _default_section_order(),
            claims=claims,
            evidence=list(evidence),
            warnings=_dedupe_strings(warnings),
        ),
        _dedupe_strings(warnings),
    )


def _claim_citations_from_payload(
    raw_claim: Mapping[str, Any],
    *,
    valid_ids: set[str],
) -> tuple[list[ClaimCitation], list[str]]:
    warnings: list[str] = []
    citations: list[ClaimCitation] = []
    raw_citations = raw_claim.get("citations")
    if isinstance(raw_citations, list):
        for raw in raw_citations:
            if isinstance(raw, Mapping):
                evidence_id = _clean_str(raw.get("evidence_id") or raw.get("id"))
                quote = _clean_str(raw.get("quote") or raw.get("citation_quote"))
            else:
                evidence_id = _clean_str(raw)
                quote = ""
            if not evidence_id:
                continue
            if evidence_id not in valid_ids:
                warnings.append(f"Removed unsupported claim citation id: {evidence_id}")
                continue
            if not quote:
                warnings.append(f"Claim citation {evidence_id} did not include a verbatim quote.")
                continue
            if evidence_id not in {citation.evidence_id for citation in citations}:
                citations.append(ClaimCitation(evidence_id=evidence_id, quote=quote))
        return citations, warnings

    evidence_ids = _sanitize_evidence_ids(raw_claim.get("evidence_ids"), valid_ids=valid_ids)
    quotes = _sanitize_citation_quotes(raw_claim.get("citation_quotes"), valid_ids=valid_ids)
    for evidence_id in evidence_ids:
        quote = quotes.get(evidence_id, "")
        if quote:
            citations.append(ClaimCitation(evidence_id=evidence_id, quote=quote))
        else:
            warnings.append(f"Claim citation {evidence_id} did not include a verbatim quote.")
    return citations, warnings


def sections_from_case_summary(summary: CaseFactSummary) -> list[DraftSection]:
    return _sections_from_case_summary(summary)


def _sections_from_case_summary(summary: CaseFactSummary) -> list[DraftSection]:
    claims_by_section: dict[str, list[FactClaim]] = {}
    for claim in summary.claims:
        claims_by_section.setdefault(claim.section_id, []).append(claim)
    ordered = list(summary.section_order)
    for section in claims_by_section:
        if section not in ordered:
            ordered.append(section)
    sections: list[DraftSection] = []
    for section_id in ordered:
        claims = claims_by_section.get(section_id, [])
        if not claims:
            continue
        body_lines: list[str] = []
        evidence_ids: list[str] = []
        citation_quotes: dict[str, str] = {}
        unsupported = section_id == "open_questions"
        seen_lines: set[str] = set()
        for claim in claims:
            citations = [
                citation
                for citation in claim.citations
                if citation.evidence_id and citation.quote
            ]
            cited = " ".join(f"[{citation.evidence_id}]" for citation in citations)
            display_text = _operator_claim_text(section_id, claim.text)
            line_key = _operator_claim_key(display_text)
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)
            line = f"- {display_text}"
            if cited:
                line += f" {cited}"
            body_lines.append(line)
            for citation in citations:
                if citation.evidence_id not in evidence_ids:
                    evidence_ids.append(citation.evidence_id)
                citation_quotes.setdefault(citation.evidence_id, citation.quote)
            verdict = claim.grounding.status if claim.grounding else ""
            if claim.claim_type != "scaffolding" and (not citations or verdict in {"uncited", "ungrounded", "not_supported", "contradicted", "unclear"}):
                unsupported = True
        body = "\n".join(body_lines)
        if not unsupported and body and not evidence_ids:
            # Defense in depth: a non-open_questions section that ended up
            # with body but zero usable (id+quote) citations is not actually
            # grounded — the per-claim loop normally flips ``unsupported`` for
            # this case, but if every claim is scaffolding the flag may stay
            # False. Mark the section unsupported so the citation-validity
            # metric cannot reward a citation-less section.
            unsupported = True
        sections.append(
            DraftSection(
                heading=_section_heading(section_id),
                body=body,
                evidence_ids=evidence_ids,
                citation_quotes=citation_quotes,
                unsupported=unsupported,
            )
        )
    return sections


def _operator_claim_text(section_id: str, text: str) -> str:
    normalized = " ".join(str(text or "").split())
    if section_id == "open_questions":
        for prefix in (
            "Items Requiring Verification:",
            "Open Questions:",
            "Operator Directive:",
        ):
            if normalized.startswith(prefix):
                return _operator_language_text(normalized[len(prefix) :].strip())
    return _operator_language_text(normalized)


def _operator_language_text(text: str) -> str:
    replacements = (
        (
            r"\bAll pages are extracted from a text layer that appears to be OCR of difficult handwriting and is largely garbled\b",
            "All pages come from degraded extracted text and are largely garbled",
        ),
        (
            r"\bText layer appears to be OCR from difficult handwriting and is largely garbled\b",
            "Extracted text is degraded and largely garbled",
        ),
        (r"\bextracted OCR\b", "extracted text"),
        (r"\bsupplied OCR\b", "supplied extracted text"),
        (r"\bOCR text\b", "extracted text"),
        (r"\bOCR name fragments\b", "extracted name fragments"),
        (r"\bOCR name fragment\b", "extracted name fragment"),
        (r"\bOCR fragments\b", "extracted text fragments"),
        (r"\bOCR fragment\b", "extracted text fragment"),
        (r"\bPage (\d+) OCR includes\b", r"Page \1 extracted text includes"),
        (r"\bextracted from a text layer that appears to be OCR of\b", "extracted as degraded text from"),
        (r"\bText layer appears to be OCR from\b", "Extracted text appears degraded from"),
        (r"\bappears to be OCR from\b", "appears degraded from"),
        (r"\bderived-field extraction\b", "extracted field summary"),
        (r"\bderived-field text\b", "extracted field text"),
        (r"\bderived-field\b", "extracted field"),
        (r"\bderived field\b", "extracted field"),
        (r"\bderived fields\b", "extracted fields"),
        (r"\bextracted text/source text\b", "extracted text"),
        (r"\btext-layer extractions\b", "PDF text extraction"),
        (r"\btext-layer extraction\b", "PDF text extraction"),
        (r"\btext-layer character\b", "extracted text character"),
        (r"\bin the text layer\b", "in the extracted text"),
        (r"\bText-layer page\b", "Extracted text page"),
        (r"\btext-layer page\b", "extracted text page"),
        (r"\bText-layer source\b", "Extracted text source"),
        (r"\btext-layer source\b", "extracted text source"),
        (r"\btext-layer\b", "extracted text"),
        (r"\btext layer\b", "extracted text"),
        (r"\bimage OCR\b", "image-based extraction"),
        (r"\bmultimodal OCR\b", "multimodal image-based extraction"),
        (r"\bOCR-heavy\b", "extraction-heavy"),
        (r"\bOCR\b", "extracted text"),
    )
    cleaned = text
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.I)
    cleaned = re.sub(r"\bextracted text/source text\b", "extracted text", cleaned, flags=re.I)
    cleaned = re.sub(
        r"\bThe extracted ([A-Za-z0-9][A-Za-z0-9 .&/-]{0,80}?) extracted text\b",
        r"The \1 extracted text",
        cleaned,
    )
    cleaned = re.sub(r"\bextracted from extracted text\b", "extracted from the PDF text", cleaned, flags=re.I)
    cleaned = re.sub(r"\bA extracted field\b", "An extracted field", cleaned)
    cleaned = re.sub(r"\ba extracted field\b", "an extracted field", cleaned)
    return cleaned


def _operator_claim_key(text: str) -> str:
    cleaned = re.sub(r"\[[A-Za-z0-9_.:-]+\]", "", text)
    cleaned = re.sub(r"^\s*(?:p\.?\s*\d+|page\s+\d+)\s*:\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned.lower())
    return " ".join(cleaned.split())


def _stable_claim_id(section_id: str, text: str) -> str:
    normalized = normalize_quote_text(f"{section_id}\n{text}")
    return "claim_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _canonical_section(value: Any) -> str:
    raw = _clean_str(value).casefold()
    normalized = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    aliases = {
        "caption": "case_caption",
        "case": "case_caption",
        "case_caption": "case_caption",
        "party": "parties",
        "parties": "parties",
        "documents": "documents_reviewed",
        "documents_reviewed": "documents_reviewed",
        "evidence_table": "evidence_table",
        "procedural_posture": "procedural_posture",
        "posture": "procedural_posture",
        "facts": "material_facts",
        "material_facts": "material_facts",
        "relief": "relief_sought",
        "relief_sought": "relief_sought",
        "requested_action": "relief_sought",
        "open_questions": "open_questions",
        "unsupported_or_unclear_facts": "open_questions",
    }
    return aliases.get(normalized, normalized or "material_facts")


def _section_heading(section_id: str) -> str:
    headings = {
        "case_caption": "Case Caption",
        "parties": "Parties",
        "documents_reviewed": "Documents Reviewed",
        "procedural_posture": "Procedural Posture",
        "material_facts": "Material Facts",
        "relief_sought": "Relief Sought / Requested Action",
        "open_questions": "Open Questions",
        "evidence_table": "Evidence Table",
    }
    return headings.get(section_id, section_id.replace("_", " ").title())


def _default_section_order() -> list[str]:
    return [
        "case_caption",
        "parties",
        "documents_reviewed",
        "procedural_posture",
        "material_facts",
        "relief_sought",
        "open_questions",
    ]


def _claim_type(value: Any, *, section: str, citations: Sequence[ClaimCitation]) -> str:
    raw = _clean_str(value).casefold()
    if section == "open_questions" and not citations:
        return "scaffolding"
    if raw in {"fact", "summary"}:
        return raw
    return "fact"


def _claim_confidence(value: Any) -> str:
    raw = _clean_str(value).casefold()
    if raw in {"high", "medium", "low"}:
        return raw
    return "medium"


def _ensure_required_sections(
    sections: list[DraftSection],
    *,
    evidence: Sequence[EvidenceChunk],
    unsupported: Sequence[str],
    guidance: set[str],
    task: DraftingTask,
) -> list[DraftSection]:
    headings = {section.heading.lower().strip() for section in sections}
    evidence_ids = [item.evidence_id for item in evidence]
    result = list(sections)

    if "issue" not in headings:
        issue = task.request or "Assess the drafting request using the cited record."
        # The Issue framing is restating the drafting request, not a factual
        # claim with retrieved-set evidence. It belongs in the operator review
        # bucket so the citation-validity metric correctly excludes it from
        # the denominator.
        result.insert(
            0,
            DraftSection(
                heading="Issue",
                body=issue,
                evidence_ids=[],
                unsupported=True,
            ),
        )

    if _needs_guidance_section("risk", guidance, task, headings):
        # Risk Notes is a scaffolding hook for the operator's follow-up review,
        # not a factual claim supported by retrieved evidence, so mark it
        # unsupported regardless of whether computed unsupported items exist.
        result.append(
            DraftSection(
                heading="Risk Notes",
                body=(
                    "Resolve the unsupported or unclear facts before treating this first-pass draft as legal advice."
                    if unsupported
                    else "Verify citations against the source record before finalizing."
                ),
                evidence_ids=[],
                unsupported=True,
            )
        )

    if _needs_guidance_section("recommendation", guidance, task, headings):
        # Recommended Next Steps is operator guidance, not a cited factual
        # claim, so mark it unsupported regardless of computed unsupported
        # state.
        result.append(
            DraftSection(
                heading="Recommended Next Steps",
                body=(
                    "Collect source support for the unsupported or unclear facts, then update the cited analysis."
                    if unsupported
                    else "Review the cited evidence and confirm whether more authority or source material is needed."
                ),
                evidence_ids=[],
                unsupported=True,
            )
        )

    if "unsupported or unclear facts" not in headings:
        result.append(
            DraftSection(
                heading="Unsupported or Unclear Facts",
                body="\n".join(f"- {item}" for item in unsupported),
                evidence_ids=[],
                unsupported=True,
            )
        )
    return result


def _needs_guidance_section(
    section: str,
    guidance: set[str],
    task: DraftingTask,
    headings: set[str],
) -> bool:
    requested = {item.lower().strip() for item in task.preferred_sections}
    heading_name = {"risk": "risk notes", "recommendation": "recommended next steps"}[section]
    return heading_name not in headings and (section in guidance or section in requested)


def _select_evidence(
    evidence: Sequence[EvidenceChunk | Mapping[str, Any] | Any],
    *,
    max_evidence_items: int,
) -> list[EvidenceChunk]:
    if max_evidence_items <= 0:
        return []
    converted = [_coerce_evidence(item, index) for index, item in enumerate(evidence)]
    return [item for item in converted if item.text.strip()][:max_evidence_items]


def _coerce_evidence(value: EvidenceChunk | Mapping[str, Any] | Any, index: int) -> EvidenceChunk:
    if isinstance(value, EvidenceChunk):
        return value

    text = _clean_str(_first_present(value, ("text", "content", "passage"), ""))
    evidence_id = _optional_str(_first_present(value, ("evidence_id", "id", "citation_id"), None)) or f"E{index + 1}"
    metadata = _mapping(_first_present(value, ("metadata", "meta"), {}))
    source = _optional_str(
        _first_present(value, ("document_id", "source_id", "source", "doc_id", "file_name"), None)
    )
    if not source:
        source = _optional_str(metadata.get("document_id") or metadata.get("source") or metadata.get("source_id"))
    filename = _optional_str(_first_present(value, ("filename", "file_name", "source_id", "source"), None))
    if not filename:
        filename = _optional_str(metadata.get("filename") or metadata.get("source")) or source or "unknown-document"
    page_number = _optional_int(_first_present(value, ("page_number", "page", "number"), None))
    if page_number is None:
        page_number = _optional_int(metadata.get("page_number"))
    score = _optional_float(_first_present(value, ("score", "relevance", "rank_score"), 0.0))

    return EvidenceChunk(
        evidence_id=evidence_id,
        document_id=source or "unknown-document",
        filename=filename,
        page_number=page_number if page_number is not None else 0,
        text=text,
        score=score if score is not None else 0.0,
        metadata=dict(metadata),
    )


def _coerce_task(value: DraftingTask | Mapping[str, Any] | str | None) -> DraftingTask:
    if isinstance(value, DraftingTask):
        return value
    if value is None:
        return DraftingTask(request="Case fact summary")
    if isinstance(value, str):
        return DraftingTask(request=value.strip())
    if isinstance(value, Mapping):
        return DraftingTask(
            request=_clean_str(_first_present(value, ("request", "question", "task", "objective"), "")),
            matter_name=_optional_str(_first_present(value, ("matter_name", "matter"), None)),
            audience=_optional_str(_first_present(value, ("audience", "recipient"), None)),
            known_facts=_string_mapping(_first_present(value, ("known_facts", "facts"), {})),
            missing_facts=_string_sequence(_first_present(value, ("missing_facts", "required_facts"), ())),
            preferred_sections=_string_sequence(
                _first_present(value, ("preferred_sections", "requested_sections", "sections"), ())
            ),
        )
    return DraftingTask(request=_clean_str(_first_present(value, ("request", "question", "task", "objective"), "")))


def _unsupported_or_unclear_facts(
    task: DraftingTask,
    evidence: Sequence[EvidenceChunk],
    processed_documents: Sequence[Any],
) -> list[str]:
    items: list[str] = []
    evidence_text = " ".join(item.text for item in evidence)

    if not evidence:
        items.append("No retrieved evidence was provided.")

    for fact_name, fact_value in task.known_facts.items():
        clean_name = _clean_str(fact_name)
        clean_value = _clean_str(fact_value)
        if not clean_value:
            if clean_name:
                items.append(f"{clean_name}: no value was supplied.")
            continue
        if not _is_supported(clean_value, evidence_text):
            label = f"{clean_name}: " if clean_name else ""
            items.append(f"{label}supplied fact is not grounded in the retrieved evidence: {clean_value}")

    for missing in task.missing_facts:
        clean_missing = _clean_str(missing)
        if clean_missing:
            items.append(f"Not established by retrieved evidence: {clean_missing}")

    items.extend(_unclear_items_from_processed(processed_documents))
    return _dedupe_strings(items)


def _unclear_items_from_processed(processed_documents: Sequence[Any]) -> list[str]:
    items: list[str] = []
    for document in processed_documents:
        for warning in _ensure_sequence(_first_present(document, ("warnings",), ())):
            message = _warning_message(warning)
            if message:
                items.append(message)
        for page in _ensure_sequence(_first_present(document, ("pages",), ())):
            if bool(_first_present(page, ("unclear",), False)):
                page_number = _first_present(page, ("page_number",), None)
                items.append(f"Page {page_number or '?'} contains unclear or unavailable text.")
            for warning in _ensure_sequence(_first_present(page, ("warnings",), ())):
                message = _warning_message(warning)
                if message:
                    items.append(message)
    return items


def _warning_message(warning: Any) -> str | None:
    if isinstance(warning, str):
        return _optional_str(warning)
    message = _optional_str(_first_present(warning, ("message",), None))
    code = _optional_str(_first_present(warning, ("code",), None))
    if message and code:
        return f"{code}: {message}"
    return message or code


def _combine_unsupported(computed: Sequence[str], generated: Any) -> list[str]:
    items = list(computed)
    if isinstance(generated, list):
        items.extend(_clean_str(item) for item in generated if _clean_str(item))
    merged = _dedupe_strings(items)
    if merged:
        return merged
    return ["No additional uncited facts were used in this first-pass draft."]


def _warnings_from_payload(payload: Mapping[str, Any]) -> list[str]:
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        return []
    return [_clean_str(item) for item in warnings if _clean_str(item)]


def visible_review_warnings(warnings: Sequence[Any], *, evidence: Sequence[Any] = ()) -> list[str]:
    """Filter run warnings down to operator-relevant review flags.

    Learned guidance is intentionally broad and durable. The model may sometimes
    echo a knowledge-layer rule as a warning even when the current evidence does
    not contain the fact pattern that rule was written for. Keep those rules in
    the prompt envelope, but avoid showing them as current-run defects.
    """

    evidence_text = _evidence_context_text(evidence)
    result: list[str] = []
    for warning in warnings:
        clean = _clean_str(warning)
        if not clean:
            continue
        if _is_operator_noise_warning(clean):
            continue
        if _is_irrelevant_decimal_financial_warning(clean, evidence_text):
            continue
        result.append(clean)
    return _dedupe_strings(result)


def _is_operator_noise_warning(warning: str) -> bool:
    text = warning.casefold()
    if text.startswith("input processing reported") or text.startswith("input processing produced"):
        return True
    if text.startswith("advisory claim-support check flagged section"):
        return True
    if "0.00" in warning and any(
        marker in text
        for marker in (
            "proof-of-service amount field",
            "amount field showing",
            "derived from extracted fields",
        )
    ):
        return True
    return False


def _is_irrelevant_decimal_financial_warning(warning: str, evidence_text: str) -> bool:
    warning_text = warning.casefold()
    if "decimal punctuation" not in warning_text:
        return False
    if not any(term in warning_text for term in ("net-income", "net income", "aggregate-revenue", "aggregate revenue", "millions")):
        return False
    financial_terms = (
        "net income",
        "net-income",
        "aggregate revenue",
        "aggregate-revenue",
        "total revenue",
        "annual audited",
        "financial statement",
        "broker-dealer",
        "form x-17a-5",
        "millions",
    )
    return not any(term in evidence_text for term in financial_terms)


def _evidence_context_text(evidence: Sequence[Any]) -> str:
    parts: list[str] = []
    for item in evidence:
        if isinstance(item, Mapping):
            values = (
                item.get("text"),
                item.get("filename"),
                item.get("document_id"),
                item.get("source"),
            )
            metadata = item.get("metadata")
            parts.extend(_clean_str(value) for value in values if _clean_str(value))
            if isinstance(metadata, Mapping):
                parts.extend(_clean_str(value) for value in metadata.values() if isinstance(value, str))
            continue
        values = (
            _first_present(item, ("text", "content", "passage"), ""),
            _first_present(item, ("filename", "file_name", "source"), ""),
            _first_present(item, ("document_id", "source_id", "doc_id"), ""),
        )
        parts.extend(_clean_str(value) for value in values if _clean_str(value))
        metadata = _first_present(item, ("metadata", "meta"), {})
        if isinstance(metadata, Mapping):
            parts.extend(_clean_str(value) for value in metadata.values() if isinstance(value, str))
    return " ".join(parts).casefold()


def _sanitize_evidence_ids(value: Any, *, valid_ids: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    ids: list[str] = []
    for item in value:
        evidence_id = _clean_str(item)
        if evidence_id in valid_ids and evidence_id not in seen:
            ids.append(evidence_id)
            seen.add(evidence_id)
    return ids


def _sanitize_citation_quotes(value: Any, *, valid_ids: set[str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    quotes: dict[str, str] = {}
    for key, raw_quote in value.items():
        evidence_id = _clean_str(key)
        if evidence_id not in valid_ids or evidence_id in quotes:
            continue
        quote = _clean_str(raw_quote)
        if quote:
            quotes[evidence_id] = quote
    return quotes


def _validate_sentence_citations(sections: list[DraftSection]) -> tuple[list[DraftSection], list[str]]:
    warnings: list[str] = []
    validated: list[DraftSection] = []
    for section in sections:
        if section.unsupported or not section.body.strip() or not section.evidence_ids:
            validated.append(section)
            continue

        uncited = [
            sentence
            for sentence in sentence_units(section.body)
            if _looks_like_factual_sentence(sentence) and not _CITATION_RE.search(sentence)
        ]
        if not uncited:
            validated.append(section)
            continue

        warnings.append(
            "Marked section "
            + repr(section.heading)
            + " for review because it contains uncited factual sentence(s): "
            + "; ".join(_truncate(sentence, 120) for sentence in uncited[:3])
        )
        validated.append(
            DraftSection(
                heading=section.heading,
                body=section.body,
                evidence_ids=section.evidence_ids,
                citation_quotes=section.citation_quotes,
                unsupported=True,
            )
        )
    return validated, warnings


def _validate_claim_support(sections: list[DraftSection]) -> list[str]:
    warnings: list[str] = []
    for section in sections:
        if section.unsupported or not section.body.strip() or not section.evidence_ids:
            continue
        flagged: list[str] = []
        for sentence in sentence_units(section.body):
            citation_ids = [
                evidence_id
                for evidence_id in _CITATION_RE.findall(sentence)
                if evidence_id in section.citation_quotes
            ]
            if not citation_ids:
                continue
            claim_tokens = _support_tokens(_CITATION_RE.sub("", sentence))
            if len(claim_tokens) < 5:
                continue
            quote_tokens: set[str] = set()
            for evidence_id in citation_ids:
                quote_tokens.update(_support_tokens(section.citation_quotes.get(evidence_id, "")))
            if not quote_tokens:
                continue
            overlap = claim_tokens & quote_tokens
            overlap_ratio = len(overlap) / max(len(claim_tokens), 1)
            if len(overlap) < 2 and overlap_ratio < 0.25:
                flagged.append(sentence)
        if flagged:
            warnings.append(
                "Advisory claim-support check flagged section "
                + repr(section.heading)
                + " for weak lexical overlap between cited sentence(s) and verbatim quote(s): "
                + "; ".join(_truncate(sentence, 120) for sentence in flagged[:3])
            )
    return warnings


def sentence_units(body: str) -> list[str]:
    """Split body text into sentence-like units, keeping trailing citations attached.

    Shared by drafting (for uncited-sentence detection) and evaluation
    (for claim-citation-rate scoring). Keeping a single implementation
    prevents drafter-evaluator disagreement on what counts as a cited
    sentence, which previously surfaced as a spurious A/B regression.
    """

    units: list[str] = []
    citation_attached = _TRAILING_CITATIONS_RE.sub(r" \2\1 ", body)
    protected = _protect_abbreviations(citation_attached)
    for raw in _SENTENCE_SPLIT_RE.split(protected):
        cleaned = _clean_str(_restore_abbreviations(raw).strip(" -*\t"))
        if cleaned:
            units.append(cleaned)
    return units


def _looks_like_factual_sentence(sentence: str) -> bool:
    text = _CITATION_RE.sub("", sentence)
    tokens = _tokens(text)
    return len(tokens) >= 4


_QUOTE_NORMALIZATION_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
        " ": " ",
        " ": " ",
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
    }
)


def normalize_quote_text(text: str) -> str:
    translated = (text or "").translate(_QUOTE_NORMALIZATION_MAP)
    return " ".join(translated.split()).casefold()


def render_section_quotes_markdown(section: DraftSection) -> list[str]:
    lines: list[str] = []
    if not section.citation_quotes:
        return lines
    for evidence_id in section.evidence_ids:
        quote = section.citation_quotes.get(evidence_id)
        if quote:
            lines.append(f"  - [{evidence_id}] \"{quote}\"")
    return lines


def _safe_title(value: Any) -> str:
    title = _clean_str(value)
    return title or "Case Fact Summary"


def _parse_guidance(learned_guidance: str | None) -> set[str]:
    text = (learned_guidance or "").lower()
    guidance: set[str] = set()
    if any(term in text for term in ("concise", "brief", "short")):
        guidance.add("concise")
    if any(term in text for term in ("formal", "conservative", "legal tone")):
        guidance.add("formal")
    if any(term in text for term in ("plain english", "plain language", "simple language")):
        guidance.add("plain_language")
    if "facts first" in text or "facts before analysis" in text:
        guidance.add("facts_first")
    if "background" in text:
        guidance.add("background")
    if "risk" in text:
        guidance.add("risk")
    if any(term in text for term in ("recommendation", "recommendations", "next steps")):
        guidance.add("recommendation")
    return guidance


def _is_supported(fact: str, evidence_text: str) -> bool:
    fact_terms = set(_tokens(fact))
    if not fact_terms:
        return False
    evidence_terms = set(_tokens(evidence_text))
    required_matches = min(len(fact_terms), max(2, (len(fact_terms) + 1) // 2))
    return len(fact_terms.intersection(evidence_terms)) >= required_matches


def _tokens(text: str) -> list[str]:
    return [token.lower().strip("'") for token in _WORD_RE.findall(text or "") if token.strip("'")]


def _support_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _tokens(text):
        if len(token) < 3 or token in _SUPPORT_STOPWORDS:
            continue
        tokens.add(_support_token_root(token))
    return tokens


def _support_token_root(token: str) -> str:
    for suffix in ("'s", "ing", "ies", "ied", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            if suffix in {"ies", "ied"}:
                return token[: -len(suffix)] + "y"
            return token[: -len(suffix)]
    return token


def _first_present(value: Any, names: Sequence[str], default: Any) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default

    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _optional_str(value: Any) -> str | None:
    text = _clean_str(value)
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _string_mapping(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _clean_str(val) for key, val in value.items()}


def _string_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_clean_str(value),) if _clean_str(value) else ()
    try:
        return tuple(_clean_str(item) for item in value if _clean_str(item))
    except TypeError:
        text = _clean_str(value)
        return (text,) if text else ()


def _ensure_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = _clean_str(value)
        if clean and clean not in seen:
            result.append(clean)
            seen.add(clean)
    return result


def _truncate(text: str, limit: int) -> str:
    cleaned = _clean_str(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."
