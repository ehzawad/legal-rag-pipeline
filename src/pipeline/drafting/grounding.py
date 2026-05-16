from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from pipeline.drafting.memo import normalize_quote_text
from pipeline.schemas import CaseFactSummary, ClaimCitation, EvidenceChunk, FactClaim, GroundingVerdict


SUPPORTED = "grounded"
UNCITED = "uncited"
UNGROUNDED = "ungrounded"
SCAFFOLDING = "scaffolding"
UNSUPPORTED = "not_supported"
CONTRADICTED = "contradicted"
UNCLEAR = "unclear"


def apply_claim_grounding(
    summary: CaseFactSummary,
    *,
    draft_type: str = "case_fact_summary",
    entailment_results: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[CaseFactSummary, dict[str, Any]]:
    """Attach claim-level grounding verdicts and return a serializable report."""

    evidence_by_id = {chunk.evidence_id: chunk for chunk in summary.evidence}
    entailment_results = entailment_results or {}
    grounded_claims: list[FactClaim] = []
    report_claims: list[dict[str, Any]] = []
    counts = {
        SUPPORTED: 0,
        UNCITED: 0,
        UNGROUNDED: 0,
        SCAFFOLDING: 0,
        UNSUPPORTED: 0,
        CONTRADICTED: 0,
        UNCLEAR: 0,
    }
    for claim in summary.claims:
        grounded, verdict = _ground_claim(claim, evidence_by_id, entailment_results.get(claim.claim_id))
        counts[verdict.status] = counts.get(verdict.status, 0) + 1
        grounded_claims.append(grounded)
        report_claims.append(
            {
                "claim_id": grounded.claim_id,
                "section_id": grounded.section_id,
                "section": grounded.section,
                "claim_type": grounded.claim_type,
                "text": grounded.text,
                "status": verdict.status,
                "action": verdict.action,
                "reason": verdict.reason,
                "cited_evidence_ids": list(verdict.cited_evidence_ids),
                "substring_grounded_count": verdict.substring_grounded_count,
                "entailed_count": verdict.entailed_count,
            }
        )
    non_scaffolding = sum(1 for claim in grounded_claims if claim.claim_type != SCAFFOLDING)
    grounded = counts.get(SUPPORTED, 0)
    failed = counts.get(UNCITED, 0) + counts.get(UNGROUNDED, 0) + counts.get(UNSUPPORTED, 0) + counts.get(CONTRADICTED, 0)
    report = {
        "case_id": summary.case_id,
        "draft_type": draft_type,
        "total_claims": len(grounded_claims),
        "non_scaffolding_claims": non_scaffolding,
        "grounded": grounded,
        "entailed": sum(
            1
            for claim in grounded_claims
            if claim.claim_type != SCAFFOLDING
            and claim.grounding is not None
            and claim.grounding.entailed_count is not None
            and claim.grounding.entailed_count > 0
        ),
        "uncited": counts.get(UNCITED, 0),
        "ungrounded": counts.get(UNGROUNDED, 0),
        "not_supported": counts.get(UNSUPPORTED, 0),
        "contradicted": counts.get(CONTRADICTED, 0),
        "unclear": counts.get(UNCLEAR, 0),
        "scaffolding": counts.get(SCAFFOLDING, 0),
        "citation_coverage": _citation_coverage(grounded_claims),
        "grounding_pass_rate": grounded / non_scaffolding if non_scaffolding else None,
        "unsupported_claim_count": failed,
        "claims": report_claims,
    }
    return replace(summary, claims=grounded_claims), report


def _ground_claim(
    claim: FactClaim,
    evidence_by_id: Mapping[str, EvidenceChunk],
    entailment: Mapping[str, Any] | None,
) -> tuple[FactClaim, GroundingVerdict]:
    if claim.claim_type == SCAFFOLDING:
        verdict = GroundingVerdict(
            status=SCAFFOLDING,
            reason="Scaffolding claim is outside factual grounding denominator.",
            action="keep",
        )
        return replace(claim, grounding=verdict), verdict
    if not claim.citations:
        verdict = GroundingVerdict(
            status=UNCITED,
            reason="Non-scaffolding claim has no citation.",
            action="move_to_open_questions",
        )
        return replace(claim, grounding=verdict), verdict

    citations: list[ClaimCitation] = []
    grounded_count = 0
    for citation in claim.citations:
        chunk = evidence_by_id.get(citation.evidence_id)
        grounded = bool(chunk and _quote_is_grounded(citation.quote, chunk.text))
        if grounded:
            grounded_count += 1
        citations.append(replace(citation, substring_grounded=grounded))

    if grounded_count < len(citations):
        verdict = GroundingVerdict(
            status=UNGROUNDED,
            reason="One or more citation quotes are not verbatim substrings of cited evidence.",
            cited_evidence_ids=[citation.evidence_id for citation in citations],
            substring_grounded_count=grounded_count,
            action="revise",
        )
        return replace(claim, citations=citations, grounding=verdict), verdict

    entailed_count = None
    if entailment:
        status = str(entailment.get("verdict") or entailment.get("status") or "").strip().lower()
        if status in {"contradicted", "contradicts", "contradiction"}:
            verdict = GroundingVerdict(
                status=CONTRADICTED,
                reason=str(entailment.get("reason") or "Entailment judge reported contradiction."),
                cited_evidence_ids=[citation.evidence_id for citation in citations],
                substring_grounded_count=grounded_count,
                entailed_count=0,
                action="move_to_open_questions",
            )
            return replace(claim, citations=citations, grounding=verdict), verdict
        if status in {"not_supported", "neutral", "unsupported"}:
            verdict = GroundingVerdict(
                status=UNSUPPORTED,
                reason=str(entailment.get("reason") or "Entailment judge did not find support."),
                cited_evidence_ids=[citation.evidence_id for citation in citations],
                substring_grounded_count=grounded_count,
                entailed_count=0,
                action="move_to_open_questions",
            )
            return replace(claim, citations=citations, grounding=verdict), verdict
        if status in {"unclear", "unknown"}:
            verdict = GroundingVerdict(
                status=UNCLEAR,
                reason=str(entailment.get("reason") or "Entailment judge marked the claim unclear."),
                cited_evidence_ids=[citation.evidence_id for citation in citations],
                substring_grounded_count=grounded_count,
                entailed_count=0,
                action="revise",
            )
            return replace(claim, citations=citations, grounding=verdict), verdict
        entailed_count = len(citations) if status in {"entailed", "supported"} else None

    verdict = GroundingVerdict(
        status=SUPPORTED,
        reason="All citation quotes are grounded in cited evidence.",
        cited_evidence_ids=[citation.evidence_id for citation in citations],
        substring_grounded_count=grounded_count,
        entailed_count=entailed_count,
        action="keep",
    )
    return replace(claim, citations=citations, grounding=verdict), verdict


def _quote_is_grounded(quote: str, chunk_text: str) -> bool:
    needle = normalize_quote_text(quote)
    haystack = normalize_quote_text(chunk_text)
    return bool(needle and needle in haystack)


def _citation_coverage(claims: list[FactClaim]) -> float | None:
    factual = [claim for claim in claims if claim.claim_type != SCAFFOLDING]
    if not factual:
        return None
    cited = [claim for claim in factual if claim.citations]
    return len(cited) / len(factual)
