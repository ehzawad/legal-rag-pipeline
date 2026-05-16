from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pipeline.drafting.grounding import SCAFFOLDING, SUPPORTED
from pipeline.drafting.memo import normalize_quote_text
from pipeline.schemas import Draft, EvidenceChunk


def validate_draft_contract(
    draft: Draft,
    grounding_report: Mapping[str, Any],
    *,
    require_entailment: bool = False,
) -> None:
    """Enforce non-relaxable draft invariants before artifacts are written."""

    summary = draft.case_summary
    if summary is None:
        raise ValueError("Draft contract failed: missing CaseFactSummary")
    if not draft.evidence:
        raise ValueError("Draft contract failed: draft evidence is empty")
    if not summary.evidence:
        raise ValueError("Draft contract failed: CaseFactSummary evidence is empty")
    if not summary.claims:
        raise ValueError("Draft contract failed: CaseFactSummary has no claims")

    draft_evidence = {chunk.evidence_id: chunk for chunk in draft.evidence}
    summary_evidence = {chunk.evidence_id: chunk for chunk in summary.evidence}
    factual_claims = [claim for claim in summary.claims if claim.claim_type != SCAFFOLDING]
    if not factual_claims:
        raise ValueError("Draft contract failed: CaseFactSummary has no factual claims")

    for claim in factual_claims:
        if not claim.citations:
            raise ValueError(f"Draft contract failed: factual claim {claim.claim_id} has no citations")
        if claim.grounding is None:
            raise ValueError(f"Draft contract failed: factual claim {claim.claim_id} was not grounded")
        if claim.grounding.status != SUPPORTED:
            raise ValueError(
                f"Draft contract failed: factual claim {claim.claim_id} has grounding status "
                f"{claim.grounding.status!r}"
            )
        if require_entailment and claim.grounding.entailed_count is None:
            raise ValueError(f"Draft contract failed: factual claim {claim.claim_id} has no entailment verdict")
        for citation in claim.citations:
            chunk = draft_evidence.get(citation.evidence_id)
            if chunk is None:
                raise ValueError(f"Draft contract failed: citation {citation.evidence_id!r} is not in draft evidence")
            if citation.evidence_id not in summary_evidence:
                raise ValueError(
                    f"Draft contract failed: citation {citation.evidence_id!r} is not in summary evidence"
                )
            if not citation.quote.strip():
                raise ValueError(f"Draft contract failed: claim {claim.claim_id} has an empty citation quote")
            if not _quote_is_grounded(citation.quote, chunk):
                raise ValueError(
                    f"Draft contract failed: quote for claim {claim.claim_id} is not grounded in "
                    f"{citation.evidence_id}"
                )

    unsupported = int(grounding_report.get("unsupported_claim_count") or 0)
    coverage = grounding_report.get("citation_coverage")
    pass_rate = grounding_report.get("grounding_pass_rate")
    if unsupported != 0:
        raise ValueError(f"Draft contract failed: unsupported_claim_count is {unsupported}")
    if coverage is not None and float(coverage) < 1.0:
        raise ValueError(f"Draft contract failed: citation coverage is {float(coverage):.2%}")
    if pass_rate is not None and float(pass_rate) < 1.0:
        raise ValueError(f"Draft contract failed: grounding pass rate is {float(pass_rate):.2%}")


def _quote_is_grounded(quote: str, chunk: EvidenceChunk) -> bool:
    return normalize_quote_text(quote) in normalize_quote_text(chunk.text)
