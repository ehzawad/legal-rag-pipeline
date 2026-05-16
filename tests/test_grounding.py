"""Verbatim-quote grounding contract."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pipeline.drafting.grounding import apply_claim_grounding
from pipeline.drafting.memo import normalize_quote_text, sections_from_case_summary
from pipeline.drafting.validation import validate_draft_contract
from pipeline.schemas import CaseFactSummary, ClaimCitation, Draft, EvidenceChunk, FactClaim


def _chunk(evidence_id: str, text: str) -> EvidenceChunk:
    return EvidenceChunk(
        evidence_id=evidence_id,
        document_id="doc1",
        filename="doc1.pdf",
        page_number=1,
        text=text,
    )


def _summary(chunk: EvidenceChunk, quote: str) -> CaseFactSummary:
    return CaseFactSummary(
        case_id="case1",
        generated_at="2026-05-17T00:00:00+00:00",
        title="Case Fact Summary",
        section_order=["material_facts"],
        claims=[
            FactClaim(
                claim_id="claim1",
                section_id="material_facts",
                section="Material Facts",
                text="The party must deliver final copies.",
                claim_type="fact",
                confidence="medium",
                citations=[ClaimCitation(evidence_id=chunk.evidence_id, quote=quote)],
            )
        ],
        evidence=[chunk],
    )


def _draft(summary: CaseFactSummary, chunk: EvidenceChunk) -> Draft:
    return Draft(
        draft_type="case_fact_summary",
        title=summary.title,
        generated_at=summary.generated_at,
        sections=[],
        evidence=[chunk],
        case_summary=summary,
    )


def test_normalize_quote_text_folds_punctuation_and_whitespace():
    raw = "  The “Party” shall—with notice—deliver ﬁnal copies.  "
    expected = 'the "party" shall-with notice-deliver final copies.'
    assert normalize_quote_text(raw) == expected


def test_grounding_accepts_substring_quote_and_contract_passes():
    chunk = _chunk("E1", "The Party shall deliver final copies on time.")
    summary = _summary(chunk, "deliver final copies")

    grounded, report = apply_claim_grounding(summary)
    draft = replace(_draft(grounded, chunk), sections=sections_from_case_summary(grounded))

    validate_draft_contract(draft, report)
    assert report["unsupported_claim_count"] == 0
    assert grounded.claims[0].citations[0].substring_grounded is True


def test_grounding_rejects_non_substring_quote_before_artifacts():
    chunk = _chunk("E1", "The Party shall deliver final copies on time.")
    summary = _summary(chunk, "agreed to indemnify the buyer")

    grounded, report = apply_claim_grounding(summary)
    draft = replace(_draft(grounded, chunk), sections=sections_from_case_summary(grounded))

    assert report["unsupported_claim_count"] == 1
    assert grounded.claims[0].grounding is not None
    assert grounded.claims[0].grounding.status == "ungrounded"
    with pytest.raises(ValueError, match="grounding status"):
        validate_draft_contract(draft, report)


def test_grounding_handles_curly_quote_and_ligature_normalization():
    chunk = _chunk("E1", 'The "Party" shall produce final copies.')
    summary = _summary(chunk, "the “Party” shall produce ﬁnal copies")

    grounded, report = apply_claim_grounding(summary)
    draft = replace(_draft(grounded, chunk), sections=sections_from_case_summary(grounded))

    validate_draft_contract(draft, report)
    assert grounded.claims[0].citations[0].substring_grounded is True
