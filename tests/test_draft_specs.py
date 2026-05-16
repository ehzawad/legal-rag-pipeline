from __future__ import annotations

import json

import pytest

from pipeline.drafting import generate_internal_memo
from pipeline.providers import ProviderUnavailable
from pipeline.schemas import EvidenceChunk


def _chunk() -> EvidenceChunk:
    return EvidenceChunk(
        evidence_id="E1",
        document_id="doc1",
        filename="doc1.pdf",
        page_number=1,
        text="The agreement requires Acme to deliver final copies by May 1.",
    )


def test_registered_draft_spec_accepts_claim_summary_payload():
    chunk = _chunk()
    payload = {
        "title": "Case Fact Summary",
        "claims": [
            {
                "section": "material_facts",
                "text": "Acme must deliver final copies by May 1.",
                "claim_type": "fact",
                "confidence": "high",
                "citations": [
                    {
                        "evidence_id": "E1",
                        "quote": "requires Acme to deliver final copies by May 1",
                    }
                ],
            }
        ],
        "warnings": [],
    }

    draft = generate_internal_memo(
        "Draft a fact summary.",
        [chunk],
        generator=lambda _prompt: json.dumps(payload),
        draft_type="case_fact_summary",
        case_id="case1",
    )

    assert draft.draft_type == "case_fact_summary"
    assert draft.case_summary is not None
    assert draft.case_summary.case_id == "case1"
    assert draft.case_summary.claims[0].citations[0].evidence_id == "E1"


def test_registered_draft_spec_rejects_old_section_payload_shape():
    chunk = _chunk()
    payload = {
        "title": "Case Fact Summary",
        "sections": [
            {
                "heading": "Facts",
                "body": "Acme must deliver final copies. [E1]",
                "evidence_ids": ["E1"],
                "citation_quotes": {"E1": "deliver final copies"},
            }
        ],
    }

    with pytest.raises(ProviderUnavailable, match="requires a case-fact claims response"):
        generate_internal_memo(
            "Draft a fact summary.",
            [chunk],
            generator=lambda _prompt: json.dumps(payload),
            draft_type="case_fact_summary",
        )


def test_non_open_question_scaffolding_claim_cannot_bypass_citation_contract():
    chunk = _chunk()
    payload = {
        "title": "Case Fact Summary",
        "claims": [
            {
                "section": "material_facts",
                "text": "Acme must deliver final copies by May 1.",
                "claim_type": "fact",
                "confidence": "high",
                "citations": [
                    {
                        "evidence_id": "E1",
                        "quote": "requires Acme to deliver final copies by May 1",
                    }
                ],
            },
            {
                "section": "material_facts",
                "text": "Acme breached the agreement.",
                "claim_type": "scaffolding",
                "confidence": "low",
                "citations": [],
            },
        ],
    }

    with pytest.raises(ProviderUnavailable, match="uncited factual claim"):
        generate_internal_memo(
            "Draft a fact summary.",
            [chunk],
            generator=lambda _prompt: json.dumps(payload),
            draft_type="case_fact_summary",
        )


def test_uncited_evidence_table_claim_cannot_bypass_citation_contract():
    chunk = _chunk()
    payload = {
        "title": "Case Fact Summary",
        "claims": [
            {
                "section": "material_facts",
                "text": "Acme must deliver final copies by May 1.",
                "claim_type": "fact",
                "confidence": "high",
                "citations": [
                    {
                        "evidence_id": "E1",
                        "quote": "requires Acme to deliver final copies by May 1",
                    }
                ],
            },
            {
                "section": "evidence_table",
                "text": "The evidence table says Acme breached the agreement.",
                "claim_type": "scaffolding",
                "confidence": "low",
                "citations": [],
            },
        ],
    }

    with pytest.raises(ProviderUnavailable, match="uncited factual claim"):
        generate_internal_memo(
            "Draft a fact summary.",
            [chunk],
            generator=lambda _prompt: json.dumps(payload),
            draft_type="case_fact_summary",
        )
