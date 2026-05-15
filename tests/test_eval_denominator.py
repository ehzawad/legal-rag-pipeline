"""citation_id_validity denominator excludes unsupported review sections."""

from __future__ import annotations

from pipeline.evaluation.report import _evaluate_draft_payload


def _section(heading: str, *, evidence_ids: list[str], unsupported: bool, body: str = "") -> dict:
    return {
        "heading": heading,
        "body": body or f"{heading} body",
        "evidence_ids": evidence_ids,
        "citation_quotes": {eid: f"quote for {eid}" for eid in evidence_ids},
        "unsupported": unsupported,
    }


def _evidence(evidence_id: str, text: str = "quote for the cited section") -> dict:
    return {
        "evidence_id": evidence_id,
        "document_id": "doc",
        "filename": "doc.pdf",
        "page_number": 1,
        "text": text,
    }


def test_unsupported_sections_excluded_from_denominator():
    # Two sections, one supported and one explicitly marked unsupported.
    # The unsupported one is a review hook (e.g., "Unsupported or Unclear Facts"),
    # so the denominator should be 1, not 2. citation_id_validity stays 100%.
    draft = {
        "title": "First-Pass Internal Memo",
        "sections": [
            _section("Issue", evidence_ids=["E1"], unsupported=False, body="Issue body [E1]"),
            _section("Unsupported or Unclear Facts", evidence_ids=[], unsupported=True),
        ],
        "evidence": [_evidence("E1", text="quote for the cited section quote for E1")],
        "warnings": [],
    }
    result = _evaluate_draft_payload(draft)
    assert result.citation_id_validity == 1.0
    assert result.unsupported_section_count == 1


def test_eligible_section_without_evidence_ids_drags_metric_down():
    # A non-unsupported section with empty evidence_ids is a real failure:
    # it claims to be a factual section but has no citations. Denominator
    # includes it; numerator excludes it. citation_id_validity = 0.
    draft = {
        "title": "First-Pass Internal Memo",
        "sections": [
            _section("Issue", evidence_ids=[], unsupported=False, body="Issue body without citations"),
        ],
        "evidence": [_evidence("E1")],
        "warnings": [],
    }
    result = _evaluate_draft_payload(draft)
    assert result.citation_id_validity == 0.0
    assert result.unsupported_section_count == 0


def test_supported_section_with_evidence_id_not_in_retrieved_fails():
    # The drafter declared E2 but E2 is not in the retrieved evidence set.
    # This must deflate citation_id_validity to 0%.
    draft = {
        "title": "First-Pass Internal Memo",
        "sections": [
            _section("Issue", evidence_ids=["E2"], unsupported=False, body="Issue body [E2]"),
        ],
        "evidence": [_evidence("E1")],
        "warnings": [],
    }
    result = _evaluate_draft_payload(draft)
    assert result.citation_id_validity == 0.0


def test_no_eligible_sections_returns_zero_without_dividing_by_zero():
    # Edge case: every section is marked unsupported. The metric is undefined
    # by construction (no denominator); the implementation reports 0.0 and
    # surfaces a note rather than raising.
    draft = {
        "title": "First-Pass Internal Memo",
        "sections": [
            _section("Review Flags", evidence_ids=[], unsupported=True),
        ],
        "evidence": [],
        "warnings": [],
    }
    result = _evaluate_draft_payload(draft)
    assert result.citation_id_validity == 0.0
    assert result.unsupported_section_count == 1
