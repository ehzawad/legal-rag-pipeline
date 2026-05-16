"""Lexical-reporting contract for the playbook risk scorer.

Codex flagged two real bugs in the old `_contains_term` matcher:

1. High-severity findings could fire with empty `evidence_ids`, because term
   matches were taken from concatenated corpus text but evidence IDs were
   sourced only from retrieved chunks. Re-introduced as a regression test
   below: a term that appears only in a page (not in the retrieved chunks)
   must still produce a finding carrying a page-level source span.
2. The boundary character class `[a-z0-9]` allowed "short-term" to match
   "term", because the hyphen is outside the boundary. Re-introduced as
   `test_boundary_rejects_hyphen_attached_words`.
"""

from __future__ import annotations

from pipeline.playbooks.risk import (
    RISK_REPORT_SCHEMA_VERSION,
    _field_anchor_page,
    _term_pattern,
    score_playbook,
)
from pipeline.schemas import (
    Draft,
    EvidenceChunk,
    ExtractedField,
    PageText,
    ProcessedDocument,
    SourceDocument,
)


# ---------- matcher unit tests ----------


def _matches(term: str, text: str) -> bool:
    pattern = _term_pattern(term)
    return pattern is not None and pattern.search(text) is not None


def test_boundary_rejects_hyphen_attached_words():
    # "term" must not match inside "short-term"; the hyphen counts as a token boundary.
    assert not _matches("term", "we agreed to a short-term lease")
    assert not _matches("term", "many terms apply")
    assert _matches("term", "the initial term shall be five years")


def test_internal_whitespace_is_flexible():
    assert _matches("termination for convenience", "Termination\nfor\nConvenience clause")
    assert _matches("termination for convenience", "termination   for   convenience")


def test_internal_hyphen_and_space_are_equivalent():
    # The playbook may say "new york" but the contract may write "New-York".
    assert _matches("new york", "Venue lies in New-York County")
    assert _matches("new york", "Venue lies in New York County")


def test_phrase_order_matters_so_negation_inversion_does_not_match():
    # Codex's correction: "uncapped liability" should NOT match "Liability shall not
    # be uncapped" because the word order is reversed.
    assert not _matches("uncapped liability", "Liability shall not be uncapped under this Section.")
    # But a literal escalation phrase still fires.
    assert _matches("uncapped liability", "The Agreement permits uncapped liability for Provider.")


def test_pattern_is_case_insensitive():
    assert _matches("indemnify", "Customer shall INDEMNIFY Provider")


def test_empty_term_returns_no_pattern():
    assert _term_pattern("") is None
    assert _term_pattern("   ") is None


# ---------- end-to-end scoring tests ----------


def _doc(*, document_id: str, filename: str, pages: list[tuple[int, str]], fields: dict[str, ExtractedField] | None = None) -> ProcessedDocument:
    return ProcessedDocument(
        source=SourceDocument(
            path=f"/tmp/{filename}",
            document_id=document_id,
            filename=filename,
            mime_type="application/pdf",
            sha256="0" * 64,
        ),
        pages=[
            PageText(
                document_id=document_id,
                page_number=page_number,
                text=text,
                extraction_method="fixture",
                confidence=0.9,
            )
            for page_number, text in pages
        ],
        fields=fields or {},
    )


def _empty_draft() -> Draft:
    return Draft(
        draft_type="case_fact_summary",
        title="test",
        generated_at="2026-05-16T00:00:00Z",
        sections=[],
        evidence=[],
    )


def test_escalation_finding_carries_source_span_from_page_only():
    """The original `sole discretion` bug: escalation matched in a page but
    not in any retrieved chunk → old scorer produced `evidence_ids: []`."""

    document = _doc(
        document_id="docA",
        filename="alpha.pdf",
        pages=[
            (1, "Recitals page."),
            (
                3,
                "Provider may, in its sole discretion, withhold indemnification of any third-party claim.",
            ),
        ],
    )
    playbook = {
        "name": "test",
        "positions": [
            {
                "id": "indemnification",
                "title": "Indemnification",
                "severity": "high",
                "required_any": ["indemnification"],
                "escalation_terms": ["sole discretion"],
            }
        ],
    }
    report = score_playbook(processed=[document], evidence=[], draft=_empty_draft(), playbook=playbook)

    findings = report["findings"]
    [finding] = [f for f in findings if f["position_id"] == "indemnification"]
    assert finding["status"] == "escalation_triggered"
    assert finding["severity"] == "high"
    assert finding["matched_terms"] == ["sole discretion"]
    assert finding["source_spans"], "high-severity finding must carry at least one source span"
    [span] = finding["source_spans"]
    assert span["source_kind"] == "page"
    assert span["page_number"] == 3
    assert span["filename"] == "alpha.pdf"
    assert "sole discretion" in span["snippet"].casefold()
    # New contract: evidence_ids is chunk-only; page IDs surface via source_ids.
    assert finding["evidence_ids"] == []
    assert finding["source_ids"] == ["docA:p3"]
    assert "audit_warning" not in finding


def test_chunk_match_is_preferred_for_evidence_ids_when_available():
    document = _doc(
        document_id="docB",
        filename="beta.pdf",
        pages=[(1, "Termination for convenience permitted with thirty days notice.")],
    )
    chunk = EvidenceChunk(
        evidence_id="docB:c1",
        document_id="docB",
        filename="beta.pdf",
        page_number=1,
        text="Termination for convenience permitted with thirty days notice.",
    )
    playbook = {
        "name": "test",
        "positions": [
            {
                "id": "term_termination",
                "title": "Term and Termination",
                "severity": "medium",
                "required_any": ["termination"],
                "preferred_any": ["termination for convenience", "thirty days"],
            }
        ],
    }
    report = score_playbook(
        processed=[document],
        evidence=[chunk],
        draft=_empty_draft(),
        playbook=playbook,
    )
    passed = report["passed_positions"]
    assert passed, "preferred phrase present → position should pass"
    [entry] = passed
    assert entry["position_id"] == "term_termination"
    # Chunk source IDs go in evidence_ids; page IDs go in source_ids.
    assert "docB:c1" in entry["evidence_ids"]
    assert all(":c" in ev or ev == "" for ev in entry["evidence_ids"])
    assert "docB:p1" in entry["source_ids"]


def test_missing_required_term_has_no_source_spans():
    document = _doc(
        document_id="docC",
        filename="gamma.pdf",
        pages=[(1, "This contract addresses unrelated topics.")],
    )
    playbook = {
        "name": "test",
        "positions": [
            {
                "id": "data_protection",
                "title": "Data Protection",
                "severity": "medium",
                "required_any": ["personal data", "data protection"],
            }
        ],
    }
    report = score_playbook(processed=[document], evidence=[], draft=_empty_draft(), playbook=playbook)
    [finding] = report["findings"]
    assert finding["status"] == "missing_or_unconfirmed"
    assert finding["source_spans"] == []
    assert finding["evidence_ids"] == []
    assert finding["matched_terms"] == []


def test_no_high_severity_finding_without_source_span_invariant():
    """Run a realistic-shaped playbook against a tiny corpus and assert the
    invariant the operator UI depends on: every fired finding (escalation or
    review_preferred_position) is anchored to at least one source span."""

    document = _doc(
        document_id="docD",
        filename="delta.pdf",
        pages=[
            (1, "Provider's liability is uncapped for breaches of confidentiality."),
            (2, "Either party may terminate this Agreement for convenience upon thirty days written notice."),
        ],
    )
    playbook = {
        "name": "test",
        "positions": [
            {
                "id": "limitation_of_liability",
                "title": "Limitation of Liability",
                "severity": "high",
                "required_any": ["liability"],
                "preferred_any": ["liability cap", "fees paid"],
                "escalation_terms": ["uncapped"],
            },
            {
                "id": "term_termination",
                "title": "Term and Termination",
                "severity": "medium",
                "required_any": ["termination"],
                "preferred_any": ["termination for convenience"],
            },
        ],
    }
    report = score_playbook(processed=[document], evidence=[], draft=_empty_draft(), playbook=playbook)
    for finding in report["findings"]:
        if finding["status"] in {"escalation_triggered", "review_preferred_position"}:
            assert finding["source_spans"], f"fired finding without source span: {finding}"
            # source_ids carries the union (chunks + pages + fields); evidence_ids
            # may be empty in page-only matches by design.
            assert finding["source_ids"], f"fired finding without source ids: {finding}"


def test_report_carries_schema_version():
    """Resume invalidation in orchestration/run.py depends on this field."""

    document = _doc(document_id="docZ", filename="zeta.pdf", pages=[(1, "anything")])
    report = score_playbook(
        processed=[document],
        evidence=[],
        draft=_empty_draft(),
        playbook={"name": "test", "positions": []},
    )
    assert report["schema_version"] == RISK_REPORT_SCHEMA_VERSION


def test_en_dash_and_soft_hyphen_normalize_to_match():
    """Codex finding: 'data–protection' / 'data­protection' must match 'data protection'."""

    document = _doc(
        document_id="dEN",
        filename="endash.pdf",
        pages=[(1, "We discuss data–protection and data­protection clauses here.")],
    )
    playbook = {
        "name": "test",
        "positions": [
            {
                "id": "data_protection",
                "title": "Data Protection",
                "severity": "low",
                "required_any": ["data protection"],
            }
        ],
    }
    report = score_playbook(processed=[document], evidence=[], draft=_empty_draft(), playbook=playbook)
    assert report["passed_positions"], "en-dash and soft-hyphen variants should both match the required term"


def test_ligature_in_text_matches_ascii_term():
    """OCR ligatures like 'ﬁ' must not silently miss the 'fi' in playbook terms."""

    document = _doc(
        document_id="dLIG",
        filename="lig.pdf",
        pages=[(1, "The party shall pay all ﬁnal amounts within thirty days.")],
    )
    playbook = {
        "name": "test",
        "positions": [
            {
                "id": "final_amount",
                "title": "Final Amount",
                "severity": "low",
                "required_any": ["final amounts"],
            }
        ],
    }
    report = score_playbook(processed=[document], evidence=[], draft=_empty_draft(), playbook=playbook)
    assert report["passed_positions"], "ligature ﬁ must normalize to 'fi' before matching"


def test_unicode_letter_blocks_false_positive_at_word_boundary():
    """Boundary class uses Unicode \\w so 'term' does NOT match inside 'termíno'."""

    document = _doc(
        document_id="dUNI",
        filename="uni.pdf",
        pages=[(1, "The contract uses Spanish termíno frequently. No other usage.")],
    )
    playbook = {
        "name": "test",
        "positions": [
            {
                "id": "term_termination",
                "title": "Term and Termination",
                "severity": "low",
                "required_any": ["term"],
            }
        ],
    }
    report = score_playbook(processed=[document], evidence=[], draft=_empty_draft(), playbook=playbook)
    findings = report["findings"]
    assert findings and findings[0]["status"] == "missing_or_unconfirmed", (
        "'term' should not match inside 'termíno'; current findings: " + str(findings)
    )


def test_field_anchor_page_handles_chunk_suffix():
    """Codex finding: evidence IDs like 'doc:p3:c0' must yield page 3, not 0."""

    assert _field_anchor_page(["docX:p3:c0"]) == 3
    assert _field_anchor_page(["docX:p12"]) == 12
    assert _field_anchor_page(["docX:fields:parties", "docX:p7:c2"]) == 7
    assert _field_anchor_page(["docX:malformed"]) == 0
    assert _field_anchor_page([]) == 0


def test_evidence_ids_excludes_page_and_field_ids():
    """Forward-compat: downstream consumers may assume evidence_ids resolve to chunks."""

    document = _doc(
        document_id="dMIX",
        filename="mix.pdf",
        pages=[(2, "Provider may, in its sole discretion, refuse service.")],
    )
    chunk = EvidenceChunk(
        evidence_id="dMIX:p2:c0",
        document_id="dMIX",
        filename="mix.pdf",
        page_number=2,
        text="Provider may, in its sole discretion, refuse service.",
    )
    playbook = {
        "name": "test",
        "positions": [
            {
                "id": "indemnification",
                "title": "Indemnification",
                "severity": "high",
                "required_any": ["service"],
                "escalation_terms": ["sole discretion"],
            }
        ],
    }
    report = score_playbook(processed=[document], evidence=[chunk], draft=_empty_draft(), playbook=playbook)
    [finding] = [f for f in report["findings"] if f["position_id"] == "indemnification"]
    assert finding["evidence_ids"] == ["dMIX:p2:c0"]
    for evidence_id in finding["evidence_ids"]:
        assert ":c" in evidence_id, f"evidence_ids must be chunk-shaped; got {evidence_id}"
    # source_ids carries both the chunk and the page.
    assert "dMIX:p2:c0" in finding["source_ids"]
    assert "dMIX:p2" in finding["source_ids"]


def test_field_value_can_be_a_source():
    document = _doc(
        document_id="docE",
        filename="epsilon.pdf",
        pages=[(1, "Body text without the magic phrase.")],
        fields={
            "governing_law": ExtractedField(
                name="governing_law",
                value="The State of New York",
                confidence=0.9,
                evidence_ids=["docE:p1"],
            )
        },
    )
    playbook = {
        "name": "test",
        "positions": [
            {
                "id": "governing_law",
                "title": "Governing Law",
                "severity": "low",
                "required_any": ["new york"],
            }
        ],
    }
    report = score_playbook(processed=[document], evidence=[], draft=_empty_draft(), playbook=playbook)
    passed = report["passed_positions"]
    assert passed, "field-only match should satisfy the required check"
