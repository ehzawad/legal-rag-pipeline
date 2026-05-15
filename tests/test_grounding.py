"""Verbatim-quote grounding contract."""

from __future__ import annotations

from pipeline.drafting.memo import (
    _validate_quote_grounding,
    normalize_quote_text,
)
from pipeline.schemas import DraftSection, EvidenceChunk


def _chunk(evidence_id: str, text: str) -> EvidenceChunk:
    return EvidenceChunk(
        evidence_id=evidence_id,
        document_id="doc1",
        filename="doc1.pdf",
        page_number=1,
        text=text,
    )


def test_normalize_quote_text_folds_punctuation_and_whitespace():
    raw = "  The “Party” shall—with notice—deliver ﬁnal copies.  "
    expected = 'the "party" shall-with notice-deliver final copies.'
    assert normalize_quote_text(raw) == expected


def test_validate_keeps_substring_quote_and_lowers_unsupported_flag():
    chunk = _chunk("E1", "The Party shall deliver final copies on time.")
    section = DraftSection(
        heading="Issue",
        body="The party must deliver. [E1]",
        evidence_ids=["E1"],
        citation_quotes={"E1": "deliver final copies"},
        unsupported=False,
    )
    validated, warnings = _validate_quote_grounding([section], evidence_by_id={"E1": chunk})
    [result] = validated
    assert result.evidence_ids == ["E1"]
    assert result.citation_quotes == {"E1": "deliver final copies"}
    assert result.unsupported is False
    assert warnings == []


def test_validate_strips_non_substring_quote_and_flips_to_unsupported():
    chunk = _chunk("E1", "The Party shall deliver final copies on time.")
    section = DraftSection(
        heading="Analysis",
        body="The party agreed to indemnify. [E1]",
        evidence_ids=["E1"],
        citation_quotes={"E1": "agreed to indemnify the buyer"},  # not in chunk
        unsupported=False,
    )
    validated, warnings = _validate_quote_grounding([section], evidence_by_id={"E1": chunk})
    [result] = validated
    assert result.evidence_ids == []
    assert result.citation_quotes == {}
    assert result.unsupported is True
    assert "[E1]" not in result.body
    assert warnings, "expected a draft warning when a quote was stripped"


def test_validate_handles_curly_quote_and_ligature_normalization():
    chunk = _chunk("E1", 'The "Party" shall produce final copies.')
    section = DraftSection(
        heading="Issue",
        body="See evidence. [E1]",
        evidence_ids=["E1"],
        # Curly quote + ligature should normalize through to the chunk text.
        citation_quotes={"E1": "the “Party” shall produce ﬁnal copies"},
        unsupported=False,
    )
    validated, _ = _validate_quote_grounding([section], evidence_by_id={"E1": chunk})
    assert validated[0].evidence_ids == ["E1"]
    assert validated[0].unsupported is False
