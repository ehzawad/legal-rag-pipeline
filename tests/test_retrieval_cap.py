"""Field-chunk cap and per-document diversity cap in retrieval."""

from __future__ import annotations

from pipeline.retrieval.engine import _cap_field_chunks, _cap_per_document
from pipeline.schemas import EvidenceChunk


def _raw(evidence_id: str, score: float = 1.0, document_id: str = "doc", filename: str | None = None) -> EvidenceChunk:
    return EvidenceChunk(
        evidence_id=evidence_id,
        document_id=document_id,
        filename=filename or f"{document_id}.pdf",
        page_number=1,
        text="raw page text",
        score=score,
        metadata={"evidence_kind": "source-page"},
    )


def _field(evidence_id: str, score: float = 1.0) -> EvidenceChunk:
    return EvidenceChunk(
        evidence_id=evidence_id,
        document_id="doc",
        filename="doc.pdf",
        page_number=0,
        text="extracted fields",
        score=score,
        metadata={"evidence_kind": "derived-fields", "is_field_chunk": True},
    )


def test_cap_zero_drops_field_chunks_to_overflow_then_caps_top_k():
    candidates = [_field("a:fields"), _raw("a:p1:c0"), _field("b:fields"), _raw("b:p1:c0")]
    result = _cap_field_chunks(candidates, top_k=3, max_field_chunks=0)
    # raw pages keep their slots; field chunks slide to overflow but still allowed up to top_k.
    assert [c.evidence_id for c in result[:2]] == ["a:p1:c0", "b:p1:c0"]
    assert len(result) == 3
    assert result[2].metadata.get("is_field_chunk") is True


def test_cap_one_keeps_exactly_one_field_chunk_before_overflow():
    candidates = [_field("a:fields"), _field("b:fields"), _raw("a:p1:c0"), _raw("b:p1:c0")]
    result = _cap_field_chunks(candidates, top_k=3, max_field_chunks=1)
    ids = [c.evidence_id for c in result]
    # The first field chunk wins primary slot; the second falls into overflow.
    assert "a:fields" in ids
    assert ids[0] == "a:fields"
    # The two raw chunks fit in the remaining slots.
    assert "a:p1:c0" in ids and "b:p1:c0" in ids
    assert len(result) == 3


def test_cap_respects_top_k_when_primary_already_full():
    candidates = [_raw("a:p1:c0"), _raw("b:p1:c0"), _raw("c:p1:c0"), _field("a:fields")]
    result = _cap_field_chunks(candidates, top_k=2, max_field_chunks=1)
    assert [c.evidence_id for c in result] == ["a:p1:c0", "b:p1:c0"]


def test_cap_per_document_promotes_diverse_docs_when_one_dominates():
    # Eight chunks from doc A, one each from B and C. Without a per-document
    # cap, the top-k would be entirely from A and gold retrieval against a
    # multi-document review query would collapse to one source. The cap
    # guarantees B and C get into the primary slots even though they score
    # lower than every A chunk; remaining slots up to top_k are filled from
    # the overflow (same fill-to-top_k policy the field-chunk cap uses).
    candidates = (
        [_raw(f"a:p{i}:c0", score=1.0 - i * 0.001, document_id="A") for i in range(8)]
        + [_raw("b:p1:c0", score=0.5, document_id="B")]
        + [_raw("c:p1:c0", score=0.4, document_id="C")]
    )
    result = _cap_per_document(candidates, top_k=6, max_chunks_per_document=3)
    assert len(result) == 6
    by_doc: dict[str, int] = {}
    for chunk in result:
        by_doc[chunk.document_id] = by_doc.get(chunk.document_id, 0) + 1
    # Critical property: B and C survive into the top-k even though A
    # outscores them. That is the diversity guarantee.
    assert "B" in by_doc and "C" in by_doc
    # The first three primary slots are the first three A chunks in score order.
    assert [c.evidence_id for c in result[:3]] == ["a:p0:c0", "a:p1:c0", "a:p2:c0"]
    # B and C come next in primary order before overflow fills the rest.
    assert {result[3].document_id, result[4].document_id} == {"B", "C"}


def test_cap_per_document_with_zero_disables_cap():
    candidates = [_raw(f"a:p{i}:c0", document_id="A") for i in range(5)]
    result = _cap_per_document(candidates, top_k=4, max_chunks_per_document=0)
    assert len(result) == 4
    assert all(c.document_id == "A" for c in result)


def test_cap_per_document_uses_filename_when_document_id_blank():
    # Defensive: an old-format chunk missing document_id should fall back to
    # filename so the diversity cap still applies.
    a1 = EvidenceChunk(evidence_id="x", document_id="", filename="legacy.pdf", page_number=1, text="t")
    a2 = EvidenceChunk(evidence_id="y", document_id="", filename="legacy.pdf", page_number=2, text="t")
    a3 = EvidenceChunk(evidence_id="z", document_id="", filename="legacy.pdf", page_number=3, text="t")
    b1 = EvidenceChunk(evidence_id="w", document_id="", filename="other.pdf", page_number=1, text="t")
    result = _cap_per_document([a1, a2, a3, b1], top_k=3, max_chunks_per_document=1)
    filenames = [c.filename for c in result]
    assert filenames == ["legacy.pdf", "other.pdf", "legacy.pdf"]


def test_cap_recognizes_field_chunks_via_suffix_when_metadata_missing():
    # Defensive: an old-format chunk without is_field_chunk metadata but with
    # the conventional ":fields" suffix should still be treated as a field chunk.
    legacy_field = EvidenceChunk(
        evidence_id="x:fields",
        document_id="x",
        filename="x.pdf",
        page_number=0,
        text="extracted fields",
        score=1.0,
        metadata={},
    )
    candidates = [legacy_field, _raw("x:p1:c0")]
    result = _cap_field_chunks(candidates, top_k=2, max_field_chunks=0)
    assert result[0].evidence_id == "x:p1:c0"
    assert result[1].evidence_id == "x:fields"
