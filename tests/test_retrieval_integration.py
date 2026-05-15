"""End-to-end retrieve() exercising doc-cap and field-cap interaction.

Uses the fixture embedding provider so no live OpenAI call happens. Builds
a small in-memory RetrievalIndex from synthetic ProcessedDocuments, then
verifies the per-document diversity cap and the field-chunk cap behave as
intended together.
"""

from __future__ import annotations

from pipeline.config import ProviderConfig
from pipeline.retrieval.engine import build_index, retrieve
from pipeline.schemas import ExtractedField, PageText, ProcessedDocument, SourceDocument


def _doc(document_id: str, *, page_texts: list[str], field_value: str | None = None) -> ProcessedDocument:
    """Build a ProcessedDocument with one PageText per entry in page_texts."""
    source = SourceDocument(
        path=f"/fake/{document_id}.txt",
        document_id=document_id,
        filename=f"{document_id}.txt",
        mime_type="text/plain",
        sha256=document_id,
    )
    pages = [
        PageText(
            document_id=document_id,
            page_number=index + 1,
            text=text,
            extraction_method="fixture",
            confidence=0.95,
        )
        for index, text in enumerate(page_texts)
    ]
    fields: dict[str, ExtractedField] = {}
    if field_value is not None:
        fields["clause"] = ExtractedField(
            name="clause",
            value=field_value,
            confidence=0.9,
            evidence_ids=[f"{document_id}:p1:c0"],
        )
    return ProcessedDocument(source=source, pages=pages, fields=fields)


def _build_fixture_index_and_retrieve(
    *,
    documents: list[ProcessedDocument],
    task: str,
    top_k: int,
    max_field_chunks: int,
    max_chunks_per_document: int,
):
    config = ProviderConfig()
    index = build_index(documents, provider="fixture", config=config)
    return retrieve(
        index,
        task,
        top_k=top_k,
        provider="fixture",
        config=config,
        max_field_chunks=max_field_chunks,
        max_chunks_per_document=max_chunks_per_document,
    )


def test_retrieve_diversity_and_field_caps_apply_together():
    # Document A has many pages; B-E each have one page. Without the
    # per-document cap, A would crowd out B-E. The query terms appear in
    # every document so all chunks score positively under hybrid retrieval.
    long_pages = [
        f"Indemnification obligations clause page {n}: the supplier indemnifies the buyer for losses."
        for n in range(1, 9)
    ]
    documents = [
        _doc("A", page_texts=long_pages, field_value="Supplier indemnifies buyer for losses"),
        _doc("B", page_texts=["Indemnification clause: buyer indemnifies supplier for breach."]),
        _doc("C", page_texts=["Indemnification provisions cap the supplier obligations at fees paid."]),
        _doc("D", page_texts=["Indemnification clause applies only after written notice from buyer."]),
        _doc("E", page_texts=["Indemnification survives termination for two years past close."]),
    ]
    results = _build_fixture_index_and_retrieve(
        documents=documents,
        task="indemnification obligations between supplier and buyer",
        top_k=10,
        max_field_chunks=1,
        max_chunks_per_document=3,
    )

    assert len(results) == 10
    field_chunks = [chunk for chunk in results if chunk.metadata.get("is_field_chunk")]
    assert len(field_chunks) <= 1
    distinct_docs = {chunk.document_id for chunk in results}
    assert len(distinct_docs) >= 3, f"diversity cap should surface multiple docs, got {distinct_docs}"


def test_retrieve_without_doc_cap_collapses_to_dominant_document():
    # With max_chunks_per_document=0 the diversity cap is disabled. Document A
    # has many tightly-similar pages, so it dominates the top_k except for the
    # single permitted field chunk that survives the field-chunk cap.
    # B and C use vocabulary that shares no tokens with the query so neither
    # the fixture dense score nor BM25 give them positive ranking weight.
    long_pages = [
        f"Indemnification clause page {n}: the supplier indemnifies the buyer for losses incurred."
        for n in range(1, 11)
    ]
    documents = [
        _doc("A", page_texts=long_pages, field_value="Supplier indemnifies buyer for losses"),
        _doc(
            "B",
            page_texts=[
                "Cooking recipes: chopped carrots onions garlic bell pepper steaming pot stovetop nineteen."
            ],
        ),
        _doc(
            "C",
            page_texts=[
                "Grocery list: bread milk eggs cheese butter cilantro thyme rosemary nineteen morning."
            ],
        ),
    ]
    results = _build_fixture_index_and_retrieve(
        documents=documents,
        task="indemnification obligations between supplier and buyer",
        top_k=6,
        max_field_chunks=1,
        max_chunks_per_document=0,
    )

    assert len(results) == 6
    non_field = [chunk for chunk in results if not chunk.metadata.get("is_field_chunk")]
    # Every non-field chunk in the top-k comes from the dominant document A.
    assert all(chunk.document_id == "A" for chunk in non_field), (
        f"without the per-doc cap, the dominant doc should fill non-field slots, got {[c.document_id for c in non_field]}"
    )
