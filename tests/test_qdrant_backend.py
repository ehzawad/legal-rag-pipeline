from __future__ import annotations

from pathlib import Path

from pipeline.config import ProviderConfig
from pipeline.retrieval import engine
from pipeline.retrieval.engine import build_index, load_index, retrieve, save_index
from pipeline.schemas import PageText, ProcessedDocument, SourceDocument


def _doc(document_id: str, text: str) -> ProcessedDocument:
    return ProcessedDocument(
        source=SourceDocument(
            path=f"/fake/{document_id}.txt",
            document_id=document_id,
            filename=f"{document_id}.txt",
            mime_type="text/plain",
            sha256=document_id,
        ),
        pages=[
            PageText(
                document_id=document_id,
                page_number=1,
                text=text,
                extraction_method="fixture",
                confidence=0.99,
            )
        ],
        fields={},
    )


def _documents() -> list[ProcessedDocument]:
    return [
        _doc("alpha", "Termination notice and cure period appear in the supplier agreement."),
        _doc("bravo", "Governing law is Delaware and venue is New Castle County."),
        _doc("charlie", "Termination for convenience includes notice and transition assistance."),
    ]


def _qdrant_config(tmp_path: Path, *, collection: str = "test_qdrant") -> ProviderConfig:
    return ProviderConfig(
        index_backend="qdrant",
        retrieval_mode="dense",
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection=collection,
    )


def _memory_config() -> ProviderConfig:
    return ProviderConfig(index_backend="memory", retrieval_mode="dense")


def _delete_collection(config: ProviderConfig, collection: str) -> None:
    client, _models = engine._qdrant_client_and_models(config)
    try:
        if engine._qdrant_collection_exists(client, collection):
            client.delete_collection(collection_name=collection)
    finally:
        engine._close_qdrant_client(client)


def test_qdrant_path_backend_matches_memory_dense_ranking(tmp_path: Path):
    documents = _documents()
    task = "termination notice cure period"
    qdrant_config = _qdrant_config(tmp_path, collection="roundtrip")
    memory_config = _memory_config()

    qdrant_index = build_index(documents, provider="fixture", config=qdrant_config, include_field_chunks=False)
    memory_index = build_index(documents, provider="fixture", config=memory_config, include_field_chunks=False)

    qdrant_results = retrieve(qdrant_index, task, top_k=3, provider="fixture", config=qdrant_config)
    memory_results = retrieve(memory_index, task, top_k=3, provider="fixture", config=memory_config)

    assert qdrant_index.index_backend == "qdrant"
    assert qdrant_index.qdrant_collection.startswith("roundtrip_")
    assert qdrant_index.qdrant_collection != "roundtrip"
    assert [chunk.document_id for chunk in qdrant_results] == [chunk.document_id for chunk in memory_results]


def test_qdrant_rebuilds_missing_collection_from_persisted_index(tmp_path: Path):
    config = _qdrant_config(tmp_path, collection="reload")
    index_path = tmp_path / "retrieval_index.json"
    index = build_index(_documents(), provider="fixture", config=config, include_field_chunks=False)
    save_index(index, index_path)
    _delete_collection(config, index.qdrant_collection)

    loaded = load_index(index_path)
    results = retrieve(loaded, "termination rights", top_k=2, provider="fixture", config=config)

    assert [chunk.document_id for chunk in results]
    assert loaded.qdrant_collection == index.qdrant_collection
    client, _models = engine._qdrant_client_and_models(config)
    try:
        assert engine._qdrant_collection_exists(client, index.qdrant_collection)
    finally:
        engine._close_qdrant_client(client)


def test_qdrant_rebuilds_stale_collection_with_same_name(tmp_path: Path):
    config = _qdrant_config(tmp_path, collection="stale")
    index = build_index(_documents(), provider="fixture", config=config, include_field_chunks=False)
    collection = index.qdrant_collection
    vector_size = len(index.embeddings[0])

    client, models = engine._qdrant_client_and_models(config)
    try:
        client.delete_collection(collection_name=collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=position,
                    vector=[1.0] + [0.0] * (vector_size - 1),
                    payload={
                        "index_digest": "stale",
                        "position": position,
                        "evidence_id": "wrong",
                        "chunk_hash": "wrong",
                        "embedding_model": index.embedding_model,
                        "embedding_backend_id": index.embedding_backend_id,
                        "vector_size": vector_size,
                    },
                )
                for position in range(len(index.chunks))
            ],
            wait=True,
        )
    finally:
        engine._close_qdrant_client(client)

    results = retrieve(index, "termination notice cure period", top_k=1, provider="fixture", config=config)

    assert results[0].document_id == "alpha"
    client, _models = engine._qdrant_client_and_models(config)
    try:
        records = client.retrieve(collection_name=collection, ids=[0], with_payload=True, with_vectors=False)
    finally:
        engine._close_qdrant_client(client)
    expected_digest = engine._qdrant_index_digest(
        index.chunks,
        embedding_model=index.embedding_model,
        embedding_backend_id=index.embedding_backend_id,
    )
    assert records[0].payload["index_digest"] == expected_digest
