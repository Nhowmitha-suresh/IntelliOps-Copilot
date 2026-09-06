import datetime
import pytest
from sqlalchemy import Column, String, Integer, DateTime, Text, Index
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector
from app.embeddings.chunker import chunk_incident, estimate_tokens
from app.embeddings import pgvector_store, embedder
from app.embeddings.index_pipeline import run_indexing_pipeline

TestBase = declarative_base()
TestBase.__test__ = False


class TestIncidentChunk8(TestBase):
    __test__ = False
    __tablename__ = "test_incident_chunks_8"

    id = Column(String, primary_key=True)
    incident_id = Column(String, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(8), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def test_chunker_counts_and_sliding_window():
    # Small incident
    small_inc = {
        "id": "INC-TEST-1",
        "title": "Small bug",
        "description": "Short description",
        "error_log": "ERROR 404",
        "resolution": "Fixed",
    }
    chunks = chunk_incident(small_inc, target_tokens=300, overlap_tokens=50)
    assert len(chunks) == 1
    assert chunks[0].incident_id == "INC-TEST-1"
    assert chunks[0].chunk_index == 0

    # Long incident text
    long_desc = " ".join([f"word{i}" for i in range(500)])
    long_inc = {
        "id": "INC-TEST-2",
        "title": "Long incident",
        "description": long_desc,
        "error_log": "Long log content here",
        "resolution": "Long resolution description",
    }
    long_chunks = chunk_incident(long_inc, target_tokens=300, overlap_tokens=50)
    assert len(long_chunks) > 1
    for idx, c in enumerate(long_chunks):
        assert c.chunk_index == idx
        assert c.incident_id == "INC-TEST-2"
        assert c.token_count > 0


def test_embedder_provider_error():
    # Calling embed_texts with no key should raise EmbeddingProviderError
    with pytest.raises(embedder.EmbeddingProviderError):
        embedder.embed_texts(["Test embedding text string"])


def test_pgvector_store_roundtrip_dimension_8():
    engine = pgvector_store.get_engine()
    pgvector_store.init_db(engine=engine)
    TestBase.metadata.create_all(bind=engine)

    fake_vector_1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fake_vector_2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    test_payload = [
        {
            "id": "test-chunk-1",
            "incident_id": "INC-888",
            "chunk_index": 0,
            "chunk_text": "First test chunk text for dim 8",
            "embedding": fake_vector_1,
        },
        {
            "id": "test-chunk-2",
            "incident_id": "INC-888",
            "chunk_index": 1,
            "chunk_text": "Second test chunk text for dim 8",
            "embedding": fake_vector_2,
        },
    ]

    session = pgvector_store.get_session()
    try:
        inserted_ids = pgvector_store.upsert_chunks(
            test_payload, session=session, model_cls=TestIncidentChunk8
        )
        assert len(inserted_ids) == 2

        # Query vector close to fake_vector_1
        query_vec = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        results = pgvector_store.similarity_search(
            query_embedding=query_vec,
            top_k=2,
            session=session,
            model_cls=TestIncidentChunk8,
        )

        assert len(results) == 2
        top_chunk, top_distance = results[0]
        assert top_chunk["id"] == "test-chunk-1"
        assert top_distance < 0.2
    finally:
        session.close()


def test_index_pipeline_idempotency():
    # Mock embedding function returning 1536-dim dummy vectors
    def mock_embed(texts):
        return [[0.01] * 1536 for _ in texts]

    res1 = run_indexing_pipeline(embed_fn=mock_embed, force_reindex=True)
    assert res1["processed_incidents"] >= 1

    # Second run without force_reindex should skip unchanged incidents
    res2 = run_indexing_pipeline(embed_fn=mock_embed, force_reindex=False)
    assert res2["skipped_incidents"] == res1["processed_incidents"]
    assert res2["processed_incidents"] == 0
