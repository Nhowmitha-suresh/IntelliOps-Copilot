from typing import List, Dict, Any
from app.retrieval.reranker import rerank_incidents, extract_keywords_from_query
from app.retrieval.retriever import retrieve_similar_incidents


def test_extract_keywords_from_query():
    query = "Production deployment failed due to docker port 8000 conflict"
    keywords = extract_keywords_from_query(query)
    assert "production" in keywords
    assert "docker" in keywords
    assert "port" in keywords
    assert "8000" in keywords


def test_reranker_metadata_boosting():
    candidates = [
        {
            "incident_id": "INC-001",
            "similarity_score": 0.75,
            "incident_details": {
                "environment": "production",
                "tags": ["docker", "networking"],
            },
        },
        {
            "incident_id": "INC-002",
            "similarity_score": 0.78,
            "incident_details": {
                "environment": "staging",
                "tags": ["python", "pydantic"],
            },
        },
    ]

    # Query matching production and docker
    query = "Production container deployment issue with docker tag"
    reranked = rerank_incidents(query, candidates, boost_amount=0.1)

    # INC-001 should be boosted above INC-002 due to production + docker matches
    assert len(reranked) == 2
    assert reranked[0]["incident_id"] == "INC-001"
    assert reranked[0]["similarity_score"] > 0.75


def test_retriever_deduplication_and_threshold_filtering(caplog):
    # Mock embedding function returning dummy vector
    def mock_embed(texts: List[str]) -> List[List[float]]:
        return [[0.1] * 1536]

    # Mock search function returning multiple chunks from same incident + low score items
    # Distance d -> similarity = 1 - d
    # d = 0.1 -> sim = 0.9 (INC-100 chunk 0)
    # d = 0.2 -> sim = 0.8 (INC-100 chunk 1, duplicate, should be discarded in favor of chunk 0)
    # d = 0.5 -> sim = 0.5 (INC-200, below threshold 0.7, should be filtered out)
    def mock_search(query_vec: List[float], top_k: int) -> List[Any]:
        return [
            ({"incident_id": "INC-100", "chunk_index": 0, "chunk_text": "Top chunk"}, 0.1),
            ({"incident_id": "INC-100", "chunk_index": 1, "chunk_text": "Duplicate chunk"}, 0.2),
            ({"incident_id": "INC-200", "chunk_index": 0, "chunk_text": "Low score chunk"}, 0.5),
        ]

    results = retrieve_similar_incidents(
        query_text="production failure log",
        top_k=5,
        threshold=0.7,
        embed_fn=mock_embed,
        search_fn=mock_search,
    )

    # INC-200 should be filtered out because similarity 0.5 < threshold 0.7
    # INC-100 should be deduplicated so only 1 result is returned
    assert len(results) == 1
    assert results[0]["incident_id"] == "INC-100"
    assert results[0]["chunk_index"] == 0
