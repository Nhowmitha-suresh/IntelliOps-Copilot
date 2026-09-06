import time
import structlog
from typing import List, Dict, Any, Optional, Callable
from app.config import settings
from app.embeddings import embedder, pgvector_store
from app.ingestion import mongo_client
from app.retrieval import reranker
from app import observability

logger = structlog.get_logger(__name__)


def retrieve_similar_incidents(
    query_text: str,
    top_k: int = 5,
    environment: Optional[str] = None,
    threshold: Optional[float] = None,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    search_fn: Optional[Callable[[List[float], int], List[Any]]] = None,
) -> List[Dict[str, Any]]:
    """Retrieves top matching incident records for a given query text.

    Pipeline:
    1. Embeds query text.
    2. Queries PGVector store for similarity matches.
    3. Converts cosine distance to similarity score (1.0 - distance).
    4. Deduplicates chunks by incident_id (retaining highest similarity score).
    5. Filters out low-confidence matches below threshold and logs them.
    6. Applies keyword & environment metadata reranking.
    7. Enriches candidates with full incident details from MongoDB.
    """
    if not query_text or not query_text.strip():
        return []

    start_time = time.time()
    if threshold is None:
        threshold = settings.similarity_threshold

    if embed_fn is None:
        embed_fn = embedder.embed_texts

    # 1. Embed query
    try:
        query_embeddings = embed_fn([query_text])
    except Exception as emb_err:
        logger.warning(
            "Primary embedding provider unavailable; using fallback deterministic vector search.",
            error=str(emb_err),
        )
        query_embeddings = [embedder._deterministic_embedding(query_text)]

    if not query_embeddings:
        return []
    query_vector = query_embeddings[0]

    # 2. Similarity search in pgvector
    if search_fn is not None:
        raw_results = search_fn(query_vector, top_k * 4)
    else:
        raw_results = pgvector_store.similarity_search(query_vector, top_k=top_k * 4)

    # 3 & 4. Deduplicate by incident_id & convert cosine distance to similarity score
    best_chunks_by_incident: Dict[str, Dict[str, Any]] = {}

    for chunk_dict, distance in raw_results:
        sim_score = max(0.0, min(1.0, 1.0 - float(distance)))
        inc_id = chunk_dict["incident_id"]

        if (
            inc_id not in best_chunks_by_incident
            or sim_score > best_chunks_by_incident[inc_id]["similarity_score"]
        ):
            best_chunks_by_incident[inc_id] = {
                "incident_id": inc_id,
                "chunk_index": chunk_dict.get("chunk_index", 0),
                "chunk_text": chunk_dict.get("chunk_text", ""),
                "similarity_score": round(sim_score, 4),
            }

    # 5. Threshold filtering
    valid_candidates = []
    for inc_id, item in best_chunks_by_incident.items():
        if item["similarity_score"] < threshold:
            logger.warning(
                "Low confidence retrieval",
                correlation_id=observability.get_correlation_id(),
                incident_id=inc_id,
                similarity_score=item["similarity_score"],
                threshold=threshold,
                query=query_text[:50],
            )
        else:
            valid_candidates.append(item)

    if not valid_candidates:
        observability.log_trace_event(
            event_type="retrieval_complete",
            stage="retrieval",
            details={"candidates_found": 0, "retrieved_ids": []},
            latency=time.time() - start_time,
        )
        return []

    # 6. Rerank candidates
    reranked = reranker.rerank_incidents(query_text, valid_candidates)

    # 7. Fetch MongoDB full incident details & apply optional environment filter
    final_results = []
    for candidate in reranked:
        inc_id = candidate["incident_id"]
        doc = mongo_client.get_incident(inc_id)
        candidate["incident_details"] = doc or {}

        # Environment filter check
        if environment:
            doc_env = (doc.get("environment") if doc else "").lower()
            if doc_env != environment.lower():
                continue

        final_results.append(candidate)
        if len(final_results) >= top_k:
            break

    ret_latency = time.time() - start_time
    retrieved_ids = [res["incident_id"] for res in final_results]

    observability.log_trace_event(
        event_type="retrieval_complete",
        stage="retrieval",
        details={
            "query_snippet": query_text[:60],
            "retrieved_count": len(final_results),
            "retrieved_ids": retrieved_ids,
            "top_similarity": final_results[0]["similarity_score"] if final_results else 0.0,
        },
        latency=ret_latency,
    )

    return final_results
