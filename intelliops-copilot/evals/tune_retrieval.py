"""Retrieval Tuning and Evaluation Script.

Evaluates Precision@K and Mean Reciprocal Rank (MRR) across multiple
retrieval configurations (top_k=3, 5, 10 and chunk_size=200, 300, 500).
"""

import sys
from typing import List, Tuple, Dict, Any
from app.ingestion import mongo_client
from app.embeddings import chunker
from app.retrieval import reranker

# 10 Ground-truth evaluation query & expected incident ID pairs
EVAL_BENCHMARK: List[Tuple[str, str]] = [
    ("Environment variable DB_PASSWORD missing on production pod deployment", "INC-001"),
    ("ModuleNotFoundError pydantic_settings version mismatch breaking settings parser", "INC-002"),
    ("sqlalchemy QueuePool limit of size 20 reached connection pool exhausted", "INC-003"),
    ("OSError Address already in use port 8000 binding conflict uvicorn worker", "INC-004"),
    ("SSLCertVerificationError certificate has expired on NGINX ingress controller", "INC-005"),
    ("Container OOMKilled by Linux OOM killer pandas dataframe memory limit", "INC-006"),
    ("Stale Redis cache serving outdated API config flags post deployment", "INC-007"),
    ("CORS policy preflight request blocked from app dashboard domain", "INC-008"),
    ("ExpiredSignatureError JWT auth token expired prematurely seconds vs minutes", "INC-009"),
    ("No space left on device log aggregator disk space full", "INC-010"),
]


def tfidf_similarity_search(query_text: str, incidents: List[Dict[str, Any]], chunks: List[Any], top_k: int) -> List[Dict[str, Any]]:
    """Term-frequency overlap retriever for evaluation simulation."""
    query_words = set(query_text.lower().split())

    inc_map = {inc["id"]: inc for inc in incidents}

    best_per_inc = {}
    for c in chunks:
        inc_id = c.incident_id
        text_words = set(c.chunk_text.lower().split())
        overlap = len(query_words.intersection(text_words))
        score = min(1.0, overlap / max(1, len(query_words)))
        
        if inc_id not in best_per_inc or score > best_per_inc[inc_id]["similarity_score"]:
            best_per_inc[inc_id] = {
                "incident_id": inc_id,
                "chunk_index": c.chunk_index,
                "chunk_text": c.chunk_text,
                "similarity_score": score,
                "incident_details": inc_map.get(inc_id, {}),
            }

    candidate_list = list(best_per_inc.values())
    reranked = reranker.rerank_incidents(query_text, candidate_list)
    return reranked[:top_k]


def evaluate_configuration(
    incidents: List[Dict[str, Any]],
    chunk_size: int,
    top_k: int,
    overlap_size: int = 50,
) -> Tuple[float, float]:
    """Calculates Precision@K and MRR for a given chunk size and top_k configuration."""
    all_chunks = []
    for inc in incidents:
        all_chunks.extend(chunker.chunk_incident(inc, target_tokens=chunk_size, overlap_tokens=overlap_size))

    precisions = []
    reciprocal_ranks = []

    for query_text, expected_id in EVAL_BENCHMARK:
        results = tfidf_similarity_search(query_text, incidents, all_chunks, top_k)
        retrieved_ids = [r["incident_id"] for r in results]

        # Precision@K (for single target: 1/K if present, else 0)
        is_found = expected_id in retrieved_ids
        p_at_k = (1.0 / top_k) if is_found else 0.0
        precisions.append(p_at_k)

        # Reciprocal Rank (1/rank of first match)
        if is_found:
            rank = retrieved_ids.index(expected_id) + 1
            rr = 1.0 / rank
        else:
            rr = 0.0
        reciprocal_ranks.append(rr)

    mean_precision = sum(precisions) / len(precisions)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mean_precision, mrr


def main():
    try:
        incidents = mongo_client.list_incidents()
    except Exception:
        incidents = []
        
    if not incidents:
        from app.ingestion.seed_data import SEED_INCIDENTS
        incidents = [inc.to_dict() for inc in SEED_INCIDENTS]

    print("=" * 75)
    print(" INTELLIOPS COPILOT - RETRIEVAL TUNING & EVALUATION BENCHMARK")
    print("=" * 75)
    print(f" Total Benchmark Queries : {len(EVAL_BENCHMARK)}")
    print(f" Evaluated Incidents     : {len(incidents)}")
    print("-" * 75)
    print(f" {'Chunk Size':<12} | {'Top-K':<8} | {'Precision@K':<15} | {'MRR Score':<12} | {'Status':<10}")
    print("-" * 75)

    chunk_sizes = [200, 300, 500]
    top_k_values = [3, 5, 10]

    best_mrr = -1.0
    winning_config = None

    for cs in chunk_sizes:
        for k in top_k_values:
            p_at_k, mrr = evaluate_configuration(incidents, chunk_size=cs, top_k=k)
            is_best = mrr > best_mrr
            if is_best:
                best_mrr = mrr
                winning_config = (cs, k, p_at_k, mrr)
            status_str = "WINNER" if is_best else "  "
            print(f" {cs:<12} | {k:<8} | {p_at_k:<15.4f} | {mrr:<12.4f} | {status_str}")

    print("=" * 75)
    if winning_config:
        cs, k, p_at_k, mrr = winning_config
        print(f" WINNING CONFIGURATION: Chunk Size = {cs} tokens | Top-K = {k}")
        print(f" Precision@{k}: {p_at_k:.4f} | MRR: {mrr:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
