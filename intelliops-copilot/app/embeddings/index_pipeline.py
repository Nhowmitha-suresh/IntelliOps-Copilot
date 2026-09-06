import hashlib
import structlog
from typing import List, Dict, Any, Callable, Optional
from app.ingestion import mongo_client
from app.embeddings import chunker, embedder, pgvector_store

logger = structlog.get_logger(__name__)


def compute_incident_hash(incident: Dict[str, Any]) -> str:
    """Computes a SHA-256 hash of an incident's core content fields."""
    title = incident.get("title", "")
    description = incident.get("description", "")
    error_log = incident.get("error_log", "")
    resolution = incident.get("resolution", "")
    content = f"{title}|{description}|{error_log}|{resolution}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def run_indexing_pipeline(
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    force_reindex: bool = False,
) -> Dict[str, int]:
    """Reads all incidents from MongoDB, chunks, embeds, and stores them into PGVector.

    Idempotent: skips incidents whose content hash has not changed unless force_reindex=True.
    """
    if embed_fn is None:
        embed_fn = embedder.embed_texts

    incidents = mongo_client.list_incidents()
    logger.info("Starting indexing pipeline", total_incidents=len(incidents))

    stats = {
        "total_incidents": len(incidents),
        "skipped_incidents": 0,
        "processed_incidents": 0,
        "chunks_indexed": 0,
    }

    db = mongo_client.get_database()

    for inc in incidents:
        inc_id = inc.get("id")
        current_hash = compute_incident_hash(inc)
        stored_hash = inc.get("indexed_content_hash")

        if not force_reindex and stored_hash == current_hash:
            logger.debug("Skipping unchanged incident", incident_id=inc_id)
            stats["skipped_incidents"] += 1
            continue

        # Chunk incident
        chunks = chunker.chunk_incident(inc)
        if not chunks:
            continue

        chunk_texts = [c.chunk_text for c in chunks]

        # Embed chunks
        logger.info(
            "Embedding incident chunks", incident_id=inc_id, num_chunks=len(chunks)
        )
        embeddings = embed_fn(chunk_texts)

        # Prepare payload for pgvector
        payload = []
        for c, emb in zip(chunks, embeddings):
            payload.append(
                {
                    "incident_id": c.incident_id,
                    "chunk_index": c.chunk_index,
                    "chunk_text": c.chunk_text,
                    "embedding": emb,
                }
            )

        # Upsert into PostgreSQL
        pgvector_store.upsert_chunks(payload)
        stats["chunks_indexed"] += len(payload)
        stats["processed_incidents"] += 1

        # Update MongoDB with content hash
        db["incidents"].update_one(
            {"id": inc_id}, {"$set": {"indexed_content_hash": current_hash}}
        )

    logger.info("Indexing pipeline completed", **stats)
    return stats


if __name__ == "__main__":
    run_indexing_pipeline()
