import math
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class IncidentChunk(BaseModel):
    incident_id: str
    chunk_index: int
    chunk_text: str
    token_count: int

    def to_dict(self) -> dict:
        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")
        return self.dict()


def estimate_tokens(text: str) -> int:
    """Estimates token count using a whitespace/char-based estimator (~1.3 tokens per word)."""
    if not text:
        return 0
    words = text.split()
    return max(1, int(math.ceil(len(words) * 1.3)))


def chunk_incident(
    incident: Dict[str, Any],
    target_tokens: int = 300,
    overlap_tokens: int = 50,
) -> List[IncidentChunk]:
    """Combines incident title, description, error_log, and resolution,

    then chunks using a sliding window (~target_tokens with ~overlap_tokens).
    """
    incident_id = str(incident.get("id") or incident.get("_id") or "unknown")
    title = incident.get("title", "")
    description = incident.get("description", "")
    error_log = incident.get("error_log", "")
    resolution = incident.get("resolution", "")

    # Combine fields into a unified text block
    combined_text = (
        f"Title: {title}\n"
        f"Description: {description}\n"
        f"Error Log: {error_log}\n"
        f"Resolution: {resolution}"
    ).strip()

    words = combined_text.split()
    if not words:
        return []

    # Calculate word window and overlap (approx 1 word = 1.3 tokens)
    words_per_chunk = max(1, int(target_tokens / 1.3))
    overlap_words = max(0, int(overlap_tokens / 1.3))
    stride = max(1, words_per_chunk - overlap_words)

    total_words = len(words)
    chunks: List[IncidentChunk] = []
    chunk_index = 0
    start_idx = 0

    while start_idx < total_words:
        end_idx = min(start_idx + words_per_chunk, total_words)
        chunk_words = words[start_idx:end_idx]
        chunk_text = " ".join(chunk_words)
        token_count = estimate_tokens(chunk_text)

        chunks.append(
            IncidentChunk(
                incident_id=incident_id,
                chunk_index=chunk_index,
                chunk_text=chunk_text,
                token_count=token_count,
            )
        )
        chunk_index += 1

        if end_idx >= total_words:
            break
        start_idx += stride

    return chunks


def chunk_incidents(
    incidents: List[Dict[str, Any]],
    target_tokens: int = 300,
    overlap_tokens: int = 50,
) -> List[IncidentChunk]:
    """Chunks a list of incident records."""
    all_chunks = []
    for inc in incidents:
        all_chunks.extend(chunk_incident(inc, target_tokens, overlap_tokens))
    return all_chunks
