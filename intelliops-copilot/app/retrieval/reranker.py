import re
from typing import List, Dict, Any, Set

ENV_KEYWORDS = {"production", "prod", "staging", "stage", "development", "dev", "local", "test"}

KNOWN_TECH_TAGS = {
    "docker", "kubernetes", "k8s", "postgres", "postgresql", "mongo", "mongodb",
    "redis", "kafka", "vault", "pydantic", "ssl", "tls", "cors", "jwt", "auth",
    "memory", "oom", "disk", "dns", "liveness", "probe", "rate-limit", "openai",
    "python", "fastapi", "uvicorn", "sqlalchemy", "env-var", "config"
}


def extract_keywords_from_query(query_text: str) -> Set[str]:
    """Extracts metadata keywords, environment names, and tech tags from query text."""
    if not query_text:
        return set()
    tokens = set(re.findall(r"\b[a-z0-9_-]+\b", query_text.lower()))
    return tokens


def rerank_incidents(
    query_text: str,
    candidates: List[Dict[str, Any]],
    boost_amount: float = 0.1,
) -> List[Dict[str, Any]]:
    """Reranks candidate incidents by boosting similarity scores if query keywords match

    the incident's environment or tags metadata.
    """
    if not candidates:
        return []

    query_tokens = extract_keywords_from_query(query_text)
    reranked = []

    for candidate in candidates:
        item = dict(candidate)
        inc_details = item.get("incident_details") or {}
        
        # Check environment match
        inc_env = str(inc_details.get("environment", "") or item.get("environment", "")).lower()
        env_match = inc_env in query_tokens and inc_env != ""

        # Check tags match
        inc_tags = [str(t).lower() for t in (inc_details.get("tags") or item.get("tags") or [])]
        tag_matches = set(inc_tags).intersection(query_tokens)

        # Apply boost if matching environment or tags found
        boost = 0.0
        if env_match:
            boost += boost_amount
        if tag_matches:
            boost += boost_amount * min(len(tag_matches), 2)

        original_score = item.get("similarity_score", 0.0)
        boosted_score = min(1.0, original_score + boost)
        item["similarity_score"] = round(boosted_score, 4)
        item["original_score"] = original_score
        item["boost_applied"] = round(boost, 4)

        reranked.append(item)

    # Sort descending by boosted similarity score
    reranked.sort(key=lambda x: x["similarity_score"], reverse=True)
    return reranked
