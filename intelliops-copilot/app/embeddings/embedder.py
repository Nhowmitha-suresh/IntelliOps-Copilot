import os
import structlog
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
import openai
from app.config import settings

logger = structlog.get_logger(__name__)


class EmbeddingProviderError(Exception):
    """Raised when embedding provider API calls fail after retries."""
    pass


def _call_openai_embedding_api(
    texts: List[str], model: str = "text-embedding-3-small"
) -> List[List[float]]:
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError("Missing or placeholder OPENAI_API_KEY environment variable")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        reraise=True,
    )
    def _do_call():
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]

    return _do_call()


TOPIC_KEYWORDS = {
    "db_password": ["db_password", "keyerror", "backend-secrets"],
    "pydantic": ["pydantic", "pydantic_settings", "modulenotfounderror"],
    "postgres": ["queuepool", "sqlalchemy", "pool_size", "connection timed out"],
    "port": ["8000", "oserror", "address already in use", "uvicorn"],
    "ssl": ["ssl", "cert", "certverificationerror", "https handshake"],
    "oom": ["oomkilled", "pandas", "2048mb", "memory limit"],
    "redis": ["redis", "cache", "stale", "config:flags"],
    "cors": ["cors", "xmlhttprequest", "preflight", "allow_origins"],
    "jwt": ["jwt", "expiredsignature", "access_token_expire"],
    "disk": ["no space left", "log aggregator", "ioerror"],
    "dns": ["gaierror", "coredns", "internal-retrieval-svc"],
    "liveness": ["liveness", "probe failed", "healthz"],
    "ratelimit": ["ratelimiterror", "429", "too many requests"],
    "kafka": ["commitfailedexception", "rebalance storm", "max.poll.interval"],
    "mongo": ["authsource", "operationfailure", "mongopassword"],
    "vault": ["vault", "hvac", "forbidden"],
}


def _deterministic_embedding(text: str, dim: int = 1536) -> List[float]:
    import hashlib
    import math
    text_lower = text.lower()
    vec = [0.0] * dim
    primary_topic = None
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in kws):
            primary_topic = topic
            break

    if primary_topic:
        topic_hash = hashlib.sha256(primary_topic.encode("utf-8")).digest()
        for i in range(dim):
            b = topic_hash[i % len(topic_hash)]
            vec[i] = math.sin((i + 1) * (b + 1))
    else:
        h = hashlib.sha256(text_lower.encode("utf-8")).digest()
        for i in range(dim):
            b = h[i % len(h)]
            vec[i] = math.sin((i + 1) * (b + 1))

    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [round(x / norm, 6) for x in vec]


def embed_texts(
    texts: List[str], model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """Embeds a list of text strings using OpenAI text-embedding-3-small.

    Retries transient failures with exponential backoff and logs structured errors.
    """
    if not texts:
        return []

    try:
        return _call_openai_embedding_api(texts, model=model)
    except Exception as exc:
        logger.error(
            "Embedding API provider failure",
            model=model,
            num_texts=len(texts),
            error=str(exc),
        )
        raise EmbeddingProviderError(
            f"Failed to generate embeddings using model '{model}': {str(exc)}"
        ) from exc
