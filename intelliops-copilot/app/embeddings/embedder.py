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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    reraise=True,
)
def _call_openai_embedding_api(
    texts: List[str], model: str = "text-embedding-3-small"
) -> List[List[float]]:
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError("Missing or placeholder OPENAI_API_KEY environment variable")

    client = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


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
