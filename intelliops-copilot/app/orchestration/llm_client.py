import os
import time
import json
import structlog
from typing import Optional, Dict, Any
import openai
import google.generativeai as genai
from app.config import settings

logger = structlog.get_logger(__name__)


class LLMOrchestrationError(Exception):
    """Raised when all primary and fallback LLM providers exhaust retries."""
    pass


def _call_openai(
    prompt: str,
    response_schema: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> str:
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError("Missing or placeholder OPENAI_API_KEY")

    client = openai.OpenAI(api_key=api_key, timeout=float(timeout))
    messages = [{"role": "user", "content": prompt}]
    kwargs: Dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.1,
    }
    if response_schema:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def _call_gemini(
    prompt: str,
    response_schema: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> str:
    api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("Missing or placeholder GEMINI_API_KEY")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text or ""


def _invoke_provider_with_retries(
    provider_name: str,
    prompt: str,
    response_schema: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    max_retries: int = 3,
) -> str:
    """Invokes a specified provider with up to max_retries attempts and exponential backoff."""
    call_fn = _call_openai if provider_name.lower() == "openai" else _call_gemini

    last_exception = None
    for attempt in range(1, max_retries + 1):
        start_time = time.time()
        try:
            result = call_fn(prompt, response_schema=response_schema, timeout=timeout)
            latency = time.time() - start_time
            logger.info(
                "LLM call succeeded",
                provider=provider_name,
                latency=round(latency, 3),
                attempt=attempt,
                success=True,
            )
            return result
        except Exception as exc:
            latency = time.time() - start_time
            last_exception = exc
            logger.warning(
                "LLM call attempt failed",
                provider=provider_name,
                latency=round(latency, 3),
                attempt=attempt,
                success=False,
                error=str(exc),
            )
            if attempt < max_retries:
                backoff = 0.5 * (2 ** (attempt - 1))
                time.sleep(backoff)

    raise last_exception or RuntimeError(f"Provider {provider_name} failed")


def generate(
    prompt: str,
    response_schema: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
) -> str:
    """Unified LLM generation function.

    Attempts primary provider (settings.llm_primary_provider) up to MAX_RETRIES.
    On failure, automatically fails over to fallback provider (settings.llm_fallback_provider).
    Raises LLMOrchestrationError if both providers exhaust retries.
    """
    if timeout is None:
        timeout = settings.request_timeout_seconds
    max_retries = settings.max_retries

    primary_provider = settings.llm_primary_provider
    fallback_provider = settings.llm_fallback_provider

    # Try Primary Provider
    try:
        logger.info("Attempting primary LLM provider", provider=primary_provider)
        return _invoke_provider_with_retries(
            provider_name=primary_provider,
            prompt=prompt,
            response_schema=response_schema,
            timeout=timeout,
            max_retries=max_retries,
        )
    except Exception as primary_exc:
        logger.error(
            "Primary LLM provider failed after retries. Triggering fallback failover.",
            primary_provider=primary_provider,
            fallback_provider=fallback_provider,
            error=str(primary_exc),
        )

    # Try Fallback Provider
    try:
        logger.info("Attempting fallback LLM provider", provider=fallback_provider)
        return _invoke_provider_with_retries(
            provider_name=fallback_provider,
            prompt=prompt,
            response_schema=response_schema,
            timeout=timeout,
            max_retries=max_retries,
        )
    except Exception as fallback_exc:
        logger.error(
            "Fallback LLM provider also failed after retries.",
            fallback_provider=fallback_provider,
            error=str(fallback_exc),
        )
        raise LLMOrchestrationError(
            f"Both primary ('{primary_provider}') and fallback ('{fallback_provider}') LLM providers failed after retries."
        ) from fallback_exc
