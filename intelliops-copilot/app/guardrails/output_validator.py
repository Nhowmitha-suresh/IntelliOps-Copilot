import json
import structlog
from typing import Optional, Dict, Any, Callable
from app.models import DiagnosisResult
from app.orchestration import llm_client, prompts
from app.guardrails import content_filters

logger = structlog.get_logger(__name__)


class GuardrailViolationError(Exception):
    """Raised when output validation or schema correction re-asks fail."""
    pass


def _parse_and_validate_json(raw_text: str) -> Dict[str, Any]:
    """Helper function to parse raw JSON and ensure required schema fields exist."""
    data = json.loads(raw_text)
    if not isinstance(data, dict):
        raise ValueError("JSON output must be a JSON object (dictionary).")

    required_fields = ["root_cause", "confidence", "suggested_fix", "evidence_chunks"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in JSON schema.")
    return data


def validate_diagnosis(
    raw_llm_output: str,
    reask_fn: Optional[Callable[[str], str]] = None,
) -> DiagnosisResult:
    """Validates raw LLM JSON output against DiagnosisResult guardrails.

    Rules:
    1. Parses JSON. On failure, triggers a single re-ask attempt with error feedback.
    2. Raises GuardrailViolationError if re-ask also fails.
    3. If confidence < 0.5, forces needs_human_review = True.
    4. If evidence_chunks is empty, forces needs_human_review = True and prepends low context warning.
    5. Filters suggested_fix for destructive commands.
    """
    if reask_fn is None:
        reask_fn = lambda prompt: llm_client.generate(
            prompt=prompt, response_schema=prompts.DIAGNOSIS_RESULT_SCHEMA
        )

    parsed_data = None
    try:
        parsed_data = _parse_and_validate_json(raw_llm_output)
    except Exception as parse_error:
        logger.warning(
            "Initial LLM output failed JSON validation. Triggering schema correction re-ask.",
            error=str(parse_error),
            raw_output=raw_llm_output[:100],
        )
        correction_prompt = (
            f"Your previous response did not match the required schema: {str(parse_error)}.\n"
            f"Return ONLY valid JSON matching this schema:\n"
            f"{json.dumps(prompts.DIAGNOSIS_RESULT_SCHEMA, indent=2)}"
        )
        try:
            reask_output = reask_fn(correction_prompt)
            parsed_data = _parse_and_validate_json(reask_output)
        except Exception as reask_error:
            logger.error(
                "Guardrail violation: Re-ask correction failed to produce valid schema.",
                error=str(reask_error),
            )
            raise GuardrailViolationError(
                f"Output validation failed after re-ask: {str(reask_error)}"
            ) from reask_error

    # Extract fields
    root_cause = str(parsed_data.get("root_cause", "insufficient evidence"))
    confidence = float(parsed_data.get("confidence", 0.0))
    suggested_fix = str(parsed_data.get("suggested_fix", ""))
    evidence_chunks = list(parsed_data.get("evidence_chunks") or [])
    needs_human_review = bool(parsed_data.get("needs_human_review", False))

    # Guardrail Rule 3: If confidence < 0.5, force needs_human_review = True
    if confidence < 0.5:
        logger.info(
            "Low confidence score (< 0.5). Forcing needs_human_review=True.",
            confidence=confidence,
        )
        needs_human_review = True

    # Guardrail Rule 4: If evidence_chunks is empty, force needs_human_review = True & prepend warning
    if not evidence_chunks:
        logger.info(
            "Empty evidence_chunks detected. Forcing needs_human_review=True and prepending warning."
        )
        needs_human_review = True
        low_context_prefix = "⚠️ Low context — no direct evidence chunks retrieved. "
        if not suggested_fix.startswith(low_context_prefix):
            suggested_fix = f"{low_context_prefix}{suggested_fix}"

    # Guardrail Rule 5: Filter destructive commands in suggested_fix
    suggested_fix = content_filters.filter_destructive_commands(suggested_fix)

    return DiagnosisResult(
        root_cause=root_cause,
        confidence=confidence,
        suggested_fix=suggested_fix,
        evidence_chunks=evidence_chunks,
        needs_human_review=needs_human_review,
    )
