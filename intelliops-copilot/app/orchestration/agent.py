import json
import structlog
from typing import Optional, Dict, Any, List, Callable
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from app.models import DiagnosisResult
from app.retrieval import retriever
from app.ingestion import mongo_client
from app.orchestration import llm_client, prompts
from app.guardrails import output_validator

logger = structlog.get_logger(__name__)


class FetchSimilarIncidentsInput(BaseModel):
    query_text: str = Field(
        description="Search string or log snippet to find similar historical incidents."
    )
    top_k: int = Field(
        default=3, description="Number of top similar incident matches to return."
    )


class FetchRawLogsInput(BaseModel):
    incident_id: str = Field(
        description="Specific incident ID (e.g., INC-001) to fetch full raw logs for."
    )


@tool("fetch_similar_incidents", args_schema=FetchSimilarIncidentsInput)
def fetch_similar_incidents_tool(query_text: str, top_k: int = 3) -> str:
    """Retrieves top similar historical incident reports matching the query text."""
    results = retriever.retrieve_similar_incidents(query_text, top_k=top_k)
    return json.dumps(results, default=str)


@tool("fetch_raw_logs", args_schema=FetchRawLogsInput)
def fetch_raw_logs_tool(incident_id: str) -> str:
    """Fetches full raw incident document and error logs for a given incident ID."""
    doc = mongo_client.get_incident(incident_id)
    if not doc:
        return f"No incident record found for ID '{incident_id}'."
    return json.dumps(doc, default=str)


TOOLS = [fetch_similar_incidents_tool, fetch_raw_logs_tool]


def diagnose_issue(
    issue_description: str,
    environment: Optional[str] = None,
    custom_llm_fn: Optional[Callable[[str], str]] = None,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
) -> DiagnosisResult:
    """Orchestrates root-cause analysis by retrieving evidence via tools, synthesizing

    prompt context, calling LLM, and validating results through guardrails.
    """
    logger.info(
        "Starting issue diagnosis workflow",
        issue=issue_description[:60],
        environment=environment,
    )

    # Step 1: Retrieve evidence using vector & metadata search (handling provider errors gracefully)
    try:
        similar_incidents = retriever.retrieve_similar_incidents(
            query_text=issue_description,
            top_k=3,
            environment=environment,
            embed_fn=embed_fn,
        )
    except Exception as ret_exc:
        logger.warning(
            "Vector retrieval provider unavailable. Proceeding with empty evidence.",
            error=str(ret_exc),
        )
        similar_incidents = []

    evidence_texts = []
    evidence_ids = []
    for inc in similar_incidents:
        inc_id = inc.get("incident_id", "")
        evidence_ids.append(inc_id)
        details = inc.get("incident_details", {})
        err = details.get("error_log") or inc.get("chunk_text", "")
        res = details.get("resolution", "")
        evidence_texts.append(
            f"[Incident {inc_id}]: {details.get('title', '')}\nError: {err}\nResolution: {res}"
        )

    combined_evidence = (
        "\n\n".join(evidence_texts) if evidence_texts else "No matching historical incidents found."
    )

    # Step 2: Construct prompt
    user_prompt = (
        f"{prompts.SYSTEM_PROMPT}\n\n"
        f"--- USER REPORTED ISSUE ---\n{issue_description}\n"
        f"Environment: {environment or 'unspecified'}\n\n"
        f"--- RETRIEVED EVIDENCE CHUNKS ---\n{combined_evidence}\n\n"
        f"Generate a diagnosis as a JSON object adhering to this schema:\n"
        f"{json.dumps(prompts.DIAGNOSIS_RESULT_SCHEMA, indent=2)}\n"
    )

    # Step 3: Call LLM & pass through output_validator guardrail
    try:
        if custom_llm_fn:
            response_raw = custom_llm_fn(user_prompt)
            reask_fn = custom_llm_fn
        else:
            response_raw = llm_client.generate(
                prompt=user_prompt,
                response_schema=prompts.DIAGNOSIS_RESULT_SCHEMA,
            )
            reask_fn = lambda p: llm_client.generate(
                prompt=p, response_schema=prompts.DIAGNOSIS_RESULT_SCHEMA
            )

        # Validate response through output_validator (which also applies content_filters)
        return output_validator.validate_diagnosis(response_raw, reask_fn=reask_fn)

    except Exception as exc:
        logger.warning(
            "Failed to generate or validate LLM diagnosis. Returning safe fallback.",
            error=str(exc),
        )
        return DiagnosisResult(
            root_cause="insufficient evidence",
            confidence=0.2,
            suggested_fix="Automated diagnosis unavailable. Please review raw logs.",
            evidence_chunks=evidence_texts,
            needs_human_review=True,
        )
