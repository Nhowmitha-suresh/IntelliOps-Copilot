import time
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
from app import observability

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
    observability.log_trace_event(
        event_type="tool_execution",
        stage="orchestration_agent",
        details={"tool_name": "fetch_similar_incidents", "query_text": query_text[:50]},
    )
    results = retriever.retrieve_similar_incidents(query_text, top_k=top_k)
    return json.dumps(results, default=str)


@tool("fetch_raw_logs", args_schema=FetchRawLogsInput)
def fetch_raw_logs_tool(incident_id: str) -> str:
    """Fetches full raw incident document and error logs for a given incident ID."""
    observability.log_trace_event(
        event_type="tool_execution",
        stage="orchestration_agent",
        details={"tool_name": "fetch_raw_logs", "incident_id": incident_id},
    )
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
    start_time = time.time()
    cid = observability.get_correlation_id()

    logger.info(
        "Starting issue diagnosis workflow",
        correlation_id=cid,
        issue=issue_description[:60],
        environment=environment,
    )

    observability.log_trace_event(
        event_type="workflow_start",
        stage="orchestration_agent",
        details={"issue_snippet": issue_description[:60], "environment": environment},
    )

    # Step 1: Retrieve evidence using vector & metadata search
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
            correlation_id=cid,
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
    llm_start_time = time.time()
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

        llm_latency = time.time() - llm_start_time
        observability.log_trace_event(
            event_type="llm_generation_complete",
            stage="orchestration_llm",
            details={"prompt_length": len(user_prompt), "response_length": len(response_raw)},
            latency=llm_latency,
        )

        # Validate response through output_validator (which also applies content_filters)
        val_start_time = time.time()
        result = output_validator.validate_diagnosis(response_raw, reask_fn=reask_fn)
        val_latency = time.time() - val_start_time

        observability.log_trace_event(
            event_type="guardrail_validation_complete",
            stage="guardrails",
            details={
                "confidence": result.confidence,
                "needs_human_review": result.needs_human_review,
                "root_cause_snippet": result.root_cause[:60],
            },
            latency=val_latency,
        )

        total_latency = time.time() - start_time
        observability.log_trace_event(
            event_type="workflow_complete",
            stage="orchestration_agent",
            details={"status": "success", "confidence": result.confidence},
            latency=total_latency,
        )

        return result

    except Exception as exc:
        logger.warning(
            "Failed to generate or validate LLM diagnosis. Returning safe fallback.",
            correlation_id=cid,
            error=str(exc),
        )
        total_latency = time.time() - start_time
        top_doc = similar_incidents[0].get("incident_details", {}) if similar_incidents else {}
        if top_doc and evidence_texts:
            fallback_rc = top_doc.get("title") or "Historical incident match found."
            fallback_fix = top_doc.get("resolution") or "Review retrieved incident resolution."
            fallback_conf = round(float(similar_incidents[0].get("similarity_score", 0.85)), 2)
            fallback_review = False
        else:
            fallback_rc = "insufficient evidence"
            fallback_fix = "Automated diagnosis unavailable. Please review raw logs."
            fallback_conf = 0.2
            fallback_review = True

        fallback_result = DiagnosisResult(
            root_cause=fallback_rc,
            confidence=fallback_conf,
            suggested_fix=fallback_fix,
            evidence_chunks=evidence_texts,
            needs_human_review=fallback_review,
        )
        observability.log_trace_event(
            event_type="workflow_fallback",
            stage="orchestration_agent",
            details={"error": str(exc)},
            latency=total_latency,
        )
        return fallback_result
