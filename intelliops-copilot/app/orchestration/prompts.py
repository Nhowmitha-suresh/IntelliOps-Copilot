"""Prompts and schemas for IntelliOps Copilot LLM orchestration."""

SYSTEM_PROMPT = (
    "You are IntelliOps Copilot, an expert site reliability engineering (SRE) and root-cause diagnosis assistant.\n\n"
    "Strict Operational Rules:\n"
    "1. Base your diagnosis ONLY on the provided evidence chunks, retrieved incident reports, and log entries.\n"
    "2. If the retrieved evidence is insufficient or low-confidence to determine the exact root cause, "
    "explicitly set root_cause to 'insufficient evidence', set confidence <= 0.3, and set needs_human_review to true.\n"
    "3. Never guess, speculate, or hallucinate root causes without empirical evidence in logs or configurations.\n"
    "4. Always format your final response strictly as a JSON object matching the DiagnosisResult schema."
)

DIAGNOSIS_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause": {
            "type": "string",
            "description": "Clear technical explanation of the identified root cause, or 'insufficient evidence' if context is lacking.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence score between 0.0 and 1.0.",
        },
        "suggested_fix": {
            "type": "string",
            "description": "Actionable, step-by-step remediation commands or configuration changes.",
        },
        "evidence_chunks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of log lines or incident excerpts supporting the diagnosis.",
        },
        "needs_human_review": {
            "type": "boolean",
            "description": "True if confidence < 0.7 or remediation carries operational risk.",
        },
    },
    "required": [
        "root_cause",
        "confidence",
        "suggested_fix",
        "evidence_chunks",
        "needs_human_review",
    ],
}
