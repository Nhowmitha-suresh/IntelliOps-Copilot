"""IntelliOps Copilot — CLI Trace Analysis & Debugging Tool.

Reconstructs a human-readable operational timeline for a given correlation_id
from local trace logs or MongoDB telemetry collections.
"""

import sys
import json
import os
import argparse
from typing import List, Dict, Any

LOG_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "logs", "app_traces.log")
)


def fetch_trace_events(correlation_id: str) -> List[Dict[str, Any]]:
    """Retrieves trace log events matching correlation_id from local log file and MongoDB."""
    events: List[Dict[str, Any]] = []

    # 1. Read local log file
    if os.path.exists(LOG_FILE_PATH):
        try:
            with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if data.get("correlation_id") == correlation_id:
                                events.append(data)
                        except Exception:
                            continue
        except Exception:
            pass

    # 2. Query MongoDB trace_logs collection
    try:
        from app.ingestion import mongo_client
        db = mongo_client.get_database()
        cursor = db["trace_logs"].find({"correlation_id": correlation_id})
        for doc in cursor:
            doc.pop("_id", None)
            if doc not in events:
                events.append(doc)
    except Exception:
        pass

    # Sort chronological by timestamp
    events.sort(key=lambda x: x.get("timestamp", ""))
    return events


def format_trace_timeline(correlation_id: str, events: List[Dict[str, Any]]) -> str:
    """Formats trace events into an ASCII timeline breakdown."""
    lines = []
    lines.append("=" * 80)
    lines.append(" INTELLIOPS COPILOT - CORRELATION TRACE ANALYSIS")
    lines.append("=" * 80)
    lines.append(f" Correlation ID : {correlation_id}")
    lines.append(f" Total Events   : {len(events)}")

    if not events:
        lines.append("-" * 80)
        lines.append(f" No trace events found for Correlation ID '{correlation_id}'.")
        lines.append("=" * 80)
        return "\n".join(lines)

    total_latency = sum(e.get("latency", 0.0) for e in events if e.get("stage") == "api_gateway")
    if total_latency == 0:
        total_latency = sum(e.get("latency", 0.0) for e in events)

    lines.append(f" End-to-End Latency : {total_latency:.4f}s")
    lines.append("-" * 80)
    lines.append(" STAGE TIMELINE BREAKDOWN")
    lines.append("-" * 80)

    stage_latencies: Dict[str, float] = {}

    for idx, event in enumerate(events, 1):
        stage = event.get("stage", "unknown")
        event_type = event.get("event_type", "unknown")
        latency = event.get("latency", 0.0)
        details = event.get("details", {})
        ts = event.get("timestamp", "")

        stage_latencies[stage] = stage_latencies.get(stage, 0.0) + latency

        lines.append(f" [{idx}] [{ts}] Stage: {stage} | Event: {event_type}")

        if stage == "api_gateway":
            lines.append(f"     - Method: {details.get('method')} {details.get('path')} | Status: {details.get('status_code')}")
        elif stage == "retrieval":
            lines.append(f"     - Query Snippet: {details.get('query_snippet')}")
            lines.append(f"     - Retrieved Incidents ({details.get('retrieved_count')}): {', '.join(details.get('retrieved_ids', []))}")
            lines.append(f"     - Top Similarity: {details.get('top_similarity')}")
        elif stage == "orchestration_llm":
            lines.append(f"     - Prompt Chars: {details.get('prompt_length')} | Response Chars: {details.get('response_length')}")
        elif stage == "guardrails":
            lines.append(f"     - Confidence Score: {details.get('confidence')} | Needs Human Review: {details.get('needs_human_review')}")
            lines.append(f"     - Root Cause: {details.get('root_cause_snippet')}")
        elif stage == "orchestration_agent":
            if event_type == "tool_execution":
                lines.append(f"     - Tool Executed: {details.get('tool_name')} ({details.get('query_text') or details.get('incident_id')})")
            else:
                lines.append(f"     - Workflow Status: {details.get('status') or details.get('issue_snippet')}")
        else:
            lines.append(f"     - Details: {json.dumps(details)}")

        lines.append(f"     - Stage Latency: {latency:.4f}s")
        lines.append("")

    lines.append("-" * 80)
    lines.append(" STAGE LATENCY SUMMARY")
    lines.append("-" * 80)
    for stg, lat in stage_latencies.items():
        pct = (lat / max(0.001, total_latency)) * 100.0 if total_latency > 0 else 0.0
        lines.append(f"  - {stg:<25}: {lat:.4f}s ({pct:.1f}%)")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct operational trace timeline for a given correlation_id."
    )
    parser.add_argument("correlation_id", help="UUID correlation_id to analyze.")
    args = parser.parse_args()

    events = fetch_trace_events(args.correlation_id)
    report = format_trace_timeline(args.correlation_id, events)
    print(report)


if __name__ == "__main__":
    main()
