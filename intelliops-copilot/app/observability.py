import os
import uuid
import json
import time
import contextvars
from datetime import datetime
from typing import Optional, Dict, Any
import structlog

# Context variable to hold correlation_id per request execution context
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)

LOG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "app_traces.log")


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Sets the active correlation_id for the current request execution context."""
    new_id = cid or str(uuid.uuid4())
    correlation_id_var.set(new_id)
    structlog.contextvars.bind_contextvars(correlation_id=new_id)
    return new_id


def get_correlation_id() -> str:
    """Returns the current active correlation_id, initializing a new one if unset."""
    cid = correlation_id_var.get()
    if not cid:
        cid = set_correlation_id()
    return cid


def configure_logging(log_format: Optional[str] = None) -> None:
    """Configures structlog to output JSON for production or colorized console for dev."""
    fmt = log_format or os.getenv("LOG_FORMAT", "console").lower()

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def log_trace_event(
    event_type: str,
    stage: str,
    details: Dict[str, Any],
    latency: float = 0.0,
) -> Dict[str, Any]:
    """Records a structured telemetry event to both local trace log file and MongoDB."""
    cid = get_correlation_id()
    entry = {
        "correlation_id": cid,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "stage": stage,
        "latency": round(latency, 4),
        "details": details,
    }

    # 1. Append to local log file
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

    # 2. Append to MongoDB trace_logs collection
    try:
        from app.ingestion import mongo_client
        db = mongo_client.get_database()
        doc = entry.copy()
        doc["_id"] = str(uuid.uuid4())
        db["trace_logs"].insert_one(doc)
    except Exception:
        pass

    return entry


# Run default configuration
configure_logging()
