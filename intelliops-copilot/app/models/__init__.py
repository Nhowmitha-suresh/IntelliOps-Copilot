import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class IncidentTicket(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    error_log: str
    config_snippet: Optional[str] = None
    environment: str
    resolution: str
    resolved_at: datetime
    tags: List[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")
        return self.dict()


class LogEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "app"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: LogLevel = LogLevel.INFO
    message: str
    raw_line: str
    searchable_text: Optional[str] = None

    def to_dict(self) -> dict:
        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")
        return self.dict()


class DiagnosisResult(BaseModel):
    root_cause: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_fix: str
    evidence_chunks: List[str] = Field(default_factory=list)
    needs_human_review: bool = False

    def to_dict(self) -> dict:
        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")
        return self.dict()


__all__ = [
    "LogLevel",
    "IncidentTicket",
    "LogEntry",
    "DiagnosisResult",
]
