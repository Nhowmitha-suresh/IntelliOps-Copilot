import re
from datetime import datetime
from typing import List, Optional
import nltk
from app.models import LogEntry, LogLevel

# Download stopwords safely if missing
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    try:
        nltk.download("stopwords", quiet=True)
    except Exception:
        pass

try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
        "is", "are", "was", "were", "be", "been", "being", "and", "or", "not",
        "this", "that", "it", "from", "as", "at"
    }


def clean_text_for_search(text: str) -> str:
    """Lowercases text and removes punctuation & stopwords for searchable_text."""
    if not text:
        return ""
    text_lower = text.lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", text_lower)
    filtered = [w for w in tokens if w not in STOP_WORDS]
    return " ".join(filtered)


# Regex patterns matching common log formats:
# 1. [2026-09-06 12:00:00] [ERROR] Message
# 2. 2026-09-06 12:00:00 - ERROR - Message
# 3. 2026-09-06T12:00:00Z ERROR Message
LOG_PATTERNS = [
    re.compile(
        r"^\s*\[?(?P<timestamp>\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?)\]?\s*[-:]?\s*\[?(?P<level>DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL)\]?\s*[-:]?\s*(?P<message>.*)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*\[?(?P<level>DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL)\]?\s*[-:]?\s*\[?(?P<timestamp>\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?)\]?\s*[-:]?\s*(?P<message>.*)$",
        re.IGNORECASE,
    ),
]


def parse_timestamp(ts_str: str) -> datetime:
    """Attempts to parse timestamp strings in common ISO / standard formats."""
    ts_clean = ts_str.strip("[]").replace(",", ".")
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(ts_clean, fmt)
        except ValueError:
            continue
    return datetime.utcnow()


def parse_log_level(level_str: str) -> LogLevel:
    """Normalizes string log level to LogLevel enum."""
    lvl = level_str.upper().strip("[]")
    if lvl in ("WARN", "WARNING"):
        return LogLevel.WARN
    try:
        return LogLevel(lvl)
    except ValueError:
        return LogLevel.UNKNOWN


def parse_log_line(line: str, source: str = "app") -> LogEntry:
    """Parses a single line of log into a LogEntry record."""
    raw_line = line.rstrip("\r\n")
    if not raw_line.strip():
        return LogEntry(
            source=source,
            timestamp=datetime.utcnow(),
            level=LogLevel.UNKNOWN,
            message="",
            raw_line=raw_line,
            searchable_text="",
        )

    for pattern in LOG_PATTERNS:
        match = pattern.match(raw_line)
        if match:
            groups = match.groupdict()
            ts = parse_timestamp(groups["timestamp"])
            lvl = parse_log_level(groups["level"])
            msg = groups["message"].strip()
            searchable = clean_text_for_search(msg)
            return LogEntry(
                source=source,
                timestamp=ts,
                level=lvl,
                message=msg,
                raw_line=raw_line,
                searchable_text=searchable,
            )

    # If regex does not match, handle as malformed line gracefully
    searchable = clean_text_for_search(raw_line)
    return LogEntry(
        source=source,
        timestamp=datetime.utcnow(),
        level=LogLevel.UNKNOWN,
        message=raw_line.strip(),
        raw_line=raw_line,
        searchable_text=searchable,
    )


def parse_raw_logs(raw_log_text: str, source: str = "app") -> List[LogEntry]:
    """Accepts multi-line raw log text and returns a list of LogEntry items."""
    if not raw_log_text:
        return []
    lines = raw_log_text.splitlines()
    entries = []
    for line in lines:
        if line.strip():
            entries.append(parse_log_line(line, source=source))
    return entries
