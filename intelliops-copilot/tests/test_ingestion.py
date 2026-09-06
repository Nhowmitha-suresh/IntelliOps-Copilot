from datetime import datetime
import pytest
from app.models import IncidentTicket, LogEntry, LogLevel
from app.ingestion.log_parser import parse_log_line, parse_raw_logs, clean_text_for_search
from app.ingestion import mongo_client


def test_clean_text_for_search():
    text = "The Database Connection Failed Unexpectedly!"
    cleaned = clean_text_for_search(text)
    # Stopwords like 'the' should be removed, lowercase preserved
    assert "database" in cleaned
    assert "connection" in cleaned
    assert "failed" in cleaned
    assert "unexpectedly" in cleaned
    assert "the" not in cleaned.split()


def test_log_parser_well_formed_lines():
    line1 = "[2026-09-06 12:00:00] [ERROR] Database connection timed out"
    entry1 = parse_log_line(line1, source="db-service")
    assert entry1.level == LogLevel.ERROR
    assert entry1.message == "Database connection timed out"
    assert entry1.raw_line == line1
    assert entry1.source == "db-service"
    assert "database" in entry1.searchable_text

    line2 = "2026-09-06T14:30:15Z - WARN - High memory consumption detected"
    entry2 = parse_log_line(line2, source="monitor")
    assert entry2.level == LogLevel.WARN
    assert "High memory consumption" in entry2.message
    assert entry2.raw_line == line2


def test_log_parser_malformed_lines():
    malformed_line = "Random crash log text without any timestamp or level header"
    entry = parse_log_line(malformed_line)
    assert entry.level == LogLevel.UNKNOWN
    assert entry.message == malformed_line
    assert entry.raw_line == malformed_line
    assert entry.searchable_text != ""

    raw_logs = f"{malformed_line}\n[2026-09-06 10:00:00] INFO Normal log"
    parsed_entries = parse_raw_logs(raw_logs)
    assert len(parsed_entries) == 2
    assert parsed_entries[0].level == LogLevel.UNKNOWN
    assert parsed_entries[1].level == LogLevel.INFO


def test_mongo_client_incident_roundtrip():
    ticket = IncidentTicket(
        id="TEST-INC-999",
        title="Test Incident for Pytest",
        description="Unit test incident description",
        error_log="[2026-09-06 12:00:00] ERROR test error",
        environment="test",
        resolution="Resolved in unit test",
        resolved_at=datetime.utcnow(),
        tags=["pytest", "unit-test"],
    )

    inserted_id = mongo_client.insert_incident(ticket)
    assert inserted_id == "TEST-INC-999"

    fetched = mongo_client.get_incident("TEST-INC-999")
    assert fetched is not None
    assert fetched["id"] == "TEST-INC-999"
    assert fetched["title"] == "Test Incident for Pytest"
    assert fetched["environment"] == "test"
    assert "pytest" in fetched["tags"]

    filtered = mongo_client.list_incidents({"environment": "test"})
    assert len(filtered) >= 1
    assert any(i["id"] == "TEST-INC-999" for i in filtered)


def test_mongo_client_log_batch_insert():
    logs = [
        LogEntry(
            source="test-service",
            timestamp=datetime.utcnow(),
            level=LogLevel.INFO,
            message="Test log entry 1",
            raw_line="Test log line 1",
        ),
        LogEntry(
            source="test-service",
            timestamp=datetime.utcnow(),
            level=LogLevel.ERROR,
            message="Test log entry 2",
            raw_line="Test log line 2",
        ),
    ]

    inserted_ids = mongo_client.insert_log_batch(logs)
    assert len(inserted_ids) == 2
