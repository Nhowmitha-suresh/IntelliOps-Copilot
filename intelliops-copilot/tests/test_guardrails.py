from unittest.mock import MagicMock
import pytest
from app.guardrails.content_filters import filter_destructive_commands
from app.guardrails.output_validator import validate_diagnosis, GuardrailViolationError
from app.models import DiagnosisResult


def test_destructive_command_detection():
    # Test rm -rf
    cmd1 = "rm -rf /var/log/app/*"
    filtered1 = filter_destructive_commands(cmd1)
    assert filtered1.startswith("⚠️ Destructive action")
    assert "rm -rf" in filtered1

    # Test DROP TABLE
    cmd2 = "Execute DROP TABLE incident_logs;"
    filtered2 = filter_destructive_commands(cmd2)
    assert filtered2.startswith("⚠️ Destructive action")
    assert "DROP TABLE" in filtered2

    # Test DELETE FROM without WHERE
    cmd3 = "DELETE FROM user_sessions"
    filtered3 = filter_destructive_commands(cmd3)
    assert filtered3.startswith("⚠️ Destructive action")

    # Test safe command (should remain untouched)
    safe_cmd = "docker-compose restart postgres"
    filtered_safe = filter_destructive_commands(safe_cmd)
    assert filtered_safe == safe_cmd
    assert "⚠️" not in filtered_safe


def test_malformed_json_triggers_reask_success():
    malformed_initial = "This is raw unformatted text instead of JSON"
    valid_reask_response = (
        '{"root_cause": "OOM Killed", "confidence": 0.85, '
        '"suggested_fix": "Increase container memory limit", '
        '"evidence_chunks": ["OOMKilled status 137"], "needs_human_review": false}'
    )

    mock_reask = MagicMock(return_value=valid_reask_response)
    result = validate_diagnosis(malformed_initial, reask_fn=mock_reask)

    assert mock_reask.call_count == 1
    assert result.root_cause == "OOM Killed"
    assert result.confidence == 0.85
    assert result.needs_human_review is False


def test_reask_failure_raises_guardrail_violation_error():
    malformed_initial = "Invalid JSON 1"
    malformed_reask = "Invalid JSON 2"

    mock_reask = MagicMock(return_value=malformed_reask)
    with pytest.raises(GuardrailViolationError):
        validate_diagnosis(malformed_initial, reask_fn=mock_reask)


def test_low_confidence_forces_human_review():
    raw_json = (
        '{"root_cause": "Uncertain cause", "confidence": 0.3, '
        '"suggested_fix": "Inspect logs manually", '
        '"evidence_chunks": ["log line 1"], "needs_human_review": false}'
    )
    result = validate_diagnosis(raw_json)
    assert result.confidence == 0.3
    # Guardrail must force needs_human_review to True despite false in model output
    assert result.needs_human_review is True


def test_empty_evidence_chunks_forces_human_review_and_warning():
    raw_json = (
        '{"root_cause": "Unknown", "confidence": 0.8, '
        '"suggested_fix": "Restart service", '
        '"evidence_chunks": [], "needs_human_review": false}'
    )
    result = validate_diagnosis(raw_json)
    assert result.evidence_chunks == []
    assert result.needs_human_review is True
    assert "⚠️ Low context" in result.suggested_fix
