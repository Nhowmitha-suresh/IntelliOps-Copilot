from unittest.mock import patch, MagicMock
import pytest
from app.orchestration.llm_client import generate, LLMOrchestrationError
from app.orchestration.agent import (
    fetch_similar_incidents_tool,
    fetch_raw_logs_tool,
    diagnose_issue,
)
from app.models import DiagnosisResult


def test_llm_client_primary_success():
    with patch(
        "app.orchestration.llm_client._call_openai",
        return_value='{"root_cause": "test"}',
    ) as mock_primary:
        res = generate("Test prompt")
        assert res == '{"root_cause": "test"}'
        assert mock_primary.call_count == 1


def test_llm_client_retry_on_primary_failure_then_succeeds():
    mock_primary = MagicMock(
        side_effect=[
            Exception("Timeout 1"),
            Exception("Timeout 2"),
            '{"root_cause": "success"}',
        ]
    )
    with patch("app.orchestration.llm_client._call_openai", mock_primary):
        with patch("time.sleep", return_value=None):
            res = generate("Test prompt")
            assert res == '{"root_cause": "success"}'
            assert mock_primary.call_count == 3


def test_llm_client_failover_to_fallback():
    mock_primary = MagicMock(side_effect=Exception("OpenAI API Down"))
    mock_fallback = MagicMock(return_value='{"root_cause": "fallback_success"}')
    with patch("app.orchestration.llm_client._call_openai", mock_primary):
        with patch("app.orchestration.llm_client._call_gemini", mock_fallback):
            with patch("time.sleep", return_value=None):
                res = generate("Test prompt")
                assert res == '{"root_cause": "fallback_success"}'
                assert mock_primary.call_count == 3
                assert mock_fallback.call_count == 1


def test_llm_client_both_providers_fail_raises_error():
    mock_primary = MagicMock(side_effect=Exception("OpenAI Down"))
    mock_fallback = MagicMock(side_effect=Exception("Gemini Down"))
    with patch("app.orchestration.llm_client._call_openai", mock_primary):
        with patch("app.orchestration.llm_client._call_gemini", mock_fallback):
            with patch("time.sleep", return_value=None):
                with pytest.raises(LLMOrchestrationError):
                    generate("Test prompt")


def test_agent_tools_execution():
    with patch(
        "app.retrieval.retriever.retrieve_similar_incidents",
        return_value=[{"incident_id": "INC-001"}],
    ):
        res = fetch_similar_incidents_tool.invoke({"query_text": "port binding error"})
        assert "INC-001" in res

    with patch(
        "app.ingestion.mongo_client.get_incident",
        return_value={"id": "INC-001", "title": "Port conflict"},
    ):
        res = fetch_raw_logs_tool.invoke({"incident_id": "INC-001"})
        assert "INC-001" in res


def test_diagnose_issue_workflow():
    dummy_llm_response = (
        '{"root_cause": "Port 8000 address already in use", "confidence": 0.95, '
        '"suggested_fix": "Kill process on 8000", "evidence_chunks": ["OSError 98"], "needs_human_review": false}'
    )
    with patch("app.retrieval.retriever.retrieve_similar_incidents", return_value=[]):
        diag = diagnose_issue(
            "Port 8000 bound", custom_llm_fn=lambda prompt: dummy_llm_response
        )
        assert isinstance(diag, DiagnosisResult)
        assert diag.root_cause == "Port 8000 address already in use"
        assert diag.confidence == 0.95
        assert diag.needs_human_review is False
