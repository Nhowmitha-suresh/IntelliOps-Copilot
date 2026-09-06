from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models import DiagnosisResult, IncidentTicket

client = TestClient(app)


def test_health_endpoint_success():
    with patch("app.ingestion.mongo_client.get_database") as mock_mongo_db:
        with patch("app.embeddings.pgvector_store.get_engine") as mock_pg_engine:
            mock_db = MagicMock()
            mock_db.command.return_value = {"ok": 1}
            mock_mongo_db.return_value = mock_db

            mock_conn = MagicMock()
            mock_engine = MagicMock()
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            mock_pg_engine.return_value = mock_engine

            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["mongodb"] == "connected"
            assert data["postgres"] == "connected"


def test_diagnose_endpoint_success():
    mock_diagnosis = DiagnosisResult(
        root_cause="PostgreSQL connection pool exhausted",
        confidence=0.92,
        suggested_fix="Increase pool_size in SQLAlchemy engine configuration.",
        evidence_chunks=["sqlalchemy.exc.TimeoutError"],
        needs_human_review=False,
    )

    with patch("app.orchestration.agent.diagnose_issue", return_value=mock_diagnosis) as mock_agent:
        payload = {
            "issue_description": "Connection pool timeout error under load",
            "environment": "production",
        }
        response = client.post("/diagnose", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["root_cause"] == "PostgreSQL connection pool exhausted"
        assert data["confidence"] == 0.92
        assert data["needs_human_review"] is False
        assert mock_agent.call_count == 1


def test_get_incident_endpoint_success_and_not_found():
    with patch("app.ingestion.mongo_client.get_incident") as mock_get_inc:
        # Success case
        mock_get_inc.return_value = {
            "id": "INC-001",
            "title": "DB Password missing",
            "environment": "production",
        }
        resp1 = client.get("/incidents/INC-001")
        assert resp1.status_code == 200
        assert resp1.json()["id"] == "INC-001"

        # Not found case
        mock_get_inc.return_value = None
        resp2 = client.get("/incidents/NONEXISTENT")
        assert resp2.status_code == 404
        assert "not found" in resp2.json()["detail"].lower()


def test_ingest_incident_endpoint():
    ticket_payload = {
        "id": "INC-API-TEST",
        "title": "API test incident title",
        "description": "API test description",
        "error_log": "ERROR 500 API test",
        "environment": "staging",
        "resolution": "Resolved in API test",
        "resolved_at": "2026-09-06T12:00:00Z",
        "tags": ["api", "test"],
    }

    with patch("app.ingestion.mongo_client.insert_incident", return_value="INC-API-TEST") as mock_insert:
        with patch("app.embeddings.chunker.chunk_incident") as mock_chunk:
            with patch("app.embeddings.embedder.embed_texts", return_value=[[0.1] * 1536]):
                with patch("app.embeddings.pgvector_store.upsert_chunks", return_value=["id1"]):
                    mock_chunk.return_value = [
                        MagicMock(
                            incident_id="INC-API-TEST",
                            chunk_index=0,
                            chunk_text="Chunk text",
                        )
                    ]
                    response = client.post("/ingest/incident", json=ticket_payload)
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "success"
                    assert data["incident_id"] == "INC-API-TEST"
                    assert data["chunks_indexed"] == 1
