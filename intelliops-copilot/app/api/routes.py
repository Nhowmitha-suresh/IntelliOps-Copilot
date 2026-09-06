from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from app.models import IncidentTicket, DiagnosisResult
from app.ingestion import mongo_client
from app.embeddings import chunker, embedder, pgvector_store
from app.orchestration import agent

router = APIRouter()


class DiagnoseRequest(BaseModel):
    issue_description: str = Field(
        ..., description="Description of the issue or error encountered."
    )
    raw_log: Optional[str] = Field(
        None, description="Optional raw log snippet to append to analysis."
    )
    environment: Optional[str] = Field(
        None, description="Optional environment filter (e.g. production, staging)."
    )


class IngestResponse(BaseModel):
    status: str
    incident_id: str
    chunks_indexed: int


@router.post("/diagnose", response_model=DiagnosisResult)
def diagnose_issue_endpoint(request: DiagnoseRequest) -> DiagnosisResult:
    """Runs the full RAG root-cause diagnosis pipeline for a reported issue."""
    combined_query = request.issue_description
    if request.raw_log and request.raw_log.strip():
        combined_query += f"\nRaw Log:\n{request.raw_log.strip()}"

    try:
        return agent.diagnose_issue(
            issue_description=combined_query,
            environment=request.environment,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagnosis pipeline error: {str(exc)}",
        )


@router.post("/ingest/incident", response_model=IngestResponse)
def ingest_incident_endpoint(ticket: IncidentTicket) -> IngestResponse:
    """Stores an IncidentTicket in MongoDB and triggers chunking, embedding,

    and upserting into PGVector.
    """
    # 1. Store in MongoDB
    inc_id = mongo_client.insert_incident(ticket)

    # 2. Chunk incident
    inc_dict = ticket.to_dict()
    chunks = chunker.chunk_incident(inc_dict)

    # 3. Generate embeddings
    if chunks:
        chunk_texts = [c.chunk_text for c in chunks]
        embeddings = embedder.embed_texts(chunk_texts)

        payload = [
            {
                "incident_id": c.incident_id,
                "chunk_index": c.chunk_index,
                "chunk_text": c.chunk_text,
                "embedding": emb,
            }
            for c, emb in zip(chunks, embeddings)
        ]
        pgvector_store.upsert_chunks(payload)

    return IngestResponse(
        status="success",
        incident_id=inc_id,
        chunks_indexed=len(chunks),
    )


@router.get("/health")
def health_check_endpoint() -> Dict[str, Any]:
    """Returns connectivity status for MongoDB and PostgreSQL databases."""
    health = {"status": "healthy", "mongodb": "disconnected", "postgres": "disconnected"}

    # Check MongoDB
    try:
        db = mongo_client.get_database()
        db.command("ping")
        health["mongodb"] = "connected"
    except Exception as exc:
        health["status"] = "unhealthy"
        health["mongodb_error"] = str(exc)

    # Check PostgreSQL
    try:
        engine = pgvector_store.get_engine()
        with engine.connect() as conn:
            conn.execute(pgvector_store.text("SELECT 1;"))
        health["postgres"] = "connected"
    except Exception as exc:
        health["status"] = "unhealthy"
        health["postgres_error"] = str(exc)

    return health


@router.get("/incidents/{incident_id}")
def get_incident_endpoint(incident_id: str) -> Dict[str, Any]:
    """Retrieves a stored incident record by its ID."""
    doc = mongo_client.get_incident(incident_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID '{incident_id}' not found.",
        )
    return doc
