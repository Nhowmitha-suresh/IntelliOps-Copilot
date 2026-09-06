import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    DateTime,
    Text,
    Index,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from app.config import settings

Base = declarative_base()


class IncidentChunkModel(Base):
    __tablename__ = "incident_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    incident_id = Column(String, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index(
            "idx_incident_chunks_embedding",
            embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.postgres_dsn, pool_pre_ping=True)
    return _engine


def get_session():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


def init_db(engine=None):
    """Ensures vector extension exists and creates database tables and indexes."""
    eng = engine or get_engine()
    with eng.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    Base.metadata.create_all(bind=eng)


def upsert_chunks(
    chunks_with_embeddings: List[Dict[str, Any]],
    session=None,
    model_cls=IncidentChunkModel,
) -> List[str]:
    """Upserts chunks with embeddings into PostgreSQL.

    Each dict must contain 'incident_id', 'chunk_index', 'chunk_text', 'embedding'.
    """
    if not chunks_with_embeddings:
        return []

    close_session = False
    if session is None:
        session = get_session()
        close_session = True

    inserted_ids = []
    try:
        init_db(engine=session.get_bind())
        for item in chunks_with_embeddings:
            chunk_id = item.get("id") or str(uuid.uuid4())
            existing = (
                session.query(model_cls)
                .filter(
                    model_cls.incident_id == item["incident_id"],
                    model_cls.chunk_index == item["chunk_index"],
                )
                .first()
            )
            if existing:
                existing.chunk_text = item["chunk_text"]
                existing.embedding = item["embedding"]
                inserted_ids.append(existing.id)
            else:
                record = model_cls(
                    id=chunk_id,
                    incident_id=item["incident_id"],
                    chunk_index=item["chunk_index"],
                    chunk_text=item["chunk_text"],
                    embedding=item["embedding"],
                    created_at=item.get("created_at") or datetime.utcnow(),
                )
                session.add(record)
                inserted_ids.append(chunk_id)

        session.commit()
        return inserted_ids
    except Exception:
        session.rollback()
        raise
    finally:
        if close_session:
            session.close()


def similarity_search(
    query_embedding: List[float],
    top_k: int = 5,
    environment_filter: Optional[str] = None,
    session=None,
    model_cls=IncidentChunkModel,
    incident_id_subquery: Optional[List[str]] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """Performs cosine similarity search using pgvector.

    Returns a list of tuples: (chunk_dict, cosine_distance_score).
    """
    close_session = False
    if session is None:
        session = get_session()
        close_session = True

    try:
        init_db(engine=session.get_bind())
        distance_col = model_cls.embedding.cosine_distance(query_embedding).label(
            "distance"
        )
        query = session.query(model_cls, distance_col)

        if incident_id_subquery is not None:
            query = query.filter(model_cls.incident_id.in_(incident_id_subquery))

        results = query.order_by(distance_col).limit(top_k).all()

        output = []
        for record, dist in results:
            chunk_dict = {
                "id": record.id,
                "incident_id": record.incident_id,
                "chunk_index": record.chunk_index,
                "chunk_text": record.chunk_text,
                "created_at": record.created_at,
            }
            output.append((chunk_dict, float(dist)))
        return output
    finally:
        if close_session:
            session.close()
