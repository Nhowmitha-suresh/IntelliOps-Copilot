from typing import Optional, List, Dict, Any
from pymongo import MongoClient, ASCENDING, DESCENDING
from app.config import settings
from app.models import IncidentTicket, LogEntry

_client: Optional[MongoClient] = None


def get_mongo_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.mongo_uri)
    return _client


def get_database(db_name: Optional[str] = None):
    client = get_mongo_client()
    if db_name:
        return client[db_name]
    return client.get_default_database("intelliops")


def setup_indexes() -> None:
    """Creates indexes on 'tags', 'environment', and 'resolved_at' for the incidents collection."""
    db = get_database()
    incidents_col = db["incidents"]
    incidents_col.create_index([("tags", ASCENDING)], name="idx_tags")
    incidents_col.create_index([("environment", ASCENDING)], name="idx_environment")
    incidents_col.create_index([("resolved_at", DESCENDING)], name="idx_resolved_at")

    logs_col = db["logs"]
    logs_col.create_index([("timestamp", DESCENDING)], name="idx_timestamp")
    logs_col.create_index([("level", ASCENDING)], name="idx_level")


def insert_incident(ticket: IncidentTicket) -> str:
    """Inserts a single IncidentTicket into MongoDB."""
    db = get_database()
    setup_indexes()
    ticket_dict = ticket.to_dict()
    # Use custom id as _id in mongo or keep both
    ticket_dict["_id"] = ticket.id
    db["incidents"].replace_one({"_id": ticket.id}, ticket_dict, upsert=True)
    return ticket.id


def get_incident(incident_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves an incident by its ID."""
    db = get_database()
    doc = db["incidents"].find_one({"_id": incident_id})
    if not doc:
        doc = db["incidents"].find_one({"id": incident_id})
    if doc:
        doc.pop("_id", None)
        return doc
    return None


def list_incidents(filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Lists incidents matching an optional filter dictionary."""
    db = get_database()
    query = filter_dict or {}
    cursor = db["incidents"].find(query)
    results = []
    for doc in cursor:
        doc.pop("_id", None)
        results.append(doc)
    return results


def insert_log_batch(logs: List[LogEntry]) -> List[str]:
    """Inserts a batch of LogEntry items into MongoDB."""
    if not logs:
        return []
    db = get_database()
    setup_indexes()
    log_dicts = []
    inserted_ids = []
    for log in logs:
        l_dict = log.to_dict()
        l_dict["_id"] = log.id
        log_dicts.append(l_dict)
        inserted_ids.append(log.id)
    
    db["logs"].insert_many(log_dicts, ordered=False)
    return inserted_ids
