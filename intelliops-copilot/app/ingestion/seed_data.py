from datetime import datetime, timedelta
from typing import List
from app.models import IncidentTicket
from app.ingestion.mongo_client import insert_incident, setup_indexes

SEED_INCIDENTS = [
    IncidentTicket(
        id="INC-001",
        title="Environment variable DB_PASSWORD missing on production pod deployment",
        description="Backend API container crash-looped after deploying v2.4.1 due to KeyError: DB_PASSWORD during app initialization.",
        error_log="[2026-09-01 10:15:22] [CRITICAL] KeyError: 'DB_PASSWORD' not found in environment variables. Process exiting with code 1.",
        config_snippet="db_pass = os.environ['DB_PASSWORD']",
        environment="production",
        resolution="Updated Kubernetes Secret 'backend-secrets' with DB_PASSWORD and redeployed deployment/backend-api.",
        resolved_at=datetime.utcnow() - timedelta(days=5),
        tags=["env-var", "config", "production", "backend"],
    ),
    IncidentTicket(
        id="INC-002",
        title="Pydantic dependency version mismatch breaking settings parser",
        description="Deploy failed during container startup because pydantic-settings was missing after upgrading to Pydantic v2.",
        error_log="[2026-09-02 08:30:11] [ERROR] ModuleNotFoundError: No module named 'pydantic_settings'",
        config_snippet="from pydantic_settings import BaseSettings",
        environment="staging",
        resolution="Added pydantic-settings>=2.0.0 to requirements.txt and rebuilt Docker image.",
        resolved_at=datetime.utcnow() - timedelta(days=4, hours=2),
        tags=["dependency", "python", "pydantic", "staging"],
    ),
    IncidentTicket(
        id="INC-003",
        title="PostgreSQL connection pool exhausted under heavy load",
        description="API latency spiked to 10s and 500 errors were returned due to DB connection pool exhaustion.",
        error_log="[2026-09-02 14:22:05] [ERROR] sqlalchemy.exc.TimeoutError: QueuePool limit of size 20 overflow 10 reached, connection timed out",
        config_snippet="engine = create_engine(POSTGRES_DSN, pool_size=20, max_overflow=10)",
        environment="production",
        resolution="Increased pool_size to 50 and enabled pool_pre_ping in SQLAlchemy engine configuration.",
        resolved_at=datetime.utcnow() - timedelta(days=4),
        tags=["database", "postgresql", "connection-pool", "production"],
    ),
    IncidentTicket(
        id="INC-004",
        title="Port 8000 binding conflict on host during service container startup",
        description="FastAPI service failed to bind to port 8000 because a stale uvicorn worker process occupied the address.",
        error_log="[2026-09-03 09:12:00] [CRITICAL] OSError: [Errno 98] Address already in use: ('0.0.0.0', 8000)",
        config_snippet="ports:\n  - '8000:8000'",
        environment="staging",
        resolution="Killed stale process on port 8000 using fuser -k 8000/tcp and adjusted docker-compose.yml restart policy.",
        resolved_at=datetime.utcnow() - timedelta(days=3, hours=5),
        tags=["port-conflict", "uvicorn", "docker", "staging"],
    ),
    IncidentTicket(
        id="INC-005",
        title="Expired TLS/SSL certificate causing HTTPS handshake errors on ingress",
        description="External client requests to https://api.intelliops.io failed with SSLCertVerificationError.",
        error_log="[2026-09-03 11:45:30] [ERROR] ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate has expired",
        config_snippet="tls:\n  - secretName: api-intelliops-tls-cert\n    hosts:\n      - api.intelliops.io",
        environment="production",
        resolution="Triggered cert-manager certificate renewal and reloaded NGINX ingress controller.",
        resolved_at=datetime.utcnow() - timedelta(days=3),
        tags=["ssl", "tls", "ingress", "cert-manager", "production"],
    ),
    IncidentTicket(
        id="INC-006",
        title="Container OOMKilled due to unconstrained pandas dataframe memory usage",
        description="Worker container crashed with status 137 (OOMKilled) while processing large log dataset.",
        error_log="[2026-09-04 03:00:15] [CRITICAL] Container intelliops-worker killed by Linux OOM killer (used: 2048MB, limit: 2048MB)",
        config_snippet="resources:\n  limits:\n    memory: 2Gi",
        environment="production",
        resolution="Chunked dataframe processing batch size from 500,000 to 50,000 rows and increased memory limit to 4Gi.",
        resolved_at=datetime.utcnow() - timedelta(days=2, hours=10),
        tags=["memory", "oom", "pandas", "worker", "production"],
    ),
    IncidentTicket(
        id="INC-007",
        title="Stale Redis cache serving outdated API responses post-deployment",
        description="Users reported seeing old feature flag values because Redis cache keys lacked version prefixing.",
        error_log="[2026-09-04 15:10:00] [WARN] Cache hit for key 'config:flags' returning legacy v1 format.",
        config_snippet="redis_client.get('config:flags')",
        environment="production",
        resolution="Flushed legacy keys using redis-cli EVAL script and appended deploy commit hash to cache key prefixes.",
        resolved_at=datetime.utcnow() - timedelta(days=2),
        tags=["redis", "cache", "deployment", "production"],
    ),
    IncidentTicket(
        id="INC-008",
        title="CORS policy blocking frontend requests from new dashboard domain",
        description="Browser web app threw CORS error preflight request failed when communicating with API backend.",
        error_log="[2026-09-05 07:20:44] [WARN] Access to XMLHttpRequest at 'https://api.intelliops.io/v1/query' from origin 'https://app.intelliops.io' has been blocked by CORS policy.",
        config_snippet="app.add_middleware(CORSMiddleware, allow_origins=['https://dashboard.intelliops.io'])",
        environment="production",
        resolution="Added 'https://app.intelliops.io' to CORSMiddleware allow_origins in app/main.py.",
        resolved_at=datetime.utcnow() - timedelta(days=1, hours=8),
        tags=["cors", "fastapi", "frontend", "production"],
    ),
    IncidentTicket(
        id="INC-009",
        title="JWT Auth token expiry discrepancy causing premature user logout",
        description="Tokens issued by auth service expired after 15 minutes instead of configured 8 hours due to seconds vs minutes unit confusion.",
        error_log="[2026-09-05 13:05:12] [ERROR] jwt.exceptions.ExpiredSignatureError: Signature has expired",
        config_snippet="exp = datetime.utcnow() + timedelta(seconds=ACCESS_TOKEN_EXPIRE_MINUTES)",
        environment="production",
        resolution="Corrected token expiration calculation to timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES).",
        resolved_at=datetime.utcnow() - timedelta(days=1),
        tags=["auth", "jwt", "token-expiry", "production"],
    ),
    IncidentTicket(
        id="INC-010",
        title="Disk space exhausted on log aggregator node",
        description="Elasticsearch and Fluentd failed to write incoming application logs due to 100% disk utilization on /var/log.",
        error_log="[2026-09-05 18:40:02] [CRITICAL] IOError: [Errno 28] No space left on device: '/var/log/app/output.log'",
        config_snippet="logrotate.conf: maxsize 100M, retain 5",
        environment="production",
        resolution="Cleaned up old Docker log files, enabled automatic log rotation compression, and resized disk volume.",
        resolved_at=datetime.utcnow() - timedelta(hours=18),
        tags=["disk-space", "logging", "infrastructure", "production"],
    ),
    IncidentTicket(
        id="INC-011",
        title="DNS resolution failure for internal microservice hostname",
        description="Ingestion worker failed to resolve hostname 'internal-retrieval-svc.local' inside CoreDNS.",
        error_log="[2026-09-06 01:12:33] [ERROR] socket.gaierror: [Errno -2] Name or service not known: 'internal-retrieval-svc.local'",
        config_snippet="RETRIEVAL_URL = 'http://internal-retrieval-svc.local:8000'",
        environment="staging",
        resolution="Updated Kubernetes Service object selector to match active pod labels and restarted CoreDNS pods.",
        resolved_at=datetime.utcnow() - timedelta(hours=14),
        tags=["dns", "kubernetes", "networking", "staging"],
    ),
    IncidentTicket(
        id="INC-012",
        title="Kubernetes Liveness probe failed due to blocking synchronous DB call",
        description="K8s kubelet restarted API pods repeatedly because /health endpoint executed a blocking DB ping that timed out.",
        error_log="[2026-09-06 04:30:19] [WARN] Liveness probe failed: HTTP probe failed with statuscode 500 for http://10.244.1.4:8000/health",
        config_snippet="livenessProbe:\n  httpGet:\n    path: /health\n    port: 8000\n  timeoutSeconds: 1",
        environment="production",
        resolution="Separated /healthz lightweight check from /readiness DB check and increased timeoutSeconds to 3.",
        resolved_at=datetime.utcnow() - timedelta(hours=10),
        tags=["kubernetes", "liveness-probe", "health-check", "production"],
    ),
    IncidentTicket(
        id="INC-013",
        title="Third-party OpenAI API rate limit exceeded (429 Too Many Requests)",
        description="LLM orchestration workflow failed during batch evaluation due to HTTP 429 rate limit errors without retry backoff.",
        error_log="[2026-09-06 06:15:40] [ERROR] openai.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4'}}",
        config_snippet="response = openai.ChatCompletion.create(...)",
        environment="staging",
        resolution="Wrapped LLM calls with tenacity Exponential Backoff retry handler and switched fallback provider to Gemini.",
        resolved_at=datetime.utcnow() - timedelta(hours=8),
        tags=["openai", "rate-limit", "tenacity", "llm", "staging"],
    ),
    IncidentTicket(
        id="INC-014",
        title="Kafka consumer group rebalance storm caused by long processing delay",
        description="Event stream processor exceeded max.poll.interval.ms while generating embeddings, triggering consumer eviction.",
        error_log="[2026-09-06 08:00:55] [WARN] org.apache.kafka.clients.consumer.internals.ConsumerCoordinator: Commit failed on partition 2: CommitFailedException",
        config_snippet="max.poll.interval.ms = 300000",
        environment="production",
        resolution="Increased max.poll.interval.ms to 900000 and reduced max.poll.records to 50.",
        resolved_at=datetime.utcnow() - timedelta(hours=6),
        tags=["kafka", "streaming", "consumer-group", "production"],
    ),
    IncidentTicket(
        id="INC-015",
        title="MongoDB authentication failed due to authSource query parameter missing in URI",
        description="Ingestion worker failed to connect to MongoDB container because root credentials required authSource=admin.",
        error_log="[2026-09-06 10:22:18] [CRITICAL] pymongo.errors.OperationFailure: Authentication failed.",
        config_snippet="MONGO_URI = 'mongodb://root:mongopassword@localhost:27017/intelliops'",
        environment="development",
        resolution="Updated MONGO_URI string to include ?authSource=admin parameter.",
        resolved_at=datetime.utcnow() - timedelta(hours=2),
        tags=["mongodb", "auth", "connection", "development"],
    ),
    IncidentTicket(
        id="INC-016",
        title="Vault secrets engine permission denied on production app role",
        description="App instance failed to retrieve DB password from HashiCorp Vault on boot.",
        error_log="[2026-09-06 11:50:00] [ERROR] hvac.exceptions.Forbidden: 403 Client Error: Forbidden for url: http://vault:8200/v1/secret/data/db",
        config_snippet="client.read('secret/data/db')",
        environment="production",
        resolution="Updated Vault ACL policy 'app-readonly' to allow read access on path 'secret/data/db'.",
        resolved_at=datetime.utcnow() - timedelta(minutes=30),
        tags=["vault", "secrets", "security", "production"],
    ),
]


def seed() -> List[str]:
    """Inserts all synthetic incidents into MongoDB."""
    print("Initializing MongoDB indexes...")
    setup_indexes()
    inserted_ids = []
    print(f"Seeding {len(SEED_INCIDENTS)} synthetic incidents into MongoDB...")
    for ticket in SEED_INCIDENTS:
        tid = insert_incident(ticket)
        inserted_ids.append(tid)
        print(f"  - Inserted incident [{tid}]: {ticket.title}")
    print(f"Successfully seeded {len(inserted_ids)} incidents.")
    return inserted_ids


if __name__ == "__main__":
    seed()
