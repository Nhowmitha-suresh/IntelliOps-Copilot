# IntelliOps Copilot 🚀

**IntelliOps Copilot** is a Retrieval-Augmented Generation (RAG) based root-cause diagnosis copilot specifically designed for investigating, troubleshooting, and resolving complex deployment and operations (Ops) issues.

---

## 🎯 Purpose

Modern cloud-native operations generate vast amounts of log streams, metrics, deployment events, and incident histories. When a production deployment fails or experiences anomalies, pinpointing the root cause manually requires sifting through multiple observability tools.

IntelliOps Copilot automates root-cause analysis by combining:
- **Vector search** over historical incident reports, runbooks, and log patterns.
- **Structured metadata retrieval** for deployment configurations and system metrics.
- **LLM orchestration** with automatic fallback mechanisms to provide actionable diagnostic reports and remediation recommendations.

---

## 🏗️ Architecture Summary

The repository is structured into focused modules:

```text
intelliops-copilot/
  ├── app/
  │   ├── config.py         # Application configuration & env parsing (Pydantic Settings)
  │   ├── main.py           # FastAPI application entry point & structlog middleware
  │   ├── observability.py  # Structlog JSON/console logging & correlation_id tracing
  │   ├── debug_trace.py    # CLI tool to reconstruct timeline for a correlation_id
  │   ├── ingestion/        # Log parsing, metric streaming, incident ingestion & MongoDB client
  │   ├── embeddings/       # Chunking, vector embedding & PGVector storage engine
  │   ├── retrieval/        # Hybrid search, similarity thresholding & metadata reranker
  │   ├── orchestration/    # Multi-provider LLM client (OpenAI/Gemini), retries & agent tools
  │   ├── guardrails/       # Output validator, JSON schema re-ask & destructive command filter
  │   ├── api/              # RESTful API endpoints & static glassmorphism frontend
  │   └── models/           # Pydantic & ORM data models
  ├── evals/
  │   ├── dataset/          # Labeled benchmark test cases (eval_cases.json)
  │   ├── run_eval.py       # Full pipeline evaluation runner (Precision@5, MRR, Accuracy)
  │   ├── tune_retrieval.py # Retrieval hyperparameter tuning benchmark
  │   └── eval_report.md    # Summary evaluation report
  ├── tests/                # Automated unit & integration test suite (28 tests)
  ├── docker-compose.yml    # Database infrastructure (PostgreSQL pgvector + MongoDB)
  ├── requirements.txt      # Core Python dependencies
  └── .env.example          # Environment variable template
```

### Key Components

- **Ingestion (`app/ingestion/`)**: Parsers for logs, kubernetes events, deployment manifests, and telemetry data.
- **Embeddings (`app/embeddings/`)**: Transforms ops documentation and historical logs into vector embeddings using PGVector.
- **Retrieval (`app/retrieval/`)**: Hybrid queries using **PostgreSQL (pgvector)** and **MongoDB**, with similarity thresholding (`0.7`) and metadata reranking.
- **Orchestration (`app/orchestration/`)**: Multi-provider LLM agentic workflow (OpenAI + Gemini fallback) with tenacity retries.
- **Guardrails (`app/guardrails/`)**: Output validator enforcing JSON schema re-asks, low-confidence review flags, and destructive command warning filters.
- **Observability (`app/observability.py`)**: Distributed request context tracing via `correlation_id` (uuid4).

---

## ⚡ Quickstart & Setup

### Prerequisites
- Python 3.11+
- Docker & Docker Compose

### Step 1: Start Infrastructure Containers
Launch PostgreSQL (with `pgvector`) and MongoDB using Docker Compose:
```bash
docker-compose up -d
```

### Step 2: Environment Configuration
Copy `.env.example` to create your local `.env` file:
```bash
cp .env.example .env
```

### Step 3: Install Dependencies & Seed Data
```bash
pip install -r requirements.txt

# Seed synthetic ops incidents into MongoDB
python -m app.ingestion.seed_data
```

### Step 4: Run the Application
Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```
- **Web Interface**: Open `http://localhost:8000/` in your browser.
- **API Docs**: Open `http://localhost:8000/docs`.

---

## 🔍 Debugging This System

IntelliOps Copilot includes built-in observability with request correlation IDs (`correlation_id`). Every request generates or inherits a unique UUID header (`X-Correlation-ID`) that is automatically threaded through retrieval, orchestration, and guardrail validation steps.

### CLI Debug Trace Tool
Use `app/debug_trace.py` to reconstruct a step-by-step operational timeline and stage latency breakdown for any request:

```bash
python -m app.debug_trace test-trace-1234
```

### Sample Output:
```text
================================================================================
 INTELLIOPS COPILOT - CORRELATION TRACE ANALYSIS
================================================================================
 Correlation ID : test-trace-1234
 Total Events   : 4
 End-to-End Latency : 0.8500s
--------------------------------------------------------------------------------
 STAGE TIMELINE BREAKDOWN
--------------------------------------------------------------------------------
 [1] [2026-09-06T15:04:05.653309Z] Stage: api_gateway | Event: http_request
     - Method: POST /diagnose | Status: 200
     - Stage Latency: 0.8500s

 [2] [2026-09-06T15:04:06.479141Z] Stage: retrieval | Event: retrieval_complete
     - Query Snippet: Database connection timeout
     - Retrieved Incidents (1): INC-003
     - Top Similarity: 0.92
     - Stage Latency: 0.1200s

 [3] [2026-09-06T15:04:06.482064Z] Stage: orchestration_llm | Event: llm_generation_complete
     - Prompt Chars: 850 | Response Chars: 240
     - Stage Latency: 0.6500s

 [4] [2026-09-06T15:04:06.484265Z] Stage: guardrails | Event: guardrail_validation_complete
     - Confidence Score: 0.92 | Needs Human Review: False
     - Root Cause: PostgreSQL connection pool exhausted
     - Stage Latency: 0.0800s

--------------------------------------------------------------------------------
 STAGE LATENCY SUMMARY
--------------------------------------------------------------------------------
  - api_gateway              : 0.8500s (100.0%)
  - retrieval                : 0.1200s (14.1%)
  - orchestration_llm        : 0.6500s (76.5%)
  - guardrails               : 0.0800s (9.4%)
================================================================================
```

---

## 🧪 Testing & Evals

- **Run full unit test suite (28 tests)**:
  ```bash
  python -m pytest -v
  ```
- **Run end-to-end evaluation harness**:
  ```bash
  python -m evals.run_eval
  ```
- **Run retrieval tuning benchmark**:
  ```bash
  python -m evals.tune_retrieval
  ```
