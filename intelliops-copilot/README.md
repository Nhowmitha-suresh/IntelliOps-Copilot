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
  │   ├── main.py           # FastAPI application entry point
  │   ├── ingestion/        # Log parsing, metric streaming, incident ingestion
  │   ├── embeddings/       # Vector embedding generators for ops documents & logs
  │   ├── retrieval/        # Hybrid search (pgvector for semantic search + MongoDB for metadata)
  │   ├── orchestration/    # Multi-provider LLM agentic workflow & retry/fallback handlers
  │   ├── guardrails/       # Safety filters, validation rules & prompt guardrails
  │   ├── api/              # RESTful & WebSocket API endpoints
  │   └── models/           # Pydantic & ORM data models
  ├── evals/
  │   ├── dataset/          # Evaluation benchmarks & incident test suites
  │   └── run_eval.py       # Evaluation runner for diagnostic accuracy
  ├── tests/                # Automated unit & integration tests
  ├── docker-compose.yml    # Database infrastructure (PostgreSQL pgvector + MongoDB)
  ├── requirements.txt      # Core Python dependencies
  └── .env.example          # Environment variable template
```

### Key Components

- **Ingestion (`app/ingestion/`)**: Parsers for logs, kubernetes events, deployment manifests, and telemetry data.
- **Embeddings (`app/embeddings/`)**: Transforms ops documentation and historical logs into vector embeddings.
- **Retrieval (`app/retrieval/`)**: Executes hybrid queries using **PostgreSQL (pgvector)** for semantic similarities and **MongoDB** for document store metadata.
- **Orchestration (`app/orchestration/`)**: Manages diagnostic agent workflows using OpenAI and Google Gemini models with tenacity retries and fallback logic.
- **Guardrails (`app/guardrails/`)**: Ensures diagnostic recommendations strictly adhere to safety constraints before execution or display.

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
Update `.env` with your API keys (e.g. `OPENAI_API_KEY`, `GEMINI_API_KEY`) and connection strings if needed.

### Step 3: Install Dependencies
Create a virtual environment (optional but recommended) and install dependencies:
```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate

pip install -r requirements.txt
```

### Step 4: Run the Application
Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```
Access API documentation at `http://localhost:8000/docs`.

---

## 🧪 Testing & Evals

- **Run unit tests**:
  ```bash
  pytest
  ```
- **Run evaluation suite**:
  ```bash
  python evals/run_eval.py
  ```
