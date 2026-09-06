from fastapi import FastAPI
from app.config import settings

app = FastAPI(
    title="IntelliOps Copilot",
    description="RAG-based root-cause diagnosis copilot for deployment and ops issues",
    version="0.1.0",
)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to IntelliOps Copilot API",
        "status": "online",
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "primary_provider": settings.llm_primary_provider,
        "fallback_provider": settings.llm_fallback_provider,
        "max_retries": settings.max_retries,
        "timeout_seconds": settings.request_timeout_seconds,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
