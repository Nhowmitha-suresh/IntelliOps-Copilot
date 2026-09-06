import time
import os
import structlog
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.routes import router as api_router
from app.config import settings
from app import observability

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="IntelliOps Copilot",
    description="RAG-based root-cause diagnosis copilot for deployment and ops issues",
    version="0.1.0",
)


@app.middleware("http")
async def log_request_middleware(request: Request, call_next):
    """Structlog middleware logging correlation_id, HTTP method, path, status_code, and latency."""
    client_cid = request.headers.get("X-Correlation-ID")
    cid = observability.set_correlation_id(client_cid)

    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    response.headers["X-Correlation-ID"] = cid

    observability.log_trace_event(
        event_type="http_request",
        stage="api_gateway",
        details={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
        },
        latency=latency,
    )

    logger.info(
        "HTTP Request",
        correlation_id=cid,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency=round(latency, 4),
    )
    return response


# Include API routes
app.include_router(api_router)

# Mount static files & root page
static_dir = os.path.join(os.path.dirname(__file__), "api", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=FileResponse)
def read_root_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to IntelliOps Copilot API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
