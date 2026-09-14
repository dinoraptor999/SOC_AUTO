"""FastAPI application exposing anomaly prediction."""

from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_model, predict_one
from src.utils import setup_logger

logger = setup_logger("soc_auto.api")
_artifacts = None


class EventInput(BaseModel):
    """Validated security event payload."""

    src_ip: str
    dst_ip: str
    src_port: int = Field(ge=0, le=65535)
    dst_port: int = Field(ge=0, le=65535)
    protocol: str
    event_type: str
    failed_count: int = Field(default=0, ge=0)
    bytes: float = Field(default=0, ge=0)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load model artifacts once when available."""
    global _artifacts
    try:
        _artifacts = load_model()
        logger.info("model artifacts loaded")
    except FileNotFoundError:
        logger.warning("model artifacts are missing; train the model before prediction")
    yield


app = FastAPI(title="SOC Auto AI", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request, call_next):
    """Log method, path, status, and duration for every HTTP request."""
    started_at = perf_counter()
    response = await call_next(request)
    duration_ms = (perf_counter() - started_at) * 1000
    logger.info(
        "%s %s -> %d (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health")
def health() -> dict:
    """Return service liveness."""
    return {"status": "ok"}


@app.post("/predict")
def predict(event: EventInput) -> dict:
    """Predict whether one security event is anomalous."""
    if _artifacts is None:
        raise HTTPException(status_code=503, detail="Model is not trained")
    result = predict_one(event.model_dump(), artifacts=_artifacts)
    logger.info("prediction completed for %s -> %s", event.src_ip, result["is_anomaly"])
    return result
