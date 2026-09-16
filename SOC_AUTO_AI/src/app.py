"""FastAPI application exposing anomaly prediction."""

from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import EVALUATION_LOG_PATH, THRESHOLD_PATH
from src.predict import load_model, predict_one
from src.utils import load_json, setup_logger

logger = setup_logger("soc_auto.api", EVALUATION_LOG_PATH.parent / "api.log")
_artifacts = None


class EventInput(BaseModel):
    """Validated security event payload."""

    src_ip: str
    dst_ip: str
    src_port: int = Field(ge=0, le=65535)
    dst_port: int = Field(ge=0, le=65535)
    protocol: str
    event_type: str
    failed_count: int = Field(ge=0)
    bytes: float = Field(ge=0)
    timestamp: str | None = None


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


app = FastAPI(title="SOC Auto AI", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status, and duration for every HTTP request."""
    started_at = perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = (perf_counter() - started_at) * 1000
        logger.info("%s %s (%.2f ms)", request.method, request.url.path, duration_ms)


@app.get("/health")
def health() -> dict:
    """Return service status and the loaded model threshold."""
    threshold = float(_artifacts[3]) if _artifacts is not None else float(load_json(THRESHOLD_PATH)["threshold"])
    return {
        "status": "ok",
        "model": "IsolationForest",
        "threshold": threshold,
        "version": "1.0",
    }


@app.post("/predict")
def predict(event: EventInput) -> dict:
    """Predict whether one security event is anomalous."""
    if _artifacts is None:
        raise HTTPException(status_code=500, detail="Model artifacts are unavailable")
    try:
        result = predict_one(event.model_dump(), artifacts=_artifacts)
        logger.info("prediction completed for %s -> %s", event.src_ip, result["is_anomaly"])
        return result
    except Exception as error:
        logger.exception("prediction failed for %s", event.src_ip)
        raise HTTPException(status_code=500, detail="Prediction failed") from error
