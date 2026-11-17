
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, HTTPException, status

from api.schemas import (
    PredictionResponse,
    ResultResponse,
    WineInput,
)

# 1. Configure logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# 2. Redis configuration

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_QUEUE = "wine_prediction_jobs"

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    socket_connect_timeout=3,
    socket_timeout=3,
)


# 3. Application lifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting White Wine Quality Prediction API")

    try:
        redis_client.ping()
        logger.info("Redis connection established")
    except redis.RedisError:
        logger.warning("Redis is unavailable at startup")

    yield

    logger.info("Shutting down API")


app = FastAPI(
    title="White Wine Quality Prediction API",
    description=(
        "Asynchronous ML inference service using "
        "FastAPI, Redis, and Random Forest."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# 4. Root endpoint

@app.get("/")
def root():
    return {
        "message": "White Wine Quality Prediction API",
        "documentation": "/docs",
    }

# 5. Health endpoint

@app.get("/health")
def health():
    try:
        redis_client.ping()
        return {
            "status": "healthy",
            "redis": "connected",
        }
    except redis.RedisError as exc:
        logger.error("Redis health check failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis is unavailable",
        ) from exc


# 6. Submit prediction job

@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def predict(wine: WineInput):
    task_id = str(uuid.uuid4())

    job = {
        "task_id": task_id,
        "features": wine.model_dump(),
    }

    task_key = f"result:{task_id}"

    initial_result = {
        "task_id": task_id,
        "status": "queued",
    }

    try:
        # Store the initial task status.
        redis_client.set(
            task_key,
            json.dumps(initial_result),
        )

        # Push the job into the Redis list.
        redis_client.lpush(
            REDIS_QUEUE,
            json.dumps(job),
        )

        logger.info("Queued prediction task %s", task_id)

    except redis.RedisError as exc:
        logger.error("Failed to queue task: %s", exc)

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction queue is unavailable",
        ) from exc

    return PredictionResponse(
        task_id=task_id,
        status="queued",
    )

# 7. Retrieve prediction result

@app.get(
    "/result/{task_id}",
    response_model=ResultResponse,
)
def get_result(task_id: str):
    task_key = f"result:{task_id}"

    try:
        stored_result = redis_client.get(task_key)

    except redis.RedisError as exc:
        logger.error("Failed to retrieve task: %s", exc)

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Result storage is unavailable",
        ) from exc

    if stored_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    logger.info("Retrieved status for task %s", task_id)

    return ResultResponse(**json.loads(stored_result))