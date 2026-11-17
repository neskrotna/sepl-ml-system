
import json
import logging
import os
import time

import redis

from ml_model.model import load_model, preprocess_input

# 1. Logging configuration

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
    socket_connect_timeout=5,
)

# 3. Convert API feature names

def convert_feature_names(features: dict) -> dict:
    """Convert API field names into training feature names."""

    converted = {}

    for key, value in features.items():
        if key == "pH":
            converted[key] = value
        else:
            converted[key.replace("_", " ")] = value

    return converted


# 4. Process a prediction job

def process_job(job: dict, model):
    task_id = job["task_id"]
    result_key = f"result:{task_id}"

    logger.info("Starting processing for task %s", task_id)

    redis_client.set(
        result_key,
        json.dumps({
            "task_id": task_id,
            "status": "processing",
        }),
    )

    start_time = time.perf_counter()

    try:
        # Convert incoming API feature names.
        features = convert_feature_names(job["features"])

        # Apply the same preprocessing used locally.
        processed = preprocess_input(features)

        # Perform inference.
        prediction = int(model.predict(processed)[0])

        # Calculate confidence.
        probabilities = model.predict_proba(processed)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = float(probabilities[class_index])

        # Convert numerical class into readable label.
        quality_label = (
            "good_quality"
            if prediction == 1
            else "lower_quality"
        )

        elapsed = time.perf_counter() - start_time

        result = {
            "task_id": task_id,
            "status": "completed",
            "prediction": quality_label,
            "class_id": prediction,
            "confidence": round(confidence, 4),
        }

        # Save the result in Redis.
        redis_client.set(result_key, json.dumps(result))

        logger.info(
            "Completed task %s | prediction=%s | confidence=%.4f "
            "| inference_time=%.4fs",
            task_id,
            quality_label,
            confidence,
            elapsed,
        )

    except Exception as exc:
        logger.exception("Task %s failed", task_id)

        redis_client.set(
            result_key,
            json.dumps({
                "task_id": task_id,
                "status": "failed",
                "error": str(exc),
            }),
        )

# 5. Worker main

def main():
    logger.info("Starting White Wine Quality Worker")

    # Load the model once at startup.
    model = load_model()
    logger.info("Random Forest model loaded successfully")

    # Verify Redis connection.
    redis_client.ping()
    logger.info("Connected to Redis")

    logger.info("Waiting for prediction jobs...")

    while True:
        # BRPOP waits until a new job is available.
        job_item = redis_client.brpop(
            REDIS_QUEUE,
            timeout=0,
        )

        if job_item is None:
            continue

        _, job_json = job_item

        job = json.loads(job_json)

        logger.info(
            "Received task %s from Redis",
            job["task_id"],
        )

        process_job(job, model)


if __name__ == "__main__":
    main()