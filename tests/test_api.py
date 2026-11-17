import json
from unittest.mock import patch

import pytest
import redis
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

@pytest.fixture
def valid_wine():
    """Valid white wine measurements for API testing."""
    return {
        "fixed_acidity": 7.0,
        "volatile_acidity": 0.27,
        "citric_acid": 0.36,
        "residual_sugar": 20.7,
        "chlorides": 0.045,
        "free_sulfur_dioxide": 45.0,
        "total_sulfur_dioxide": 170.0,
        "density": 1.001,
        "pH": 3.0,
        "sulphates": 0.45,
        "alcohol": 8.8,
    }


# Test 1: Root endpoint

def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["documentation"] == "/docs"


# Test 2: Health endpoint with Redis available

@patch("api.main.redis_client")
def test_health_success(mock_redis):
    mock_redis.ping.return_value = True

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "redis": "connected",
    }


# Test 3: Health endpoint with Redis unavailable

@patch("api.main.redis_client")
def test_health_redis_unavailable(mock_redis):
    mock_redis.ping.side_effect = redis.RedisError(
        "Connection failed"
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "Redis is unavailable"


# Test 4: Valid prediction request

@patch("api.main.redis_client")
def test_predict_valid_request(mock_redis, valid_wine):
    response = client.post("/predict", json=valid_wine)

    assert response.status_code == 202

    data = response.json()

    assert data["status"] == "queued"
    assert "task_id" in data

    mock_redis.set.assert_called_once()
    mock_redis.lpush.assert_called_once()

    queued_job = json.loads(
        mock_redis.lpush.call_args.args[1]
    )

    assert queued_job["task_id"] == data["task_id"]
    assert queued_job["features"] == valid_wine


# Test 5: Missing required features

def test_predict_missing_features():
    response = client.post(
        "/predict",
        json={
            "fixed_acidity": 7.0,
            "alcohol": 8.8,
        },
    )

    assert response.status_code == 422


# Test 6: Invalid feature type

def test_predict_invalid_feature_type(valid_wine):
    invalid_wine = valid_wine.copy()
    invalid_wine["alcohol"] = "not_a_number"

    response = client.post(
        "/predict",
        json=invalid_wine,
    )

    assert response.status_code == 422


# Test 7: Redis unavailable during prediction

@patch("api.main.redis_client")
def test_predict_redis_unavailable(mock_redis, valid_wine):
    mock_redis.set.side_effect = redis.RedisError(
        "Connection failed"
    )

    response = client.post("/predict", json=valid_wine)

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Prediction queue is unavailable"
    )


# Test 8: Retrieve queued task

@patch("api.main.redis_client")
def test_get_queued_result(mock_redis):
    task_id = "test-task-123"

    mock_redis.get.return_value = json.dumps({
        "task_id": task_id,
        "status": "queued",
    })

    response = client.get(f"/result/{task_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "queued"


# Test 9: Retrieve completed task

@patch("api.main.redis_client")
def test_get_completed_result(mock_redis):
    task_id = "test-task-456"

    mock_redis.get.return_value = json.dumps({
        "task_id": task_id,
        "status": "completed",
        "prediction": "good_quality",
        "class_id": 1,
        "confidence": 0.75,
        "error": None,
    })

    response = client.get(f"/result/{task_id}")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "completed"
    assert data["prediction"] == "good_quality"
    assert data["class_id"] == 1
    assert data["confidence"] == 0.75


# Test 10: Nonexistent task

@patch("api.main.redis_client")
def test_get_nonexistent_task(mock_redis):
    mock_redis.get.return_value = None

    response = client.get("/result/nonexistent-task-123")

    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"


# Test 11: Redis unavailable during result retrieval

@patch("api.main.redis_client")
def test_get_result_redis_unavailable(mock_redis):
    mock_redis.get.side_effect = redis.RedisError(
        "Connection failed"
    )

    response = client.get("/result/test-task")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Result storage is unavailable"
    )