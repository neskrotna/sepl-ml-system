import time

import httpx
import pytest

BASE_URL = "http://127.0.0.1:8000"

VALID_WINE = {
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


@pytest.mark.integration
def test_complete_prediction_pipeline():
    """Test API -> Redis -> Worker -> Result."""

    with httpx.Client(
        base_url=BASE_URL,
        timeout=10.0,
    ) as client:

        # Step 1: Verify service health
        health = client.get("/health")

        assert health.status_code == 200
        assert health.json()["redis"] == "connected"

        # Step 2: Submit prediction
        response = client.post(
            "/predict",
            json=VALID_WINE,
        )

        assert response.status_code == 202

        task = response.json()
        task_id = task["task_id"]

        assert task["status"] == "queued"

        # Step 3: Poll until worker finishes
        deadline = time.monotonic() + 15

        while time.monotonic() < deadline:
            result_response = client.get(
                f"/result/{task_id}"
            )

            assert result_response.status_code == 200

            result = result_response.json()

            if result["status"] == "completed":
                break

            if result["status"] == "failed":
                pytest.fail(
                    f"Worker failed: {result.get('error')}"
                )

            time.sleep(0.25)
        else:
            pytest.fail(
                "Prediction was not completed within 15 seconds"
            )

        # Step 4: Validate completed result
        assert result["task_id"] == task_id
        assert result["prediction"] == "good_quality"
        assert result["class_id"] == 1
        assert result["confidence"] == pytest.approx(
            0.75,
            abs=0.0001,
        )

@pytest.mark.integration
def test_unknown_task_returns_404():
    """Verify unknown task handling through real API."""

    response = httpx.get(
        f"{BASE_URL}/result/nonexistent-integration-task",
        timeout=10.0,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"