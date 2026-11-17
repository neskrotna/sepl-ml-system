# White Wine Quality Prediction

A containerized machine learning application for predicting whether a white wine belongs to the **good quality** or **lower quality** category based on 11 physicochemical measurements.

**Course:** Software Engineering Practices in ML (SEPL)  
**Author:** Gabriela Rehakova BSc  
**Repository:** https://github.com/neskrotna/sepl-ml-system

## Overview

The system uses a trained **Random Forest** classifier and an asynchronous prediction workflow:

```text
Client / Swagger UI
       |
       v
FastAPI REST API  -- POST /predict -->  Redis job queue
                                             |
                                             v
                                      Background worker
                                             |
                                             v
                                      Random Forest model
                                             |
                                             v
                                      Redis result store
                                             |
                                             v
Client <-- GET /result/{task_id} -- FastAPI REST API
```

The API validates requests with Pydantic, places prediction jobs in Redis, and returns a task ID. A background worker performs inference and stores the result in Redis. Clients retrieve the result using the task ID.

## Dataset

UCI Wine Quality Dataset (White Wine)
- 4,898 observations
- 11 input features
- Binary classification:
  - Good quality: score >= 6
  - Lower quality: score < 6

## Technologies

- Python 3.12
- scikit-learn
- FastAPI
- Redis
- Docker and Docker Compose
- pytest and Ruff
- GitHub Actions
- Slurm

### Key results

| Item | Result |
| --- | --- |
| Dataset | UCI White Wine Quality |
| Algorithm | Random Forest classifier |
| Training records | 3,168 |
| Test records | 793 |
| Test accuracy | 75.16% |
| Automated tests | 23 passed locally |
| Static analysis | Ruff checks passed |
| Services | FastAPI, Redis, worker |

## Prerequisites

For the recommended Docker setup:

- **Git**
- **Docker Desktop** with Docker Compose v2
- Docker Desktop running before starting the application

For local training and testing:

- **Python 3.12**
- `pip`

## Quick start: Run the complete system with Docker

### 1. Clone the repository

```powershell
git clone https://github.com/neskrotna/sepl-ml-system.git
cd sepl-ml-system
```

### 2. Start the services

Make sure Docker Desktop is running, then execute:

```powershell
docker compose up -d --build
```

Docker Compose builds and starts the **API**, **Redis**, and **worker** services. The repository includes the trained model at `ml_model/model.pkl`, so retraining is **not required** to use the API.

### 3. Check the containers

```powershell
docker compose ps
```

All three services should be running. To inspect logs:

```powershell
docker compose logs api
docker compose logs worker
docker compose logs redis
```

### 4. Open the API

- Interactive Swagger UI: http://localhost:8000/docs
- API information: http://localhost:8000/
- Health check: http://localhost:8000/health

The health endpoint checks connectivity to Redis.

## Make a prediction

### Step 1: Submit a prediction

Open **http://localhost:8000/docs**, expand **POST `/predict`**, select **Try it out**, and submit the following example JSON:

```json
{
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
  "alcohol": 8.8
}
```

The API responds with **HTTP 202 Accepted** and a generated task ID. The initial task status is `queued`.

> **Note:** The JSON API uses underscore-separated names, such as `fixed_acidity`. The underlying dataset uses column names containing spaces, such as `fixed acidity`.

### Step 2: Retrieve the result

In Swagger UI, expand **GET `/result/{task_id}`**, enter the task ID returned by `/predict`, and execute the request.

- If the worker has not finished, the task remains `queued`.
- Once processing finishes, the response includes the predicted quality label, numeric class ID, and confidence score.

Possible prediction labels are `good_quality` (class `1`) and `lower_quality` (class `0`). The classification threshold used during training is **quality >= 6** for `good_quality`.

### API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Application information |
| GET | `/health` | Check Redis connectivity |
| POST | `/predict` | Validate input and enqueue a prediction (HTTP 202) |
| GET | `/result/{task_id}` | Retrieve queued or completed task status |

Invalid prediction inputs return HTTP **422**, unknown task IDs return HTTP **404**, and Redis connection failures are handled with HTTP **503**.

## Stop the application

```powershell
docker compose down
```

To rebuild after modifying application code:

```powershell
docker compose up -d --build
```

## Local Python setup

The Docker workflow above is sufficient for inference. Use a local Python environment to retrain the model, run tests, or perform linting.

From the repository root, in **Windows PowerShell**:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If `py -3.12` is unavailable but Python 3.12 is installed, use `python -m venv .venv` instead. On macOS/Linux, activate with `source .venv/bin/activate`.

### Retrain the model

The dataset is included at `data/winequality-white.csv`.

```powershell
python ml_model/train.py
```

The training script removes duplicate rows, converts the original wine quality score into a binary target, trains and evaluates the Random Forest classifier, and saves the trained model to `ml_model/model.pkl`.

To test inference without the API:

```powershell
python ml_model/test_local_prediction.py
```

## Run automated tests

Install dependencies and activate the virtual environment first.

### Model unit tests (10)

```powershell
python -m pytest tests/test_model.py -v
```

### API tests (11)

```powershell
python -m pytest tests/test_api.py -v
```

These use a FastAPI test client and mocked Redis interactions; Docker is not required.

### Integration tests (2)

Integration tests communicate with the **running Docker application**. Start the services before running them:

```powershell
docker compose up -d --build
python -m pytest tests/test_integration.py -v
```

### Full test suite (23)

With Docker services running:

```powershell
python -m pytest tests/ -v
```

### Unit and API tests only (21)

This option excludes integration tests and does not require Docker:

```powershell
python -m pytest tests/ -v -m "not integration"
```

**Locally verified result:** 23/23 tests passed. A Starlette dependency deprecation warning was observed but did not fail the tests.

## Code quality checks

Run Ruff from the repository root:

```powershell
python -m ruff check api/ ml_model/ worker/ tests/
```

The last local check returned `All checks passed!`.

## GitHub Actions CI

The workflow is defined in `.github/workflows/ci.yml` and is configured for pushes to `main`, pull requests targeting `main`, and manual execution.

It defines two jobs:

1. **Lint and Unit Tests:** Install dependencies, run Ruff, and execute the 21 non-integration tests.
2. **Docker Integration Tests:** Start the Docker Compose services and execute the two integration tests.

**Current status (17 November 2026):** GitHub recognized the workflow, but its hosted jobs did not start because GitHub reported an account-level billing lock. Therefore, **hosted CI execution has not yet been verified**. All 23 tests and Ruff checks passed locally.

## Project structure

```text
sepl-ml-system/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions workflow
├── api/
│   ├── main.py                 # FastAPI endpoints and Redis interaction
│   └── schemas.py              # Pydantic request/response models
├── data/
│   └── winequality-white.csv   # Training dataset
├── docs/
│   ├── report/                 # LaTeX technical report
│   └── screenshots/            # Implementation and testing evidence
├── ml_model/
│   ├── explore_data.py         # Dataset exploration
│   ├── model.py                # Preprocessing and prediction functions
│   ├── model.pkl               # Trained model artifact
│   ├── test_local_prediction.py
│   └── train.py                # Model training and evaluation
├── tests/
│   ├── test_api.py
│   ├── test_integration.py
│   └── test_model.py
├── worker/
│   └── worker.py               # Redis background consumer
├── .dockerignore
├── .gitignore
├── Dockerfile
├── compose.yaml
├── pytest.ini
├── requirements.txt
└── README.md
```

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `docker compose` fails | Start Docker Desktop and verify `docker compose version`. |
| API not reachable | Check `docker compose ps` and `docker compose logs api`; confirm port 8000 is free. |
| Prediction remains queued | Check `docker compose logs worker` and `docker compose logs redis`. |
| Redis health check fails | Check that the Redis container is running using `docker compose ps`. |
| Integration tests fail | Start the complete stack and wait until the API is available at `http://localhost:8000/health`. |
| `ModuleNotFoundError` | Activate `.venv` and reinstall dependencies using `python -m pip install -r requirements.txt`. |
| GitHub Actions jobs do not start | Check the workflow annotations and GitHub account billing status; local test success does not establish hosted CI success. |

## Dataset and documentation

- [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Redis documentation](https://redis.io/docs/)
- [Docker documentation](https://docs.docker.com/)
- [pytest documentation](https://docs.pytest.org/)
- [Ruff documentation](https://docs.astral.sh/ruff/)

**Dataset reference:** Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547–553.
