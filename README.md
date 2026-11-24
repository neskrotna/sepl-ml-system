
# White Wine Quality Prediction

A containerized machine learning application for predicting whether a white wine belongs to the **good quality** or **lower quality** category based on 11 physicochemical measurements.

**Course:** Software Engineering Practices in ML 
**Author:** Gabriela Rehakova BSc  
**Programme:** Master's in AI & Data Science  
**University:** FH Technikum Wien  
**Repository:** https://github.com/neskrotna/sepl-ml-system

---

## Overview

The system uses a trained **Random Forest** classifier and an asynchronous prediction workflow:

```text
Client / Swagger UI
       |
       v
FastAPI REST API -- POST /predict --> Redis job queue
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

The application demonstrates software engineering practices for machine learning, including:

- ML model development and serialization
- REST API development
- Input validation and error handling
- Asynchronous job processing
- Redis-based communication
- Docker containerization
- Environment-based configuration
- Model artifact management through bind mounts
- Automated testing
- Static code analysis
- Continuous integration
- Application logging and health monitoring

---

## Dataset

The project uses the UCI Wine Quality Dataset (White Wine).

- Original observations: 4,898
- Duplicate observations removed: 937
- Observations after preprocessing: 3,961
- Input features: 11
- Training observations: 3,168
- Test observations: 793

The original wine quality score is converted into a binary classification target:

- Good quality: score >= 6
- Lower quality: score < 6

### Input features

| Feature | Description |
| --- | --- |
| Fixed acidity | Non-volatile acidity |
| Volatile acidity | Volatile acid concentration |
| Citric acid | Citric acid concentration |
| Residual sugar | Remaining sugar after fermentation |
| Chlorides | Salt concentration |
| Free sulfur dioxide | Free SO2 concentration |
| Total sulfur dioxide | Total SO2 concentration |
| Density | Wine density |
| pH | Acidity level |
| Sulphates | Sulphate concentration |
| Alcohol | Alcohol content |

---

## Technologies

| Technology | Purpose |
| --- | --- |
| Python | Application and ML development |
| scikit-learn | Model training and inference |
| NumPy | Numerical operations |
| pandas | Dataset processing |
| Joblib | Model serialization |
| FastAPI | REST API |
| Pydantic | Input and output validation |
| Redis | Job queue and result storage |
| Docker | Application containerization |
| Docker Compose | Multi-container orchestration |
| pytest | Automated testing |
| Ruff | Static code analysis |
| Git | Version control |
| GitHub Actions | Continuous integration |
| Python logging | Application logging and monitoring |

The Docker and GitHub Actions environments use Python 3.12. Local testing has also been successfully performed with Python 3.13.14.

---

## Key Results

| Item | Result |
| --- | --- |
| Dataset | UCI White Wine Quality |
| Algorithm | Random Forest classifier |
| Original dataset size | 4,898 |
| Dataset size after preprocessing | 3,961 |
| Training records | 3,168 |
| Test records | 793 |
| Test accuracy | 75.16% |
| Model unit tests | 10/10 passed |
| API tests | 11/11 passed |
| Integration tests | 2/2 passed |
| Total automated tests | 23/23 passed |
| Static analysis | All Ruff checks passed |
| Docker services | FastAPI, Redis, worker |
| Model bind mounts | Verified |
| Environment configuration | Verified |
| Application logging | Verified |

---

## Prerequisites

For the recommended Docker setup:

- Git
- Docker Desktop with Docker Compose v2
- Docker Desktop running before starting the application

For local training and testing:

- Python 3.12 recommended
- pip

Python 3.13.14 has also been successfully tested locally.

Verify the installed tools:

```powershell
git --version
docker --version
docker compose version
python --version
```

---

## Quick Start: Run the Complete System with Docker

### 1. Clone the Repository

```powershell
git clone https://github.com/neskrotna/sepl-ml-system.git
cd sepl-ml-system
```

### 2. Configure Environment Variables

Before starting the application, create a local `.env` file from the provided configuration template.

Windows (PowerShell):

```powershell
Copy-Item .env.example .env
```

Linux / macOS:

```bash
cp .env.example .env
```

The `.env` file contains the Redis connection settings:

```dotenv
REDIS_HOST=redis
REDIS_PORT=6379
```

| Variable | Default value | Purpose |
| --- | --- | --- |
| REDIS_HOST | redis | Redis service hostname |
| REDIS_PORT | 6379 | Redis connection port |

Docker Compose loads these environment variables into both the API and worker containers through the `env_file` configuration.

The actual `.env` file is excluded from version control. The `.env.example` template is included in the repository to support reproducibility.

### 3. Validate the Docker Compose Configuration

```powershell
docker compose config
```

This command validates and displays the resolved Docker Compose configuration, including environment variables, networking, volumes and service definitions.

### 4. Start the Services

Make sure Docker Desktop is running, then execute:

```powershell
docker compose up -d --build
```

Docker Compose builds and starts the following services:

| Service | Responsibility |
| --- | --- |
| api | FastAPI REST API |
| worker | Background ML inference |
| redis | Job queue and result storage |

The repository includes the trained model at `ml_model/model.pkl`, so retraining is not required to use the API.

### 5. Check the Containers

```powershell
docker compose ps
```

All three services should be running. The Redis container should be marked as healthy.

To inspect the logs:

```powershell
docker compose logs api
docker compose logs worker
docker compose logs redis
```

### 6. Open the API

- Interactive Swagger UI: http://localhost:8000/docs
- API information: http://localhost:8000/
- Health check: http://localhost:8000/health

The health endpoint checks connectivity to Redis.

---

## Docker Architecture

The application consists of three independently running Docker services.

```text
                 Host Machine
                      |
               localhost:8000
                      |
                      v
              +---------------+
              |  FastAPI API  |
              +---------------+
                      |
                      v
              +---------------+
              |     Redis     |
              +---------------+
                      |
                      v
              +---------------+
              |    Worker     |
              +---------------+
                      |
                      v
              Random Forest Model
```

### Service-to-Service Networking

Docker Compose creates a shared default network for the application services.

The API and worker communicate with Redis using its service name:

```text
redis:6379
```

The Redis service does not require a publicly exposed host port.

The API is accessible locally through:

```text
127.0.0.1:8000
```

### Model Bind Mount

The trained Random Forest model is mounted into both the API and worker containers using a read-only bind mount.

```yaml
volumes:
  - ./ml_model/model.pkl:/app/ml_model/model.pkl:ro
```

The bind mount connects the model artifact on the host filesystem to the corresponding file location inside the containers.

Benefits include:

- Runtime access to the trained model artifact
- Consistent model access across containers
- Protection against container-side modification
- Model replacement without rebuilding the Docker image
- Separation of runtime model management from application execution

After replacing the model artifact, restart the relevant services to ensure the new model is loaded.

The Dockerfile also includes the model directory during image construction. The runtime bind mount overrides the corresponding model file inside the container.

### Verify Model Bind Mounts

Windows (PowerShell):

```powershell
docker inspect (docker compose ps -q api) --format '{{json .Mounts}}'

docker inspect (docker compose ps -q worker) --format '{{json .Mounts}}'
```

The expected mount properties include:

```json
{
  "Type": "bind",
  "Destination": "/app/ml_model/model.pkl",
  "Mode": "ro",
  "RW": false
}
```

Both container mounts were successfully verified during local testing.

### Verify Environment Variables

```powershell
docker compose exec api printenv REDIS_HOST REDIS_PORT

docker compose exec worker printenv REDIS_HOST REDIS_PORT
```

Expected output:

```text
redis
6379
```

The environment variables were successfully verified inside both running containers.

### Redis Persistence

Redis uses a named Docker volume:

```yaml
volumes:
  - redis_data:/data
```

Append-only persistence is enabled through:

```yaml
command: ["redis-server", "--appendonly", "yes"]
```

This preserves Redis data across normal container restarts and recreation.

### Redis Health Check

The Redis service includes a health check:

```yaml
healthcheck:
  test: ["CMD", "redis-cli", "ping"]
  interval: 5s
  timeout: 3s
  retries: 10
```

The API and worker depend on Redis becoming healthy before starting.

---

## Make a Prediction

### Step 1: Submit a Prediction

Open:

http://localhost:8000/docs

Expand **POST `/predict`**, select **Try it out**, and submit the following example JSON:

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

The API responds with **HTTP 202 Accepted** and a generated task ID.

The initial task status is `queued`.

> Note: The JSON API uses underscore-separated feature names, such as `fixed_acidity`. The underlying dataset uses column names containing spaces, such as `fixed acidity`.

### Step 2: Retrieve the Result

In Swagger UI, expand **GET `/result/{task_id}`**, enter the task ID returned by `/predict`, and execute the request.

- If the worker has not finished, the task remains `queued`.
- Once processing finishes, the response includes the predicted quality label, numeric class ID and confidence score.

Example prediction values:

```json
{
  "prediction": "good_quality",
  "class_id": 1,
  "confidence": 0.75
}
```

The exact prediction and confidence depend on the submitted feature values.

### Prediction Classes

| Class ID | Label | Quality threshold |
| --- | --- | --- |
| 0 | lower_quality | Quality score < 6 |
| 1 | good_quality | Quality score >= 6 |

### API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Application information |
| GET | `/health` | Check Redis connectivity |
| POST | `/predict` | Validate input and enqueue a prediction |
| GET | `/result/{task_id}` | Retrieve queued or completed task status |

### HTTP Status Codes

| Status code | Meaning |
| --- | --- |
| 200 OK | Successful request |
| 202 Accepted | Prediction task accepted |
| 404 Not Found | Unknown task ID |
| 422 Unprocessable Entity | Invalid prediction input |
| 503 Service Unavailable | Redis connection unavailable |

---

## Asynchronous Prediction Workflow

The prediction pipeline follows these steps:

1. The client submits an HTTP POST request.
2. FastAPI validates the input using Pydantic.
3. The API generates a unique task ID.
4. The prediction job is added to the Redis queue.
5. The background worker receives the job.
6. The worker preprocesses the input features.
7. The trained Random Forest model performs inference.
8. The prediction result is stored in Redis.
9. The client retrieves the result using the task ID.

The asynchronous architecture separates request handling from model inference.

This allows the API to accept prediction requests without waiting for inference to finish within the original HTTP request.

---

## Logging and Monitoring

The API and worker use Python's built-in `logging` module to record important application events.

### Log Format

Custom application logs include:

- Timestamp
- Logging level
- Log message

Example:

```text
2026-11-24 10:14:31,793 | INFO | Starting processing for task ...
```

### API Logging

The API records important events, including:

- Application startup
- Redis connection establishment
- Prediction task submission
- Prediction result retrieval
- Application errors

View API logs:

```powershell
docker compose logs api --tail 30
```

### Worker Logging

The worker records important processing events, including:

- Worker startup
- Successful model loading
- Redis connection establishment
- Waiting for prediction jobs
- Task reception
- Processing start
- Completed predictions
- Prediction confidence
- Inference execution time

View worker logs:

```powershell
docker compose logs worker --tail 30
```

Example of a completed prediction:

```text
INFO | Completed task ... | prediction=good_quality |
confidence=0.7500 | inference_time=0.3005s
```

### Task Traceability

The same unique task ID appears in both the API and worker logs.

This supports tracking individual prediction requests across the asynchronous processing pipeline.

### Live Monitoring

To follow application logs continuously:

```powershell
docker compose logs -f api worker
```

Press `Ctrl+C` to stop following the logs without stopping the containers.

### Health Monitoring

The application exposes the following health endpoint:

```http
GET /health
```

Access it through:

http://localhost:8000/health

The endpoint checks Redis connectivity and reports the application's health status.

Redis availability is also monitored through the Docker Compose health check.

### Observed Inference Performance

During local verification on 24 November 2026, one prediction produced:

| Metric | Observed value |
| --- | --- |
| Prediction | good_quality |
| Confidence | 0.75 |
| Inference time | 0.3005 seconds |

This measurement demonstrates successful inference-time logging. It represents a single observed prediction and is not a comprehensive performance benchmark.

---

## Stop the Application

```powershell
docker compose down
```

The named Redis volume remains available after normal shutdown.

To rebuild after modifying application code:

```powershell
docker compose up -d --build
```

To restart the worker after replacing the trained model:

```powershell
docker compose restart worker
```

---

## Local Python Setup

The Docker workflow is sufficient for inference.

Use a local Python environment to retrain the model, run automated tests or perform linting.

### Windows PowerShell

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python 3.12 and the Windows Python Launcher are available, create the environment using:

```powershell
py -3.12 -m venv .venv
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Verify the Python Environment

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

The virtual environment is excluded from version control.

When transferring the project to another computer, recreate the virtual environment instead of copying the existing `.venv` directory.

---

## Model Training

The dataset is included at:

```text
data/winequality-white.csv
```

### Retrain the Model

```powershell
python ml_model/train.py
```

The training script:

1. Loads the white wine dataset.
2. Removes duplicate observations.
3. Converts the original quality score into a binary target.
4. Splits the dataset into training and test sets.
5. Trains the Random Forest classifier.
6. Evaluates model performance.
7. Saves the trained model artifact.

The generated model is stored at:

```text
ml_model/model.pkl
```

### Local Prediction

To test inference without the REST API:

```powershell
python ml_model/test_local_prediction.py
```

The model module is responsible for loading the serialized model, preprocessing input data and performing predictions.

---

## Run Automated Tests

Install the project dependencies and activate the virtual environment before running the tests.

The project contains 23 automated tests.

### Model Unit Tests (10)

```powershell
python -m pytest tests/test_model.py -v
```

The model tests verify:

- Successful model loading
- Valid input preprocessing
- Missing input features
- Unexpected input features
- Non-numeric input values
- NaN input values
- Infinite input values
- Prediction response structure
- Prediction confidence
- Known wine prediction

### API Tests (11)

```powershell
python -m pytest tests/test_api.py -v
```

The API tests verify:

- Root endpoint
- Successful health check
- Redis unavailability during health checks
- Valid prediction requests
- Missing prediction features
- Invalid feature types
- Redis unavailability during prediction
- Queued prediction results
- Completed prediction results
- Unknown task IDs
- Redis unavailability during result retrieval

These tests use a FastAPI test client and mocked Redis interactions. Running Docker containers is not required.

### Integration Tests (2)

Integration tests communicate with the running Docker application.

Create the environment configuration if it does not already exist, then start the services:

```powershell
Copy-Item .env.example .env
docker compose up -d --build
```

Wait until the API is available at:

http://localhost:8000/health

Run the integration tests:

```powershell
python -m pytest tests/test_integration.py -v
```

The integration tests verify:

- Complete end-to-end prediction pipeline
- HTTP 404 response for an unknown task ID

### Full Test Suite (23)

With Docker services running:

```powershell
python -m pytest tests/ -v
```

To enable diagnostic output for tests taking longer than 30 seconds:

```powershell
python -m pytest tests/ -v -o faulthandler_timeout=30
```

### Unit and API Tests Only (21)

This option excludes integration tests and does not require Docker:

```powershell
python -m pytest tests/ -v -m "not integration"
```

### Verified Test Results

| Test category | Passed | Total |
| --- | --- | --- |
| Model unit tests | 10 | 10 |
| API tests | 11 | 11 |
| Integration tests | 2 | 2 |
| Total | 23 | 23 |

All 23 automated tests passed locally following the introduction of environment-based configuration and read-only model bind mounts.

The updated configuration was successfully tested on Windows using Python 3.13.14, while the Docker application continued to use Python 3.12.

---

## Code Quality Checks

The project uses Ruff for static code analysis.

Run Ruff from the repository root:

```powershell
python -m ruff check api/ ml_model/ worker/ tests/
```

Expected output:

```text
All checks passed!
```

To automatically correct supported linting issues:

```powershell
python -m ruff check api/ ml_model/ worker/ tests/ --fix
```

All Ruff checks passed following the latest local configuration changes.

---

## GitHub Actions CI

The continuous integration workflow is defined in:

```text
.github/workflows/ci.yml
```

The workflow is triggered by:

- Pushes to the `main` branch
- Pull requests targeting the `main` branch
- Manual workflow execution

### Job 1: Lint and Unit Tests

The first job performs the following steps:

1. Check out the repository.
2. Set up Python 3.12.
3. Install project dependencies.
4. Run Ruff linting.
5. Execute the 21 non-integration tests.

### Job 2: Docker Integration Tests

The second job performs the following steps:

1. Check out the repository.
2. Create `.env` from `.env.example`.
3. Set up Python 3.12.
4. Install test dependencies.
5. Build and start the Docker Compose services.
6. Wait for API readiness.
7. Execute the two integration tests.
8. Display Docker logs if a failure occurs.
9. Stop the Docker services.

### CI Execution Flow

```text
Push / Pull Request / Manual Trigger
                  |
                  v
          GitHub Actions CI
                  |
        +---------+---------+
        |                   |
        v                   v
 Lint and Unit Tests   Docker Integration Tests
        |                   |
        v                   v
      Ruff             Create .env
        |                   |
        v                   v
    21 Tests          Build Containers
        |                   |
        |                   v
        |              Start Services
        |                   |
        |                   v
        |              Health Check
        |                   |
        |                   v
        |                2 Tests
        |                   |
        +---------+---------+
                  |
                  v
              CI Result
```

### Environment Configuration in CI

The actual `.env` file is excluded from Git.

The Docker integration job therefore creates its own environment configuration using:

```yaml
- name: Create environment configuration
  run: cp .env.example .env
```

This ensures that the Docker services receive the required Redis connection settings on GitHub-hosted runners.

### CI Status

**Status as of 24 November 2026:**

The previous GitHub account billing restriction has been resolved. Both CI jobs successfully executed on GitHub-hosted runners before the latest container configuration changes.

Previously verified successful workflow:

https://github.com/neskrotna/sepl-ml-system/actions/runs/45367264475

Following the introduction of read-only model bind mounts and environment-based configuration:

- All 23 automated tests passed locally.
- All Ruff checks passed.
- All three Docker services started successfully.
- Model bind mounts were verified.
- Environment variables were verified.
- End-to-end prediction processing was verified.

The CI workflow has been updated to support the new environment configuration. Remote verification of this latest configuration is pending.

GitHub Actions dashboard:

https://github.com/neskrotna/sepl-ml-system/actions

---

## Project Structure

```text
sepl-ml-system/
├── .github/
│   └── workflows/
│       └── ci.yml
├── api/
│   ├── main.py
│   └── schemas.py
├── data/
│   └── winequality-white.csv
├── docs/
│   ├── report/
│   └── screenshots/
├── ml_model/
│   ├── explore_data.py
│   ├── model.py
│   ├── model.pkl
│   ├── test_local_prediction.py
│   └── train.py
├── tests/
│   ├── test_api.py
│   ├── test_integration.py
│   └── test_model.py
├── worker/
│   └── worker.py
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── compose.yaml
├── pytest.ini
├── requirements.txt
└── README.md
```

### Directory Responsibilities

| Directory / File | Responsibility |
| --- | --- |
| api/ | REST API implementation and validation |
| data/ | Training dataset |
| docs/report/ | LaTeX technical report |
| docs/screenshots/ | Implementation and testing evidence |
| ml_model/ | Model training, preprocessing and inference |
| tests/ | Unit, API and integration tests |
| worker/ | Background prediction processing |
| .github/workflows/ | Continuous integration workflow |
| .env.example | Environment configuration template |
| Dockerfile | Docker image definition |
| compose.yaml | Multi-container orchestration |
| requirements.txt | Python dependencies |
| pytest.ini | Test configuration |
| README.md | Project documentation |

---

## Version Control

Git is used to track source code, configuration files, documentation and implementation changes.

The following local files and directories are excluded from version control:

- `.env`
- `.venv/`
- Python cache files
- Temporary development files

The `.env.example` configuration template is included in the repository.

To verify ignored files:

```powershell
git check-ignore .env
git check-ignore .venv/
```

To inspect the current repository status:

```powershell
git status
```

---

## Troubleshooting

| Problem | Recommended solution |
| --- | --- |
| Docker Compose fails | Start Docker Desktop and verify `docker compose version`. |
| `.env` file not found | Create `.env` from `.env.example` before starting Docker Compose. |
| Model file mount fails | Verify that `ml_model/model.pkl` exists and Docker Desktop has access to the project directory. |
| API not reachable | Check `docker compose ps` and `docker compose logs api`. |
| Port 8000 already in use | Stop the conflicting application or container. |
| Prediction remains queued | Inspect `docker compose logs worker` and `docker compose logs redis`. |
| Redis health check fails | Verify the Redis container status and inspect its logs. |
| Integration tests fail | Start the complete Docker stack and verify API readiness. |
| ModuleNotFoundError | Activate `.venv` and reinstall dependencies. |
| Virtual environment references an old computer | Remove and recreate `.venv` using the locally installed Python interpreter. |
| Model loading takes unusually long | Run the model-loading test separately with diagnostic output. |
| GitHub Actions cannot find `.env` | Verify that `.env.example` is committed and the CI environment creation step is present. |
| GitHub Actions jobs do not start | Inspect workflow annotations and GitHub account status. |

### Inspect All Docker Logs

```powershell
docker compose logs
```

### Restart the Complete System

```powershell
docker compose down
docker compose up -d --build
```

### Diagnose Model Loading

```powershell
python -m pytest tests/test_model.py::test_model_loads_successfully -v -s -o faulthandler_timeout=30
```

### Verify Model Dependencies

```powershell
python -c "import numpy; print(numpy.__version__)"
python -c "import scipy; print(scipy.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
python -c "import joblib; print(joblib.__version__)"
```

---

## Documentation

Implementation screenshots and technical documentation are maintained in the `docs/` directory.

The technical report documents:

- Dataset exploration and preprocessing
- Model training and evaluation
- System architecture
- REST API implementation
- Redis queue and worker processing
- Docker containerization
- Environment configuration
- Model bind mounts
- Automated testing
- Static code analysis
- Version control
- Continuous integration
- Application logging and monitoring

The final report will also include the HPC simulation and performance reflection.

---

## References

### Dataset

- [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547–553.

### Technical Documentation

- [Python](https://docs.python.org/3/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Pydantic](https://docs.pydantic.dev/)
- [Redis](https://redis.io/docs/)
- [Docker](https://docs.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [pytest](https://docs.pytest.org/)
- [Ruff](https://docs.astral.sh/ruff/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [Python Logging](https://docs.python.org/3/library/logging.html)