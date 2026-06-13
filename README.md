# Backend-FastAPI Project Description

## Project Description

This project is a backend for the **Earthquake Prediction Machine Learning** project.

## Project Setup

To set up the project, you need to install the required dependencies. You can do this by running the following command in your terminal:

## Virtual Environment
- Create Virtual ENV (Python 3.11)
  
    [Click here for installation guide](https://fastapi.tiangolo.com/virtual-environments/#create-a-virtual-environment)

## Build FastAPI
- Install FastAPI
    ```bash
    pip install "fastapi[standard]"
    ```
     [Click here for installation guide](https://fastapi.tiangolo.com/#installation)

## Prerequisites

- Python 3.11
- DBMS PostgreSQL
- FastAPI
- Pydantic
- SQLModel
- SQLAlchemy

| Konsep Penelitian   | App Loc                          |
| ------------------- | -------------------------------- |
| Database init       | `core/db/`                       |
| Data gempa          | `models/earthquake.py`           |
| Pra-pemrosesan      | `preprocessing/`                 |
| Algoritma K-Medoids | `algorithms/k_medoids.py`        |
| Perhitungan jarak   | `algorithms/distance.py`         |
| Proses clustering   | `services/clustering_service.py` |
| Hasil cluster       | `models/clustering_result.py`    |
| Visualisasi peta    | `api/endpoints/visualization.py` |

## Deploy checklist

Before deploying to production, ensure:

- **Secrets are set**: `POSTGRES_URL` (for app runtime) and `SQLALCHEMY_URL` (for Alembic migrations) must be provided via environment or CI secrets. Do NOT commit `.env`.
- **Migrations**: run `alembic upgrade head` against your production DB before routing traffic.
- **FASTAPI_DEV** should be unset or false in production.
- **RUN_RETRAINER** should be disabled for app containers; schedule retraining as an external job.
- **ML artifacts**: keep `mlruns/` or configure MLflow to use remote storage (S3/GCS) for production.
- **Health checks**: configure readiness/liveness to call `/health` and require `db: true` for readiness if you want to block traffic until DB ready.
- **Build & smoke test**: CI will build a Docker image and run a quick smoke test (see .github/workflows/ci.yml).

Example migration command in CI (use secret `SQLALCHEMY_URL`):

```bash
export SQLALCHEMY_URL='postgresql://user:pass@host/db'
alembic upgrade head
```

