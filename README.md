# Backend-FastAPI Project Description

## Project Description

This project is a backend for the **Earthquake Prediction Machine Learning** project.

## Project Setup

To set up the project, you need to install the required dependencies. You can do this by running the following command in your terminal:

## Virtual Environment
- Create Virtual ENV (Python 3.10)
  
    [Click here for installation guide](https://fastapi.tiangolo.com/virtual-environments/#create-a-virtual-environment)

## Build FastAPI
- Install FastAPI
    ```bash
    pip install "fastapi[standard]"
    ```
     [Click here for installation guide](https://fastapi.tiangolo.com/#installation)

## Prerequisites

- Python 3.10
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
