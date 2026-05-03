from typing import Union
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.main import init_db
from app.routes import mapping_risk, realtime

# Define lifespan event handlers 
@asynccontextmanager
async def lifespan(_: FastAPI):
    print("server is starting")
    await init_db()
    yield
    print("server is shutting down")

app = FastAPI(
    title="Machine Learning Earthquake Prediction API",
    version="1.0.0",
    description="A FastAPI service for machine learning earthquake prediction models.",
    lifespan=lifespan,
)

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


#  daftar route
app.include_router(realtime.router, prefix="/api", tags=["Realtime Prediction",])
app.include_router(mapping_risk.router, prefix="/api", tags=["Region Risk",])
