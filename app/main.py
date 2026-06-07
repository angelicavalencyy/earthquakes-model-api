from typing import Union
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.main import init_db
from app.routes import mapping_risk, realtime

import asyncio
import subprocess
import sys
from pathlib import Path

async def schedule_model_training():
    """Run model training every 24 hours in the background."""
    while True:
        try:
            # Wait 24 hours before first background run (or adjust if you want it immediately)
            await asyncio.sleep(24 * 60 * 60)
            
            print("[Scheduler] Starting automated 24-hour K-Medoids model retraining...")
            script_path = Path(__file__).resolve().parents[2] / "src" / "train_kmed.py"
            # run training script via subprocess
            subprocess.run([sys.executable, str(script_path)], check=True)
            print("[Scheduler] Model retraining completed successfully.")
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[Scheduler] Error during model retraining: {e}")

# Define lifespan event handlers 
@asynccontextmanager
async def lifespan(_: FastAPI):
    print("server is starting")
    await init_db()
    
    # Start the 24-hour training loop
    task = asyncio.create_task(schedule_model_training())
    
    yield
    print("server is shutting down")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

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
