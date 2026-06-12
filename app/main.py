from typing import Union
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.main import init_db, is_db_available
from app.routes import mapping_risk, realtime

import asyncio
import logging
import sys
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Konfigurasi Scheduler ────────────────────────────────────────────
RETRAIN_INTERVAL_DAYS = 7  # Retrain setiap 7 hari
RETRAIN_INTERVAL_SECONDS = RETRAIN_INTERVAL_DAYS * 24 * 60 * 60

async def schedule_model_training():
    """Run realtime model retraining every 7 days in the background (non-blocking)."""
    script_path = Path(__file__).resolve().parents[1] / "src" / "train_realtime.py"

    while True:
        try:
            logger.info(
                "[Scheduler] Next retrain in %d days. Sleeping...",
                RETRAIN_INTERVAL_DAYS,
            )
            await asyncio.sleep(RETRAIN_INTERVAL_SECONDS)

            logger.info(
                "[Scheduler] Starting automated %d-day K-Medoids realtime model retraining...",
                RETRAIN_INTERVAL_DAYS,
            )

            # Non-blocking subprocess — tidak membekukan event loop / API server
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info("[Scheduler] Model retraining completed successfully.")
                if stdout:
                    logger.debug("[Scheduler] stdout: %s", stdout.decode()[-500:])
            else:
                logger.error(
                    "[Scheduler] Retraining failed (exit code %d): %s",
                    process.returncode,
                    stderr.decode()[-1000:] if stderr else "no stderr",
                )

        except asyncio.CancelledError:
            logger.info("[Scheduler] Training scheduler cancelled.")
            break
        except Exception as e:
            logger.exception("[Scheduler] Unexpected error during model retraining: %s", e)
            # Tetap lanjut loop, jangan crash scheduler
            await asyncio.sleep(60)  # Tunggu 1 menit sebelum retry loop


# Define lifespan event handlers
@asynccontextmanager
async def lifespan(_: FastAPI):
    print("server is starting")
    await init_db()

    # Start the 7-day training loop only when explicitly enabled to avoid
    # duplicated retraining across multiple workers/replicas. Use the
    # RUN_RETRAINER=1 environment variable to enable the in-process scheduler
    # (recommended: run retraining as a separate cron/job instead).
    retrain_enabled = os.getenv("RUN_RETRAINER", "0").lower() in ("1", "true", "yes")
    task = None
    if retrain_enabled:
        task = asyncio.create_task(schedule_model_training())

    yield
    print("server is shutting down")
    if task:
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


@app.get("/health")
async def health():
    """Simple health endpoint reporting DB availability."""
    return {
        "status": "ok",
        "db": is_db_available()
    }


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


#  daftar route
app.include_router(realtime.router, prefix="/api", tags=["Realtime Prediction",])
app.include_router(mapping_risk.router, prefix="/api", tags=["Region Risk",])
