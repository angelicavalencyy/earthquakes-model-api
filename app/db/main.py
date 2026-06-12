from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, text
import os
import logging
from typing import Optional

from app.core.config import settings
from app.db.models import EarthquakeRaw

logger = logging.getLogger(__name__)

# Lazily create the async engine to avoid constructing it at import time
# (which would require `POSTGRES_URL` to be present during imports such
# as when Alembic loads `app.db.models`). Use `get_async_engine()` to
# obtain the engine at runtime.
_async_engine = None
_db_available = False


def get_async_engine():
    """Return a cached AsyncEngine, creating it if necessary.

    Returns None when no POSTGRES_URL is configured. Callers should handle
    a missing engine (e.g., skip DB work or return degraded health).
    """
    global _async_engine, _db_available
    if _async_engine is None:
        if not settings.POSTGRES_URL:
            _db_available = False
            return None
        _async_engine = create_async_engine(settings.POSTGRES_URL, echo=True)
        _db_available = True
    return _async_engine


def is_db_available() -> bool:
    """Return whether a usable DB engine has been created."""
    return bool(_db_available)

async def ensure_realtime_updated_at(conn):
    await conn.execute(
        text(
            "ALTER TABLE realtime_predictions "
            "ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ"
        )
    )
    await conn.execute(
        text(
            "UPDATE realtime_predictions "
            "SET updated_at = COALESCE(updated_at, created_at) "
            "WHERE updated_at IS NULL"
        )
    )
    await conn.execute(
        text(
            "ALTER TABLE realtime_predictions "
            "ALTER COLUMN updated_at SET DEFAULT now()"
        )
    )

async def init_db():
    # Distinguish dev vs production: in development we may skip DB init when
    # POSTGRES_URL is not provided. In production (FASTAPI_DEV unset/false) we
    # require a configured DB and should fail fast if missing.
    is_dev = os.getenv("FASTAPI_DEV", "0").lower() in ("1", "true", "yes")

    if not settings.POSTGRES_URL:
        if is_dev:
            logger.info("POSTGRES_URL not set and FASTAPI_DEV=true; skipping database initialization.")
            return
        # In production, log an error but do not raise to allow the process to
        # start (useful for orchestrators); health checks will indicate DB
        # is unavailable.
        logger.error("POSTGRES_URL is required in production. Continuing startup with DB disabled.")
        return

    engine = get_async_engine()
    if engine is None:
        logger.warning("Database engine could not be created; skipping initialization.")
        return

    async with engine.begin() as conn:
        # By default do NOT auto-create DB schema on app startup in production.
        # Use a proper migration workflow (Alembic) instead. To enable the
        # legacy create_all behavior explicitly set `ALLOW_SCHEMA_AUTOCREATE=1`.
        if os.getenv("ALLOW_SCHEMA_AUTOCREATE", "0") in ("1", "true", "yes"):
            await conn.run_sync(SQLModel.metadata.create_all)
        else:
            logger.debug(
                "Schema auto-creation disabled; set ALLOW_SCHEMA_AUTOCREATE=1 to enable."
            )
        
