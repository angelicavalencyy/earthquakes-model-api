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

def get_async_engine():
    """Return a cached AsyncEngine, creating it if necessary.

    Raises:
        RuntimeError: if `POSTGRES_URL` is not configured.
    """
    global _async_engine
    if _async_engine is None:
        if not settings.POSTGRES_URL:
            raise RuntimeError(
                "POSTGRES_URL is not set. Set the environment variable or provide a config file."
            )
        _async_engine = create_async_engine(settings.POSTGRES_URL, echo=True)
    return _async_engine

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
    # If no DB URL configured, skip initialization to avoid crashing
    # application startup in environments where a DB is optional.
    if not settings.POSTGRES_URL:
        logger.info(
            "POSTGRES_URL not set; skipping database initialization. Set POSTGRES_URL to enable DB features."
        )
        return

    try:
        engine = get_async_engine()
    except RuntimeError as err:
        # Engine couldn't be created (e.g., invalid config); log and skip
        logger.warning("Database engine unavailable, skipping init: %s", err)
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
        
