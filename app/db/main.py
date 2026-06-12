from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, text
import os
import logging

from app.core.config import settings
from app.db.models import EarthquakeRaw

logger = logging.getLogger(__name__)

async_engine = create_async_engine(
    settings.POSTGRES_URL,
    echo=True,
)

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
    async with async_engine.begin() as conn:
        # By default do NOT auto-create DB schema on app startup in production.
        # Use a proper migration workflow (Alembic) instead. To enable the
        # legacy create_all behavior explicitly set `ALLOW_SCHEMA_AUTOCREATE=1`.
        if os.getenv("ALLOW_SCHEMA_AUTOCREATE", "0") in ("1", "true", "yes"):
            await conn.run_sync(SQLModel.metadata.create_all)
        else:
            logger.debug(
                "Schema auto-creation disabled; set ALLOW_SCHEMA_AUTOCREATE=1 to enable."
            )
        
