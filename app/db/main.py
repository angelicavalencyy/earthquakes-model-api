from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, text

from app.core.config import settings
from app.db.models import EarthquakeRaw

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
        await conn.run_sync(SQLModel.metadata.create_all)
        await ensure_realtime_updated_at(conn)
        
