"""Async database session management."""
from typing import AsyncGenerator

from sqlmodel.ext.asyncio.session import AsyncSession
from app.db.main import get_async_engine


# Dependency FastAPI
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async DB session.

    The engine is created lazily by `get_async_engine()` to avoid requiring
    DB environment variables at import time.

    Yields:
        AsyncSession: an async SQLModel/SQLAlchemy session instance.
    """
    engine = get_async_engine()
    async with AsyncSession(engine) as session:
        yield session
