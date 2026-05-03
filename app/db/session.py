"""Async database session management."""
from typing import AsyncGenerator

from sqlmodel.ext.asyncio.session import AsyncSession
from app.db.main import async_engine

# Session factory
async_session = AsyncSession(async_engine, expire_on_commit=False)

# Dependency FastAPI
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async DB session.

    Yields:
        AsyncSession: an async SQLModel/SQLAlchemy session instance.
    """
    async with AsyncSession(async_engine) as session:
        yield session
