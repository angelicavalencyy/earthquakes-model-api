"""Alembic env configuration.

This module intentionally uses dynamic Alembic runtime features which static
linters (pylint/mypy) may not fully understand. Disable `no-member` warnings
here as they are false positives for the Alembic API.
"""

# pylint: disable=E1101,no-member

import asyncio
from logging.config import fileConfig
import os
import sys

# Ensure project package can be imported when Alembic runs from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context
from typing import Any, cast

# Cast Alembic objects to `Any` so linters (pylint/mypy) don't raise
# false-positive 'no-member' errors for dynamically provided API.
context = cast(Any, context)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
fileConfig(config.config_file_name)

from sqlmodel import SQLModel
# Import models so that SQLModel.metadata is populated without importing app settings
import app.db.models  # noqa: F401


# If using an async URL (postgresql+asyncpg), Alembic needs a sync URL for
# migration operations. Convert by removing the async driver suffix.
def _get_sync_url(async_url: str) -> str:
    return async_url.replace("+asyncpg", "")


target_metadata = SQLModel.metadata


def run_migrations_offline():
    url = _get_sync_url(os.getenv("SQLALCHEMY_URL") or config.get_main_option("sqlalchemy.url"))
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    # Prefer an env var override for runtime operations
    url = _get_sync_url(os.getenv("SQLALCHEMY_URL") or config.get_main_option("sqlalchemy.url"))
    connectable = engine_from_config(
        {
            "url": url
        },
        prefix="",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        do_run_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
