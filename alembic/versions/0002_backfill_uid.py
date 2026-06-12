"""backfill uid for existing rows

Revision ID: 0002_backfill_uid
Revises: d6f588e2c016
Create Date: 2026-06-13 00:00:00
"""

"""Migration: backfill uid for existing rows."""

# pylint: disable=E1101,no-member

from alembic import op
from typing import Any, cast

# Cast `op` to Any to avoid linter complaints about dynamic Alembic API.
op = cast(Any, op)

revision = '0002_backfill_uid'
down_revision = 'd6f588e2c016'
branch_labels = None
depends_on = None


def upgrade():
    # Ensure pgcrypto (provides gen_random_uuid) is available; if not, create it.
    op.execute('CREATE EXTENSION IF NOT EXISTS pgcrypto')

    # Backfill uid for rows that don't have one yet.
    op.execute(
        """
        UPDATE dibi_earthquakes_raw
        SET uid = gen_random_uuid()
        WHERE uid IS NULL
        """
    )

    # Set a default so new rows get UUIDs automatically.
    op.execute(
        "ALTER TABLE dibi_earthquakes_raw ALTER COLUMN uid SET DEFAULT gen_random_uuid()"
    )


def downgrade():
    # Remove the default but keep filled UUIDs (downgrade won't erase generated ids).
    op.execute(
        "ALTER TABLE dibi_earthquakes_raw ALTER COLUMN uid DROP DEFAULT"
    )