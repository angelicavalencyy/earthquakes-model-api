"""make uid NOT NULL and add unique constraint

Revision ID: 0003_uid_not_null_unique
Revises: 0002_backfill_uid
Create Date: 2026-06-13 00:00:00
"""

"""Migration: make uid NOT NULL and add unique constraint."""

# pylint: disable=E1101,no-member

from alembic import op
from typing import Any, cast
import sqlalchemy as sa

# Cast `op` to Any to avoid linter complaints about dynamic Alembic API.
op = cast(Any, op)

revision = '0003_uid_not_null_unique'
down_revision = '0002_backfill_uid'
branch_labels = None
depends_on = None


def upgrade():
    # Add unique constraint on uid (assumes backfill ensured uniqueness)
    op.create_unique_constraint('uq_dibi_earthquakes_raw_uid', 'dibi_earthquakes_raw', ['uid'])

    # Make uid NOT NULL
    op.alter_column('dibi_earthquakes_raw', 'uid', existing_type=sa.UUID(), nullable=False)


def downgrade():
    # Revert NOT NULL and drop unique constraint
    op.alter_column('dibi_earthquakes_raw', 'uid', existing_type=sa.UUID(), nullable=True)
    op.drop_constraint('uq_dibi_earthquakes_raw_uid', 'dibi_earthquakes_raw', type_='unique')
