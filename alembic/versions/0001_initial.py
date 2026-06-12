"""initial migration

Revision ID: 0001_initial
Revises: 
Create Date: 2026-06-13 00:00:00
"""

from alembic import op
import sqlalchemy as sa

revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Empty initial migration placeholder.

    To populate actual schema changes, run locally:
        alembic revision --autogenerate -m "initial"
    then review the generated file and apply with `alembic upgrade head`.
    """
    pass


def downgrade():
    pass
