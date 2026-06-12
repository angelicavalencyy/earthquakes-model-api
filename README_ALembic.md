Alembic migration quickstart

1. Configure connection (preferred via env var):

   - Set `SQLALCHEMY_URL` to a sync SQLAlchemy URL, e.g.
     `postgresql://user:pass@host:5432/dbname` (note: remove `+asyncpg`)

2. Initialize (already scaffolded in repo). Create a revision:

```bash
alembic revision --autogenerate -m "initial"
```

3. Apply migrations:

```bash
alembic upgrade head
```

Notes:
- This project uses `sqlmodel.SQLModel.metadata` as target metadata in `alembic/env.py`.
- For CI/CD deploys, set `SQLALCHEMY_URL` or update `alembic.ini` before running migrations.
- Do NOT enable `ALLOW_SCHEMA_AUTOCREATE` in production; prefer migrations.
