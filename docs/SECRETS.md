# GitHub Secrets & Migration Instructions

This file describes the repository secrets and the safe migration steps to run in CI or by an administrator.

## Required repository secrets

- `POSTGRES_URL` — SQLAlchemy async URL used by the running app. Example:

  `postgresql+asyncpg://user:password@db-host:5432/database`

- `SQLALCHEMY_URL` — synchronous SQLAlchemy URL used by Alembic (for migrations).

  `postgresql://user:password@db-host:5432/database`

- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` — optional, only if CI pushes the image to Docker Hub.

- `MLFLOW_TRACKING_URI` — optional, if MLflow is used remotely.

## CI: run migrations safely

CI should run migrations against a controlled database (staging or a maintenance window). Example steps for a job that runs migrations:

```bash
# make sure SQLALCHEMY_URL is set in repository secrets
export SQLALCHEMY_URL="$SQLALCHEMY_URL"
alembic upgrade head
```

If your Alembic config expects different env var names, adapt accordingly.

## DVC remote and CI

This repository uses DVC (`dvc.yaml`) to track large data artifacts. Current
local config points to a local path (used for development). For CI and
production you should configure a remote store (S3/GCS/Azure) and set
credentials as repository secrets.

Recommended secrets for DVC (example for S3):

- `DVC_REMOTE_URL` → `s3://my-bucket/path` (or the remote name configured in `.dvc/config`)
- `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (if using S3)

CI runner steps should install DVC and run `dvc pull` after checkout and before
building the Docker image. Example (in the workflow):

```yaml
- name: Install DVC
  run: |
    python -m pip install --upgrade pip
    pip install "dvc[s3]"  # or dvc[gdrive], dvc[azure], etc. depending on remote

- name: Pull DVC data
  run: |
    dvc pull
```

If DVC remote requires authentication, ensure the necessary cloud credentials
are added to repository Secrets and available to the runner.

## How to set secrets using GitHub CLI

Run locally (you need `gh` installed and authenticated):

```bash
gh secret set POSTGRES_URL --body 'postgresql+asyncpg://user:pass@host:5432/dbname'
gh secret set SQLALCHEMY_URL --body 'postgresql://user:pass@host:5432/dbname'
gh secret set DOCKERHUB_USERNAME --body 'your-user'
gh secret set DOCKERHUB_TOKEN --body 'token'
gh secret set MLFLOW_TRACKING_URI --body 'http://mlflow:5000'
```

## Post-deploy checklist

- Rotate credentials if any secrets were committed previously.
- Run `alembic upgrade head` in a maintenance window.
- Verify `/health` returns `db: true` on your target environment (or use readiness probe that requires DB).

## Contact
If you prefer, provide repository admin access and I can open the PR for you, or follow the `gh` steps above to set secrets yourself.
