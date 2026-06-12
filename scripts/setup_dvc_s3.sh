#!/usr/bin/env bash
# Setup a DVC S3 remote and push data to it.
# Usage: ./scripts/setup_dvc_s3.sh s3://my-bucket/path


set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 s3://bucket/path" >&2
  exit 2
fi

REMOTE_URL="$1"

echo "Ensuring DVC S3 support is installed..."
# try installing dvc[s3] (works inside virtualenv)
python -m pip install --upgrade pip
if ! python -c "import dvc_s3" >/dev/null 2>&1; then
  echo "Installing dvc[s3]..."
  python -m pip install "dvc[s3]"
fi

echo "Configuring DVC remote: $REMOTE_URL"
dvc remote add -d ci_remote "$REMOTE_URL" || true

echo "Pushing tracked data to remote..."
dvc push

echo "Done. Ensure CI runner has access to this remote (set secrets)."
