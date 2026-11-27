#!/usr/bin/env bash
# One-time setup before the first `docker compose up`.
# Creates the folders Airflow mounts and an .env file with your user ID and a
# Fernet key (Airflow uses it to encrypt connection passwords).
# Safe to run again: it never deletes anything.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p dags logs artifacts tests

if [[ -f .env ]]; then
  echo ".env already exists, leaving it as is"
else
  {
    echo "AIRFLOW_UID=$(id -u)"
    echo "FERNET_KEY=$(openssl rand -base64 32 | tr '+/' '-_')"
  } > .env
  echo "Created .env"
fi
