#!/usr/bin/env bash
# Start RAGScope PostgreSQL (Docker Compose). Must be run from a terminal
# where Docker Desktop WSL integration is enabled.
set -e

cd "$(dirname "$0")"

echo "Starting RAGScope PostgreSQL on host port 5433 ..."
docker-compose up -d postgres
echo "Done. Verify: docker ps --filter name=ragscope-postgres"