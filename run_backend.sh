#!/usr/bin/env bash
# Start RAGScope backend (FastAPI) using the fast-disk venv.
# Requires PostgreSQL running on port 5433 (see start_postgres.sh).
set -e

cd "$(dirname "$0")/backend"

export UV_PROJECT_ENVIRONMENT=/home/shahina/.venvs/ragscope

# Export API keys into the process environment so libraries that read the
# environment (e.g. RAGAS's internal OpenAI() client) work too, in addition
# to the explicit `api_key=` passing done in our own code.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

echo "Starting RAGScope backend on http://localhost:8000 ..."
echo "First start takes ~60-90s (torch CUDA library load)."
exec /home/shahina/.venvs/ragscope/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000