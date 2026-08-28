#!/usr/bin/env bash
# Start RAGScope frontend (Streamlit) using the fast-disk venv.
# Requires the backend running on port 8000.
set -e

cd "$(dirname "$0")/frontend"

export UV_PROJECT_ENVIRONMENT=/home/shahina/.venvs/ragscope-frontend

echo "Starting RAGScope dashboard on http://localhost:8501 ..."
exec /home/shahina/.venvs/ragscope-frontend/bin/streamlit run app.py