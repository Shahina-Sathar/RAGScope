# Single-container image for Hugging Face Spaces (free CPU tier).
# Runs FastAPI on 127.0.0.1:8000 (private) and Streamlit on 0.0.0.0:7860 (public).
# Build locally from the repo root:  docker build -f deploy/huggingface/Dockerfile .
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

WORKDIR /app

# Backend dependencies, with CPU-only torch (the locked CUDA build adds several GB)
COPY backend/pyproject.toml backend/uv.lock backend/
RUN cd backend \
    && uv export --frozen --no-dev --no-hashes --no-emit-project -o /tmp/requirements.txt \
    && grep -Ev '^(torch|triton|nvidia-|cuda-)' /tmp/requirements.txt > /tmp/requirements-cpu.txt \
    && uv venv /opt/backend-venv --python /usr/local/bin/python3.12 \
    && uv pip install --python /opt/backend-venv/bin/python torch \
        --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --python /opt/backend-venv/bin/python -r /tmp/requirements-cpu.txt

# Frontend dependencies
COPY frontend/pyproject.toml frontend/uv.lock frontend/
RUN cd frontend && UV_PROJECT_ENVIRONMENT=/opt/frontend-venv uv sync --frozen --no-dev --no-install-project

USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    UPLOAD_DIR=/home/user/data/uploads \
    RAGSCOPE_API_URL=http://127.0.0.1:8000 \
    RAGSCOPE_DEFAULT_DATASET=/app/evaluation/questions_mini.json

# Pre-download the reranker model so the first reranked query is fast
RUN /opt/backend-venv/bin/python -c \
    "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user backend/app backend/app
COPY --chown=user frontend frontend
COPY --chown=user evaluation/questions.json evaluation/questions_mini.json evaluation/
COPY --chown=user deploy/huggingface/start.sh start.sh

EXPOSE 7860
CMD ["bash", "start.sh"]
