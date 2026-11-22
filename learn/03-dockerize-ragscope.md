# 03: Dockerize RAGScope

## Why
You built a toy image in lesson 02. RAGScope's real Dockerfiles use the same instructions, just more of them.
In this lesson you'll read them line by line, build the backend image, and find **a real security bug** in this project.

## Concept: the build context is the project root

Both Dockerfiles are built from the **project root** (that's what `context: .` means in
`docker-compose.yml`). So paths inside them start with `backend/…` or `frontend/…`:

```
RAGScope/                  ← build context (".")
├── Dockerfile.backend
├── Dockerfile.frontend
├── .dockerignore          ← what NOT to send to Docker
├── backend/  (pyproject.toml, uv.lock, app/, .env !)
└── frontend/ (pyproject.toml, uv.lock, app.py, pages/)
```

## Walkthrough: [Dockerfile.backend](../Dockerfile.backend)

```dockerfile
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
```
- `apt-get` is Debian's package installer, like `pip` but for Linux programs.
  `build-essential` = C compilers (in case a library must be compiled), `curl` = used by the healthcheck.
- Everything is in **one `RUN`**, ending with deleting the apt cache. Each `RUN` makes a layer, and
  deleting files in a *later* layer doesn't shrink earlier ones.

```dockerfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
```
- Copies the `uv` program out of **another image** (published on GitHub's registry `ghcr.io`). It's a quick
  way to install a tool without downloading an installer.

```dockerfile
COPY backend/pyproject.toml backend/uv.lock* ./
RUN uv sync --frozen --no-dev
```
- Same trick as lesson 02: dependency files first, install, **then** the code, so code edits don't
  re-install everything.
- `--frozen` = install exactly what `uv.lock` says and fail if the lock is out of date. `--no-dev` = skip
  test tools (pytest, ruff).

```dockerfile
COPY backend/ ./
RUN mkdir -p /app/data/uploads
EXPOSE 8000
```
- Copies **the whole `backend/` folder** into `/app`. Keep this line in mind for the bug hunt below.
- `EXPOSE` is documentation only ("this app uses port 8000"). It does **not** publish the port; `-p` does.

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
- Every 30 s Docker calls the `/health` endpoint from [backend/app/main.py](../backend/app/main.py).
  `docker ps` then shows `(healthy)` or `(unhealthy)`.
- `--host 0.0.0.0`: required inside containers (you proved this in lesson 02).

## Walkthrough: [Dockerfile.frontend](../Dockerfile.frontend)

It's the same pattern with three differences:
- Port **8501**, and the healthcheck uses Streamlit's built-in `/_stcore/health`.
- It writes a Streamlit config file with `echo` (`headless = true` = don't try to open a browser).
- It doesn't contain the backend's address. That comes from the `RAGSCOPE_API_URL` environment
  variable at **run time** (see [frontend/utils/api_client.py](../frontend/utils/api_client.py)).
  The same image can talk to any backend.

## Try it

**Step 0. Fix your ghcr.io login.** Your Docker has an old, expired login saved for `ghcr.io`,
which makes the `COPY --from=ghcr.io/...` line fail with `denied`. Log out so Docker downloads anonymously:
```bash
docker logout ghcr.io
```

**Step 1. Build the backend image** from the project root. **The first build is slow (10–20 min)**:
```bash
cd /mnt/d/Work/RAGScope
docker build -f Dockerfile.backend -t ragscope-backend .
```
`-f` picks the Dockerfile (because it isn't named just `Dockerfile`).

**Step 2. How big is it?**
```bash
docker images ragscope-backend
```
It's **many GB**. Why? Open [backend/uv.lock](../backend/uv.lock) and search for `name = "nvidia-`.
`sentence-transformers` needs PyTorch, and by default PyTorch on Linux ships **GPU (CUDA) libraries**:

| Download | Size |
|---|---|
| torch | 527 MB |
| nvidia-cublas | 423 MB |
| nvidia-cudnn | 366 MB |
| nvidia-cufft, -nccl, -cusolver, … | ~1.4 GB more |
| **Total GPU-related** | **≈ 2.7 GB** compressed |

The free host has **no GPU**, so all of this is dead weight. Lesson 05 shows how the Hugging Face
Dockerfile installs the CPU-only PyTorch instead: about a 200 MB download in place of 2.7 GB.

**Step 3. Run it with your settings:**
```bash
docker run --rm -p 8000:8000 --env-file .env ragscope-backend
```
`--env-file .env` loads every line of `.env` as an environment variable inside the container.
This is the safe way to give secrets to a container.

## Break it on purpose: `localhost` inside a container

Step 3 fails at startup with something like:
```
ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 5433)
```
Your `.env` says `DATABASE_URL=...@localhost:5433/...`. Inside the container, **`localhost` is the container
itself**, and there's no Postgres in there. To reach a port on your laptop from a container, Docker Desktop
gives you a special name, `host.docker.internal`:

```bash
docker run --rm -p 8000:8000 --env-file .env \
  -e DATABASE_URL=postgresql+asyncpg://ragscope:ragscope@host.docker.internal:5433/ragscope \
  ragscope-backend
```
(Postgres from lesson 01 must be running.) Wait for `Application startup complete`, then open
http://localhost:8000/docs. **Your backend is running in a container.** 🎉 Stop it with `Ctrl+C`.

`-e` after `--env-file` overrides that one variable. Lesson 04 fixes this more cleanly with Compose.

## Bug hunt: are your API keys inside the image? 🔍

Try this, **without** `--env-file`:
```bash
docker run --rm ragscope-backend ls -a /app
docker run --rm ragscope-backend cat /app/.env
```
Your **real API keys** are printed. They're baked into the image.

**Why?** [.dockerignore](../.dockerignore) contains `.env`, but `.dockerignore` patterns match from the
**root of the build context** only. `.env` excludes `RAGScope/.env` and **not** `RAGScope/backend/.env`.
Then `COPY backend/ ./` copies `backend/.env` into the image.

Why it matters: if you ever pushed this image to a registry or shared it, anyone could read your keys.
(Your Hugging Face deployment is **not** affected. Its Dockerfile copies only `backend/app`, and the deploy
script sends only files committed to git, where `.env` is ignored.)

**Fix it yourself:** in `.dockerignore`, change the Environment section to:
```
# Environment
**/.env
**/.env.*
!**/.env.example
```
`**/` means "in any folder". Rebuild (fast, since most layers are cached) and run the `cat` command again.
You should get `No such file or directory`. Tell Claude when you've done it, and it will check your change.

## Check yourself
1. Why is `uv.lock` copied and installed *before* `COPY backend/ ./`?
2. Does `EXPOSE 8000` make the app reachable from your laptop?
3. The frontend image doesn't contain the backend's address. How does it find the backend?

<details><summary>Answers</summary>

1. Layer caching. Libraries change rarely and code changes often, so code edits don't trigger a 10-minute re-install.
2. No. It's documentation. Publishing needs `-p 8000:8000` (or `ports:` in Compose).
3. From the `RAGSCOPE_API_URL` environment variable when the container starts.
</details>

## Common errors
| Symptom | Fix |
|---|---|
| `failed to fetch oauth token: denied` on the `ghcr.io/astral-sh/uv` line | `docker logout ghcr.io` (expired saved login). |
| `no space left on device` | Docker's disk is full. Run `docker system df` to see usage, then `docker image prune` / `docker builder prune` to free space. |
| Build very slow every time | Did you edit `pyproject.toml`/`uv.lock`? That invalidates the install layer. Code-only edits should be fast. |
| `Connect call failed ('127.0.0.1', 5433)` | `localhost` in a container is the container. Use `host.docker.internal`, or Compose service names (lesson 04). |
