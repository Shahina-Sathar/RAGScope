# 01: Run RAGScope on your laptop

## Why
Before packaging something, you need to know what "running it" involves: which programs start,
which ports they use, and which settings they read. Anything you do by hand here, a Dockerfile
will later have to do automatically.

## Concept: three processes, three ports

| Process | Started by | Listens on | Check it at |
|---|---|---|---|
| PostgreSQL | [start_postgres.sh](../start_postgres.sh) (runs it in Docker) | `localhost:5433` | — |
| FastAPI backend | [run_backend.sh](../run_backend.sh) → `uvicorn` | `0.0.0.0:8000` | http://localhost:8000/docs |
| Streamlit frontend | [run_frontend.sh](../run_frontend.sh) → `streamlit` | port `8501` | http://localhost:8501 |

- **uvicorn** is the web server that runs a FastAPI app. `uvicorn app.main:app` means
  "in the file `app/main.py`, find the variable `app` and serve it."
- **streamlit run app.py** runs the dashboard script and serves it as a web page.

Each one runs in its own terminal and keeps running until you press `Ctrl+C`.

## Walkthrough: `run_backend.sh`

```bash
cd "$(dirname "$0")/backend"          # go into backend/, wherever the script is
export UV_PROJECT_ENVIRONMENT=/home/shahina/.venvs/ragscope   # which virtual env to use
if [ -f .env ]; then                  # if backend/.env exists…
  set -a; . ./.env; set +a            # …load every line as an environment variable
fi
exec /home/shahina/.venvs/ragscope/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Notice that the script depends on things that **only exist on your laptop**:
- a virtual environment at `/home/shahina/.venvs/ragscope` with all libraries installed,
- the file `backend/.env`,
- Python 3.12.

A host has none of these. Docker exists to package them up (lesson 02).

## Walkthrough: how the backend reads settings

[backend/app/config/settings.py](../backend/app/config/settings.py):

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", ...)
    openai_api_key: str                    # no default → REQUIRED
    pinecone_api_key: str                  # REQUIRED
    pinecone_host: str                     # REQUIRED
    database_url: str = "postgresql+asyncpg://...localhost:5432/ragscope"   # has a default
    upload_dir: Path = Path("./data/uploads")
```

`pydantic-settings` fills each field from the environment variable with the same name
(case doesn't matter: `OPENAI_API_KEY` → `openai_api_key`). It also reads a `.env` file in the
**current folder**. That's why there are two `.env` files in this project:

| File | Who reads it |
|---|---|
| `backend/.env` | `run_backend.sh` and the backend's settings (it runs inside `backend/`) |
| `.env` (project root) | Docker Compose (lesson 04) |

## Try it

You'll need **4 terminals**. In VS Code, click the **+** in the terminal panel to open more.

**Terminal 1: database.** Start **Docker Desktop** on Windows first (Start menu → Docker Desktop) and wait
until it says "Engine running". Lesson 02 explains what it is. Then:
```bash
cd /mnt/d/Work/RAGScope
./start_postgres.sh
docker ps --filter name=ragscope-postgres
```
Expected: a line showing `ragscope-postgres` with `Up ... (healthy)` and `0.0.0.0:5433->5432/tcp`.

**Terminal 2: backend.**
```bash
cd /mnt/d/Work/RAGScope
./run_backend.sh
```
Wait for `Application startup complete.` The first start can take a minute.

**Terminal 3: frontend.**
```bash
cd /mnt/d/Work/RAGScope
./run_frontend.sh
```
Open http://localhost:8501. The sidebar should say **✅ API Connected**.

**Terminal 4: look around.**
```bash
curl http://localhost:8000/health                 # → {"status":"healthy"}
ss -ltn | grep -E ':(5433|8000|8501)'             # the three listening ports
```
Then open http://localhost:8000/docs in your browser. FastAPI generates this page from the code.
Try `GET /health` there with **Try it out → Execute**.

## Break it on purpose
1. Stop the backend (`Ctrl+C` in terminal 2). Refresh the Streamlit page and you'll see **❌ API Offline**.
   The frontend is fine; it just can't reach port 8000.
2. Start the backend again. Now, in terminal 2, try starting a **second** backend:
   ```bash
   cd /mnt/d/Work/RAGScope/backend && /home/shahina/.venvs/ragscope/bin/uvicorn app.main:app --port 8000
   ```
   You'll get `[Errno 98] address already in use`. **Only one process can listen on a port.**
   Remember this error: you'll meet it again.

When finished: `Ctrl+C` in terminals 2 and 3. You can leave Postgres running.

## Check yourself
1. What does `--host 0.0.0.0` mean, compared with the default `127.0.0.1`?
2. The backend crashes on start with `Field required ... openai_api_key`. What's wrong?
3. Which three things does `run_backend.sh` need that only exist on your laptop?

<details><summary>Answers</summary>

1. `0.0.0.0` accepts connections from other machines too. `127.0.0.1` accepts only connections from the same machine.
   This matters in Docker, where "another machine" includes your laptop talking to a container.
2. The `OPENAI_API_KEY` environment variable isn't set: `backend/.env` is missing or doesn't contain it.
3. The virtual environment with all libraries, the `backend/.env` file, and Python 3.12.
</details>

## Common errors
| Symptom | Fix |
|---|---|
| `The command 'docker' could not be found in this WSL 2 distro` | Start Docker Desktop on Windows. If it's already running: Docker Desktop → Settings → Resources → WSL integration → turn on your Ubuntu distro → Apply. |
| `address already in use` | Something already uses that port. Find it with `ss -ltnp \| grep 8000` and stop it, or use another port. |
| Backend: `connection refused` to Postgres | Postgres isn't running. Run `./start_postgres.sh` and check `docker ps`. |
| Sidebar says **API Offline** | The backend isn't running, or it's still starting. Check terminal 2. |
