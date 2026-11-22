# Glossary

Plain-English definitions, in roughly the order you'll meet them.

| Word | Meaning |
|---|---|
| **Server** | A program that waits for requests and answers them. Also used for "the computer that runs it". |
| **Process** | A program that is currently running. `uvicorn` running your backend is one process. |
| **Port** | A numbered "door" on a computer. A server listens on a port, e.g. FastAPI on 8000 and Streamlit on 8501. Only one process can use a port at a time. |
| **localhost / 127.0.0.1** | "This computer itself". `localhost:8000` means "port 8000 on the machine I'm on". |
| **0.0.0.0** | When a server *listens* on `0.0.0.0`, it accepts connections from anywhere, not only from the same machine. |
| **Environment variable** | A named setting handed to a process from outside the code, e.g. `OPENAI_API_KEY`. Lets the same code run with different settings. |
| **`.env` file** | A text file of `NAME=value` lines that tools load as environment variables. Holds secrets, so it must **never** be committed. |
| **Secret** | A password-like value (API key, database password). Anyone who has it can act as you. |
| **Dependency** | A library your code needs (fastapi, torch, …). Listed in `pyproject.toml`. |
| **Lock file** | `uv.lock`. The exact version of every dependency, so every install is identical. |
| **Docker** | A tool that packages an app *and everything it needs* (Python, libraries, files) so it runs the same on any computer. |
| **Image** | The package Docker builds: a frozen, read-only snapshot of an app plus its environment. Like a recipe that's already been cooked and frozen. |
| **Container** | A running copy of an image. You can start many containers from one image. Like heating up a portion of the frozen meal. |
| **Dockerfile** | The text file of instructions Docker follows to build an image. |
| **Layer** | Each Dockerfile instruction makes one layer. Docker caches layers, so unchanged steps are skipped on the next build. |
| **Base image** | The starting point of a Dockerfile (`FROM python:3.12-slim`): someone else's image you build on. |
| **Registry** | A website that stores images, e.g. Docker Hub (`docker.io`) or GitHub's `ghcr.io`. `docker pull` downloads from one. |
| **Build context** | The folder you send to Docker when you run `docker build`. Only files in it can be `COPY`'d. |
| **`.dockerignore`** | Lists files to leave out of the build context (like `.gitignore`, but for Docker). |
| **Port mapping** | `-p 8080:8000` means "laptop port 8080 → container port 8000". |
| **Volume** | Storage that lives outside a container, so data survives when the container is deleted. |
| **Docker Compose** | A tool to start several containers together from one file (`docker-compose.yml`). |
| **Service** | One container definition inside `docker-compose.yml` (e.g. `postgres`, `backend`). |
| **Healthcheck** | A command Docker runs regularly to ask "is this container working?". |
| **Deploy** | Put your app on a computer that's always on and reachable from the internet. |
| **Host / hosting provider** | The company whose computers run your app (Hugging Face, Google Cloud, …). |
| **Free tier** | The free amount of a paid service. Has limits (RAM, hours, storage). |
| **Managed database** | A database someone else runs for you (Neon). You only get a connection string. |
| **Connection string** | One URL with everything needed to reach a database: `postgresql://user:password@host:port/dbname?options`. |
| **SSL / TLS** | Encryption for network traffic. Cloud databases require it. HTTPS is HTTP + TLS. |
| **Ephemeral disk** | Storage that's wiped when the app restarts. Free Hugging Face Spaces have this. |
| **Cold start** | The wait while a sleeping app starts up again. |
| **Build logs / runtime logs** | What the host prints while building your image / while your app runs. Your main debugging tool. |
| **Access token** | A password for tools instead of people (e.g. for `git push` to Hugging Face). |
| **CI/CD** | Automating test + deploy whenever you push code (e.g. GitHub Actions). |
