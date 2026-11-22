# 02: Docker first steps

## Why
In lesson 01 the backend needed a virtual env, Python 3.12 and lots of libraries that are only on
your laptop. A host doesn't have them. You *could* log into a server and install everything by hand,
but you'd have to remember every step, and "it works on my machine" would bite you.

**Docker** packages your app *together with* the operating system bits, the Python version and every library
into one bundle that runs the same on any computer that has Docker. Hugging Face (lesson 07) runs
exactly the bundle you build.

## Concept: image vs container

Think of a **shipping container**: whatever is inside, every ship and crane handles it the same way.

| | Image | Container |
|---|---|---|
| What | A frozen, read-only package: OS files + Python + libraries + your code | A **running** copy of an image |
| Made by | `docker build` (following a **Dockerfile**) | `docker run <image>` |
| Analogy | A recipe that's been cooked and frozen | A portion you heated up and are eating |
| How many | One | As many as you like, from the same image |
| Changes | Never change. Rebuild to get a new one | Changes inside are **lost** when the container is deleted |

Where do images come from? Either you **build** one, or you **pull** someone else's from a
**registry** (Docker Hub). `python:3.12-slim` is the official Python image, which is a small
Debian Linux with Python 3.12 already installed.

**On your machine:** Docker Desktop runs the Docker engine inside a small Linux VM on Windows.
The `docker` command in WSL is just a remote control that talks to that engine. So
**Docker Desktop must be running** for any `docker` command to work.

## Try it, part 1: your first containers

**Step 0. Start Docker Desktop** (Windows Start menu). Wait for "Engine running". Then in WSL:
```bash
docker version
```
You should see a `Client:` section **and** a `Server:` section. If you only see an error, check
the common-errors table at the bottom.

**Step 1. Hello world:**
```bash
docker run hello-world
```
Roughly what you'll see:
```
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
...
Hello from Docker!
This message shows that your installation appears to be working correctly.
```
What happened: Docker didn't have the image, so it **pulled** it from Docker Hub, **created a container**,
the container printed a message, and then **exited**.

**Step 2. A Python that isn't on your laptop:**
```bash
docker run -it --rm python:3.12-slim python
```
- `-it` = interactive terminal (so you can type).
- `--rm` = delete the container when it exits.

You're now in a Python prompt *inside a container*. Try:
```python
import platform; print(platform.platform())
import fastapi
```
The second line fails with `ModuleNotFoundError`. This container is a clean, empty Python with **only what's in the image**.
Exit with `exit()`.

**Step 3. Look inside a container's Linux:**
```bash
docker run -it --rm python:3.12-slim bash
cat /etc/os-release        # Debian, not your Ubuntu!
ls /                       # its own separate filesystem
exit
```

**Step 4. The commands you'll use most:**
```bash
docker images              # images you have
docker ps                  # running containers
docker ps -a               # all containers, including stopped ones
```

## Try it, part 2: your first Dockerfile

Open [learn/examples/hello-docker/](examples/hello-docker/). It's a one-file FastAPI app:

[main.py](examples/hello-docker/main.py):
```python
@app.get("/")
def hello():
    return {"message": os.getenv("GREETING", "Hello from inside a container!")}
```

[Dockerfile](examples/hello-docker/Dockerfile), line by line:
```dockerfile
FROM python:3.12-slim                 # start from the official Python image
WORKDIR /app                          # make /app the current folder inside the image
COPY requirements.txt .               # copy one file from your laptop into /app
RUN pip install --no-cache-dir -r requirements.txt   # run a command while BUILDING
COPY main.py .                        # copy the code
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]   # what to run when a CONTAINER starts
```

The key difference: **`RUN` happens once, at build time** (installing libraries into the image).
**`CMD` happens every time a container starts** (starting your server).

**Build the image:**
```bash
cd /mnt/d/Work/RAGScope/learn/examples/hello-docker
docker build -t hello-docker .
```
- `-t hello-docker` = give the image a name (a "tag").
- `.` = the **build context**: the folder whose files `COPY` can see.

You'll see each step run (`[2/5] WORKDIR /app`, `[4/5] RUN pip install ...`).

**Run a container from it:**
```bash
docker run -d --name hello -p 8080:8000 hello-docker
```
- `-d` = detached (runs in the background, gives you your terminal back).
- `--name hello` = a name so you can refer to it.
- `-p 8080:8000` = **port mapping**: "laptop port 8080 → container port 8000".

```bash
curl http://localhost:8080             # → {"message":"Hello from inside a container!"}
docker ps                              # see it running, with 0.0.0.0:8080->8000/tcp
docker logs hello                      # the uvicorn output from inside the container
docker exec -it hello bash             # open a shell INSIDE the running container
#   ls        → Dockerfile? No! Only main.py and requirements.txt were copied
#   exit
docker stop hello                      # stop it
docker rm hello                        # delete the stopped container
```

**Pass a setting with an environment variable:**
```bash
docker run -d --rm --name hello -p 8080:8000 -e GREETING="Hi Shahina" hello-docker
curl http://localhost:8080             # → {"message":"Hi Shahina"}
docker stop hello                      # --rm deletes it automatically after stopping
```
This is exactly how RAGScope will get its API keys on the host: as environment variables,
never baked into the image.

**Watch the cache:** run `docker build -t hello-docker .` again. It finishes almost instantly and
says `CACHED` on each step. Now change the message in `main.py` and build again. Only the steps
**from `COPY main.py` onward** re-run; `pip install` stays cached. That's why the Dockerfile copies
`requirements.txt` and installs it *before* copying the code. Code changes often, libraries rarely.

## Break it on purpose
1. **Forget the port mapping:** `docker run -d --name hello hello-docker`, then `curl localhost:8080`.
   You get `Connection refused`. The app is running, but nothing connects your laptop's port to the container.
   `docker rm -f hello` to clean up.
2. **Listen only on localhost:** in the Dockerfile, change `0.0.0.0` to `127.0.0.1`, rebuild, run with
   `-p 8080:8000`, then `curl localhost:8080`. It fails (`Connection reset` or an empty reply). Inside a container, `127.0.0.1`
   means *the container itself*, so connections forwarded from your laptop are rejected. **Servers in
   containers must listen on `0.0.0.0`.** Change it back and rebuild.

## Check yourself
1. What's the difference between `RUN` and `CMD`?
2. You run `docker run -p 9000:8000 hello-docker`. Which URL do you open on your laptop?
3. You `docker exec` into a container, create a file, then `docker rm` the container and start a new one
   from the same image. Is the file there?

<details><summary>Answers</summary>

1. `RUN` executes while **building** the image (e.g. installing libraries). `CMD` is what runs each time a **container starts**.
2. `http://localhost:9000`. The left side of `-p` is your laptop's port.
3. No. Containers start fresh from the image every time. To keep data you need a **volume** (lesson 04).
   This is also why free hosts lose uploaded files on restart (lesson 05).
</details>

## Common errors
| Symptom | Fix |
|---|---|
| `The command 'docker' could not be found in this WSL 2 distro` | Docker Desktop isn't running. Start it. If it is running: Settings → Resources → WSL integration → enable your Ubuntu → Apply & restart. |
| `Cannot connect to the Docker daemon` / `Input/output error` | Same cause. Docker Desktop is stopped or still starting. |
| `port is already allocated` | Another container already uses that laptop port. `docker ps` to find it, then `docker rm -f <name>`. |
| `Conflict. The container name "/hello" is already in use` | A stopped container still has that name. `docker rm hello`. |
| `COPY failed: file not found` | You ran `docker build` from the wrong folder. The build context (`.`) must contain the file. |
