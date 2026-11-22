# 00: The big picture

## Why
Right now RAGScope only works on **your laptop**. If you send a friend `http://localhost:8501`,
it won't work for them: `localhost` means *their own* laptop, which isn't running RAGScope.

**Deploying** means running your app on a computer that is:
1. **always on**, even when your laptop is closed,
2. **reachable from the internet** at a public address like `https://you-ragscope.hf.space`,
3. **set up correctly**, with the right Python, libraries, settings and secrets.

Point 3 is the hard part, and it's what Docker solves. That's why we learn Docker first.

## Concept: the pieces of RAGScope

```
   You (browser)
        │  http://localhost:8501
        ▼
┌──────────────────┐   HTTP requests    ┌───────────────────┐
│ FRONTEND         │ ─────────────────▶ │ BACKEND           │
│ Streamlit        │  localhost:8000    │ FastAPI (uvicorn) │
│ frontend/app.py  │ ◀───────────────── │ backend/app/      │
└──────────────────┘   JSON answers     └─────────┬─────────┘
                                                  │
                    ┌─────────────────────────────┼──────────────────────┐
                    ▼                             ▼                      ▼
          ┌──────────────────┐        ┌────────────────────┐   ┌──────────────────┐
          │ PostgreSQL       │        │ Pinecone           │   │ OpenAI           │
          │ documents,       │        │ (cloud) vectors    │   │ (cloud) embeddings│
          │ traces,          │        │ for search         │   │ + answers        │
          │ experiments      │        └────────────────────┘   └──────────────────┘
          │ (Docker, :5433)  │
          └──────────────────┘
```

| Piece | What it does | Where it runs today |
|---|---|---|
| Streamlit frontend | The web pages you click on | Your laptop, port **8501** |
| FastAPI backend | Uploads, indexing, search, answers | Your laptop, port **8000** |
| PostgreSQL | Stores documents, traces and experiment results | A Docker container on your laptop, port **5433** |
| Pinecone | Stores the vectors for semantic search | Already in the cloud |
| OpenAI | Makes embeddings and writes answers | Already in the cloud |

Pinecone and OpenAI are **already** "deployed": they're services on the internet that you reach
with an API key. To go live we need to move the other three pieces off your laptop.

## Concept: settings and secrets

The code never contains your API keys. It reads them from **environment variables**: named values
given to the program from outside. Locally they come from the `.env` file:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
DATABASE_URL=postgresql+asyncpg://ragscope:ragscope@localhost:5433/ragscope
```

Why not write them into the code?
- **Safety:** code gets shared (GitHub, Hugging Face). Anyone who sees a key can spend your money.
- **Flexibility:** the same code can use your laptop's database today and a cloud database tomorrow.
  You change one setting, not the code.

`.env` is listed in [.gitignore](../.gitignore), so git never uploads it. Keep it that way.

## The road ahead

```
Lesson 01  run it on the laptop      (understand the pieces)
Lesson 02  Docker basics              (learn packaging on a toy app)
Lesson 03  package RAGScope           (real Dockerfiles)
Lesson 04  Docker Compose             (all pieces with one command)
Lesson 05  production differences     (what breaks on a host, and why)
Lesson 06  cloud database (Neon)      (replace the laptop's Postgres)
Lesson 07  Hugging Face Spaces        (go live!)
Lesson 08  operate & debug            (keep it running)
```

## Try it
Nothing to run yet. Open these files in VS Code and find:
1. In [frontend/utils/api_client.py](../frontend/utils/api_client.py), the line that decides **which address**
   the frontend uses to reach the backend. (Hint: `RAGSCOPE_API_URL`.)
2. In [backend/app/config/settings.py](../backend/app/config/settings.py), the list of settings the backend reads.
   Which ones have **no default value**? Those are the ones the app can't start without.

## Check yourself
1. Your friend opens `http://localhost:8501`. Why doesn't it show your app?
2. Name the two pieces of RAGScope that are already in the cloud.
3. Why are API keys environment variables instead of being written in the code?

<details><summary>Answers</summary>

1. `localhost` always means "the computer I'm on". On their laptop nothing is running on port 8501.
2. Pinecone and OpenAI.
3. So they're never shared along with the code, and so you can change them (e.g. the database address)
   without editing code.
</details>

## Common confusions
| You might think… | Actually… |
|---|---|
| "Deploying = uploading my code" | You also need the right Python, libraries, settings and secrets on that computer, and something must *run* the code. |
| "localhost is my laptop" | It's "whichever machine the program is running on". This matters a lot inside Docker (lesson 03). |
| "The frontend talks to the database" | Only the backend does. The frontend only talks to the backend. |
