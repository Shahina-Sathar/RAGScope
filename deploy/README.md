# Deploying RAGScope for free

| Piece | Where it runs | Free tier |
|---|---|---|
| FastAPI backend + Streamlit dashboard (one container) | Hugging Face Spaces (Docker, CPU basic) | 2 vCPU, 16 GB RAM; sleeps after 48 h without visitors |
| PostgreSQL | Neon | 0.5 GB storage; suspends when idle and wakes on connect |
| Vectors | Pinecone (your existing index) | Starter plan |
| LLM + embeddings | OpenAI (your existing key) | **Not free.** Pay per use (a few cents per session). Set a budget limit (step 0). |

Only port 7860 (Streamlit) is public. The FastAPI backend listens on `127.0.0.1` inside the
container, so it can't be reached from the internet.

---

## Step 0: Protect your OpenAI key

Anyone who can open your app can make it call OpenAI with your key.

1. Go to https://platform.openai.com/settings/organization/limits and set a low monthly budget (e.g. $5).
2. Plan to set `APP_PASSWORD` in step 3. It shows a password screen before the dashboard.

## Step 1: Create the database (Neon)

1. Sign up at https://neon.tech (GitHub login works). No card needed.
2. Create a project (any name, pick the region closest to you).
3. On the project dashboard click **Connect**.
   - Turn **Connection pooling OFF**. The app's `asyncpg` driver works best with the direct connection.
   - Copy the connection string. It looks like
     `postgresql://neondb_owner:xxxx@ep-something.region.aws.neon.tech/neondb?sslmode=require&channel_binding=require`
4. Keep it for step 3. You can paste it as-is: the app converts it to the `asyncpg` format.
   The app creates its tables on first start.

## Step 2: Create the Space (Hugging Face)

1. Sign up at https://huggingface.co.
2. Create an access token: **Settings → Access Tokens → Create new token → type "Write"**. Copy it.
3. Create the Space at https://huggingface.co/new-space:
   - Space name: `ragscope`
   - SDK: **Docker** → **Blank**
   - Hardware: **CPU basic (free)**
   - Visibility: **Public** means anyone can see your code *and* use the app.
     **Private** means only you (logged in) can use it. Your keys are never visible either way.

## Step 3: Add your secrets to the Space

In the Space, go to **Settings → Variables and secrets → New secret** and add each of these:

| Name | Value |
|---|---|
| `OPENAI_API_KEY` | from your `.env` |
| `PINECONE_API_KEY` | from your `.env` |
| `PINECONE_HOST` | from your `.env` |
| `PINECONE_INDEX_NAME` | from your `.env` (e.g. `ragscope`) |
| `DATABASE_URL` | the Neon string from step 1 |
| `APP_PASSWORD` | any password you choose (recommended) |

Never commit `.env`. It is git-ignored, and the deploy script only sends committed files.

## Step 4: Deploy

From the project folder in your WSL terminal:

```bash
git add -A
git commit -m "Add Hugging Face deployment"
./deploy/push_to_hf.sh <your-hf-username>/ragscope
```

When git asks for credentials, use your **Hugging Face username** and paste the **token** from
step 2 as the password.

The Space page then shows **Building**. The first build takes about 10 minutes, and later builds are faster.
Click **Logs** to watch. It's ready when the log shows `Uvicorn running` and then
`You can now view your Streamlit app`.

Your app is live at `https://<your-hf-username>-ragscope.hf.space`.

## Step 5: Try it

1. Open the URL and enter your `APP_PASSWORD`.
2. The sidebar should show **✅ API Connected**.
3. **Documents** → upload the PDFs from `evaluation/sample_docs/` → index them.
4. **Query Playground** → ask "How much does the Pro plan cost?"
5. **Experiments**: the dataset path already points to `/app/evaluation/questions_mini.json`.

## Updating the app later

Commit your changes, then run `./deploy/push_to_hf.sh <your-hf-username>/ragscope` again.

## Good to know

- **The Space disk is temporary.** Uploaded PDFs are deleted when the Space restarts (after sleeping, or on
  every deploy). Your data isn't lost: documents, traces and experiments live in Neon, vectors live
  in Pinecone, and the BM25 index is rebuilt from Pinecone on startup. The only thing you can't do after a restart is
  **re-index a document that was uploaded before the restart**. To change it, delete it and upload it again.
- **Cold starts:** after 48 h without visitors the Space sleeps. The next visit wakes it (about 1–2 minutes).
- **Testing the image locally** (optional):
  ```bash
  docker build -f deploy/huggingface/Dockerfile -t ragscope-space .
  docker run --rm -p 7860:7860 --env-file .env -e UPLOAD_DIR=/home/user/data/uploads \
    -e DATABASE_URL='<your Neon URL>' ragscope-space
  ```
  Then open http://localhost:7860.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `git push` rejected: "binary files" | Push with `deploy/push_to_hf.sh`, not a plain `git push`. It strips PDFs. |
| `git push` auth fails | Use your HF **username** and a **Write** token as the password. |
| Logs: `Field required ... openai_api_key` | A secret is missing or misspelled in step 3. |
| Logs: database / SSL / connection errors | Re-copy the Neon string with pooling **off**. |
| Sidebar says **API Offline** | Check the logs for a backend traceback above the Streamlit lines. |
| Build fails pulling `ghcr.io/astral-sh/uv` (local builds only) | Run `docker logout ghcr.io`. Your saved ghcr login has expired. |
