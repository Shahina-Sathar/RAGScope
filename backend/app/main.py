from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database.session import init_db, close_db
from app.api.documents import router as documents_router
from app.api.query import router as query_router
from app.api.traces import router as traces_router
from app.api.experiments import router as experiments_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title="RAGScope",
    description="RAG Evaluation & Observability Platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents_router)
app.include_router(query_router)
app.include_router(traces_router)
app.include_router(experiments_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
