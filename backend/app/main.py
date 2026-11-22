import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database.session import init_db, close_db
from app.ingestion.indexer import restore_bm25_chunks_from_pinecone
from app.api.documents import router as documents_router
from app.api.query import router as query_router
from app.api.traces import router as traces_router
from app.api.experiments import router as experiments_router


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    try:
        restored = await asyncio.to_thread(restore_bm25_chunks_from_pinecone)
        if restored:
            logger.warning("Restored %d BM25 chunks from Pinecone", restored)
    except Exception:
        logger.exception("Could not restore BM25 chunks from Pinecone; BM25/hybrid may be empty")
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
