import json
from enum import Enum
from pathlib import Path

from app.config.settings import settings
from app.retrieval.vector import VectorRetriever, RetrievedChunk
from app.retrieval.bm25 import BM25Retriever, BM25Result
from app.retrieval.hybrid import HybridRetriever, HybridChunk


class RetrievalMode(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


class RetrievalFactory:
    def __init__(self):
        self.vector_retriever = VectorRetriever()
        self.bm25_retriever = BM25Retriever()
        self.hybrid_retriever = HybridRetriever(self.vector_retriever, self.bm25_retriever)
        self._bm25_index_loaded = False

    def _ensure_bm25_index(self):
        if self._bm25_index_loaded:
            return

        bm25_file = Path(settings.upload_dir) / "bm25_chunks.json"
        if bm25_file.exists():
            with open(bm25_file, "r") as f:
                documents = json.load(f)
            self.bm25_retriever.build_index(documents)
            self._bm25_index_loaded = True

    def retrieve(
        self, query: str, mode: RetrievalMode, top_k: int = 5
    ) -> tuple[list[RetrievedChunk | BM25Result | HybridChunk], str]:
        if mode == RetrievalMode.VECTOR:
            results = self.vector_retriever.retrieve(query, top_k=top_k)
            return results, "vector"
        elif mode == RetrievalMode.BM25:
            self._ensure_bm25_index()
            results = self.bm25_retriever.retrieve(query, top_k=top_k)
            return results, "bm25"
        elif mode == RetrievalMode.HYBRID:
            self._ensure_bm25_index()
            results = self.hybrid_retriever.retrieve(query, top_k=top_k)
            return results, "hybrid"
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

    def build_bm25_index(self, documents: list[dict]):
        self.bm25_retriever.build_index(documents)
        self._bm25_index_loaded = True
