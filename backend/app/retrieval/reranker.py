from dataclasses import dataclass

from app.retrieval.vector import RetrievedChunk
from app.retrieval.bm25 import BM25Result
from app.retrieval.hybrid import HybridChunk


@dataclass
class RerankedChunk:
    chunk_id: str
    text: str
    document_name: str
    page_number: int
    original_score: float | None
    reranker_score: float
    original_rank: int
    reranker_rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk | BM25Result | HybridChunk],
        top_k: int = 5,
    ) -> list[RerankedChunk]:
        if not chunks:
            return []

        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self.model.predict(pairs)

        reranked = []
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            original_score = None
            if isinstance(chunk, RetrievedChunk):
                original_score = chunk.similarity_score
            elif isinstance(chunk, BM25Result):
                original_score = chunk.bm25_score
            elif isinstance(chunk, HybridChunk):
                original_score = chunk.rrf_score

            reranked.append(
                RerankedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    original_score=original_score,
                    reranker_score=float(score),
                    original_rank=i + 1,
                    reranker_rank=0,
                )
            )

        reranked.sort(key=lambda x: x.reranker_score, reverse=True)

        for i, chunk in enumerate(reranked):
            chunk.reranker_rank = i + 1

        return reranked[:top_k]
