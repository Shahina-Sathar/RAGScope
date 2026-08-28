from dataclasses import dataclass

from app.retrieval.vector import VectorRetriever, RetrievedChunk
from app.retrieval.bm25 import BM25Retriever, BM25Result


@dataclass
class HybridChunk:
    chunk_id: str
    text: str
    document_name: str
    page_number: int
    vector_score: float | None
    bm25_score: float | None
    rrf_score: float


class HybridRetriever:
    def __init__(self, vector_retriever: VectorRetriever, bm25_retriever: BM25Retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 20) -> list[HybridChunk]:
        vector_results = self.vector_retriever.retrieve(query, top_k=candidate_k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=candidate_k)

        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)

        fused.sort(key=lambda x: x.rrf_score, reverse=True)

        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[RetrievedChunk],
        bm25_results: list[BM25Result],
        k: int = 60,
    ) -> list[HybridChunk]:
        chunk_map: dict[str, HybridChunk] = {}

        for rank, chunk in enumerate(vector_results, start=1):
            rrf_score = 1.0 / (k + rank)
            chunk_map[chunk.chunk_id] = HybridChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                vector_score=chunk.similarity_score,
                bm25_score=None,
                rrf_score=rrf_score,
            )

        for rank, chunk in enumerate(bm25_results, start=1):
            rrf_score = 1.0 / (k + rank)
            if chunk.chunk_id in chunk_map:
                chunk_map[chunk.chunk_id].bm25_score = chunk.bm25_score
                chunk_map[chunk.chunk_id].rrf_score += rrf_score
            else:
                chunk_map[chunk.chunk_id] = HybridChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    vector_score=None,
                    bm25_score=chunk.bm25_score,
                    rrf_score=rrf_score,
                )

        return list(chunk_map.values())
