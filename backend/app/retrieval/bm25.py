from dataclasses import dataclass
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class BM25Result:
    chunk_id: str
    text: str
    document_name: str
    page_number: int
    bm25_score: float


class BM25Retriever:
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.documents: list[dict] = []

    def build_index(self, documents: list[dict]):
        self.documents = documents
        corpus = [doc["text"] for doc in documents]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[BM25Result]:
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx]
                results.append(
                    BM25Result(
                        chunk_id=doc["chunk_id"],
                        text=doc["text"],
                        document_name=doc["document_name"],
                        page_number=doc["page_number"],
                        bm25_score=float(scores[idx]),
                    )
                )

        return results
