import pytest
from unittest.mock import Mock

from app.retrieval.hybrid import HybridRetriever, HybridChunk
from app.retrieval.vector import RetrievedChunk
from app.retrieval.bm25 import BM25Result


def test_hybrid_retriever_rrf():
    vector_retriever = Mock()
    bm25_retriever = Mock()

    vector_retriever.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="chunk_1",
            text="Refund policy text",
            document_name="policy.pdf",
            page_number=1,
            similarity_score=0.95,
        ),
        RetrievedChunk(
            chunk_id="chunk_2",
            text="Cancellation policy",
            document_name="policy.pdf",
            page_number=2,
            similarity_score=0.85,
        ),
    ]

    bm25_retriever.retrieve.return_value = [
        BM25Result(
            chunk_id="chunk_3",
            text="Refund process details",
            document_name="policy.pdf",
            page_number=3,
            bm25_score=15.2,
        ),
        BM25Result(
            chunk_id="chunk_1",
            text="Refund policy text",
            document_name="policy.pdf",
            page_number=1,
            bm25_score=12.5,
        ),
    ]

    hybrid = HybridRetriever(vector_retriever, bm25_retriever)
    results = hybrid.retrieve("refund policy", top_k=3)

    assert len(results) == 3
    assert all(isinstance(r, HybridChunk) for r in results)
    
    # chunk_1 should have highest RRF score since it appears in both
    chunk_1 = next(r for r in results if r.chunk_id == "chunk_1")
    assert chunk_1.vector_score == 0.95
    assert chunk_1.bm25_score == 12.5
    assert chunk_1.rrf_score > 0


def test_hybrid_retriever_rrf_formula():
    vector_retriever = Mock()
    bm25_retriever = Mock()

    vector_retriever.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="chunk_1",
            text="Text 1",
            document_name="doc.pdf",
            page_number=1,
            similarity_score=0.9,
        ),
    ]

    bm25_retriever.retrieve.return_value = [
        BM25Result(
            chunk_id="chunk_1",
            text="Text 1",
            document_name="doc.pdf",
            page_number=1,
            bm25_score=10.0,
        ),
    ]

    hybrid = HybridRetriever(vector_retriever, bm25_retriever)
    results = hybrid.retrieve("query", top_k=1)

    # RRF score should be 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.0328
    expected_rrf = 2.0 / 61.0
    assert abs(results[0].rrf_score - expected_rrf) < 0.001


def test_hybrid_retriever_top_k():
    vector_retriever = Mock()
    bm25_retriever = Mock()

    vector_retriever.retrieve.return_value = [
        RetrievedChunk(
            chunk_id=f"chunk_{i}",
            text=f"Text {i}",
            document_name="doc.pdf",
            page_number=i,
            similarity_score=0.9 - i * 0.1,
        )
        for i in range(10)
    ]

    bm25_retriever.retrieve.return_value = []

    hybrid = HybridRetriever(vector_retriever, bm25_retriever)
    results = hybrid.retrieve("query", top_k=3)

    assert len(results) == 3


def test_hybrid_retriever_candidate_k():
    vector_retriever = Mock()
    bm25_retriever = Mock()

    vector_retriever.retrieve.return_value = []
    bm25_retriever.retrieve.return_value = []

    hybrid = HybridRetriever(vector_retriever, bm25_retriever)
    hybrid.retrieve("query", top_k=5, candidate_k=20)

    vector_retriever.retrieve.assert_called_once_with("query", top_k=20)
    bm25_retriever.retrieve.assert_called_once_with("query", top_k=20)
