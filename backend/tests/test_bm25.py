import pytest
import numpy as np

from app.retrieval.bm25 import BM25Retriever, BM25Result


def test_bm25_retriever_build_index():
    retriever = BM25Retriever()
    documents = [
        {
            "chunk_id": "chunk_1",
            "text": "The refund policy allows returns within 30 days",
            "document_name": "policy.pdf",
            "page_number": 1,
        },
        {
            "chunk_id": "chunk_2",
            "text": "Customers can cancel their subscription at any time",
            "document_name": "policy.pdf",
            "page_number": 2,
        },
    ]

    retriever.build_index(documents)

    assert retriever.bm25 is not None
    assert len(retriever.documents) == 2


def test_bm25_retriever_retrieve():
    retriever = BM25Retriever()
    documents = [
        {
            "chunk_id": "chunk_1",
            "text": "The refund policy allows returns within 30 days of purchase",
            "document_name": "policy.pdf",
            "page_number": 1,
        },
        {
            "chunk_id": "chunk_2",
            "text": "Customers can cancel their subscription at any time",
            "document_name": "policy.pdf",
            "page_number": 2,
        },
        {
            "chunk_id": "chunk_3",
            "text": "Refund process takes 5-7 business days to complete",
            "document_name": "policy.pdf",
            "page_number": 3,
        },
    ]

    retriever.build_index(documents)
    results = retriever.retrieve("refund policy", top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, BM25Result) for r in results)
    assert results[0].chunk_id == "chunk_1"
    assert results[0].bm25_score > 0


def test_bm25_retriever_empty_index():
    retriever = BM25Retriever()
    results = retriever.retrieve("test query", top_k=5)

    assert len(results) == 0


def test_bm25_retriever_no_matches():
    retriever = BM25Retriever()
    documents = [
        {
            "chunk_id": "chunk_1",
            "text": "The weather is sunny today",
            "document_name": "weather.pdf",
            "page_number": 1,
        },
    ]

    retriever.build_index(documents)
    results = retriever.retrieve("refund policy xyz123", top_k=5)

    assert len(results) == 0
