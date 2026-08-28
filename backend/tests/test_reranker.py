from unittest.mock import Mock, patch

import pytest

from app.retrieval.reranker import CrossEncoderReranker, RerankedChunk
from app.retrieval.vector import RetrievedChunk
from app.retrieval.bm25 import BM25Result
from app.retrieval.hybrid import HybridChunk


@patch("sentence_transformers.CrossEncoder")
def test_reranker_basic(mock_cross_encoder_class):
    mock_model = Mock()
    mock_cross_encoder_class.return_value = mock_model
    mock_model.predict.return_value = [0.9, 0.3, 0.7]

    reranker = CrossEncoderReranker()

    chunks = [
        RetrievedChunk(
            chunk_id="chunk_1",
            text="First chunk",
            document_name="doc.pdf",
            page_number=1,
            similarity_score=0.95,
        ),
        RetrievedChunk(
            chunk_id="chunk_2",
            text="Second chunk",
            document_name="doc.pdf",
            page_number=2,
            similarity_score=0.85,
        ),
        RetrievedChunk(
            chunk_id="chunk_3",
            text="Third chunk",
            document_name="doc.pdf",
            page_number=3,
            similarity_score=0.75,
        ),
    ]

    results = reranker.rerank("test query", chunks, top_k=3)

    assert len(results) == 3
    assert all(isinstance(r, RerankedChunk) for r in results)
    
    assert results[0].chunk_id == "chunk_1"
    assert results[0].reranker_score == 0.9
    assert results[0].reranker_rank == 1
    assert results[0].original_rank == 1

    assert results[1].chunk_id == "chunk_3"
    assert results[1].reranker_score == 0.7
    assert results[1].reranker_rank == 2
    assert results[1].original_rank == 3

    assert results[2].chunk_id == "chunk_2"
    assert results[2].reranker_score == 0.3
    assert results[2].reranker_rank == 3
    assert results[2].original_rank == 2


@patch("sentence_transformers.CrossEncoder")
def test_reranker_top_k(mock_cross_encoder_class):
    mock_model = Mock()
    mock_cross_encoder_class.return_value = mock_model
    mock_model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]

    reranker = CrossEncoderReranker()

    chunks = [
        RetrievedChunk(
            chunk_id=f"chunk_{i}",
            text=f"Chunk {i}",
            document_name="doc.pdf",
            page_number=i,
            similarity_score=0.9 - i * 0.1,
        )
        for i in range(5)
    ]

    results = reranker.rerank("test query", chunks, top_k=3)

    assert len(results) == 3


@patch("sentence_transformers.CrossEncoder")
def test_reranker_empty_chunks(mock_cross_encoder_class):
    mock_model = Mock()
    mock_cross_encoder_class.return_value = mock_model

    reranker = CrossEncoderReranker()
    results = reranker.rerank("test query", [], top_k=5)

    assert len(results) == 0
    mock_model.predict.assert_not_called()


@patch("sentence_transformers.CrossEncoder")
def test_reranker_preserves_metadata(mock_cross_encoder_class):
    mock_model = Mock()
    mock_cross_encoder_class.return_value = mock_model
    mock_model.predict.return_value = [0.8]

    reranker = CrossEncoderReranker()

    chunks = [
        RetrievedChunk(
            chunk_id="chunk_1",
            text="Test text",
            document_name="policy.pdf",
            page_number=4,
            similarity_score=0.91,
        ),
    ]

    results = reranker.rerank("test query", chunks, top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk_1"
    assert results[0].text == "Test text"
    assert results[0].document_name == "policy.pdf"
    assert results[0].page_number == 4
    assert results[0].original_score == 0.91
    assert results[0].reranker_score == 0.8


@patch("sentence_transformers.CrossEncoder")
def test_reranker_with_bm25_results(mock_cross_encoder_class):
    mock_model = Mock()
    mock_cross_encoder_class.return_value = mock_model
    mock_model.predict.return_value = [0.7, 0.9]

    reranker = CrossEncoderReranker()

    chunks = [
        BM25Result(
            chunk_id="chunk_1",
            text="First chunk",
            document_name="doc.pdf",
            page_number=1,
            bm25_score=15.2,
        ),
        BM25Result(
            chunk_id="chunk_2",
            text="Second chunk",
            document_name="doc.pdf",
            page_number=2,
            bm25_score=12.5,
        ),
    ]

    results = reranker.rerank("test query", chunks, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "chunk_2"
    assert results[0].original_score == 12.5
    assert results[1].chunk_id == "chunk_1"
    assert results[1].original_score == 15.2


@patch("sentence_transformers.CrossEncoder")
def test_reranker_with_hybrid_results(mock_cross_encoder_class):
    mock_model = Mock()
    mock_cross_encoder_class.return_value = mock_model
    mock_model.predict.return_value = [0.85]

    reranker = CrossEncoderReranker()

    chunks = [
        HybridChunk(
            chunk_id="chunk_1",
            text="Hybrid chunk",
            document_name="doc.pdf",
            page_number=1,
            vector_score=0.91,
            bm25_score=12.5,
            rrf_score=0.0328,
        ),
    ]

    results = reranker.rerank("test query", chunks, top_k=1)

    assert len(results) == 1
    assert results[0].original_score == 0.0328
