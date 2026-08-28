from unittest.mock import AsyncMock, patch

import pytest

from app.retrieval.vector import RetrievedChunk
from app.tracing.tracer import QueryTracer


def test_tracer_initialization():
    tracer = QueryTracer(original_query="What is the refund policy?")

    assert tracer.original_query == "What is the refund policy?"
    assert tracer.trace_id is not None
    assert len(tracer.trace_id) == 36
    assert tracer.rewritten_query is None
    assert tracer.retrieval_results == []
    assert tracer.generation_result == {}
    assert tracer.retrieval_ms == 0
    assert tracer.generation_ms == 0


def test_tracer_record_retrieval():
    tracer = QueryTracer(original_query="Test query")

    chunks = [
        RetrievedChunk(
            chunk_id="chunk_1",
            text="Refund policy text",
            document_name="policy.pdf",
            page_number=1,
            similarity_score=0.95,
        ),
        RetrievedChunk(
            chunk_id="chunk_2",
            text="Another policy text",
            document_name="policy.pdf",
            page_number=2,
            similarity_score=0.85,
        ),
    ]

    tracer.start_retrieval()
    tracer.record_retrieval(chunks)

    assert len(tracer.retrieval_results) == 2
    assert tracer.retrieval_results[0]["chunk_id"] == "chunk_1"
    assert tracer.retrieval_results[0]["vector_score"] == 0.95
    assert tracer.retrieval_results[0]["final_rank"] == 1
    assert tracer.retrieval_results[1]["chunk_id"] == "chunk_2"
    assert tracer.retrieval_results[1]["final_rank"] == 2
    assert tracer.retrieval_ms >= 0


def test_tracer_record_generation():
    tracer = QueryTracer(original_query="Test query")

    tracer.start_generation()
    tracer.record_generation(
        answer="The refund policy is 30 days.",
        citations=["chunk_1", "chunk_2"],
        prompt_tokens=100,
        completion_tokens=20,
    )

    assert tracer.generation_result["answer"] == "The refund policy is 30 days."
    assert tracer.generation_result["citations"] == ["chunk_1", "chunk_2"]
    assert tracer.generation_result["prompt_tokens"] == 100
    assert tracer.generation_result["completion_tokens"] == 20
    assert tracer.generation_ms >= 0


@pytest.mark.asyncio
async def test_tracer_persist():
    tracer = QueryTracer(original_query="Test query")

    chunks = [
        RetrievedChunk(
            chunk_id="chunk_1",
            text="Test text",
            document_name="test.pdf",
            page_number=1,
            similarity_score=0.9,
        )
    ]

    tracer.start_retrieval()
    tracer.record_retrieval(chunks)
    tracer.start_generation()
    tracer.record_generation(answer="Test answer", citations=["chunk_1"])

    mock_session = AsyncMock()
    trace_id = await tracer.persist(mock_session)

    assert trace_id == tracer.trace_id
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_tracer_persist_with_latency():
    tracer = QueryTracer(original_query="Test query")

    tracer.retrieval_ms = 100
    tracer.generation_ms = 200

    mock_session = AsyncMock()
    await tracer.persist(mock_session)

    added_trace = mock_session.add.call_args[0][0]
    assert added_trace.latency["retrieval_ms"] == 100
    assert added_trace.latency["generation_ms"] == 200
    assert added_trace.latency["total_ms"] == 300
