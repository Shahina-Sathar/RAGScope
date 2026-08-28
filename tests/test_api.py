from unittest.mock import Mock, patch, AsyncMock

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.database.session import get_session


@pytest.fixture
def mock_session():
    session = AsyncMock()
    # `execute` is awaited (AsyncMock); its result is a plain Mock so the
    # sync `.scalars().all()` / `.scalar_one_or_none()` chain resolves.
    result_mock = Mock()
    result_mock.scalars.return_value.all.return_value = []
    result_mock.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result_mock)
    return session


@pytest.mark.asyncio
@patch("app.api.documents.get_session")
async def test_upload_document_rejects_non_pdf(mock_get_session, mock_session):
    mock_get_session.return_value = mock_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test.txt", b"content", "text/plain")}
        response = await client.post("/documents/upload", files=files)

    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


@pytest.mark.asyncio
@patch("app.api.query.get_session")
@patch("app.api.query.RetrievalFactory")
async def test_query_endpoint_returns_response(mock_factory_class, mock_get_session, mock_session):
    mock_get_session.return_value = mock_session

    # Mock retrieval: return one vector chunk
    mock_factory = Mock()
    mock_chunk = Mock()
    mock_chunk.chunk_id = "chunk_1"
    mock_chunk.text = "Refund policy text"
    mock_chunk.document_name = "policy.pdf"
    mock_chunk.page_number = 1
    mock_chunk.similarity_score = 0.91
    mock_factory.retrieve.return_value = ([mock_chunk], "vector")
    mock_factory_class.return_value = mock_factory

    # Mock tracer persistence to avoid DB writes
    with patch("app.api.query.QueryTracer") as mock_tracer_class:
        mock_tracer = Mock()
        mock_tracer.retrieval_ms = 50
        mock_tracer.reranker_ms = 0
        mock_tracer.generation_ms = 100
        mock_tracer.evaluation_ms = 0
        mock_tracer.persist = AsyncMock(return_value="123456789012345678901234567890123456")
        mock_tracer_class.return_value = mock_tracer

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/query", json={"question": "What is the refund policy?"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "latency_ms" in data
    assert "trace_id" in data
    assert data["answer"] == "I couldn't find sufficient information in the provided documents to answer this question."
    assert len(data["trace_id"]) == 36


@pytest.mark.asyncio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_list_documents_returns_empty_list(mock_session):
    async def override_get_session():
        yield mock_session

    app.dependency_overrides[get_session] = override_get_session
    try:
        mock_session.execute.return_value.scalars.return_value.all.return_value = []

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/documents")

        assert response.status_code == 200
        assert response.json() == []
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_delete_document_not_found(mock_session):
    async def override_get_session():
        yield mock_session

    app.dependency_overrides[get_session] = override_get_session
    try:
        mock_session.execute.return_value.scalar_one_or_none.return_value = None

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete("/documents/nonexistent-id")

        assert response.status_code == 404
        assert response.json()["detail"] == "Document not found"
    finally:
        app.dependency_overrides.clear()