from unittest.mock import Mock, patch
from types import SimpleNamespace

from app.retrieval.vector import VectorRetriever, RetrievedChunk


def make_query_response(matches):
    """Build a Pinecone-QueryResponse-like object with a .matches attribute."""
    return SimpleNamespace(matches=matches)


@patch("app.retrieval.vector.OpenAIEmbeddings")
@patch("app.retrieval.vector.Pinecone")
def test_vector_retriever_retrieve(mock_pinecone, mock_embeddings):
    mock_embeddings_instance = Mock()
    mock_embeddings.return_value = mock_embeddings_instance
    mock_embeddings_instance.embed_query.return_value = [0.1] * 1536

    mock_index = Mock()
    mock_pinecone_instance = Mock()
    mock_pinecone.return_value = mock_pinecone_instance
    mock_pinecone_instance.Index.return_value = mock_index

    mock_index.query.return_value = make_query_response(
        [
            SimpleNamespace(
                id="chunk_1",
                score=0.95,
                metadata={
                    "text": "Test text",
                    "document_name": "test.pdf",
                    "page_number": 1,
                    "chunk_id": "chunk_1",
                },
            )
        ]
    )

    retriever = VectorRetriever()
    results = retriever.retrieve("test query", top_k=5)

    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].chunk_id == "chunk_1"
    assert results[0].document_name == "test.pdf"
    assert results[0].page_number == 1
    assert results[0].similarity_score == 0.95


@patch("app.retrieval.vector.OpenAIEmbeddings")
@patch("app.retrieval.vector.Pinecone")
def test_vector_retriever_passes_correct_top_k(mock_pinecone, mock_embeddings):
    mock_embeddings_instance = Mock()
    mock_embeddings.return_value = mock_embeddings_instance
    mock_embeddings_instance.embed_query.return_value = [0.1] * 1536

    mock_index = Mock()
    mock_pinecone_instance = Mock()
    mock_pinecone.return_value = mock_pinecone_instance
    mock_pinecone_instance.Index.return_value = mock_index

    mock_index.query.return_value = make_query_response([])

    retriever = VectorRetriever()
    retriever.retrieve("test query", top_k=10)

    mock_index.query.assert_called_once()
    call_kwargs = mock_index.query.call_args[1]
    assert call_kwargs["top_k"] == 10


@patch("app.retrieval.vector.OpenAIEmbeddings")
@patch("app.retrieval.vector.Pinecone")
def test_vector_retriever_handles_empty_results(mock_pinecone, mock_embeddings):
    mock_embeddings_instance = Mock()
    mock_embeddings.return_value = mock_embeddings_instance
    mock_embeddings_instance.embed_query.return_value = [0.1] * 1536

    mock_index = Mock()
    mock_pinecone_instance = Mock()
    mock_pinecone.return_value = mock_pinecone_instance
    mock_pinecone_instance.Index.return_value = mock_index

    mock_index.query.return_value = make_query_response([])

    retriever = VectorRetriever()
    results = retriever.retrieve("test query", top_k=5)

    assert len(results) == 0