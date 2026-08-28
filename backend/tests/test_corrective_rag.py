"""
Tests for corrective RAG workflow.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from app.query.corrective_rag import CorrectiveRAG, CorrectiveRAGState
from app.retrieval.vector import RetrievedChunk


class TestCorrectiveRAGState:
    """Tests for CorrectiveRAGState."""

    def test_state_has_required_fields(self):
        """Test that state has all required fields."""
        state = CorrectiveRAGState(
            query="test query",
            rewritten_query=None,
            documents=[],
            relevance_scores=[],
            is_relevant=False,
            retry_count=0,
            max_retries=2,
            final_answer=None,
        )
        
        assert state["query"] == "test query"
        assert state["rewritten_query"] is None
        assert state["documents"] == []
        assert state["relevance_scores"] == []
        assert state["is_relevant"] is False
        assert state["retry_count"] == 0
        assert state["max_retries"] == 2
        assert state["final_answer"] is None


class TestCorrectiveRAG:
    """Tests for CorrectiveRAG workflow."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        retriever = Mock()
        retriever.retrieve.return_value = [
            RetrievedChunk(
                chunk_id="chunk_1",
                text="This is relevant content about refunds",
                document_name="policy.pdf",
                page_number=1,
                similarity_score=0.85,
            )
        ]
        return retriever

    @pytest.fixture
    def corrective_rag(self, mock_retriever):
        """Create a CorrectiveRAG instance with mock retriever."""
        with patch("app.query.corrective_rag.ChatOpenAI"):
            rag = CorrectiveRAG(
                retriever=mock_retriever,
                max_retries=2,
                relevance_threshold=0.7,
            )
        return rag

    def test_init_sets_parameters(self, mock_retriever):
        """Test that initialization sets parameters correctly."""
        with patch("app.query.corrective_rag.ChatOpenAI"):
            rag = CorrectiveRAG(
                retriever=mock_retriever,
                max_retries=3,
                relevance_threshold=0.8,
            )
        
        assert rag.retriever == mock_retriever
        assert rag.max_retries == 3
        assert rag.relevance_threshold == 0.8

    def test_retrieve_node_retrieves_documents(self, corrective_rag, mock_retriever):
        """Test that retrieve node calls the retriever."""
        state = {
            "query": "test query",
            "rewritten_query": None,
            "documents": [],
            "relevance_scores": [],
            "is_relevant": False,
            "retry_count": 0,
            "max_retries": 2,
            "final_answer": None,
        }
        
        result = corrective_rag._retrieve_node(state)
        
        mock_retriever.retrieve.assert_called_once_with("test query", top_k=5)
        assert len(result["documents"]) == 1

    def test_retrieve_node_uses_rewritten_query(self, corrective_rag, mock_retriever):
        """Test that retrieve node uses rewritten query if available."""
        state = {
            "query": "original query",
            "rewritten_query": "rewritten query",
            "documents": [],
            "relevance_scores": [],
            "is_relevant": False,
            "retry_count": 0,
            "max_retries": 2,
            "final_answer": None,
        }
        
        result = corrective_rag._retrieve_node(state)
        
        mock_retriever.retrieve.assert_called_once_with("rewritten query", top_k=5)

    def test_grade_relevance_node_with_high_scores(self, corrective_rag):
        """Test grading with high relevance scores."""
        state = {
            "query": "test query",
            "rewritten_query": None,
            "documents": [
                RetrievedChunk(
                    chunk_id="chunk_1",
                    text="content",
                    document_name="doc.pdf",
                    page_number=1,
                    similarity_score=0.85,
                ),
                RetrievedChunk(
                    chunk_id="chunk_2",
                    text="content",
                    document_name="doc.pdf",
                    page_number=2,
                    similarity_score=0.90,
                ),
            ],
            "relevance_scores": [],
            "is_relevant": False,
            "retry_count": 0,
            "max_retries": 2,
            "final_answer": None,
        }
        
        result = corrective_rag._grade_relevance_node(state)
        
        assert result["is_relevant"] is True
        assert len(result["relevance_scores"]) == 2
        assert result["relevance_scores"][0] == 0.85
        assert result["relevance_scores"][1] == 0.90

    def test_grade_relevance_node_with_low_scores(self, corrective_rag):
        """Test grading with low relevance scores."""
        state = {
            "query": "test query",
            "rewritten_query": None,
            "documents": [
                RetrievedChunk(
                    chunk_id="chunk_1",
                    text="content",
                    document_name="doc.pdf",
                    page_number=1,
                    similarity_score=0.3,
                ),
            ],
            "relevance_scores": [],
            "is_relevant": False,
            "retry_count": 0,
            "max_retries": 2,
            "final_answer": None,
        }
        
        result = corrective_rag._grade_relevance_node(state)
        
        assert result["is_relevant"] is False

    def test_grade_relevance_node_with_no_documents(self, corrective_rag):
        """Test grading with no documents."""
        state = {
            "query": "test query",
            "rewritten_query": None,
            "documents": [],
            "relevance_scores": [],
            "is_relevant": False,
            "retry_count": 0,
            "max_retries": 2,
            "final_answer": None,
        }
        
        result = corrective_rag._grade_relevance_node(state)
        
        assert result["is_relevant"] is False
        assert result["relevance_scores"] == []

    def test_should_retry_when_not_relevant_and_retries_remain(self, corrective_rag):
        """Test should_retry returns retry when not relevant and retries remain."""
        state = {
            "query": "test",
            "rewritten_query": None,
            "documents": [],
            "relevance_scores": [],
            "is_relevant": False,
            "retry_count": 0,
            "max_retries": 2,
            "final_answer": None,
        }
        
        result = corrective_rag._should_retry(state)
        assert result == "retry"

    def test_should_retry_generate_when_relevant(self, corrective_rag):
        """Test should_retry returns generate when relevant."""
        state = {
            "query": "test",
            "rewritten_query": None,
            "documents": [],
            "relevance_scores": [],
            "is_relevant": True,
            "retry_count": 0,
            "max_retries": 2,
            "final_answer": None,
        }
        
        result = corrective_rag._should_retry(state)
        assert result == "generate"

    def test_should_retry_generate_when_max_retries_reached(self, corrective_rag):
        """Test should_retry returns generate when max retries reached."""
        state = {
            "query": "test",
            "rewritten_query": None,
            "documents": [],
            "relevance_scores": [],
            "is_relevant": False,
            "retry_count": 2,
            "max_retries": 2,
            "final_answer": None,
        }
        
        result = corrective_rag._should_retry(state)
        assert result == "generate"

    @patch("app.query.corrective_rag.LLMQueryRewriter")
    def test_rewrite_query_node(self, mock_rewriter_class, corrective_rag):
        """Test rewrite query node calls the rewriter."""
        mock_rewriter = Mock()
        mock_rewriter.rewrite.return_value = "improved query"
        mock_rewriter_class.return_value = mock_rewriter
        
        # Recreate corrective_rag with mocked rewriter
        with patch("app.query.corrective_rag.ChatOpenAI"):
            rag = CorrectiveRAG(
                retriever=corrective_rag.retriever,
                max_retries=2,
                relevance_threshold=0.7,
            )
            rag.query_rewriter = mock_rewriter
        
        state = {
            "query": "original query",
            "rewritten_query": None,
            "documents": [],
            "relevance_scores": [],
            "is_relevant": False,
            "retry_count": 0,
            "max_retries": 2,
            "final_answer": None,
        }
        
        result = rag._rewrite_query_node(state)
        
        mock_rewriter.rewrite.assert_called_once_with("original query")
        assert result["rewritten_query"] == "improved query"
        assert result["retry_count"] == 1

    @patch("app.query.corrective_rag.ChatOpenAI")
    def test_generate_answer_node_with_documents(self, mock_chat_openai, corrective_rag):
        """Test generate answer node with documents."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        mock_response = Mock()
        mock_response.content = "The answer is 30 days."
        mock_llm.invoke.return_value = mock_response
        
        # Recreate with mocked LLM
        with patch("app.query.corrective_rag.ChatOpenAI") as mock_co:
            mock_co.return_value = mock_llm
            rag = CorrectiveRAG(
                retriever=corrective_rag.retriever,
                max_retries=2,
                relevance_threshold=0.7,
            )
        
        state = {
            "query": "What is the refund period?",
            "rewritten_query": None,
            "documents": [
                RetrievedChunk(
                    chunk_id="chunk_1",
                    text="Refunds are processed within 30 days.",
                    document_name="policy.pdf",
                    page_number=1,
                    similarity_score=0.85,
                ),
            ],
            "relevance_scores": [0.85],
            "is_relevant": True,
            "retry_count": 0,
            "max_retries": 2,
            "final_answer": None,
        }
        
        result = rag._generate_answer_node(state)
        
        assert "final_answer" in result
        assert mock_llm.invoke.called

    @patch("app.query.corrective_rag.ChatOpenAI")
    def test_generate_answer_node_without_documents(self, mock_chat_openai, corrective_rag):
        """Test generate answer node without documents."""
        state = {
            "query": "What is the refund period?",
            "rewritten_query": None,
            "documents": [],
            "relevance_scores": [],
            "is_relevant": False,
            "retry_count": 2,
            "max_retries": 2,
            "final_answer": None,
        }
        
        result = corrective_rag._generate_answer_node(state)
        
        assert "couldn't find relevant information" in result["final_answer"].lower()

    def test_run_executes_workflow(self, mock_retriever):
        """Test that run executes the full workflow."""
        with patch("app.query.corrective_rag.ChatOpenAI") as mock_chat:
            mock_llm = Mock()
            mock_chat.return_value = mock_llm
            mock_response = Mock()
            mock_response.content = "The answer is 30 days."
            mock_llm.invoke.return_value = mock_response
            
            rag = CorrectiveRAG(
                retriever=mock_retriever,
                max_retries=2,
                relevance_threshold=0.7,
            )
            
            result = rag.run("What is the refund period?")
            
            assert "final_answer" in result
            assert "documents" in result
            assert "retry_count" in result
            assert "rewritten_query" in result
