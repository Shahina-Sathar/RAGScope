"""
Tests for query rewriting strategies.
"""
import pytest
from unittest.mock import Mock, patch

from app.query.rewriter import (
    LLMQueryRewriter,
    HyDERewriter,
    QueryDecomposer,
    NoOpRewriter,
    get_rewriter,
)


class TestNoOpRewriter:
    """Tests for NoOpRewriter."""

    def test_returns_query_unchanged(self):
        """Test that NoOpRewriter returns the query unchanged."""
        rewriter = NoOpRewriter()
        query = "What is the refund policy?"
        result = rewriter.rewrite(query)
        assert result == query


class TestLLMQueryRewriter:
    """Tests for LLMQueryRewriter."""

    @patch("app.query.rewriter.ChatOpenAI")
    def test_rewrite_calls_llm(self, mock_chat_openai):
        """Test that rewrite calls the LLM with correct messages."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        mock_response = Mock()
        mock_response.content = "What is the current refund policy for customers?"
        mock_llm.invoke.return_value = mock_response

        rewriter = LLMQueryRewriter()
        result = rewriter.rewrite("What is the refund policy?")

        assert mock_llm.invoke.called
        assert result == "What is the current refund policy for customers?"

    @patch("app.query.rewriter.ChatOpenAI")
    def test_rewrite_strips_whitespace(self, mock_chat_openai):
        """Test that rewrite strips whitespace from result."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        mock_response = Mock()
        mock_response.content = "  Rewritten query  \n"
        mock_llm.invoke.return_value = mock_response

        rewriter = LLMQueryRewriter()
        result = rewriter.rewrite("Original query")

        assert result == "Rewritten query"


class TestHyDERewriter:
    """Tests for HyDERewriter."""

    @patch("app.query.rewriter.ChatOpenAI")
    def test_rewrite_generates_hypothetical_answer(self, mock_chat_openai):
        """Test that HyDE generates a hypothetical answer."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        mock_response = Mock()
        mock_response.content = "The refund policy allows returns within 30 days of purchase."
        mock_llm.invoke.return_value = mock_response

        rewriter = HyDERewriter()
        result = rewriter.rewrite("What is the refund policy?")

        assert mock_llm.invoke.called
        assert "refund policy" in result.lower()


class TestQueryDecomposer:
    """Tests for QueryDecomposer."""

    @patch("app.query.rewriter.ChatOpenAI")
    def test_decompose_returns_list(self, mock_chat_openai):
        """Test that decompose returns a list of sub-queries."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        mock_response = Mock()
        mock_response.content = "What is the refund period?\nHow do I request a refund?"
        mock_llm.invoke.return_value = mock_response

        rewriter = QueryDecomposer()
        result = rewriter.decompose("Tell me about the refund policy")

        assert isinstance(result, list)
        assert len(result) == 2
        assert "refund period" in result[0].lower()
        assert "request a refund" in result[1].lower()

    @patch("app.query.rewriter.ChatOpenAI")
    def test_decompose_filters_empty_lines(self, mock_chat_openai):
        """Test that decompose filters out empty lines."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        mock_response = Mock()
        mock_response.content = "Query 1\n\nQuery 2\n"
        mock_llm.invoke.return_value = mock_response

        rewriter = QueryDecomposer()
        result = rewriter.decompose("Complex query")

        assert len(result) == 2
        assert all(q.strip() for q in result)


class TestGetRewriter:
    """Tests for get_rewriter factory function."""

    def test_get_none_rewriter(self):
        """Test getting the none rewriter."""
        rewriter = get_rewriter("none")
        assert isinstance(rewriter, NoOpRewriter)

    def test_get_llm_rewriter(self):
        """Test getting the LLM rewriter."""
        rewriter = get_rewriter("llm")
        assert isinstance(rewriter, LLMQueryRewriter)

    def test_get_hyde_rewriter(self):
        """Test getting the HyDE rewriter."""
        rewriter = get_rewriter("hyde")
        assert isinstance(rewriter, HyDERewriter)

    def test_get_decompose_rewriter(self):
        """Test getting the decompose rewriter."""
        rewriter = get_rewriter("decompose")
        assert isinstance(rewriter, QueryDecomposer)

    def test_get_invalid_rewriter_raises_error(self):
        """Test that getting an invalid rewriter raises ValueError."""
        with pytest.raises(ValueError, match="Unknown rewriting strategy"):
            get_rewriter("invalid_strategy")
