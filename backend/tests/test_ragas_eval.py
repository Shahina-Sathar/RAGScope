"""
Tests for RAGAS evaluation module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from app.evaluation.ragas_eval import RAGASEvaluator, RAGASEvaluationResult
from app.config.settings import settings


class TestRAGASEvaluationResult:
    """Tests for RAGASEvaluationResult dataclass."""

    def test_creation_with_all_scores(self):
        """Test creating result with all scores."""
        result = RAGASEvaluationResult(
            faithfulness_score=0.9,
            answer_relevancy_score=0.85,
            context_precision_score=0.8,
            context_recall_score=0.75,
        )
        assert result.faithfulness_score == 0.9
        assert result.answer_relevancy_score == 0.85
        assert result.context_precision_score == 0.8
        assert result.context_recall_score == 0.75
        assert result.error is None

    def test_creation_with_error(self):
        """Test creating result with error."""
        result = RAGASEvaluationResult(error="Evaluation failed")
        assert result.faithfulness_score is None
        assert result.answer_relevancy_score is None
        assert result.context_precision_score is None
        assert result.context_recall_score is None
        assert result.error == "Evaluation failed"

    def test_creation_with_partial_scores(self):
        """Test creating result with partial scores."""
        result = RAGASEvaluationResult(
            faithfulness_score=0.9,
            answer_relevancy_score=0.85,
        )
        assert result.faithfulness_score == 0.9
        assert result.answer_relevancy_score == 0.85
        assert result.context_precision_score is None
        assert result.context_recall_score is None


class TestRAGASEvaluator:
    """Tests for RAGASEvaluator class."""

    @patch("app.evaluation.ragas_eval.ChatOpenAI")
    @patch("app.evaluation.ragas_eval.OpenAIEmbeddings")
    def test_init_default_models(self, mock_embeddings_class, mock_llm_class):
        """Test initialization with default models."""
        evaluator = RAGASEvaluator()
        mock_llm_class.assert_called_once_with(
            model=settings.openai_llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        mock_embeddings_class.assert_called_once_with(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
        assert evaluator.llm is not None
        assert evaluator.embeddings is not None

    @patch("app.evaluation.ragas_eval.ChatOpenAI")
    @patch("app.evaluation.ragas_eval.OpenAIEmbeddings")
    def test_init_custom_models(self, mock_embeddings_class, mock_llm_class):
        """Test initialization with custom models."""
        custom_llm = Mock()
        custom_embeddings = Mock()
        evaluator = RAGASEvaluator(llm=custom_llm, embeddings=custom_embeddings)
        assert evaluator.llm == custom_llm
        assert evaluator.embeddings == custom_embeddings

    @patch("app.evaluation.ragas_eval.evaluate")
    @patch("app.evaluation.ragas_eval.ChatOpenAI")
    @patch("app.evaluation.ragas_eval.OpenAIEmbeddings")
    async def test_evaluate_success(self, mock_embeddings_class, mock_llm_class, mock_evaluate):
        """Test successful evaluation."""
        # Mock the evaluate function
        mock_result = Mock()
        mock_df = pd.DataFrame({
            "faithfulness": [0.9],
            "answer_relevancy": [0.85],
            "context_precision": [0.8],
            "context_recall": [0.75],
        })
        mock_result.to_pandas.return_value = mock_df
        mock_evaluate.return_value = mock_result

        evaluator = RAGASEvaluator()
        result = await evaluator.evaluate(
            question="What is the refund policy?",
            answer="The refund policy is 30 days.",
            contexts=["Refunds are processed within 30 days of purchase."],
            ground_truth="30 days",
        )

        assert result.faithfulness_score == 0.9
        assert result.answer_relevancy_score == 0.85
        assert result.context_precision_score == 0.8
        assert result.context_recall_score == 0.75
        assert result.error is None

    @patch("app.evaluation.ragas_eval.evaluate")
    @patch("app.evaluation.ragas_eval.ChatOpenAI")
    @patch("app.evaluation.ragas_eval.OpenAIEmbeddings")
    async def test_evaluate_without_ground_truth(self, mock_embeddings_class, mock_llm_class, mock_evaluate):
        """Test evaluation without ground truth."""
        # Mock the evaluate function
        mock_result = Mock()
        mock_df = pd.DataFrame({
            "faithfulness": [0.9],
            "answer_relevancy": [0.85],
            "context_precision": [0.8],
        })
        mock_result.to_pandas.return_value = mock_df
        mock_evaluate.return_value = mock_result

        evaluator = RAGASEvaluator()
        result = await evaluator.evaluate(
            question="What is the refund policy?",
            answer="The refund policy is 30 days.",
            contexts=["Refunds are processed within 30 days of purchase."],
        )

        assert result.faithfulness_score == 0.9
        assert result.answer_relevancy_score == 0.85
        assert result.context_precision_score == 0.8
        assert result.context_recall_score is None

    @patch("app.evaluation.ragas_eval.evaluate")
    @patch("app.evaluation.ragas_eval.ChatOpenAI")
    @patch("app.evaluation.ragas_eval.OpenAIEmbeddings")
    async def test_evaluate_with_error(self, mock_embeddings_class, mock_llm_class, mock_evaluate):
        """Test evaluation with error."""
        mock_evaluate.side_effect = Exception("Evaluation failed")

        evaluator = RAGASEvaluator()
        result = await evaluator.evaluate(
            question="What is the refund policy?",
            answer="The refund policy is 30 days.",
            contexts=["Refunds are processed within 30 days of purchase."],
        )

        assert result.error is not None
        assert "Evaluation failed" in result.error
        assert result.faithfulness_score is None

    @patch("app.evaluation.ragas_eval.evaluate")
    @patch("app.evaluation.ragas_eval.ChatOpenAI")
    @patch("app.evaluation.ragas_eval.OpenAIEmbeddings")
    async def test_evaluate_faithfulness(self, mock_embeddings_class, mock_llm_class, mock_evaluate):
        """Test evaluate_faithfulness method."""
        mock_result = Mock()
        mock_df = pd.DataFrame({"faithfulness": [0.9]})
        mock_result.to_pandas.return_value = mock_df
        mock_evaluate.return_value = mock_result

        evaluator = RAGASEvaluator()
        score = await evaluator.evaluate_faithfulness(
            question="What is the refund policy?",
            answer="The refund policy is 30 days.",
            contexts=["Refunds are processed within 30 days of purchase."],
        )

        assert score == 0.9

    @patch("app.evaluation.ragas_eval.evaluate")
    @patch("app.evaluation.ragas_eval.ChatOpenAI")
    @patch("app.evaluation.ragas_eval.OpenAIEmbeddings")
    async def test_evaluate_faithfulness_with_error(self, mock_embeddings_class, mock_llm_class, mock_evaluate):
        """Test evaluate_faithfulness with error."""
        mock_evaluate.side_effect = Exception("Failed")

        evaluator = RAGASEvaluator()
        score = await evaluator.evaluate_faithfulness(
            question="What is the refund policy?",
            answer="The refund policy is 30 days.",
            contexts=["Refunds are processed within 30 days of purchase."],
        )

        assert score is None

    @patch("app.evaluation.ragas_eval.evaluate")
    @patch("app.evaluation.ragas_eval.ChatOpenAI")
    @patch("app.evaluation.ragas_eval.OpenAIEmbeddings")
    async def test_evaluate_answer_relevancy(self, mock_embeddings_class, mock_llm_class, mock_evaluate):
        """Test evaluate_answer_relevancy method."""
        mock_result = Mock()
        mock_df = pd.DataFrame({"answer_relevancy": [0.85]})
        mock_result.to_pandas.return_value = mock_df
        mock_evaluate.return_value = mock_result

        evaluator = RAGASEvaluator()
        score = await evaluator.evaluate_answer_relevancy(
            question="What is the refund policy?",
            answer="The refund policy is 30 days.",
        )

        assert score == 0.85

    @patch("app.evaluation.ragas_eval.evaluate")
    @patch("app.evaluation.ragas_eval.ChatOpenAI")
    @patch("app.evaluation.ragas_eval.OpenAIEmbeddings")
    async def test_evaluate_context_precision(self, mock_embeddings_class, mock_llm_class, mock_evaluate):
        """Test evaluate_context_precision method."""
        mock_result = Mock()
        mock_df = pd.DataFrame({"context_precision": [0.8]})
        mock_result.to_pandas.return_value = mock_df
        mock_evaluate.return_value = mock_result

        evaluator = RAGASEvaluator()
        score = await evaluator.evaluate_context_precision(
            question="What is the refund policy?",
            contexts=["Refunds are processed within 30 days of purchase."],
            ground_truth="30 days",
        )

        assert score == 0.8

    @patch("app.evaluation.ragas_eval.evaluate")
    @patch("app.evaluation.ragas_eval.ChatOpenAI")
    @patch("app.evaluation.ragas_eval.OpenAIEmbeddings")
    async def test_evaluate_context_recall(self, mock_embeddings_class, mock_llm_class, mock_evaluate):
        """Test evaluate_context_recall method."""
        mock_result = Mock()
        mock_df = pd.DataFrame({"context_recall": [0.75]})
        mock_result.to_pandas.return_value = mock_df
        mock_evaluate.return_value = mock_result

        evaluator = RAGASEvaluator()
        score = await evaluator.evaluate_context_recall(
            question="What is the refund policy?",
            contexts=["Refunds are processed within 30 days of purchase."],
            ground_truth="30 days",
        )

        assert score == 0.75
