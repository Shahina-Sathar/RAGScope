"""Tests for experiment runner."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from app.experiments.runner import ExperimentRunner
from app.models.schemas import Experiment, ExperimentResult, ExperimentConfig


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = AsyncMock()
    result_mock = Mock()
    result_mock.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result_mock)
    return session


@pytest.fixture
def sample_experiment():
    """Create a sample experiment."""
    return Experiment(
        id="test-exp-1",
        name="Test Experiment",
        description="Test description",
        config={
            "top_k": 5,
            "retrieval_mode": "vector",
            "reranker_enabled": False,
            "evaluation_enabled": True,
            "grounding_check_enabled": True,
        },
        status="created",
        dataset_path="test_dataset.json",
        total_queries=10,
        completed_queries=0,
    )


@pytest.fixture
def sample_config():
    """Create a sample experiment config."""
    return ExperimentConfig(
        top_k=5,
        retrieval_mode="vector",
        reranker_enabled=False,
        evaluation_enabled=True,
        grounding_check_enabled=True,
    )


class TestExperimentRunner:
    """Tests for ExperimentRunner class."""

    @pytest.mark.asyncio
    async def test_run_experiment_not_found(self, mock_session):
        """Test running a non-existent experiment."""
        mock_session.execute.return_value.scalar_one_or_none.return_value = None

        runner = ExperimentRunner(mock_session)

        with pytest.raises(ValueError, match="Experiment .* not found"):
            await runner.run_experiment("non-existent-id")

    @pytest.mark.asyncio
    async def test_run_experiment_updates_status(self, mock_session, sample_experiment):
        """Test that running an experiment updates its status."""
        mock_session.execute.return_value.scalar_one_or_none.return_value = sample_experiment

        runner = ExperimentRunner(mock_session)

        with patch("app.experiments.runner.load_benchmark_dataset") as mock_load:
            mock_dataset = Mock()
            mock_dataset.questions = []
            mock_load.return_value = mock_dataset

            await runner.run_experiment("test-exp-1")

        assert mock_session.execute.call_count >= 2

    @pytest.mark.asyncio
    async def test_run_experiment_handles_dataset_error(self, mock_session, sample_experiment):
        """Test that dataset loading errors are handled."""
        mock_session.execute.return_value.scalar_one_or_none.return_value = sample_experiment

        runner = ExperimentRunner(mock_session)

        with patch("app.experiments.runner.load_benchmark_dataset") as mock_load:
            mock_load.side_effect = Exception("Dataset not found")

            with pytest.raises(Exception, match="Dataset not found"):
                await runner.run_experiment("test-exp-1")

        # Failure path commits the status change
        assert mock_session.commit.await_count >= 1

    @pytest.mark.asyncio
    async def test_run_query_with_evaluation(self, mock_session, sample_config):
        """Test running a single query with evaluation enabled."""
        runner = ExperimentRunner(mock_session)
        runner._retrieval_factory = Mock()

        mock_chunk = Mock()
        mock_chunk.chunk_id = "chunk-1"
        mock_chunk.document_name = "test.pdf"
        mock_chunk.page_number = 1
        mock_chunk.similarity_score = 0.9
        mock_chunk.text = "Test content"
        runner._retrieval_factory.retrieve.return_value = ([mock_chunk], "vector")

        with patch("app.generation.generator.AnswerGenerator") as mock_generator_class:
            mock_generator = Mock()
            mock_generator.generate.return_value = ("Test answer", ["chunk-1"])
            mock_generator_class.return_value = mock_generator

            mock_eval_result = Mock()
            mock_eval_result.faithfulness_score = 0.9
            mock_eval_result.answer_relevancy_score = 0.85
            mock_eval_result.context_precision_score = 0.8
            mock_eval_result.context_recall_score = 0.75
            mock_evaluator = Mock()
            mock_evaluator.evaluate.return_value = mock_eval_result
            mock_evaluator.evaluate = AsyncMock(return_value=mock_eval_result)
            runner._ragas_evaluator = mock_evaluator

            mock_grounding = Mock()
            mock_grounding.check_grounding.return_value = Mock(
                grounding_score=0.95, is_grounded=True
            )
            runner._grounding_checker = mock_grounding

            mock_query = Mock()
            mock_query.id = "q1"
            mock_query.question = "Test question?"
            mock_query.expected_answer = "Expected answer"
            mock_query.expected_sources = ["test.pdf"]

            result = await runner._run_query("exp-1", mock_query, sample_config)

        assert result["latency_ms"] >= 0
        assert result["faithfulness_score"] == 0.9
        assert result["relevancy_score"] == 0.85
        assert result["precision_score"] == 0.8
        assert result["recall_score"] == 0.75
        assert result["grounding_score"] == 0.95

    @pytest.mark.asyncio
    async def test_run_query_without_evaluation(self, mock_session, sample_config):
        """Test running a query without evaluation."""
        sample_config.evaluation_enabled = False
        sample_config.grounding_check_enabled = False

        runner = ExperimentRunner(mock_session)
        runner._retrieval_factory = Mock()
        runner._retrieval_factory.retrieve.return_value = ([], "vector")

        with patch("app.generation.generator.AnswerGenerator") as mock_generator_class:
            mock_generator = Mock()
            mock_generator.generate.return_value = ("Test answer", [])
            mock_generator_class.return_value = mock_generator

            mock_query = Mock()
            mock_query.id = "q1"
            mock_query.question = "Test question?"
            mock_query.expected_answer = "Expected answer"
            mock_query.expected_sources = []

            result = await runner._run_query("exp-1", mock_query, sample_config)

        assert result["faithfulness_score"] is None
        assert result["relevancy_score"] is None
        assert result["grounding_score"] is None

    @pytest.mark.asyncio
    async def test_run_query_with_reranking(self, mock_session, sample_config):
        """Test running a query with reranking enabled."""
        sample_config.reranker_enabled = True
        sample_config.reranker_top_k = 3

        runner = ExperimentRunner(mock_session)
        runner._retrieval_factory = Mock()

        mock_chunks = [
            Mock(
                chunk_id=f"chunk-{i}",
                text=f"Content {i}",
                document_name="test.pdf",
                page_number=i + 1,
                similarity_score=0.9 - i * 0.1,
            )
            for i in range(5)
        ]
        runner._retrieval_factory.retrieve.return_value = (mock_chunks, "vector")

        with patch("app.experiments.runner.CrossEncoderReranker") as mock_reranker_class:
            mock_reranker = Mock()
            mock_reranker.rerank.return_value = mock_chunks[:3]
            mock_reranker_class.return_value = mock_reranker

            with patch("app.generation.generator.AnswerGenerator") as mock_generator_class:
                mock_generator = Mock()
                mock_generator.generate.return_value = ("Test answer", [])
                mock_generator_class.return_value = mock_generator

                sample_config.evaluation_enabled = False
                sample_config.grounding_check_enabled = False

                mock_query = Mock()
                mock_query.id = "q1"
                mock_query.question = "Test question?"
                mock_query.expected_answer = "Expected answer"
                mock_query.expected_sources = []

                await runner._run_query("exp-1", mock_query, sample_config)

            mock_reranker.rerank.assert_called_once()