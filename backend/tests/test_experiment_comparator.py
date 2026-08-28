"""Tests for experiment comparator."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from app.experiments.comparison import ExperimentComparator
from app.models.schemas import Experiment, ExperimentResult


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = AsyncMock()
    # `execute` is awaited; its result is a plain Mock so sync chaining works
    result_mock = Mock()
    result_mock.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result_mock)
    return session


@pytest.fixture
def sample_experiment_1():
    """Create a sample experiment."""
    return Experiment(
        id="exp-1",
        name="Baseline",
        description="Baseline configuration",
        config={
            "top_k": 5,
            "retrieval_mode": "vector",
            "reranker_enabled": False,
        },
        status="completed",
        dataset_path="test_dataset.json",
        total_queries=10,
        completed_queries=10,
        avg_latency_ms=800.0,
        avg_faithfulness=0.75,
        avg_relevancy=0.70,
        avg_precision=0.65,
        avg_recall=0.60,
        avg_grounding_score=0.72,
        failure_analysis={"total_failures": 3, "failure_types": {"RETRIEVAL_FAILURE": 2, "GROUNDING_FAILURE": 1}},
        created_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_experiment_2():
    """Create a second sample experiment."""
    return Experiment(
        id="exp-2",
        name="With Reranking",
        description="Configuration with reranking",
        config={
            "top_k": 5,
            "retrieval_mode": "hybrid",
            "reranker_enabled": True,
        },
        status="completed",
        dataset_path="test_dataset.json",
        total_queries=10,
        completed_queries=10,
        avg_latency_ms=1200.0,
        avg_faithfulness=0.85,
        avg_relevancy=0.80,
        avg_precision=0.78,
        avg_recall=0.75,
        avg_grounding_score=0.82,
        failure_analysis={"total_failures": 1, "failure_types": {"HIGH_LATENCY": 1}},
        created_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_experiment_3():
    """Create a third sample experiment."""
    return Experiment(
        id="exp-3",
        name="Full Pipeline",
        description="Full pipeline with all features",
        config={
            "top_k": 5,
            "retrieval_mode": "hybrid",
            "reranker_enabled": True,
            "query_rewriting_enabled": True,
        },
        status="completed",
        dataset_path="test_dataset.json",
        total_queries=10,
        completed_queries=10,
        avg_latency_ms=1500.0,
        avg_faithfulness=0.90,
        avg_relevancy=0.88,
        avg_precision=0.85,
        avg_recall=0.82,
        avg_grounding_score=0.88,
        failure_analysis={"total_failures": 0, "failure_types": {}},
        created_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_results_exp_1():
    """Create sample results for experiment 1."""
    return [
        ExperimentResult(
            id="result-1-1",
            experiment_id="exp-1",
            query_id="q1",
            question="Q1?",
            expected_answer="A1",
            generated_answer="GA1",
            expected_sources=["doc1.pdf"],
            retrieved_sources=[{"chunk_id": "c1", "score": 0.9}],
            latency_ms=700,
            faithfulness_score=0.8,
            relevancy_score=0.75,
            precision_score=0.7,
            recall_score=0.65,
            grounding_score=0.75,
            is_grounded=True,
            failure_type=None,
            failure_details={},
        ),
        ExperimentResult(
            id="result-1-2",
            experiment_id="exp-1",
            query_id="q2",
            question="Q2?",
            expected_answer="A2",
            generated_answer="GA2",
            expected_sources=["doc2.pdf"],
            retrieved_sources=[],
            latency_ms=900,
            faithfulness_score=None,
            relevancy_score=None,
            precision_score=None,
            recall_score=None,
            grounding_score=None,
            is_grounded=None,
            failure_type="RETRIEVAL_FAILURE",
            failure_details={"reason": "No documents retrieved"},
        ),
    ]


@pytest.fixture
def sample_results_exp_2():
    """Create sample results for experiment 2."""
    return [
        ExperimentResult(
            id="result-2-1",
            experiment_id="exp-2",
            query_id="q1",
            question="Q1?",
            expected_answer="A1",
            generated_answer="GA1",
            expected_sources=["doc1.pdf"],
            retrieved_sources=[{"chunk_id": "c1", "score": 0.95}],
            latency_ms=1100,
            faithfulness_score=0.9,
            relevancy_score=0.85,
            precision_score=0.8,
            recall_score=0.78,
            grounding_score=0.85,
            is_grounded=True,
            failure_type=None,
            failure_details={},
        ),
        ExperimentResult(
            id="result-2-2",
            experiment_id="exp-2",
            query_id="q2",
            question="Q2?",
            expected_answer="A2",
            generated_answer="GA2",
            expected_sources=["doc2.pdf"],
            retrieved_sources=[{"chunk_id": "c2", "score": 0.88}],
            latency_ms=1300,
            faithfulness_score=0.88,
            relevancy_score=0.82,
            precision_score=0.79,
            recall_score=0.76,
            grounding_score=0.80,
            is_grounded=True,
            failure_type="HIGH_LATENCY",
            failure_details={"latency_ms": 1300},
        ),
    ]


class TestExperimentComparator:
    """Tests for ExperimentComparator class."""

    @pytest.mark.asyncio
    async def test_compare_experiments_insufficient_experiments(self, mock_session):
        """Test comparing with less than 2 experiments."""
        mock_session.execute.return_value.scalar_one_or_none.return_value = None
        
        comparator = ExperimentComparator(mock_session)
        
        with pytest.raises(ValueError, match="At least 2 experiments"):
            await comparator.compare_experiments(["exp-1"])

    @pytest.mark.asyncio
    async def test_compare_experiments_not_found(self, mock_session):
        """Test comparing when experiments are not found."""
        mock_session.execute.return_value.scalar_one_or_none.return_value = None
        
        comparator = ExperimentComparator(mock_session)
        
        with pytest.raises(ValueError, match="Could not load enough experiments"):
            await comparator.compare_experiments(["exp-1", "exp-2"])

    @pytest.mark.asyncio
    async def test_compare_experiments_success(
        self, mock_session, sample_experiment_1, sample_experiment_2,
        sample_results_exp_1, sample_results_exp_2
    ):
        """Test successful experiment comparison."""
        # compare_experiments executes: 2 experiment loads, then 2 result loads
        calls = {"n": 0}
        exp_results = [sample_experiment_1, sample_experiment_2]
        failure_results = [sample_results_exp_1, sample_results_exp_2]

        def mock_execute(query):
            idx = calls["n"]
            calls["n"] += 1
            mock_result = Mock()
            if idx < len(exp_results):
                mock_result.scalar_one_or_none.return_value = exp_results[idx]
            else:
                mock_result.scalars.return_value.all.return_value = failure_results[
                    idx - len(exp_results)
                ]
            return mock_result

        mock_session.execute.side_effect = mock_execute

        comparator = ExperimentComparator(mock_session)
        comparison = await comparator.compare_experiments(["exp-1", "exp-2"])

        assert "experiment_ids" in comparison
        assert "experiments" in comparison
        assert "metrics_comparison" in comparison
        assert "failures_comparison" in comparison
        assert "recommendations" in comparison

        assert len(comparison["experiments"]) == 2
        assert comparison["experiments"][0]["id"] == "exp-1"
        assert comparison["experiments"][1]["id"] == "exp-2"

    def test_compare_metrics(self, mock_session, sample_experiment_1, sample_experiment_2):
        """Test metrics comparison."""
        comparator = ExperimentComparator(mock_session)
        comparison = comparator._compare_metrics([sample_experiment_1, sample_experiment_2])
        
        assert "latency" in comparison
        assert "faithfulness" in comparison
        assert "relevancy" in comparison
        assert "precision" in comparison
        assert "recall" in comparison
        assert "grounding" in comparison
        
        # Check latency comparison
        assert comparison["latency"]["min"] == 800.0
        assert comparison["latency"]["max"] == 1200.0
        assert comparison["latency"]["best_experiment"] == "exp-1"
        
        # Check faithfulness comparison
        assert comparison["faithfulness"]["min"] == 0.75
        assert comparison["faithfulness"]["max"] == 0.85
        assert comparison["faithfulness"]["best_experiment"] == "exp-2"

    @pytest.mark.asyncio
    async def test_compare_failures(
        self, mock_session, sample_results_exp_1, sample_results_exp_2
    ):
        """Test failure comparison."""
        # Return results in the order requested by _compare_failures
        def make_result(results):
            mock_result = Mock()
            mock_result.scalars.return_value.all.return_value = results
            return mock_result

        mock_session.execute.side_effect = [
            make_result(sample_results_exp_1),
            make_result(sample_results_exp_2),
        ]
        
        comparator = ExperimentComparator(mock_session)
        comparison = await comparator._compare_failures(["exp-1", "exp-2"])
        
        assert "exp-1" in comparison
        assert "exp-2" in comparison
        
        assert comparison["exp-1"]["total_queries"] == 2
        assert comparison["exp-1"]["queries_with_failures"] == 1
        assert "RETRIEVAL_FAILURE" in comparison["exp-1"]["failure_counts"]
        
        assert comparison["exp-2"]["total_queries"] == 2
        assert comparison["exp-2"]["queries_with_failures"] == 1
        assert "HIGH_LATENCY" in comparison["exp-2"]["failure_counts"]

    def test_generate_recommendations_best_faithfulness(
        self, mock_session, sample_experiment_1, sample_experiment_2
    ):
        """Test recommendations for best faithfulness."""
        comparator = ExperimentComparator(mock_session)
        
        metrics_comparison = {
            "faithfulness": {
                "min": 0.75,
                "max": 0.85,
                "avg": 0.80,
                "best_experiment": "exp-2",
            }
        }
        
        failures_comparison = {
            "exp-1": {"total_queries": 10, "queries_with_failures": 3, "failure_counts": {}},
            "exp-2": {"total_queries": 10, "queries_with_failures": 1, "failure_counts": {}},
        }
        
        recommendations = comparator._generate_recommendations(
            [sample_experiment_1, sample_experiment_2],
            metrics_comparison,
            failures_comparison,
        )
        
        assert len(recommendations) > 0
        assert any("With Reranking" in rec for rec in recommendations)

    def test_generate_recommendations_high_latency(
        self, mock_session, sample_experiment_1, sample_experiment_2
    ):
        """Test recommendations for high latency."""
        comparator = ExperimentComparator(mock_session)
        
        metrics_comparison = {
            "latency": {
                "min": 800.0,
                "max": 6000.0,
                "avg": 3400.0,
                "best_experiment": "exp-1",
            }
        }
        
        failures_comparison = {}
        
        recommendations = comparator._generate_recommendations(
            [sample_experiment_1, sample_experiment_2],
            metrics_comparison,
            failures_comparison,
        )
        
        assert len(recommendations) > 0
        assert any("latency" in rec.lower() for rec in recommendations)

    def test_generate_recommendations_common_failures(
        self, mock_session, sample_experiment_1, sample_experiment_2
    ):
        """Test recommendations for common failure patterns."""
        comparator = ExperimentComparator(mock_session)
        
        metrics_comparison = {}
        
        failures_comparison = {
            "exp-1": {
                "total_queries": 10,
                "queries_with_failures": 3,
                "failure_counts": {"RETRIEVAL_FAILURE": 2, "GROUNDING_FAILURE": 1},
            },
            "exp-2": {
                "total_queries": 10,
                "queries_with_failures": 2,
                "failure_counts": {"RETRIEVAL_FAILURE": 2},
            },
        }
        
        recommendations = comparator._generate_recommendations(
            [sample_experiment_1, sample_experiment_2],
            metrics_comparison,
            failures_comparison,
        )
        
        assert len(recommendations) > 0
        assert any("RETRIEVAL_FAILURE" in rec for rec in recommendations)

    def test_generate_recommendations_low_grounding(
        self, mock_session, sample_experiment_1, sample_experiment_2
    ):
        """Test recommendations for low grounding scores."""
        comparator = ExperimentComparator(mock_session)
        
        metrics_comparison = {
            "grounding": {
                "min": 0.60,
                "max": 0.65,
                "avg": 0.625,
                "best_experiment": "exp-2",
            }
        }
        
        failures_comparison = {}
        
        recommendations = comparator._generate_recommendations(
            [sample_experiment_1, sample_experiment_2],
            metrics_comparison,
            failures_comparison,
        )
        
        assert len(recommendations) > 0
        assert any("grounding" in rec.lower() for rec in recommendations)
