"""Tests for failure analyzer."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from app.experiments.analyzer import FailureAnalyzer
from app.models.schemas import Experiment, ExperimentResult


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = AsyncMock()
    # `execute` is awaited (AsyncMock); its result is a plain Mock so the
    # sync `.scalars().all()` chain resolves without returning coroutines.
    result_mock = Mock()
    result_mock.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result_mock)
    return session


@pytest.fixture
def sample_result_no_failure():
    """Create a sample result with no failures."""
    return ExperimentResult(
        id="result-1",
        experiment_id="exp-1",
        query_id="q1",
        question="What is the refund policy?",
        expected_answer="30 days",
        generated_answer="The refund policy is 30 days.",
        expected_sources=["policy.pdf"],
        retrieved_sources=[
            {"chunk_id": "chunk-1", "document_name": "policy.pdf", "page_number": 1, "score": 0.9}
        ],
        latency_ms=500,
        faithfulness_score=0.95,
        relevancy_score=0.90,
        precision_score=0.85,
        recall_score=0.80,
        grounding_score=0.92,
        is_grounded=True,
        failure_type=None,
        failure_details={},
    )


@pytest.fixture
def sample_result_retrieval_failure():
    """Create a sample result with retrieval failure."""
    return ExperimentResult(
        id="result-2",
        experiment_id="exp-1",
        query_id="q2",
        question="What is the API limit?",
        expected_answer="1000 requests/hour",
        generated_answer="I don't know.",
        expected_sources=["api.pdf"],
        retrieved_sources=[],
        latency_ms=200,
        faithfulness_score=None,
        relevancy_score=None,
        precision_score=None,
        recall_score=None,
        grounding_score=None,
        is_grounded=None,
        failure_type=None,
        failure_details={},
    )


@pytest.fixture
def sample_result_grounding_failure():
    """Create a sample result with grounding failure."""
    return ExperimentResult(
        id="result-3",
        experiment_id="exp-1",
        query_id="q3",
        question="What is the pricing?",
        expected_answer="$99/month",
        generated_answer="The pricing is $50/month.",
        expected_sources=["pricing.pdf"],
        retrieved_sources=[
            {"chunk_id": "chunk-1", "document_name": "pricing.pdf", "page_number": 1, "score": 0.8}
        ],
        latency_ms=600,
        faithfulness_score=0.85,
        relevancy_score=0.80,
        precision_score=0.75,
        recall_score=0.70,
        grounding_score=0.45,
        is_grounded=False,
        failure_type=None,
        failure_details={},
    )


@pytest.fixture
def sample_result_low_metrics():
    """Create a sample result with low evaluation metrics."""
    return ExperimentResult(
        id="result-4",
        experiment_id="exp-1",
        query_id="q4",
        question="What are the features?",
        expected_answer="Feature A, B, C",
        generated_answer="Some features are available.",
        expected_sources=["features.pdf"],
        retrieved_sources=[
            {"chunk_id": "chunk-1", "document_name": "features.pdf", "page_number": 1, "score": 0.6}
        ],
        latency_ms=800,
        faithfulness_score=0.60,
        relevancy_score=0.55,
        precision_score=0.50,
        recall_score=0.45,
        grounding_score=0.65,
        is_grounded=True,
        failure_type=None,
        failure_details={},
    )


@pytest.fixture
def sample_result_high_latency():
    """Create a sample result with high latency."""
    return ExperimentResult(
        id="result-5",
        experiment_id="exp-1",
        query_id="q5",
        question="Complex question?",
        expected_answer="Complex answer",
        generated_answer="Complex generated answer",
        expected_sources=["doc.pdf"],
        retrieved_sources=[
            {"chunk_id": "chunk-1", "document_name": "doc.pdf", "page_number": 1, "score": 0.85}
        ],
        latency_ms=6000,
        faithfulness_score=0.90,
        relevancy_score=0.85,
        precision_score=0.80,
        recall_score=0.75,
        grounding_score=0.88,
        is_grounded=True,
        failure_type=None,
        failure_details={},
    )


class TestFailureAnalyzer:
    """Tests for FailureAnalyzer class."""

    @pytest.mark.asyncio
    async def test_analyze_experiment_no_results(self, mock_session):
        """Test analyzing an experiment with no results."""
        mock_session.execute.return_value.scalars.return_value.all.return_value = []
        
        analyzer = FailureAnalyzer(mock_session)
        result = await analyzer.analyze_experiment("exp-1")
        
        assert result["total_queries"] == 0
        assert result["failures"] == []
        assert result["failure_types"] == {}

    @pytest.mark.asyncio
    async def test_analyze_experiment_no_failures(
        self, mock_session, sample_result_no_failure
    ):
        """Test analyzing an experiment with no failures."""
        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            sample_result_no_failure
        ]
        
        analyzer = FailureAnalyzer(mock_session)
        result = await analyzer.analyze_experiment("exp-1")
        
        assert result["total_queries"] == 1
        assert result["total_failures"] == 0
        assert result["failure_rate"] == 0.0
        assert result["failure_types"] == {}

    @pytest.mark.asyncio
    async def test_analyze_experiment_retrieval_failure(
        self, mock_session, sample_result_retrieval_failure
    ):
        """Test analyzing an experiment with retrieval failure."""
        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            sample_result_retrieval_failure
        ]
        
        analyzer = FailureAnalyzer(mock_session)
        result = await analyzer.analyze_experiment("exp-1")
        
        assert result["total_queries"] == 1
        assert result["total_failures"] == 1
        assert result["failure_rate"] == 1.0
        assert "RETRIEVAL_FAILURE" in result["failure_types"]
        assert result["failures"][0]["failure_type"] == "RETRIEVAL_FAILURE"

    @pytest.mark.asyncio
    async def test_analyze_experiment_grounding_failure(
        self, mock_session, sample_result_grounding_failure
    ):
        """Test analyzing an experiment with grounding failure."""
        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            sample_result_grounding_failure
        ]
        
        analyzer = FailureAnalyzer(mock_session)
        result = await analyzer.analyze_experiment("exp-1")
        
        assert result["total_queries"] == 1
        assert result["total_failures"] == 1
        assert "GROUNDING_FAILURE" in result["failure_types"]

    @pytest.mark.asyncio
    async def test_analyze_experiment_low_metrics(
        self, mock_session, sample_result_low_metrics
    ):
        """Test analyzing an experiment with low evaluation metrics."""
        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            sample_result_low_metrics
        ]
        
        analyzer = FailureAnalyzer(mock_session)
        result = await analyzer.analyze_experiment("exp-1")
        
        assert result["total_queries"] == 1
        assert result["total_failures"] == 1
        
        # Check for multiple failure types
        failure_type = result["failures"][0]["failure_type"]
        assert "FAITHFULNESS_FAILURE" in failure_type
        assert "RELEVANCY_FAILURE" in failure_type
        assert "PRECISION_FAILURE" in failure_type
        assert "RECALL_FAILURE" in failure_type

    @pytest.mark.asyncio
    async def test_analyze_experiment_high_latency(
        self, mock_session, sample_result_high_latency
    ):
        """Test analyzing an experiment with high latency."""
        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            sample_result_high_latency
        ]
        
        analyzer = FailureAnalyzer(mock_session)
        result = await analyzer.analyze_experiment("exp-1")
        
        assert result["total_queries"] == 1
        assert result["total_failures"] == 1
        assert "HIGH_LATENCY" in result["failure_types"]

    @pytest.mark.asyncio
    async def test_analyze_experiment_multiple_failures(
        self,
        mock_session,
        sample_result_no_failure,
        sample_result_retrieval_failure,
        sample_result_grounding_failure,
    ):
        """Test analyzing an experiment with multiple types of failures."""
        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            sample_result_no_failure,
            sample_result_retrieval_failure,
            sample_result_grounding_failure,
        ]
        
        analyzer = FailureAnalyzer(mock_session)
        result = await analyzer.analyze_experiment("exp-1")
        
        assert result["total_queries"] == 3
        assert result["total_failures"] == 2
        assert result["failure_rate"] == pytest.approx(2/3, rel=1e-2)
        assert "RETRIEVAL_FAILURE" in result["failure_types"]
        assert "GROUNDING_FAILURE" in result["failure_types"]

    @pytest.mark.asyncio
    async def test_get_failure_summary(self, mock_session, sample_result_no_failure, sample_result_retrieval_failure):
        """Test getting failure summary."""
        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            sample_result_no_failure,
            sample_result_retrieval_failure,
        ]
        
        # Set failure type on the result
        sample_result_retrieval_failure.failure_type = "RETRIEVAL_FAILURE"
        
        analyzer = FailureAnalyzer(mock_session)
        summary = await analyzer.get_failure_summary("exp-1")
        
        assert summary["total_queries"] == 2
        assert summary["queries_with_failures"] == 1
        assert summary["failure_rate"] == 0.5
        assert "RETRIEVAL_FAILURE" in summary["failure_counts"]
        assert summary["most_common_failure"] == "RETRIEVAL_FAILURE"

    def test_analyze_result_no_failure(self, mock_session, sample_result_no_failure):
        """Test analyzing a single result with no failures."""
        analyzer = FailureAnalyzer(mock_session)
        result = analyzer._analyze_result(sample_result_no_failure)
        
        assert result["failure_type"] is None
        assert result["details"] == {}
        assert result["recommendation"] is None

    def test_analyze_result_retrieval_failure(self, mock_session, sample_result_retrieval_failure):
        """Test analyzing a single result with retrieval failure."""
        analyzer = FailureAnalyzer(mock_session)
        result = analyzer._analyze_result(sample_result_retrieval_failure)
        
        assert result["failure_type"] == "RETRIEVAL_FAILURE"
        assert "No documents retrieved" in result["details"]["reason"]
        assert result["recommendation"] is not None

    def test_analyze_result_low_retrieval_scores(self, mock_session):
        """Test analyzing a result with low retrieval scores."""
        result = ExperimentResult(
            id="result-low-score",
            experiment_id="exp-1",
            query_id="q1",
            question="Test?",
            expected_answer="Answer",
            generated_answer="Generated",
            expected_sources=[],
            retrieved_sources=[
                {"chunk_id": "chunk-1", "document_name": "doc.pdf", "page_number": 1, "score": 0.3}
            ],
            latency_ms=500,
            faithfulness_score=None,
            relevancy_score=None,
            precision_score=None,
            recall_score=None,
            grounding_score=None,
            is_grounded=None,
            failure_type=None,
            failure_details={},
        )
        
        analyzer = FailureAnalyzer(mock_session)
        analysis = analyzer._analyze_result(result)
        
        assert analysis["failure_type"] == "LOW_RETRIEVAL_QUALITY"
        assert analysis["details"]["average_score"] == 0.3

    def test_analyze_result_compound_failure(self, mock_session):
        """Test analyzing a result with multiple failures."""
        result = ExperimentResult(
            id="result-compound",
            experiment_id="exp-1",
            query_id="q1",
            question="Test?",
            expected_answer="Answer",
            generated_answer="Generated",
            expected_sources=["doc.pdf"],
            retrieved_sources=[
                {"chunk_id": "chunk-1", "document_name": "doc.pdf", "page_number": 1, "score": 0.4}
            ],
            latency_ms=6000,
            faithfulness_score=0.5,
            relevancy_score=0.5,
            precision_score=0.5,
            recall_score=0.5,
            grounding_score=0.4,
            is_grounded=False,
            failure_type=None,
            failure_details={},
        )
        
        analyzer = FailureAnalyzer(mock_session)
        analysis = analyzer._analyze_result(result)
        
        # Should have multiple failure types
        failure_type = analysis["failure_type"]
        assert "+" in failure_type  # Compound failure
        assert "LOW_RETRIEVAL_QUALITY" in failure_type
        assert "GROUNDING_FAILURE" in failure_type
        assert "FAITHFULNESS_FAILURE" in failure_type
        assert "HIGH_LATENCY" in failure_type
