"""
Tests for grounding detection module.
"""
import pytest
from unittest.mock import Mock, patch

from app.evaluation.grounding import GroundingResult, GroundingChecker, SimpleGroundingChecker
from app.config.settings import settings


class TestGroundingResult:
    """Tests for GroundingResult dataclass."""

    def test_creation_grounded(self):
        """Test creating grounded result."""
        result = GroundingResult(
            is_grounded=True,
            grounding_score=0.9,
            hallucinated_claims=[],
            supported_claims=["claim 1", "claim 2"],
            explanation="All claims are supported",
        )
        assert result.is_grounded is True
        assert result.grounding_score == 0.9
        assert len(result.hallucinated_claims) == 0
        assert len(result.supported_claims) == 2
        assert result.explanation == "All claims are supported"

    def test_creation_not_grounded(self):
        """Test creating not grounded result."""
        result = GroundingResult(
            is_grounded=False,
            grounding_score=0.3,
            hallucinated_claims=["claim 3"],
            supported_claims=["claim 1"],
            explanation="Some claims not supported",
        )
        assert result.is_grounded is False
        assert result.grounding_score == 0.3
        assert len(result.hallucinated_claims) == 1
        assert len(result.supported_claims) == 1

    def test_creation_without_explanation(self):
        """Test creating result without explanation."""
        result = GroundingResult(
            is_grounded=True,
            grounding_score=1.0,
            hallucinated_claims=[],
            supported_claims=["claim 1"],
        )
        assert result.explanation is None


class TestGroundingChecker:
    """Tests for GroundingChecker class."""

    @patch("app.evaluation.grounding.ChatOpenAI")
    def test_init_default_llm(self, mock_llm_class):
        """Test initialization with default LLM."""
        checker = GroundingChecker()
        mock_llm_class.assert_called_once_with(
            model=settings.openai_llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        assert checker.llm is not None

    @patch("app.evaluation.grounding.ChatOpenAI")
    def test_init_custom_llm(self, mock_llm_class):
        """Test initialization with custom LLM."""
        custom_llm = Mock()
        checker = GroundingChecker(llm=custom_llm)
        assert checker.llm == custom_llm

    @patch("app.evaluation.grounding.ChatOpenAI")
    def test_check_grounding_grounded(self, mock_llm_class):
        """Test grounding check with grounded answer."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        mock_response = Mock()
        mock_response.content = """{
            "is_grounded": true,
            "grounding_score": 0.9,
            "supported_claims": ["refunds within 30 days"],
            "hallucinated_claims": [],
            "explanation": "All claims supported"
        }"""
        mock_llm.invoke.return_value = mock_response

        checker = GroundingChecker()
        result = checker.check_grounding(
            question="What is the refund policy?",
            answer="Refunds are processed within 30 days.",
            contexts=["Refunds are processed within 30 days of purchase."],
        )

        assert result.is_grounded is True
        assert result.grounding_score == 0.9
        assert len(result.supported_claims) == 1
        assert len(result.hallucinated_claims) == 0

    @patch("app.evaluation.grounding.ChatOpenAI")
    def test_check_grounding_not_grounded(self, mock_llm_class):
        """Test grounding check with hallucinated answer."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        mock_response = Mock()
        mock_response.content = """{
            "is_grounded": false,
            "grounding_score": 0.3,
            "supported_claims": ["refunds"],
            "hallucinated_claims": ["7 days", "no questions asked"],
            "explanation": "Some claims not supported"
        }"""
        mock_llm.invoke.return_value = mock_response

        checker = GroundingChecker()
        result = checker.check_grounding(
            question="What is the refund policy?",
            answer="Refunds within 7 days, no questions asked.",
            contexts=["Refunds are processed within 30 days of purchase."],
        )

        assert result.is_grounded is False
        assert result.grounding_score == 0.3
        assert len(result.hallucinated_claims) == 2

    @patch("app.evaluation.grounding.ChatOpenAI")
    def test_check_grounding_with_markdown(self, mock_llm_class):
        """Test grounding check with markdown code block response."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        mock_response = Mock()
        mock_response.content = """```json
{
    "is_grounded": true,
    "grounding_score": 0.95,
    "supported_claims": ["claim 1"],
    "hallucinated_claims": [],
    "explanation": "All claims supported"
}
```"""
        mock_llm.invoke.return_value = mock_response

        checker = GroundingChecker()
        result = checker.check_grounding(
            question="Test question",
            answer="Test answer",
            contexts=["Test context"],
        )

        assert result.is_grounded is True
        assert result.grounding_score == 0.95

    @patch("app.evaluation.grounding.ChatOpenAI")
    def test_check_grounding_with_error(self, mock_llm_class):
        """Test grounding check with LLM error."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("LLM error")

        checker = GroundingChecker()
        result = checker.check_grounding(
            question="Test question",
            answer="Test answer",
            contexts=["Test context"],
        )

        assert result.is_grounded is False
        assert result.grounding_score == 0.0
        assert "LLM error" in result.explanation

    @patch("app.evaluation.grounding.ChatOpenAI")
    def test_is_grounded_above_threshold(self, mock_llm_class):
        """Test is_grounded with score above threshold."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        mock_response = Mock()
        mock_response.content = """{
            "is_grounded": true,
            "grounding_score": 0.8,
            "supported_claims": ["claim 1"],
            "hallucinated_claims": [],
            "explanation": "All claims supported"
        }"""
        mock_llm.invoke.return_value = mock_response

        checker = GroundingChecker()
        result = checker.is_grounded(
            question="Test question",
            answer="Test answer",
            contexts=["Test context"],
            threshold=0.7,
        )

        assert result is True

    @patch("app.evaluation.grounding.ChatOpenAI")
    def test_is_grounded_below_threshold(self, mock_llm_class):
        """Test is_grounded with score below threshold."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        mock_response = Mock()
        mock_response.content = """{
            "is_grounded": false,
            "grounding_score": 0.5,
            "supported_claims": ["claim 1"],
            "hallucinated_claims": ["claim 2"],
            "explanation": "Some claims not supported"
        }"""
        mock_llm.invoke.return_value = mock_response

        checker = GroundingChecker()
        result = checker.is_grounded(
            question="Test question",
            answer="Test answer",
            contexts=["Test context"],
            threshold=0.7,
        )

        assert result is False

    @patch("app.evaluation.grounding.ChatOpenAI")
    def test_get_hallucination_report(self, mock_llm_class):
        """Test get_hallucination_report method."""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        mock_response = Mock()
        mock_response.content = """{
            "is_grounded": false,
            "grounding_score": 0.6,
            "supported_claims": ["refunds within 30 days"],
            "hallucinated_claims": ["7 days"],
            "explanation": "Time period incorrect"
        }"""
        mock_llm.invoke.return_value = mock_response

        checker = GroundingChecker()
        report = checker.get_hallucination_report(
            question="What is the refund policy?",
            answer="Refunds within 7 days.",
            contexts=["Refunds are processed within 30 days of purchase."],
        )

        assert "Grounding Score: 0.60" in report
        assert "Is Grounded: No" in report
        assert "refunds within 30 days" in report
        assert "7 days" in report
        assert "Time period incorrect" in report


class TestSimpleGroundingChecker:
    """Tests for SimpleGroundingChecker class."""

    def test_check_grounding_all_keywords_found(self):
        """Test grounding check with all keywords found."""
        checker = SimpleGroundingChecker()
        result = checker.check_grounding(
            question="What is the refund policy?",
            answer="Refunds are processed within 30 days.",
            contexts=["Refunds are processed within 30 days of purchase."],
        )

        assert result.is_grounded is True
        assert result.grounding_score >= 0.7

    def test_check_grounding_some_keywords_missing(self):
        """Test grounding check with some keywords missing."""
        checker = SimpleGroundingChecker()
        result = checker.check_grounding(
            question="What is the refund policy?",
            answer="Refunds are processed within 7 days with no questions asked.",
            contexts=["Refunds are processed within 30 days of purchase."],
        )

        # Some keywords like "7", "days", "questions" won't be in context
        assert result.grounding_score < 1.0

    def test_check_grounding_no_keywords(self):
        """Test grounding check with no meaningful keywords."""
        checker = SimpleGroundingChecker()
        result = checker.check_grounding(
            question="What?",
            answer="Yes.",
            contexts=["Some context."],
        )

        # Should handle gracefully
        assert result.grounding_score >= 0.0

    def test_check_grounding_empty_context(self):
        """Test grounding check with empty context."""
        checker = SimpleGroundingChecker()
        result = checker.check_grounding(
            question="What is the refund policy?",
            answer="Refunds are processed within 30 days.",
            contexts=[],
        )

        # No context means no keywords found
        assert result.grounding_score == 0.0
        assert result.is_grounded is False

    def test_check_grounding_case_insensitive(self):
        """Test that grounding check is case insensitive."""
        checker = SimpleGroundingChecker()
        result = checker.check_grounding(
            question="What is the REFUND policy?",
            answer="REFUNDS are processed within 30 DAYS.",
            contexts=["refunds are processed within 30 days of purchase."],
        )

        # Should match despite case differences
        assert result.grounding_score > 0.0
