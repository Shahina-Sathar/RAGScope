"""
Tests for retrieval evaluation metrics.
"""
import pytest

from app.evaluation.metrics import (
    hit_at_k,
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    average_precision_at_k,
    mean_average_precision_at_k,
)


class TestHitAtK:
    """Tests for Hit@K metric."""
    
    def test_hit_at_k_found_at_first_position(self):
        """Test Hit@K when relevant document is at rank 1."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1"]
        assert hit_at_k(retrieved, relevant, k=1) == 1
    
    def test_hit_at_k_found_at_third_position(self):
        """Test Hit@K when relevant document is at rank 3."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc3"]
        assert hit_at_k(retrieved, relevant, k=1) == 0
        assert hit_at_k(retrieved, relevant, k=2) == 0
        assert hit_at_k(retrieved, relevant, k=3) == 1
    
    def test_hit_at_k_not_found(self):
        """Test Hit@K when no relevant documents in top-K."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc5", "doc6"]
        assert hit_at_k(retrieved, relevant, k=3) == 0
    
    def test_hit_at_k_multiple_relevant(self):
        """Test Hit@K with multiple relevant documents."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc2", "doc4"]
        assert hit_at_k(retrieved, relevant, k=1) == 0
        assert hit_at_k(retrieved, relevant, k=2) == 1
    
    def test_hit_at_k_zero_k(self):
        """Test Hit@K with k=0."""
        retrieved = ["doc1", "doc2"]
        relevant = ["doc1"]
        assert hit_at_k(retrieved, relevant, k=0) == 0
    
    def test_hit_at_k_empty_relevant(self):
        """Test Hit@K with no relevant documents."""
        retrieved = ["doc1", "doc2"]
        relevant = []
        assert hit_at_k(retrieved, relevant, k=2) == 0


class TestRecallAtK:
    """Tests for Recall@K metric."""
    
    def test_recall_at_k_all_found(self):
        """Test Recall@K when all relevant documents are found."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2"]
        assert recall_at_k(retrieved, relevant, k=3) == 1.0
    
    def test_recall_at_k_half_found(self):
        """Test Recall@K when half of relevant documents are found."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc4"]
        assert recall_at_k(retrieved, relevant, k=3) == 0.5
    
    def test_recall_at_k_none_found(self):
        """Test Recall@K when no relevant documents are found."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc5", "doc6"]
        assert recall_at_k(retrieved, relevant, k=3) == 0.0
    
    def test_recall_at_k_partial_k(self):
        """Test Recall@K with limited k."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = ["doc1", "doc3"]
        assert recall_at_k(retrieved, relevant, k=2) == 0.5  # Only doc1 in top-2
        assert recall_at_k(retrieved, relevant, k=3) == 1.0  # Both in top-3
    
    def test_recall_at_k_empty_relevant(self):
        """Test Recall@K with no relevant documents."""
        retrieved = ["doc1", "doc2"]
        relevant = []
        assert recall_at_k(retrieved, relevant, k=2) == 0.0


class TestPrecisionAtK:
    """Tests for Precision@K metric."""
    
    def test_precision_at_k_all_relevant(self):
        """Test Precision@K when all retrieved are relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc3"]
        assert precision_at_k(retrieved, relevant, k=3) == 1.0
    
    def test_precision_at_k_half_relevant(self):
        """Test Precision@K when half of retrieved are relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc4"]
        assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(1/3)
    
    def test_precision_at_k_none_relevant(self):
        """Test Precision@K when none of retrieved are relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc5", "doc6"]
        assert precision_at_k(retrieved, relevant, k=3) == 0.0


class TestReciprocalRank:
    """Tests for Reciprocal Rank metric."""
    
    def test_rr_first_position(self):
        """Test RR when relevant document is at rank 1."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1"]
        assert reciprocal_rank(retrieved, relevant) == 1.0
    
    def test_rr_second_position(self):
        """Test RR when relevant document is at rank 2."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc2"]
        assert reciprocal_rank(retrieved, relevant) == 0.5
    
    def test_rr_third_position(self):
        """Test RR when relevant document is at rank 3."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc3"]
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(1/3)
    
    def test_rr_not_found(self):
        """Test RR when no relevant document is found."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc5"]
        assert reciprocal_rank(retrieved, relevant) == 0.0
    
    def test_rr_multiple_relevant(self):
        """Test RR with multiple relevant documents (takes first)."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc2", "doc3"]
        assert reciprocal_rank(retrieved, relevant) == 0.5  # doc2 at rank 2


class TestMeanReciprocalRank:
    """Tests for Mean Reciprocal Rank metric."""
    
    def test_mrr_single_query(self):
        """Test MRR with single query."""
        all_retrieved = [["doc1", "doc2", "doc3"]]
        all_relevant = [["doc2"]]
        assert mean_reciprocal_rank(all_retrieved, all_relevant) == 0.5
    
    def test_mrr_multiple_queries(self):
        """Test MRR with multiple queries."""
        all_retrieved = [
            ["doc1", "doc2", "doc3"],  # doc2 at rank 2 -> RR = 0.5
            ["doc4", "doc5", "doc6"],  # doc5 at rank 2 -> RR = 0.5
        ]
        all_relevant = [
            ["doc2"],
            ["doc5"],
        ]
        assert mean_reciprocal_rank(all_retrieved, all_relevant) == 0.5
    
    def test_mrr_mixed_results(self):
        """Test MRR with mixed results."""
        all_retrieved = [
            ["doc1", "doc2", "doc3"],  # doc1 at rank 1 -> RR = 1.0
            ["doc4", "doc5", "doc6"],  # doc8 not found -> RR = 0.0
        ]
        all_relevant = [
            ["doc1"],
            ["doc8"],
        ]
        assert mean_reciprocal_rank(all_retrieved, all_relevant) == 0.5  # (1.0 + 0.0) / 2
    
    def test_mrr_empty(self):
        """Test MRR with empty input."""
        assert mean_reciprocal_rank([], []) == 0.0


class TestAveragePrecisionAtK:
    """Tests for Average Precision@K metric."""
    
    def test_ap_perfect_ranking(self):
        """Test AP@K with perfect ranking."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc3"]
        # P@1 = 1/1 = 1.0, P@2 = 2/2 = 1.0, P@3 = 3/3 = 1.0
        # AP = (1.0 + 1.0 + 1.0) / 3 = 1.0
        assert average_precision_at_k(retrieved, relevant, k=3) == 1.0
    
    def test_ap_imperfect_ranking(self):
        """Test AP@K with imperfect ranking."""
        retrieved = ["doc1", "doc4", "doc2"]
        relevant = ["doc1", "doc2"]
        # P@1 = 1/1 = 1.0 (doc1 found)
        # P@2 = 1/2 = 0.5 (doc4 not relevant)
        # P@3 = 2/3 = 0.667 (doc2 found)
        # AP = (1.0 + 0.667) / 2 = 0.833
        ap = average_precision_at_k(retrieved, relevant, k=3)
        assert ap == pytest.approx(0.833, rel=1e-2)
    
    def test_ap_no_relevant(self):
        """Test AP@K with no relevant documents found."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc5", "doc6"]
        assert average_precision_at_k(retrieved, relevant, k=3) == 0.0


class TestMeanAveragePrecisionAtK:
    """Tests for Mean Average Precision@K metric."""
    
    def test_map_single_query(self):
        """Test MAP@K with single query."""
        all_retrieved = [["doc1", "doc2", "doc3"]]
        all_relevant = [["doc1", "doc2"]]
        map_score = mean_average_precision_at_k(all_retrieved, all_relevant, k=3)
        assert map_score > 0
    
    def test_map_multiple_queries(self):
        """Test MAP@K with multiple queries."""
        all_retrieved = [
            ["doc1", "doc2", "doc3"],
            ["doc4", "doc5", "doc6"],
        ]
        all_relevant = [
            ["doc1", "doc2"],
            ["doc5"],
        ]
        map_score = mean_average_precision_at_k(all_retrieved, all_relevant, k=3)
        assert 0 <= map_score <= 1.0
    
    def test_map_empty(self):
        """Test MAP@K with empty input."""
        assert mean_average_precision_at_k([], [], k=5) == 0.0
