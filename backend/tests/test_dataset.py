"""
Tests for benchmark dataset loading and management.
"""
import json
import tempfile
from pathlib import Path

import pytest

from app.evaluation.dataset import (
    BenchmarkQuestion,
    BenchmarkDataset,
    load_benchmark_dataset,
    save_benchmark_dataset,
    create_sample_dataset,
)


class TestBenchmarkQuestion:
    """Tests for BenchmarkQuestion dataclass."""
    
    def test_question_creation(self):
        """Test creating a BenchmarkQuestion."""
        q = BenchmarkQuestion(
            id="Q001",
            question="What is the refund policy?",
            expected_answer="30 days",
            expected_sources=["refund_policy.pdf"],
            category="simple_retrieval",
        )
        assert q.id == "Q001"
        assert q.question == "What is the refund policy?"
        assert q.expected_answer == "30 days"
        assert q.expected_sources == ["refund_policy.pdf"]
        assert q.category == "simple_retrieval"
    
    def test_question_to_dict(self):
        """Test converting BenchmarkQuestion to dictionary."""
        q = BenchmarkQuestion(
            id="Q001",
            question="Test question?",
            expected_answer="Test answer",
            expected_sources=["doc.pdf"],
            category="test",
        )
        d = q.to_dict()
        assert d["id"] == "Q001"
        assert d["question"] == "Test question?"
        assert d["expected_answer"] == "Test answer"
        assert d["expected_sources"] == ["doc.pdf"]
        assert d["category"] == "test"


class TestBenchmarkDataset:
    """Tests for BenchmarkDataset dataclass."""
    
    def test_dataset_creation(self):
        """Test creating a BenchmarkDataset."""
        questions = [
            BenchmarkQuestion(
                id="Q001",
                question="Q1?",
                expected_answer="A1",
                expected_sources=["doc1.pdf"],
                category="cat1",
            ),
            BenchmarkQuestion(
                id="Q002",
                question="Q2?",
                expected_answer="A2",
                expected_sources=["doc2.pdf"],
                category="cat2",
            ),
        ]
        dataset = BenchmarkDataset(
            questions=questions,
            name="test_dataset",
            description="Test description",
        )
        assert len(dataset) == 2
        assert dataset.name == "test_dataset"
        assert dataset.description == "Test description"
    
    def test_dataset_get_by_id(self):
        """Test getting a question by ID."""
        questions = [
            BenchmarkQuestion("Q001", "Q1?", "A1", ["doc1.pdf"], "cat1"),
            BenchmarkQuestion("Q002", "Q2?", "A2", ["doc2.pdf"], "cat2"),
        ]
        dataset = BenchmarkDataset(questions=questions)
        
        q = dataset.get_by_id("Q002")
        assert q is not None
        assert q.id == "Q002"
        
        q = dataset.get_by_id("Q999")
        assert q is None
    
    def test_dataset_get_by_category(self):
        """Test getting questions by category."""
        questions = [
            BenchmarkQuestion("Q001", "Q1?", "A1", ["doc1.pdf"], "cat1"),
            BenchmarkQuestion("Q002", "Q2?", "A2", ["doc2.pdf"], "cat2"),
            BenchmarkQuestion("Q003", "Q3?", "A3", ["doc3.pdf"], "cat1"),
        ]
        dataset = BenchmarkDataset(questions=questions)
        
        cat1_questions = dataset.get_by_category("cat1")
        assert len(cat1_questions) == 2
        assert all(q.category == "cat1" for q in cat1_questions)
    
    def test_dataset_get_categories(self):
        """Test getting all unique categories."""
        questions = [
            BenchmarkQuestion("Q001", "Q1?", "A1", ["doc1.pdf"], "cat1"),
            BenchmarkQuestion("Q002", "Q2?", "A2", ["doc2.pdf"], "cat2"),
            BenchmarkQuestion("Q003", "Q3?", "A3", ["doc3.pdf"], "cat1"),
        ]
        dataset = BenchmarkDataset(questions=questions)
        
        categories = dataset.get_categories()
        assert categories == ["cat1", "cat2"]
    
    def test_dataset_iteration(self):
        """Test iterating over dataset."""
        questions = [
            BenchmarkQuestion("Q001", "Q1?", "A1", ["doc1.pdf"], "cat1"),
            BenchmarkQuestion("Q002", "Q2?", "A2", ["doc2.pdf"], "cat2"),
        ]
        dataset = BenchmarkDataset(questions=questions)
        
        ids = [q.id for q in dataset]
        assert ids == ["Q001", "Q002"]


class TestLoadBenchmarkDataset:
    """Tests for loading benchmark datasets."""
    
    def test_load_valid_dataset(self):
        """Test loading a valid dataset file."""
        data = {
            "name": "test_dataset",
            "description": "Test description",
            "questions": [
                {
                    "id": "Q001",
                    "question": "What is the refund policy?",
                    "expected_answer": "30 days",
                    "expected_sources": ["refund_policy.pdf"],
                    "category": "simple_retrieval",
                }
            ],
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            dataset = load_benchmark_dataset(temp_path)
            assert dataset.name == "test_dataset"
            assert dataset.description == "Test description"
            assert len(dataset) == 1
            assert dataset.questions[0].id == "Q001"
        finally:
            Path(temp_path).unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_benchmark_dataset("/nonexistent/path.json")
    
    def test_load_actual_benchmark(self):
        """Test loading the actual benchmark dataset."""
        benchmark_path = Path(__file__).parent.parent.parent / "evaluation" / "questions.json"
        
        if benchmark_path.exists():
            dataset = load_benchmark_dataset(benchmark_path)
            assert len(dataset) > 0
            assert dataset.name == "novadesk_benchmark_v1"
            
            # Check that all questions have required fields
            for q in dataset:
                assert q.id
                assert q.question
                assert q.expected_answer
                assert isinstance(q.expected_sources, list)
                assert q.category


class TestSaveBenchmarkDataset:
    """Tests for saving benchmark datasets."""
    
    def test_save_and_load_roundtrip(self):
        """Test saving and loading a dataset."""
        questions = [
            BenchmarkQuestion("Q001", "Q1?", "A1", ["doc1.pdf"], "cat1"),
            BenchmarkQuestion("Q002", "Q2?", "A2", ["doc2.pdf"], "cat2"),
        ]
        original = BenchmarkDataset(
            questions=questions,
            name="test",
            description="Test dataset",
        )
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
        
        try:
            save_benchmark_dataset(original, temp_path)
            loaded = load_benchmark_dataset(temp_path)
            
            assert loaded.name == original.name
            assert loaded.description == original.description
            assert len(loaded) == len(original)
            
            for orig_q, loaded_q in zip(original, loaded):
                assert orig_q.id == loaded_q.id
                assert orig_q.question == loaded_q.question
                assert orig_q.expected_answer == loaded_q.expected_answer
                assert orig_q.expected_sources == loaded_q.expected_sources
                assert orig_q.category == loaded_q.category
        finally:
            Path(temp_path).unlink()


class TestCreateSampleDataset:
    """Tests for creating sample datasets."""
    
    def test_create_sample_dataset(self):
        """Test creating a sample dataset."""
        dataset = create_sample_dataset()
        
        assert len(dataset) > 0
        assert dataset.name == "sample_novadesk"
        
        # Check that we have different categories
        categories = dataset.get_categories()
        assert len(categories) > 1
        
        # Check that all questions are valid
        for q in dataset:
            assert q.id
            assert q.question
            assert q.expected_answer
            assert isinstance(q.expected_sources, list)
            assert q.category
