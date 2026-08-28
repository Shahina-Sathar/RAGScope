"""
Benchmark dataset loading and management for RAG evaluation.
"""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkQuestion:
    """A single benchmark question with ground truth."""
    id: str
    question: str
    expected_answer: str
    expected_sources: list[str]
    category: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "expected_sources": self.expected_sources,
            "category": self.category,
        }


@dataclass
class BenchmarkDataset:
    """Collection of benchmark questions for evaluation."""
    questions: list[BenchmarkQuestion]
    name: str = "default"
    description: str = ""
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def __iter__(self):
        return iter(self.questions)
    
    def get_by_id(self, question_id: str) -> Optional[BenchmarkQuestion]:
        """Get a question by its ID."""
        for q in self.questions:
            if q.id == question_id:
                return q
        return None
    
    def get_by_category(self, category: str) -> list[BenchmarkQuestion]:
        """Get all questions in a specific category."""
        return [q for q in self.questions if q.category == category]
    
    def get_categories(self) -> list[str]:
        """Get all unique categories in the dataset."""
        return sorted(list(set(q.category for q in self.questions)))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "questions": [q.to_dict() for q in self.questions],
        }


def load_benchmark_dataset(file_path: str | Path) -> BenchmarkDataset:
    """
    Load a benchmark dataset from a JSON file.
    
    Expected JSON format:
    {
        "name": "dataset_name",
        "description": "Dataset description",
        "questions": [
            {
                "id": "Q001",
                "question": "What is the refund policy?",
                "expected_answer": "30 days",
                "expected_sources": ["refund_policy.pdf"],
                "category": "simple_retrieval"
            }
        ]
    }
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        BenchmarkDataset object
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = [
        BenchmarkQuestion(
            id=q["id"],
            question=q["question"],
            expected_answer=q["expected_answer"],
            expected_sources=q["expected_sources"],
            category=q["category"],
        )
        for q in data.get("questions", [])
    ]
    
    return BenchmarkDataset(
        questions=questions,
        name=data.get("name", "default"),
        description=data.get("description", ""),
    )


def save_benchmark_dataset(dataset: BenchmarkDataset, file_path: str | Path):
    """
    Save a benchmark dataset to a JSON file.
    
    Args:
        dataset: BenchmarkDataset to save
        file_path: Path to save the JSON file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataset.to_dict(), f, indent=2, ensure_ascii=False)


def create_sample_dataset() -> BenchmarkDataset:
    """
    Create a sample benchmark dataset for testing.
    
    This is a minimal dataset for demonstration purposes.
    The full NovaDesk dataset should be created separately.
    """
    questions = [
        BenchmarkQuestion(
            id="Q001",
            question="What is the current refund period?",
            expected_answer="30 days",
            expected_sources=["refund_policy_2026.pdf"],
            category="simple_retrieval",
        ),
        BenchmarkQuestion(
            id="Q002",
            question="What error code indicates a rate limit exceeded?",
            expected_answer="ERR-429",
            expected_sources=["api_limits.pdf"],
            category="keyword_retrieval",
        ),
        BenchmarkQuestion(
            id="Q003",
            question="What are the differences between the 2025 and 2026 refund policies?",
            expected_answer="2025 policy had 14 days, 2026 has 30 days",
            expected_sources=["refund_policy_2025.pdf", "refund_policy_2026.pdf"],
            category="version_conflict",
        ),
        BenchmarkQuestion(
            id="Q004",
            question="What is the process for escalating a support ticket?",
            expected_answer="Contact support manager after 48 hours",
            expected_sources=["escalation_policy.pdf", "support_sla.pdf"],
            category="multi_document",
        ),
        BenchmarkQuestion(
            id="Q005",
            question="What is the CEO's favorite color?",
            expected_answer="Not found in documents",
            expected_sources=[],
            category="missing_information",
        ),
    ]
    
    return BenchmarkDataset(
        questions=questions,
        name="sample_novadesk",
        description="Sample benchmark dataset for NovaDesk knowledge base",
    )
