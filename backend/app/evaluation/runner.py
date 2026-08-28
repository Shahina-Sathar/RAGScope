"""
Evaluation runner for executing benchmark datasets and calculating metrics.
"""
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.evaluation.dataset import BenchmarkDataset, BenchmarkQuestion, load_benchmark_dataset
from app.evaluation.metrics import (
    hit_at_k,
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    average_precision_at_k,
    mean_average_precision_at_k,
)


@dataclass
class QuestionResult:
    """Result of evaluating a single question."""
    question_id: str
    question: str
    category: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    expected_answer: str
    generated_answer: str
    latency_ms: int
    
    # Per-question metrics
    hit_at_1: int = 0
    hit_at_3: int = 0
    hit_at_5: int = 0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    rr: float = 0.0
    ap_at_5: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "expected_sources": self.expected_sources,
            "retrieved_sources": self.retrieved_sources,
            "expected_answer": self.expected_answer,
            "generated_answer": self.generated_answer,
            "latency_ms": self.latency_ms,
            "metrics": {
                "hit_at_1": self.hit_at_1,
                "hit_at_3": self.hit_at_3,
                "hit_at_5": self.hit_at_5,
                "recall_at_1": self.recall_at_1,
                "recall_at_3": self.recall_at_3,
                "recall_at_5": self.recall_at_5,
                "precision_at_1": self.precision_at_1,
                "precision_at_3": self.precision_at_3,
                "precision_at_5": self.precision_at_5,
                "reciprocal_rank": self.rr,
                "average_precision_at_5": self.ap_at_5,
            },
        }


@dataclass
class EvaluationResults:
    """Aggregated results from evaluating a benchmark dataset."""
    dataset_name: str
    timestamp: str
    total_questions: int
    total_latency_ms: int
    avg_latency_ms: float
    
    # Aggregate metrics
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    map_at_5: float = 0.0
    
    # Per-category metrics
    category_metrics: dict = field(default_factory=dict)
    
    # Individual results
    question_results: list[QuestionResult] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "summary": {
                "total_questions": self.total_questions,
                "total_latency_ms": self.total_latency_ms,
                "avg_latency_ms": self.avg_latency_ms,
                "hit_at_1": self.hit_at_1,
                "hit_at_3": self.hit_at_3,
                "hit_at_5": self.hit_at_5,
                "recall_at_1": self.recall_at_1,
                "recall_at_3": self.recall_at_3,
                "recall_at_5": self.recall_at_5,
                "precision_at_1": self.precision_at_1,
                "precision_at_3": self.precision_at_3,
                "precision_at_5": self.precision_at_5,
                "mrr": self.mrr,
                "map_at_5": self.map_at_5,
            },
            "category_metrics": self.category_metrics,
            "question_results": [r.to_dict() for r in self.question_results],
        }


class RetrievalSimulator:
    """
    Simulates retrieval for evaluation purposes.
    
    In a real evaluation, this would call the actual retrieval pipeline.
    For now, it provides a mock interface for testing the evaluation framework.
    """
    
    def __init__(self, retrieval_mode: str = "vector", top_k: int = 5):
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
    
    def retrieve(self, question: str) -> tuple[list[str], str, int]:
        """
        Simulate retrieval for a question.
        
        Returns:
            Tuple of (retrieved_source_ids, generated_answer, latency_ms)
        """
        # This is a placeholder - in real implementation, this would call
        # the actual retrieval pipeline
        # For now, return empty results
        return [], "I couldn't find sufficient information in the provided documents to answer this question", 0


def evaluate_question(
    question: BenchmarkQuestion,
    simulator: RetrievalSimulator,
) -> QuestionResult:
    """
    Evaluate a single benchmark question.
    
    Args:
        question: Benchmark question with ground truth
        simulator: Retrieval simulator to use
        
    Returns:
        QuestionResult with metrics
    """
    start_time = time.time()
    
    # Simulate retrieval
    retrieved_sources, generated_answer, latency_ms = simulator.retrieve(question.question)
    
    if latency_ms == 0:
        latency_ms = int((time.time() - start_time) * 1000)
    
    # Calculate metrics
    result = QuestionResult(
        question_id=question.id,
        question=question.question,
        category=question.category,
        expected_sources=question.expected_sources,
        retrieved_sources=retrieved_sources,
        expected_answer=question.expected_answer,
        generated_answer=generated_answer,
        latency_ms=latency_ms,
    )
    
    # Calculate Hit@K
    result.hit_at_1 = hit_at_k(retrieved_sources, question.expected_sources, k=1)
    result.hit_at_3 = hit_at_k(retrieved_sources, question.expected_sources, k=3)
    result.hit_at_5 = hit_at_k(retrieved_sources, question.expected_sources, k=5)
    
    # Calculate Recall@K
    result.recall_at_1 = recall_at_k(retrieved_sources, question.expected_sources, k=1)
    result.recall_at_3 = recall_at_k(retrieved_sources, question.expected_sources, k=3)
    result.recall_at_5 = recall_at_k(retrieved_sources, question.expected_sources, k=5)
    
    # Calculate Precision@K
    result.precision_at_1 = precision_at_k(retrieved_sources, question.expected_sources, k=1)
    result.precision_at_3 = precision_at_k(retrieved_sources, question.expected_sources, k=3)
    result.precision_at_5 = precision_at_k(retrieved_sources, question.expected_sources, k=5)
    
    # Calculate Reciprocal Rank
    result.rr = reciprocal_rank(retrieved_sources, question.expected_sources)
    
    # Calculate Average Precision@K
    result.ap_at_5 = average_precision_at_k(retrieved_sources, question.expected_sources, k=5)
    
    return result


def evaluate_dataset(
    dataset: BenchmarkDataset,
    simulator: RetrievalSimulator,
) -> EvaluationResults:
    """
    Evaluate an entire benchmark dataset.
    
    Args:
        dataset: Benchmark dataset to evaluate
        simulator: Retrieval simulator to use
        
    Returns:
        EvaluationResults with aggregated metrics
    """
    question_results = []
    total_latency = 0
    
    # Evaluate each question
    for question in dataset.questions:
        result = evaluate_question(question, simulator)
        question_results.append(result)
        total_latency += result.latency_ms
    
    # Calculate aggregate metrics
    all_retrieved = [r.retrieved_sources for r in question_results]
    all_relevant = [r.expected_sources for r in question_results]
    
    results = EvaluationResults(
        dataset_name=dataset.name,
        timestamp=datetime.now().isoformat(),
        total_questions=len(question_results),
        total_latency_ms=total_latency,
        avg_latency_ms=total_latency / len(question_results) if question_results else 0,
        hit_at_1=sum(r.hit_at_1 for r in question_results) / len(question_results) if question_results else 0,
        hit_at_3=sum(r.hit_at_3 for r in question_results) / len(question_results) if question_results else 0,
        hit_at_5=sum(r.hit_at_5 for r in question_results) / len(question_results) if question_results else 0,
        recall_at_1=sum(r.recall_at_1 for r in question_results) / len(question_results) if question_results else 0,
        recall_at_3=sum(r.recall_at_3 for r in question_results) / len(question_results) if question_results else 0,
        recall_at_5=sum(r.recall_at_5 for r in question_results) / len(question_results) if question_results else 0,
        precision_at_1=sum(r.precision_at_1 for r in question_results) / len(question_results) if question_results else 0,
        precision_at_3=sum(r.precision_at_3 for r in question_results) / len(question_results) if question_results else 0,
        precision_at_5=sum(r.precision_at_5 for r in question_results) / len(question_results) if question_results else 0,
        mrr=mean_reciprocal_rank(all_retrieved, all_relevant),
        map_at_5=mean_average_precision_at_k(all_retrieved, all_relevant, k=5),
        question_results=question_results,
    )
    
    # Calculate per-category metrics
    categories = dataset.get_categories()
    for category in categories:
        category_questions = dataset.get_by_category(category)
        category_results = [r for r in question_results if r.category == category]
        
        if category_results:
            cat_retrieved = [r.retrieved_sources for r in category_results]
            cat_relevant = [r.expected_sources for r in category_results]
            
            results.category_metrics[category] = {
                "count": len(category_results),
                "hit_at_5": sum(r.hit_at_5 for r in category_results) / len(category_results),
                "recall_at_5": sum(r.recall_at_5 for r in category_results) / len(category_results),
                "precision_at_5": sum(r.precision_at_5 for r in category_results) / len(category_results),
                "mrr": mean_reciprocal_rank(cat_retrieved, cat_relevant),
                "map_at_5": mean_average_precision_at_k(cat_retrieved, cat_relevant, k=5),
            }
    
    return results


def run_evaluation(
    dataset_path: str | Path,
    retrieval_mode: str = "vector",
    top_k: int = 5,
) -> EvaluationResults:
    """
    Run evaluation on a benchmark dataset.
    
    Args:
        dataset_path: Path to the benchmark dataset JSON file
        retrieval_mode: Retrieval mode to use (vector, bm25, hybrid)
        top_k: Number of documents to retrieve
        
    Returns:
        EvaluationResults with metrics
    """
    # Load dataset
    dataset = load_benchmark_dataset(dataset_path)
    
    # Create simulator
    simulator = RetrievalSimulator(retrieval_mode=retrieval_mode, top_k=top_k)
    
    # Run evaluation
    results = evaluate_dataset(dataset, simulator)
    
    return results
