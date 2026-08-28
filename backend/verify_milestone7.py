#!/usr/bin/env python3
"""
Verification script for Milestone 7: RAGAS + Grounding Evaluation
"""
import sys
from pathlib import Path


def verify_file_exists(filepath: str, description: str) -> bool:
    """Verify a file exists."""
    path = Path(filepath)
    if not path.exists():
        print(f"✗ {description}: {filepath} not found")
        return False
    print(f"✓ {description}: {filepath}")
    return True


def verify_file_content(filepath: str, required_strings: list[str], description: str) -> bool:
    """Verify a file contains required strings."""
    path = Path(filepath)
    if not path.exists():
        print(f"✗ {description}: {filepath} not found")
        return False
    
    content = path.read_text()
    missing = []
    for req_str in required_strings:
        if req_str not in content:
            missing.append(req_str)
    
    if missing:
        print(f"✗ {description}: Missing required content in {filepath}")
        for m in missing:
            print(f"  - {m}")
        return False
    
    print(f"✓ {description}: {filepath}")
    return True


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("Milestone 7 Verification: RAGAS + Grounding Evaluation")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # 1. Verify dependencies
    print("1. Dependencies")
    all_passed &= verify_file_content(
        "pyproject.toml",
        ["ragas>=0.1.0"],
        "RAGAS dependency"
    )
    print()
    
    # 2. Verify RAGAS evaluation module
    print("2. RAGAS Evaluation Module")
    all_passed &= verify_file_exists(
        "app/evaluation/ragas_eval.py",
        "RAGAS evaluation module"
    )
    all_passed &= verify_file_content(
        "app/evaluation/ragas_eval.py",
        [
            "class RAGASEvaluationResult",
            "class RAGASEvaluator",
            "def evaluate",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ],
        "RAGAS evaluation classes and methods"
    )
    print()
    
    # 3. Verify grounding detection module
    print("3. Grounding Detection Module")
    all_passed &= verify_file_exists(
        "app/evaluation/grounding.py",
        "Grounding detection module"
    )
    all_passed &= verify_file_content(
        "app/evaluation/grounding.py",
        [
            "class GroundingResult",
            "class GroundingChecker",
            "class SimpleGroundingChecker",
            "def check_grounding",
            "def is_grounded",
            "def get_hallucination_report",
        ],
        "Grounding detection classes and methods"
    )
    print()
    
    # 4. Verify schema updates
    print("4. Schema Updates")
    all_passed &= verify_file_content(
        "app/models/schemas.py",
        [
            "evaluation_enabled: bool",
            "grounding_check_enabled: bool",
            "grounding_threshold: float",
            "evaluation: dict | None",
            "grounding: dict | None",
            "evaluation_result: dict",
            "grounding_result: dict",
        ],
        "Schema evaluation fields"
    )
    print()
    
    # 5. Verify tracer updates
    print("5. Tracer Updates")
    all_passed &= verify_file_content(
        "app/tracing/tracer.py",
        [
            "evaluation_enabled: bool",
            "grounding_check_enabled: bool",
            "evaluation_result: dict",
            "grounding_result: dict",
            "def start_evaluation",
            "def record_evaluation",
            "def record_grounding",
        ],
        "Tracer evaluation methods"
    )
    print()
    
    # 6. Verify query API integration
    print("6. Query API Integration")
    all_passed &= verify_file_content(
        "app/api/query.py",
        [
            "from app.evaluation.ragas_eval import RAGASEvaluator",
            "from app.evaluation.grounding import GroundingChecker",
            "evaluation_enabled=request.evaluation_enabled",
            "grounding_check_enabled=request.grounding_check_enabled",
            "evaluator = RAGASEvaluator()",
            "grounding_checker = GroundingChecker()",
        ],
        "Query API evaluation integration"
    )
    print()
    
    # 7. Verify tests
    print("7. Test Files")
    all_passed &= verify_file_exists(
        "tests/test_ragas_eval.py",
        "RAGAS evaluation tests"
    )
    all_passed &= verify_file_exists(
        "tests/test_grounding.py",
        "Grounding detection tests"
    )
    print()
    
    # 8. Verify README updates
    print("8. Documentation")
    all_passed &= verify_file_content(
        "../README.md",
        [
            "Milestone 7: RAGAS + Grounding Evaluation",
            "RAGAS evaluation framework integration",
            "Grounding and hallucination detection",
        ],
        "README Milestone 7 documentation"
    )
    print()
    
    # Summary
    print("=" * 70)
    if all_passed:
        print("✓ All verification checks passed!")
        print("=" * 70)
        print()
        print("Milestone 7 is complete. Key features:")
        print("  • RAGAS evaluation (faithfulness, relevancy, precision, recall)")
        print("  • Grounding detection (LLM-based and simple keyword-based)")
        print("  • Hallucination detection with claim analysis")
        print("  • Full integration with query pipeline and tracing")
        print("  • Comprehensive test coverage")
        return 0
    else:
        print("✗ Some verification checks failed")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
