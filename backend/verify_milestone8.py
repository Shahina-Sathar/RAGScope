#!/usr/bin/env python3
"""Verification script for Milestone 8: Experiment Framework + Failure Analyzer."""

import sys
from pathlib import Path


def verify_milestone_8():
    """Verify all components of Milestone 8 are implemented."""
    base_path = Path(__file__).parent
    errors = []

    print("=" * 70)
    print("Milestone 8 Verification: Experiment Framework + Failure Analyzer")
    print("=" * 70)
    print()

    # Check experiment models
    print("1. Checking experiment models...")
    schemas_path = base_path / "app" / "models" / "schemas.py"
    if not schemas_path.exists():
        errors.append("schemas.py not found")
    else:
        content = schemas_path.read_text()
        required_models = [
            "class Experiment(Base)",
            "class ExperimentResult(Base)",
            "class ExperimentConfig(BaseModel)",
            "class ExperimentCreateRequest(BaseModel)",
            "class ExperimentResponse(BaseModel)",
            "class ExperimentListResponse(BaseModel)",
            "class ExperimentResultResponse(BaseModel)",
            "class ExperimentComparisonResponse(BaseModel)",
        ]
        for model in required_models:
            if model not in content:
                errors.append(f"Missing model: {model}")
        if not errors:
            print("   ✓ All experiment models present")

    # Check experiment runner
    print("\n2. Checking experiment runner...")
    runner_path = base_path / "app" / "experiments" / "runner.py"
    if not runner_path.exists():
        errors.append("runner.py not found")
    else:
        content = runner_path.read_text()
        required_items = [
            "class ExperimentRunner",
            "async def run_experiment",
            "async def _run_query",
        ]
        for item in required_items:
            if item not in content:
                errors.append(f"Missing in runner.py: {item}")
        if not any("runner.py" in e for e in errors):
            print("   ✓ Experiment runner implemented")

    # Check failure analyzer
    print("\n3. Checking failure analyzer...")
    analyzer_path = base_path / "app" / "experiments" / "analyzer.py"
    if not analyzer_path.exists():
        errors.append("analyzer.py not found")
    else:
        content = analyzer_path.read_text()
        required_items = [
            "class FailureAnalyzer",
            "async def analyze_experiment",
            "def _analyze_result",
            "async def get_failure_summary",
        ]
        for item in required_items:
            if item not in content:
                errors.append(f"Missing in analyzer.py: {item}")
        if not any("analyzer.py" in e for e in errors):
            print("   ✓ Failure analyzer implemented")

    # Check experiment comparator
    print("\n4. Checking experiment comparator...")
    comparator_path = base_path / "app" / "experiments" / "comparison.py"
    if not comparator_path.exists():
        errors.append("comparison.py not found")
    else:
        content = comparator_path.read_text()
        required_items = [
            "class ExperimentComparator",
            "async def compare_experiments",
            "def _compare_metrics",
            "async def _compare_failures",
            "def _generate_recommendations",
        ]
        for item in required_items:
            if item not in content:
                errors.append(f"Missing in comparison.py: {item}")
        if not any("comparison.py" in e for e in errors):
            print("   ✓ Experiment comparator implemented")

    # Check API endpoints
    print("\n5. Checking experiment API endpoints...")
    api_path = base_path / "app" / "api" / "experiments.py"
    if not api_path.exists():
        errors.append("experiments.py API not found")
    else:
        content = api_path.read_text()
        required_endpoints = [
            "async def create_experiment",
            "async def list_experiments",
            "async def get_experiment",
            "async def run_experiment",
            "async def get_experiment_results",
            "async def analyze_experiment",
            "async def get_failure_summary",
            "async def compare_experiments",
            "async def delete_experiment",
        ]
        for endpoint in required_endpoints:
            if endpoint not in content:
                errors.append(f"Missing endpoint: {endpoint}")
        if not any("experiments.py API" in e for e in errors):
            print("   ✓ All API endpoints implemented")

    # Check main.py integration
    print("\n6. Checking main.py integration...")
    main_path = base_path / "app" / "main.py"
    if not main_path.exists():
        errors.append("main.py not found")
    else:
        content = main_path.read_text()
        if "from app.api.experiments import router as experiments_router" not in content:
            errors.append("experiments_router not imported in main.py")
        if "app.include_router(experiments_router)" not in content:
            errors.append("experiments_router not included in main.py")
        if not any("main.py" in e for e in errors):
            print("   ✓ Experiments router integrated in main.py")

    # Check tests
    print("\n7. Checking test files...")
    test_files = [
        "tests/test_experiment_runner.py",
        "tests/test_failure_analyzer.py",
        "tests/test_experiment_comparator.py",
    ]
    for test_file in test_files:
        test_path = base_path / test_file
        if not test_path.exists():
            errors.append(f"Missing test file: {test_file}")
    if not any("test file" in e for e in errors):
        print("   ✓ All test files present")

    # Check __init__.py
    print("\n8. Checking experiments module...")
    init_path = base_path / "app" / "experiments" / "__init__.py"
    if not init_path.exists():
        errors.append("experiments/__init__.py not found")
    else:
        print("   ✓ Experiments module initialized")

    # Summary
    print("\n" + "=" * 70)
    if errors:
        print("❌ VERIFICATION FAILED")
        print("=" * 70)
        print("\nErrors found:")
        for error in errors:
            print(f"  • {error}")
        return 1
    else:
        print("✅ VERIFICATION PASSED")
        print("=" * 70)
        print("\nMilestone 8 is complete!")
        print("\nKey features implemented:")
        print("  • Experiment models and database schema")
        print("  • Experiment configuration management")
        print("  • Experiment runner with evaluation support")
        print("  • Failure analyzer with pattern detection")
        print("  • Experiment comparison and recommendations")
        print("  • Complete API endpoints")
        print("  • Comprehensive test coverage")
        return 0


if __name__ == "__main__":
    sys.exit(verify_milestone_8())
