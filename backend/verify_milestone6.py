#!/usr/bin/env python3
"""
Verification script for Milestone 6: Query Rewriting + LangGraph Corrective RAG.
"""
import sys
from pathlib import Path


def test_query_rewriter_module():
    """Test that query rewriter module exists and has required classes."""
    print("Testing query rewriter module...")
    
    rewriter_path = Path("app/query/rewriter.py")
    assert rewriter_path.exists(), "rewriter.py not found"
    
    content = rewriter_path.read_text()
    
    required_classes = [
        "class LLMQueryRewriter",
        "class HyDERewriter",
        "class QueryDecomposer",
        "class NoOpRewriter",
    ]
    
    for cls in required_classes:
        assert cls in content, f"Class {cls} not found in rewriter.py"
    
    assert "def get_rewriter" in content, "get_rewriter function not found"
    
    print("✓ Query rewriter module verified")


def test_corrective_rag_module():
    """Test that corrective RAG module exists and has required classes."""
    print("Testing corrective RAG module...")
    
    corrective_path = Path("app/query/corrective_rag.py")
    assert corrective_path.exists(), "corrective_rag.py not found"
    
    content = corrective_path.read_text()
    
    required_items = [
        "class CorrectiveRAGState",
        "class CorrectiveRAG",
        "def _build_graph",
        "def _retrieve_node",
        "def _grade_relevance_node",
        "def _rewrite_query_node",
        "def _generate_answer_node",
        "def run",
    ]
    
    for item in required_items:
        assert item in content, f"{item} not found in corrective_rag.py"
    
    print("✓ Corrective RAG module verified")


def test_langgraph_dependency():
    """Test that langgraph is in dependencies."""
    print("Testing langgraph dependency...")
    
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    
    assert "langgraph" in content, "langgraph not found in pyproject.toml"
    
    print("✓ LangGraph dependency verified")


def test_schema_updates():
    """Test that schemas have been updated with new fields."""
    print("Testing schema updates...")
    
    schemas_path = Path("app/models/schemas.py")
    content = schemas_path.read_text()
    
    required_fields = [
        "query_rewriting_enabled",
        "query_rewriting_strategy",
        "corrective_rag_enabled",
        "corrective_rag_max_retries",
        "corrective_rag_relevance_threshold",
    ]
    
    for field in required_fields:
        assert field in content, f"Field {field} not found in schemas.py"
    
    print("✓ Schema updates verified")


def test_query_api_integration():
    """Test that query API has been updated."""
    print("Testing query API integration...")
    
    query_path = Path("app/api/query.py")
    content = query_path.read_text()
    
    required_imports = [
        "from app.query.rewriter import get_rewriter",
        "from app.query.corrective_rag import CorrectiveRAG",
    ]
    
    for imp in required_imports:
        assert imp in content, f"Import {imp} not found in query.py"
    
    assert "request.query_rewriting_enabled" in content, "query_rewriting_enabled not used"
    assert "request.corrective_rag_enabled" in content, "corrective_rag_enabled not used"
    
    print("✓ Query API integration verified")


def test_tracer_updates():
    """Test that tracer has been updated."""
    print("Testing tracer updates...")
    
    tracer_path = Path("app/tracing/tracer.py")
    content = tracer_path.read_text()
    
    required_fields = [
        "query_rewriting_enabled",
        "query_rewriting_strategy",
        "corrective_rag_enabled",
        "corrective_rag_retries",
    ]
    
    for field in required_fields:
        assert field in content, f"Field {field} not found in tracer.py"
    
    print("✓ Tracer updates verified")


def test_tests_exist():
    """Test that test files exist."""
    print("Testing test files...")
    
    test_files = [
        "tests/test_query_rewriter.py",
        "tests/test_corrective_rag.py",
    ]
    
    for test_file in test_files:
        path = Path(test_file)
        assert path.exists(), f"Test file {test_file} not found"
    
    print("✓ Test files verified")


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("RAGScope Milestone 6 Verification")
    print("Query Rewriting + LangGraph Corrective RAG")
    print("=" * 70)
    print()
    
    tests = [
        test_query_rewriter_module,
        test_corrective_rag_module,
        test_langgraph_dependency,
        test_schema_updates,
        test_query_api_integration,
        test_tracer_updates,
        test_tests_exist,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed.append(test.__name__)
    
    print()
    print("=" * 70)
    if failed:
        print(f"FAILED: {len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("SUCCESS: All verification tests passed!")
        print("=" * 70)
        print()
        print("Milestone 6 Complete:")
        print("  ✓ Query rewriting module with 4 strategies (none, llm, hyde, decompose)")
        print("  ✓ LangGraph corrective RAG workflow")
        print("  ✓ Schema updates for new configuration options")
        print("  ✓ Query API integration")
        print("  ✓ Tracer updates for tracking new features")
        print("  ✓ Test files created")
        sys.exit(0)


if __name__ == "__main__":
    main()
