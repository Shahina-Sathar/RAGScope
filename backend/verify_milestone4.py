#!/usr/bin/env python3
"""Verification script for Milestone 4 - Reranking."""

import sys


def test_reranker_module():
    """Test reranker module structure."""
    print("Testing reranker module...")
    
    with open("app/retrieval/reranker.py", "r") as f:
        content = f.read()
    
    assert "class CrossEncoderReranker:" in content, "CrossEncoderReranker class not found"
    assert "class RerankedChunk:" in content, "RerankedChunk dataclass not found"
    assert "def rerank" in content, "rerank method not found"
    assert "from sentence_transformers import CrossEncoder" in content, "CrossEncoder not imported"
    assert "original_score" in content, "original_score field not found"
    assert "reranker_score" in content, "reranker_score field not found"
    assert "original_rank" in content, "original_rank field not found"
    assert "reranker_rank" in content, "reranker_rank field not found"
    
    print("✓ Reranker module defined correctly")


def test_schemas_updated():
    """Test that schemas include reranker config."""
    print("Testing schemas...")
    
    with open("app/models/schemas.py", "r") as f:
        content = f.read()
    
    assert "reranker_enabled: bool" in content, "reranker_enabled not in QueryRequest"
    assert "reranker_top_k: int" in content, "reranker_top_k not in QueryRequest"
    assert "candidate_k: int" in content, "candidate_k not in QueryRequest"
    assert "reranker_enabled: Mapped[bool]" in content, "reranker_enabled not in Trace model"
    assert "reranker_results: Mapped[dict]" in content, "reranker_results not in Trace model"
    
    print("✓ Schemas updated correctly")


def test_query_endpoint_updated():
    """Test that query endpoint uses reranker."""
    print("Testing query endpoint...")
    
    with open("app/api/query.py", "r") as f:
        content = f.read()
    
    assert "from app.retrieval.reranker import CrossEncoderReranker" in content, "CrossEncoderReranker not imported"
    assert "request.reranker_enabled" in content, "reranker_enabled not used from request"
    assert "request.candidate_k" in content, "candidate_k not used from request"
    assert "request.reranker_top_k" in content, "reranker_top_k not used from request"
    assert "reranker.rerank" in content, "reranker.rerank not called"
    assert "tracer.start_reranker" in content, "start_reranker not called"
    assert "tracer.record_reranker" in content, "record_reranker not called"
    
    print("✓ Query endpoint updated correctly")


def test_tracer_updated():
    """Test that tracer handles reranker."""
    print("Testing tracer...")
    
    with open("app/tracing/tracer.py", "r") as f:
        content = f.read()
    
    assert "reranker_enabled: bool = False" in content, "reranker_enabled not in __init__"
    assert "reranker_results: list[dict]" in content, "reranker_results not in __init__"
    assert "reranker_start: float" in content, "reranker_start not in __init__"
    assert "reranker_ms: int" in content, "reranker_ms not in __init__"
    assert "def start_reranker" in content, "start_reranker method not found"
    assert "def record_reranker" in content, "record_reranker method not found"
    assert "from app.retrieval.reranker import RerankedChunk" in content, "RerankedChunk not imported"
    assert "reranker_enabled=self.reranker_enabled" in content, "reranker_enabled not passed to Trace"
    assert "reranker_results=" in content, "reranker_results not passed to Trace"
    assert "reranker_ms" in content, "reranker_ms not in latency"
    
    print("✓ Tracer updated correctly")


def test_tests_exist():
    """Test that test file exists."""
    print("Testing test files...")
    
    with open("tests/test_reranker.py", "r") as f:
        content = f.read()
    
    assert "def test_reranker_basic" in content, "Basic reranker test not found"
    assert "def test_reranker_top_k" in content, "Top-k test not found"
    assert "def test_reranker_empty_chunks" in content, "Empty chunks test not found"
    assert "def test_reranker_preserves_metadata" in content, "Metadata preservation test not found"
    assert "def test_reranker_with_bm25_results" in content, "BM25 results test not found"
    assert "def test_reranker_with_hybrid_results" in content, "Hybrid results test not found"
    
    print("✓ Test files exist with required tests")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("RAGScope Milestone 4 Verification")
    print("=" * 60)
    print()
    
    tests = [
        test_reranker_module,
        test_schemas_updated,
        test_query_endpoint_updated,
        test_tracer_updated,
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
    print("=" * 60)
    if failed:
        print(f"FAILED: {len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("SUCCESS: All verification tests passed!")
        print("=" * 60)
        print()
        print("Milestone 4 Complete:")
        print("  ✓ Cross-encoder reranker implemented")
        print("  ✓ QueryRequest supports reranker config")
        print("  ✓ Query pipeline: retrieve → rerank → generate")
        print("  ✓ Tracer captures reranker scores and timing")
        print("  ✓ Tests written")
        sys.exit(0)


if __name__ == "__main__":
    main()
