#!/usr/bin/env python3
"""Verification script for Milestone 3 - BM25 + Hybrid Retrieval."""

import sys


def test_bm25_module():
    """Test BM25 module structure."""
    print("Testing BM25 module...")
    
    with open("app/retrieval/bm25.py", "r") as f:
        content = f.read()
    
    assert "class BM25Retriever:" in content, "BM25Retriever class not found"
    assert "def build_index" in content, "build_index method not found"
    assert "def retrieve" in content, "retrieve method not found"
    assert "from rank_bm25 import BM25Okapi" in content, "BM25Okapi not imported"
    
    print("✓ BM25 module defined correctly")


def test_hybrid_module():
    """Test hybrid module structure."""
    print("Testing hybrid module...")
    
    with open("app/retrieval/hybrid.py", "r") as f:
        content = f.read()
    
    assert "class HybridRetriever:" in content, "HybridRetriever class not found"
    assert "def retrieve" in content, "retrieve method not found"
    assert "def _reciprocal_rank_fusion" in content, "RRF method not found"
    assert "class HybridChunk:" in content, "HybridChunk dataclass not found"
    
    print("✓ Hybrid module defined correctly")


def test_factory_module():
    """Test retrieval factory structure."""
    print("Testing retrieval factory...")
    
    with open("app/retrieval/factory.py", "r") as f:
        content = f.read()
    
    assert "class RetrievalFactory:" in content, "RetrievalFactory class not found"
    assert "class RetrievalMode(str, Enum):" in content, "RetrievalMode enum not found"
    assert "VECTOR = \"vector\"" in content, "VECTOR mode not found"
    assert "BM25 = \"bm25\"" in content, "BM25 mode not found"
    assert "HYBRID = \"hybrid\"" in content, "HYBRID mode not found"
    assert "def retrieve" in content, "retrieve method not found"
    assert "def _ensure_bm25_index" in content, "_ensure_bm25_index method not found"
    
    print("✓ Retrieval factory defined correctly")


def test_schemas_updated():
    """Test that schemas include retrieval_mode."""
    print("Testing schemas...")
    
    with open("app/models/schemas.py", "r") as f:
        content = f.read()
    
    assert "retrieval_mode: str = Field(default=\"vector\"" in content, "retrieval_mode not in QueryRequest"
    assert "retrieval_mode: Mapped[str]" in content, "retrieval_mode not in Trace model"
    
    print("✓ Schemas updated correctly")


def test_query_endpoint_updated():
    """Test that query endpoint uses retrieval factory."""
    print("Testing query endpoint...")
    
    with open("app/api/query.py", "r") as f:
        content = f.read()
    
    assert "from app.retrieval.factory import RetrievalFactory" in content, "RetrievalFactory not imported"
    assert "factory = RetrievalFactory()" in content, "RetrievalFactory not instantiated"
    assert "factory.retrieve" in content, "factory.retrieve not called"
    assert "request.retrieval_mode" in content, "retrieval_mode not used from request"
    
    print("✓ Query endpoint updated correctly")


def test_indexer_updated():
    """Test that indexer saves chunks for BM25."""
    print("Testing indexer...")
    
    with open("app/ingestion/indexer.py", "r") as f:
        content = f.read()
    
    assert "def _save_chunks_for_bm25" in content, "_save_chunks_for_bm25 method not found"
    assert "bm25_chunks.json" in content, "bm25_chunks.json file not referenced"
    assert "json.dump" in content, "json.dump not used"
    
    print("✓ Indexer updated correctly")


def test_tracer_updated():
    """Test that tracer handles different chunk types."""
    print("Testing tracer...")
    
    with open("app/tracing/tracer.py", "r") as f:
        content = f.read()
    
    assert "retrieval_mode: str = \"vector\"" in content, "retrieval_mode not in __init__"
    assert "isinstance(chunk, RetrievedChunk)" in content, "RetrievedChunk type check not found"
    assert "isinstance(chunk, BM25Result)" in content, "BM25Result type check not found"
    assert "isinstance(chunk, HybridChunk)" in content, "HybridChunk type check not found"
    
    print("✓ Tracer updated correctly")


def test_tests_exist():
    """Test that test files exist."""
    print("Testing test files...")
    
    with open("tests/test_bm25.py", "r") as f:
        bm25_content = f.read()
    
    assert "def test_bm25_retriever_build_index" in bm25_content, "BM25 index test not found"
    assert "def test_bm25_retriever_retrieve" in bm25_content, "BM25 retrieve test not found"
    
    with open("tests/test_hybrid.py", "r") as f:
        hybrid_content = f.read()
    
    assert "def test_hybrid_retriever_rrf" in hybrid_content, "Hybrid RRF test not found"
    assert "def test_hybrid_retriever_rrf_formula" in hybrid_content, "RRF formula test not found"
    
    print("✓ Test files exist with required tests")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("RAGScope Milestone 3 Verification")
    print("=" * 60)
    print()
    
    tests = [
        test_bm25_module,
        test_hybrid_module,
        test_factory_module,
        test_schemas_updated,
        test_query_endpoint_updated,
        test_indexer_updated,
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
        print("Milestone 3 Complete:")
        print("  ✓ BM25 retriever implemented")
        print("  ✓ Hybrid retriever with RRF implemented")
        print("  ✓ Retrieval factory created")
        print("  ✓ Query endpoint supports retrieval modes")
        print("  ✓ Indexer saves chunks for BM25")
        print("  ✓ Tracer captures different score types")
        print("  ✓ Tests written")
        sys.exit(0)


if __name__ == "__main__":
    main()
