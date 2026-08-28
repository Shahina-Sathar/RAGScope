#!/usr/bin/env python3
"""Verification script for Milestone 2 - Retrieval Observability."""

import sys


def test_trace_models():
    """Test trace model definitions."""
    print("Testing trace models...")
    
    with open("app/models/schemas.py", "r") as f:
        content = f.read()
    
    assert "class Trace(Base):" in content, "Trace ORM model not found"
    assert "class TraceResponse(BaseModel):" in content, "TraceResponse not found"
    assert "class TraceListResponse(BaseModel):" in content, "TraceListResponse not found"
    assert "class QueryResponseWithTrace(QueryResponse):" in content, "QueryResponseWithTrace not found"
    assert "original_query" in content, "original_query field not found"
    assert "rewritten_query" in content, "rewritten_query field not found"
    assert "retrieval_results" in content, "retrieval_results field not found"
    assert "generation_result" in content, "generation_result field not found"
    assert "latency" in content, "latency field not found"
    
    print("✓ Trace models defined correctly")


def test_tracer_module():
    """Test tracer module structure."""
    print("Testing tracer module...")
    
    with open("app/tracing/tracer.py", "r") as f:
        content = f.read()
    
    assert "class QueryTracer:" in content, "QueryTracer class not found"
    assert "def start_retrieval" in content, "start_retrieval method not found"
    assert "def record_retrieval" in content, "record_retrieval method not found"
    assert "def start_generation" in content, "start_generation method not found"
    assert "def record_generation" in content, "record_generation method not found"
    assert "async def persist" in content, "persist method not found"
    
    print("✓ QueryTracer class defined correctly")


def test_traces_api():
    """Test traces API endpoints."""
    print("Testing traces API...")
    
    with open("app/api/traces.py", "r") as f:
        content = f.read()
    
    assert "router = APIRouter(prefix=\"/traces\"" in content, "Traces router not found"
    assert "@router.get(\"\"" in content, "List traces endpoint not found"
    assert "@router.get(\"/{trace_id}\"" in content, "Get trace endpoint not found"
    assert "async def list_traces" in content, "list_traces function not found"
    assert "async def get_trace" in content, "get_trace function not found"
    
    print("✓ Traces API endpoints defined correctly")


def test_query_endpoint_updated():
    """Test that query endpoint uses tracing."""
    print("Testing query endpoint integration...")
    
    with open("app/api/query.py", "r") as f:
        content = f.read()
    
    assert "from app.tracing.tracer import QueryTracer" in content, "QueryTracer not imported"
    assert "tracer = QueryTracer" in content, "QueryTracer not instantiated"
    assert "tracer.start_retrieval()" in content, "start_retrieval not called"
    assert "tracer.record_retrieval" in content, "record_retrieval not called"
    assert "tracer.start_generation()" in content, "start_generation not called"
    assert "tracer.record_generation" in content, "record_generation not called"
    assert "await tracer.persist" in content, "persist not called"
    assert "QueryResponseWithTrace" in content, "QueryResponseWithTrace not used"
    assert "trace_id" in content, "trace_id not in response"
    
    print("✓ Query endpoint uses tracing correctly")


def test_main_includes_traces_router():
    """Test that main.py includes traces router."""
    print("Testing main.py integration...")
    
    with open("app/main.py", "r") as f:
        content = f.read()
    
    assert "from app.api.traces import router as traces_router" in content, "Traces router not imported"
    assert "app.include_router(traces_router)" in content, "Traces router not included"
    
    print("✓ Main app includes traces router")


def test_tests_exist():
    """Test that test file exists."""
    print("Testing test file...")
    
    with open("tests/test_tracing.py", "r") as f:
        content = f.read()
    
    assert "def test_tracer_initialization" in content, "Initialization test not found"
    assert "def test_tracer_record_retrieval" in content, "Record retrieval test not found"
    assert "def test_tracer_record_generation" in content, "Record generation test not found"
    assert "async def test_tracer_persist" in content, "Persist test not found"
    
    print("✓ Test file exists with required tests")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("RAGScope Milestone 2 Verification")
    print("=" * 60)
    print()
    
    tests = [
        test_trace_models,
        test_tracer_module,
        test_traces_api,
        test_query_endpoint_updated,
        test_main_includes_traces_router,
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
        print("Milestone 2 Complete:")
        print("  ✓ Trace model added to schemas.py")
        print("  ✓ QueryTracer class created")
        print("  ✓ Query endpoint uses tracing")
        print("  ✓ Traces API endpoints created")
        print("  ✓ Main app includes traces router")
        print("  ✓ Tests written")
        sys.exit(0)


if __name__ == "__main__":
    main()
