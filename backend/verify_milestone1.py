#!/usr/bin/env python3
"""Verification script for Milestone 1 components."""

import sys
from pathlib import Path


def test_parser():
    """Test PDF parser."""
    print("Testing parser...")
    from app.ingestion.parser import PDFPage, parse_pdf, get_pdf_page_count
    from pypdf import PdfWriter
    import tempfile

    # Create a test PDF
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        writer.write(tmp.name)
        pdf_path = Path(tmp.name)

    try:
        pages = parse_pdf(pdf_path)
        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert pages[1].page_number == 2

        count = get_pdf_page_count(pdf_path)
        assert count == 2

        print("✓ Parser test passed")
    finally:
        pdf_path.unlink()


def test_chunker():
    """Test text chunker."""
    print("Testing chunker...")
    from app.ingestion.parser import PDFPage
    from app.ingestion.chunker import chunk_pages, Chunk

    pages = [
        PDFPage(page_number=1, text="This is test content. " * 100),
        PDFPage(page_number=2, text="More content here. " * 100),
    ]

    chunks = chunk_pages(pages, document_name="test.pdf", chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.document_name == "test.pdf" for chunk in chunks)
    assert chunks[0].page_number in [1, 2]

    # Check unique chunk IDs
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))

    print("✓ Chunker test passed")


def test_schemas():
    """Test Pydantic schemas."""
    print("Testing schemas...")
    from app.models.schemas import (
        Source,
        QueryRequest,
        QueryResponse,
        DocumentResponse,
    )
    from datetime import datetime

    # Test Source
    source = Source(
        document="test.pdf",
        page=1,
        chunk_id="chunk_1",
        similarity_score=0.95,
    )
    assert source.document == "test.pdf"
    assert source.similarity_score == 0.95

    # Test QueryRequest
    request = QueryRequest(question="What is the refund policy?", top_k=5)
    assert request.question == "What is the refund policy?"
    assert request.top_k == 5

    # Test QueryResponse
    response = QueryResponse(
        answer="Test answer",
        sources=[source],
        latency_ms=100,
    )
    assert response.answer == "Test answer"
    assert len(response.sources) == 1

    print("✓ Schemas test passed")


def test_prompts():
    """Test prompt formatting."""
    print("Testing prompts...")
    from app.generation.prompts import format_context

    chunks = [
        {
            "text": "This is chunk 1",
            "document_name": "test.pdf",
            "page_number": 1,
            "chunk_id": "chunk_1",
        },
        {
            "text": "This is chunk 2",
            "document_name": "test.pdf",
            "page_number": 2,
            "chunk_id": "chunk_2",
        },
    ]

    context = format_context(chunks)
    assert "chunk_1" in context
    assert "chunk_2" in context
    assert "test.pdf" in context
    assert "This is chunk 1" in context

    print("✓ Prompts test passed")


def test_config():
    """Test configuration."""
    print("Testing configuration...")
    from app.config.settings import Settings

    # Settings should be importable
    assert Settings is not None

    print("✓ Configuration test passed")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("RAGScope Milestone 1 Verification")
    print("=" * 60)
    print()

    tests = [
        test_config,
        test_parser,
        test_chunker,
        test_schemas,
        test_prompts,
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
        sys.exit(0)


if __name__ == "__main__":
    main()
