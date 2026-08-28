from app.ingestion.parser import PDFPage
from app.ingestion.chunker import chunk_pages, Chunk


def test_chunk_pages_basic():
    pages = [
        PDFPage(page_number=1, text="This is a test page with some content. " * 50),
        PDFPage(page_number=2, text="Another page with different content. " * 50),
    ]

    chunks = chunk_pages(pages, document_name="test.pdf", chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.document_name == "test.pdf" for chunk in chunks)


def test_chunk_pages_preserves_metadata():
    pages = [PDFPage(page_number=1, text="Test content " * 100)]

    chunks = chunk_pages(pages, document_name="test.pdf", chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.document_name == "test.pdf"
        assert chunk.page_number == 1
        assert chunk.chunk_id.startswith("test.pdf_p1_")


def test_chunk_pages_respects_chunk_size():
    pages = [PDFPage(page_number=1, text="A" * 1000)]

    chunks = chunk_pages(pages, document_name="test.pdf", chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 250


def test_chunk_pages_handles_empty_text():
    pages = [
        PDFPage(page_number=1, text=""),
        PDFPage(page_number=2, text="   "),
        PDFPage(page_number=3, text="Valid content"),
    ]

    chunks = chunk_pages(pages, document_name="test.pdf", chunk_size=200, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0].page_number == 3


def test_chunk_pages_unique_chunk_ids():
    pages = [
        PDFPage(page_number=1, text="First page content " * 50),
        PDFPage(page_number=2, text="Second page content " * 50),
    ]

    chunks = chunk_pages(pages, document_name="test.pdf", chunk_size=200, chunk_overlap=50)

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
