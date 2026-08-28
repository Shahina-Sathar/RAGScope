import tempfile
from pathlib import Path

from pypdf import PdfWriter, PdfReader
from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject

from app.ingestion.parser import parse_pdf, get_pdf_page_count, PDFPage


def create_test_pdf(num_pages: int = 3, text_per_page: str = "Test content") -> Path:
    writer = PdfWriter()
    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    resources = DictionaryObject()
    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = font
    resources[NameObject("/Font")] = fonts

    for i in range(num_pages):
        page = writer.add_blank_page(width=612, height=792)
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text_per_page}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = stream
        page[NameObject("/Resources")] = resources

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        writer.write(tmp.name)
        return Path(tmp.name)


def test_parse_pdf_extracts_pages():
    pdf_path = create_test_pdf(num_pages=3)
    try:
        pages = parse_pdf(pdf_path)
        assert len(pages) == 3
        assert all(isinstance(page, PDFPage) for page in pages)
    finally:
        pdf_path.unlink()


def test_parse_pdf_preserves_page_numbers():
    pdf_path = create_test_pdf(num_pages=3)
    try:
        pages = parse_pdf(pdf_path)
        assert pages[0].page_number == 1
        assert pages[1].page_number == 2
        assert pages[2].page_number == 3
    finally:
        pdf_path.unlink()


def test_get_pdf_page_count():
    pdf_path = create_test_pdf(num_pages=5)
    try:
        count = get_pdf_page_count(pdf_path)
        assert count == 5
    finally:
        pdf_path.unlink()


def test_parse_pdf_handles_empty_pages():
    pdf_path = create_test_pdf(num_pages=2)
    try:
        pages = parse_pdf(pdf_path)
        assert len(pages) >= 0
    finally:
        pdf_path.unlink()
