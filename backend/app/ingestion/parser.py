from pathlib import Path

from pypdf import PdfReader


class PDFPage:
    def __init__(self, page_number: int, text: str):
        self.page_number = page_number
        self.text = text


def parse_pdf(file_path: Path | str) -> list[PDFPage]:
    reader = PdfReader(str(file_path))
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(PDFPage(page_number=i, text=text))

    return pages


def get_pdf_page_count(file_path: Path | str) -> int:
    reader = PdfReader(str(file_path))
    return len(reader.pages)
