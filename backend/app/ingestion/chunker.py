from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.parser import PDFPage


@dataclass
class Chunk:
    chunk_id: str
    text: str
    document_name: str
    page_number: int
    metadata: dict


def chunk_pages(
    pages: list[PDFPage],
    document_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_counter = 0

    for page in pages:
        if not page.text.strip():
            continue

        texts = splitter.split_text(page.text)

        for text in texts:
            if not text.strip():
                continue

            chunk_counter += 1
            chunk_id = f"{document_name}_p{page.page_number}_c{chunk_counter}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    document_name=document_name,
                    page_number=page.page_number,
                    metadata={
                        "document_name": document_name,
                        "page_number": page.page_number,
                        "chunk_id": chunk_id,
                    },
                )
            )

    return chunks
