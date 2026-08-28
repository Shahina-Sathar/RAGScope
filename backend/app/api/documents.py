import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.database.session import get_session
from app.models.schemas import Document, DocumentResponse, IndexResponse
from app.ingestion.parser import parse_pdf, get_pdf_page_count
from app.ingestion.chunker import chunk_pages
from app.ingestion.indexer import PineconeIndexer

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Document).order_by(Document.created_at.desc()))
    return result.scalars().all()


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    file_path = upload_dir / f"{doc_id}_{file.filename}"

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    num_pages = get_pdf_page_count(file_path)

    document = Document(
        id=doc_id,
        filename=file.filename,
        file_path=str(file_path),
        num_pages=num_pages,
        status="uploaded",
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)

    return document


@router.post("/index/{document_id}", response_model=IndexResponse)
async def index_document(
    document_id: str,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.status == "indexed":
        raise HTTPException(status_code=400, detail="Document already indexed")

    file_path = Path(document.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    pages = parse_pdf(file_path)
    chunks = chunk_pages(
        pages,
        document_name=document.filename,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    indexer = PineconeIndexer()
    num_chunks = indexer.index_chunks(chunks)

    document.num_chunks = num_chunks
    document.status = "indexed"
    await session.commit()
    await session.refresh(document)

    return IndexResponse(
        document_id=document.id,
        filename=document.filename,
        num_chunks=num_chunks,
        status=document.status,
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = Path(document.file_path)
    if file_path.exists():
        file_path.unlink()

    await session.delete(document)
    await session.commit()

    return {"message": "Document deleted", "document_id": document_id}
