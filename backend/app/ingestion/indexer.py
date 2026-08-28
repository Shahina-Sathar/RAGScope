import json
from pathlib import Path

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

from app.config.settings import settings
from app.ingestion.chunker import Chunk


class PineconeIndexer:
    def __init__(self):
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.pc.Index(settings.pinecone_index_name)
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
        self.bm25_chunks_file = Path(settings.upload_dir) / "bm25_chunks.json"

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> int:
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)

        records = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            records.append(
                {
                    "id": chunk.chunk_id,
                    "values": vector,
                    "metadata": {
                        "text": chunk.text,
                        "document_name": chunk.document_name,
                        "page_number": chunk.page_number,
                        "chunk_id": chunk.chunk_id,
                    },
                }
            )

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self.index.upsert(vectors=batch)

        self._save_chunks_for_bm25(chunks)

        return len(records)

    def _save_chunks_for_bm25(self, chunks: list[Chunk]):
        existing_chunks = []
        if self.bm25_chunks_file.exists():
            with open(self.bm25_chunks_file, "r") as f:
                existing_chunks = json.load(f)

        existing_ids = {c["chunk_id"] for c in existing_chunks}

        new_chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
            }
            for chunk in chunks
            if chunk.chunk_id not in existing_ids
        ]

        all_chunks = existing_chunks + new_chunks

        with open(self.bm25_chunks_file, "w") as f:
            json.dump(all_chunks, f, indent=2)
