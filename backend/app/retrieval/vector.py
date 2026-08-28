from dataclasses import dataclass

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

from app.config.settings import settings


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    document_name: str
    page_number: int
    similarity_score: float


class VectorRetriever:
    def __init__(self):
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.pc.Index(settings.pinecone_index_name)
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vector = self.embeddings.embed_query(query)

        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )

        chunks = []
        for match in results.matches:
            metadata = match.metadata
            chunks.append(
                RetrievedChunk(
                    chunk_id=metadata.get("chunk_id", match.id),
                    text=metadata.get("text", ""),
                    document_name=metadata.get("document_name", ""),
                    page_number=metadata.get("page_number", 0),
                    similarity_score=match.score,
                )
            )

        return chunks
