import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config.settings import settings
from app.generation.prompts import SYSTEM_PROMPT, USER_PROMPT, format_context
from app.retrieval.vector import RetrievedChunk


class AnswerGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
        if not chunks:
            return (
                "I couldn't find sufficient information in the provided documents to answer this question.",
                [],
            )

        context_dicts = [
            {
                "text": chunk.text,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ]

        context = format_context(context_dicts)

        system_message = SystemMessage(content=SYSTEM_PROMPT.format(context=context))
        user_message = HumanMessage(content=USER_PROMPT.format(question=question))

        response = self.llm.invoke([system_message, user_message])
        answer = response.content

        citations = self._extract_citations(answer, chunks)

        return answer, citations

    def _extract_citations(self, answer: str, chunks: list[RetrievedChunk]) -> list[str]:
        citation_pattern = r"\[chunk_(\d+)\]"
        matches = re.findall(citation_pattern, answer)

        cited_chunk_ids = []
        for match in matches:
            index = int(match) - 1
            if 0 <= index < len(chunks):
                cited_chunk_ids.append(chunks[index].chunk_id)

        return cited_chunk_ids
