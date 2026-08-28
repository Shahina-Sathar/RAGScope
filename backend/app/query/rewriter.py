"""
Query rewriting strategies to improve retrieval effectiveness.
"""
from typing import Protocol
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config.settings import settings


class QueryRewriter(Protocol):
    """Protocol for query rewriting strategies."""
    
    def rewrite(self, query: str) -> str:
        """Rewrite the query to improve retrieval."""
        ...


class LLMQueryRewriter:
    """
    Rewrites queries using LLM to improve clarity and specificity.
    
    Strategy: Ask the LLM to rephrase the query to be more specific
    and retrieval-friendly while preserving the original intent.
    """
    
    def __init__(self, model_name: str = settings.openai_llm_model):
        self.model_name = model_name
        self._llm = None
        self.system_prompt = """You are a query rewriter for a RAG (Retrieval-Augmented Generation) system.
Your task is to rewrite the user's question to make it more specific and retrieval-friendly.

Guidelines:
- Preserve the original intent and meaning
- Make implicit references explicit
- Add relevant context if the query is vague
- Use precise terminology
- Keep the rewritten query concise (1-2 sentences)
- Do NOT answer the question, only rewrite it

Return ONLY the rewritten query, nothing else."""
    
    def rewrite(self, query: str) -> str:
        """Rewrite the query using LLM."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model_name,
                api_key=settings.openai_api_key,
                temperature=0,
            )
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Original query: {query}\n\nRewritten query:")
        ]
        response = self._llm.invoke(messages)
        return response.content.strip()


class HyDERewriter:
    """
    Hypothetical Document Embeddings (HyDE) rewriter.
    
    Strategy: Generate a hypothetical answer to the query, then use
    that answer for retrieval. This can improve semantic matching.
    """
    
    def __init__(self, model_name: str = settings.openai_llm_model):
        self.model_name = model_name
        self._llm = None
        self.system_prompt = """You are a helpful assistant. Generate a brief, factual answer to the following question.
The answer should be 2-3 sentences and sound authoritative, even if you're not completely certain.
This will be used for document retrieval, so focus on key terms and concepts.

Return ONLY the hypothetical answer, nothing else."""
    
    def rewrite(self, query: str) -> str:
        """Generate a hypothetical document for the query."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model_name,
                api_key=settings.openai_api_key,
                temperature=0,
            )
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Question: {query}\n\nHypothetical answer:")
        ]
        response = self._llm.invoke(messages)
        return response.content.strip()


class QueryDecomposer:
    """
    Decomposes complex queries into simpler sub-queries.
    
    Strategy: Break down multi-part questions into individual
    queries that can be retrieved separately and combined.
    """
    
    def __init__(self, model_name: str = settings.openai_llm_model):
        self.model_name = model_name
        self._llm = None
        self.system_prompt = """You are a query decomposer for a RAG system.
Break down complex questions into 2-3 simpler sub-queries that can be answered independently.

Guidelines:
- Each sub-query should be self-contained
- Sub-queries should cover different aspects of the original question
- Use simple, direct language
- Return sub-queries separated by newlines
- If the query is already simple, return it as-is

Return ONLY the sub-queries, one per line."""
    
    def rewrite(self, query: str) -> str:
        """Decompose the query into sub-queries."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model_name,
                api_key=settings.openai_api_key,
                temperature=0,
            )
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Complex query: {query}\n\nSub-queries:")
        ]
        response = self._llm.invoke(messages)
        return response.content.strip()
    
    def decompose(self, query: str) -> list[str]:
        """Decompose and return as list."""
        result = self.rewrite(query)
        return [q.strip() for q in result.split('\n') if q.strip()]


class NoOpRewriter:
    """No-op rewriter that returns the query unchanged."""
    
    def rewrite(self, query: str) -> str:
        return query


def get_rewriter(strategy: str = "none") -> QueryRewriter:
    """
    Factory function to get a query rewriter by strategy name.
    
    Args:
        strategy: One of "none", "llm", "hyde", "decompose"
    
    Returns:
        QueryRewriter instance
    """
    rewriters = {
        "none": NoOpRewriter,
        "llm": LLMQueryRewriter,
        "hyde": HyDERewriter,
        "decompose": QueryDecomposer,
    }
    
    if strategy not in rewriters:
        raise ValueError(f"Unknown rewriting strategy: {strategy}. Available: {list(rewriters.keys())}")
    
    return rewriters[strategy]()
