"""
Corrective RAG workflow using LangGraph.

This module implements a self-correcting RAG pipeline that can:
1. Grade retrieved documents for relevance
2. Rewrite queries if documents are not relevant
3. Retry retrieval with improved queries
4. Fall back to web search if needed (future)
"""
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config.settings import settings
from app.retrieval.vector import VectorRetriever, RetrievedChunk
from app.query.rewriter import LLMQueryRewriter


class CorrectiveRAGState(TypedDict):
    """State for the corrective RAG workflow."""
    query: str
    rewritten_query: str | None
    documents: list[RetrievedChunk]
    relevance_scores: list[float]
    is_relevant: bool
    retry_count: int
    max_retries: int
    final_answer: str | None


class CorrectiveRAG:
    """
    Corrective RAG workflow that grades retrieval and retries if needed.
    
    Workflow:
    1. Retrieve documents for the query
    2. Grade document relevance
    3. If not relevant and retries remain, rewrite query and retry
    4. If relevant or max retries reached, generate answer
    """
    
    def __init__(
        self,
        retriever: VectorRetriever,
        max_retries: int = 2,
        relevance_threshold: float = 0.7,
    ):
        self.retriever = retriever
        self.max_retries = max_retries
        self.relevance_threshold = relevance_threshold
        self._llm = None
        self._query_rewriter = None
        self.graph = self._build_graph()

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=settings.openai_llm_model,
                api_key=settings.openai_api_key,
                temperature=0,
            )
        return self._llm

    @llm.setter
    def llm(self, value):
        self._llm = value

    @property
    def query_rewriter(self):
        if self._query_rewriter is None:
            self._query_rewriter = LLMQueryRewriter()
        return self._query_rewriter

    @query_rewriter.setter
    def query_rewriter(self, value):
        self._query_rewriter = value
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(CorrectiveRAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade_relevance", self._grade_relevance_node)
        workflow.add_node("rewrite_query", self._rewrite_query_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Add edges
        workflow.add_edge("retrieve", "grade_relevance")
        
        # Conditional edge after grading
        workflow.add_conditional_edges(
            "grade_relevance",
            self._should_retry,
            {
                "retry": "rewrite_query",
                "generate": "generate_answer",
            }
        )
        
        # After rewriting, go back to retrieve
        workflow.add_edge("rewrite_query", "retrieve")
        
        # After generating, end
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    def _retrieve_node(self, state: CorrectiveRAGState) -> dict:
        """Retrieve documents for the current query."""
        query = state.get("rewritten_query") or state["query"]
        documents = self.retriever.retrieve(query, top_k=5)
        
        return {
            "documents": documents,
            "retry_count": state.get("retry_count", 0),
        }
    
    def _grade_relevance_node(self, state: CorrectiveRAGState) -> dict:
        """Grade the relevance of retrieved documents."""
        documents = state["documents"]
        
        if not documents:
            return {"is_relevant": False, "relevance_scores": []}
        
        # Use similarity scores as relevance indicator
        scores = [doc.similarity_score for doc in documents]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        is_relevant = avg_score >= self.relevance_threshold
        
        return {
            "relevance_scores": scores,
            "is_relevant": is_relevant,
        }
    
    def _should_retry(self, state: CorrectiveRAGState) -> Literal["retry", "generate"]:
        """Determine whether to retry retrieval or generate answer."""
        is_relevant = state.get("is_relevant", False)
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", self.max_retries)
        
        if is_relevant:
            return "generate"
        
        if retry_count >= max_retries:
            return "generate"
        
        return "retry"
    
    def _rewrite_query_node(self, state: CorrectiveRAGState) -> dict:
        """Rewrite the query to improve retrieval."""
        original_query = state.get("rewritten_query") or state["query"]
        rewritten = self.query_rewriter.rewrite(original_query)
        
        return {
            "rewritten_query": rewritten,
            "retry_count": state.get("retry_count", 0) + 1,
        }
    
    def _generate_answer_node(self, state: CorrectiveRAGState) -> dict:
        """Generate an answer from the retrieved documents."""
        documents = state["documents"]
        query = state["query"]
        
        if not documents:
            return {
                "final_answer": "I couldn't find relevant information to answer your question."
            }
        
        # Build context from documents
        context = "\n\n".join([
            f"[Document {i+1}] (Score: {doc.similarity_score:.3f})\n{doc.text}"
            for i, doc in enumerate(documents)
        ])
        
        # Generate answer
        system_prompt = """You are a helpful assistant. Answer the question based on the provided context.
If the context doesn't contain enough information, say so clearly.
Cite the document numbers when referencing specific information."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
        ]
        
        response = self.llm.invoke(messages)
        
        return {"final_answer": response.content}
    
    def run(self, query: str) -> dict:
        """
        Run the corrective RAG workflow.
        
        Args:
            query: The user's question
        
        Returns:
            Dictionary with workflow results including:
            - final_answer: The generated answer
            - rewritten_query: The final rewritten query (if any)
            - documents: The retrieved documents
            - relevance_scores: Scores for each document
            - retry_count: Number of retries performed
        """
        initial_state = {
            "query": query,
            "rewritten_query": None,
            "documents": [],
            "relevance_scores": [],
            "is_relevant": False,
            "retry_count": 0,
            "max_retries": self.max_retries,
            "final_answer": None,
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "final_answer": final_state["final_answer"],
            "rewritten_query": final_state.get("rewritten_query"),
            "documents": final_state["documents"],
            "relevance_scores": final_state["relevance_scores"],
            "retry_count": final_state["retry_count"],
        }
