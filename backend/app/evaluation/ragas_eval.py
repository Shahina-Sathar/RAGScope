"""
RAGAS-based evaluation metrics for RAG systems.

This module provides evaluation metrics using the RAGAS framework
to assess the quality of RAG-generated answers.
"""
import asyncio
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.dataset_schema import SingleTurnSample

from app.config.settings import settings


@dataclass
class RAGASEvaluationResult:
    """Result of RAGAS evaluation."""
    faithfulness_score: Optional[float] = None
    answer_relevancy_score: Optional[float] = None
    context_precision_score: Optional[float] = None
    context_recall_score: Optional[float] = None
    error: Optional[str] = None


class RAGASEvaluator:
    """
    Evaluator using RAGAS metrics to assess RAG quality.
    
    Metrics:
    - Faithfulness: Measures factual consistency of the answer with the context
    - Answer Relevancy: Measures how relevant the answer is to the question
    - Context Precision: Measures how precise the retrieved context is
    - Context Recall: Measures how complete the retrieved context is
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        embeddings: Optional[OpenAIEmbeddings] = None,
    ):
        """
        Initialize RAGAS evaluator.
        
        Args:
            llm: Language model for evaluation (defaults to gpt-4o-mini)
            embeddings: Embedding model for evaluation (defaults to text-embedding-3-small)
        """
        self.llm = llm or ChatOpenAI(
            model=settings.openai_llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        self.embeddings = embeddings or OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
    
    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> RAGASEvaluationResult:
        """
        Evaluate a RAG response using RAGAS metrics.

        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context strings
            ground_truth: Optional ground truth answer for recall metrics

        Returns:
            RAGASEvaluationResult with scores for each metric
        """
        try:
            # Create a SingleTurnSample for RAGAS
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )

            # RAGAS's evaluate() runs its own async loop internally, which
            # conflicts with FastAPI's running (uvloop) event loop. Run it in a
            # worker thread so it gets a fresh event-loop context.
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision if ground_truth else None,
                context_recall if ground_truth else None,
            ]
            metrics = [m for m in metrics if m is not None]

            # RAGAS metrics (e.g. answer_relevancy) read embeddings from the
            # metric object and expect a LangChain-style object exposing
            # embed_query/embed_documents. Assign it explicitly so the metric
            # doesn't fall back to RAGAS's own wrapper (which lacks them).
            for metric in metrics:
                if hasattr(metric, "embeddings"):
                    metric.embeddings = self.embeddings

            dataset = EvaluationDataset(samples=[sample])

            def _run_ragas():
                return evaluate(
                    dataset=dataset,
                    metrics=metrics,
                    llm=self.llm,
                    embeddings=self.embeddings,
                )

            results = await asyncio.to_thread(_run_ragas)

            # Extract scores
            result_dict = results.to_pandas().iloc[0].to_dict()

            
            return RAGASEvaluationResult(
                faithfulness_score=result_dict.get("faithfulness"),
                answer_relevancy_score=result_dict.get("answer_relevancy"),
                context_precision_score=result_dict.get("context_precision"),
                context_recall_score=result_dict.get("context_recall"),
            )
        except Exception as e:
            return RAGASEvaluationResult(error=str(e))
    
    async def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> Optional[float]:
        """
        Evaluate faithfulness of the answer.

        Faithfulness measures whether the answer is factually consistent
        with the retrieved context.

        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context strings

        Returns:
            Faithfulness score (0.0 to 1.0) or None if evaluation fails
        """
        try:
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )

            results = await asyncio.to_thread(
                evaluate,
                dataset=EvaluationDataset(samples=[sample]),
                metrics=[faithfulness],
                llm=self.llm,
                embeddings=self.embeddings,
            )

            return results.to_pandas().iloc[0].get("faithfulness")
        except Exception:
            return None

    async def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str,
    ) -> Optional[float]:
        """
        Evaluate relevancy of the answer to the question.

        Answer relevancy measures how well the answer addresses the question.

        Args:
            question: The user's question
            answer: The generated answer

        Returns:
            Answer relevancy score (0.0 to 1.0) or None if evaluation fails
        """
        try:
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
            )

            answer_relevancy.embeddings = self.embeddings

            results = await asyncio.to_thread(
                evaluate,
                dataset=EvaluationDataset(samples=[sample]),
                metrics=[answer_relevancy],
                llm=self.llm,
                embeddings=self.embeddings,
            )

            return results.to_pandas().iloc[0].get("answer_relevancy")
        except Exception:
            return None

    async def evaluate_context_precision(
        self,
        question: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> Optional[float]:
        """
        Evaluate precision of the retrieved context.

        Context precision measures how much of the retrieved context
        is actually relevant to answering the question.

        Args:
            question: The user's question
            contexts: List of retrieved context strings
            ground_truth: Optional ground truth answer

        Returns:
            Context precision score (0.0 to 1.0) or None if evaluation fails
        """
        try:
            sample = SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )

            results = await asyncio.to_thread(
                evaluate,
                dataset=EvaluationDataset(samples=[sample]),
                metrics=[context_precision],
                llm=self.llm,
                embeddings=self.embeddings,
            )

            return results.to_pandas().iloc[0].get("context_precision")
        except Exception:
            return None

    async def evaluate_context_recall(
        self,
        question: str,
        contexts: list[str],
        ground_truth: str,
    ) -> Optional[float]:
        """
        Evaluate recall of the retrieved context.

        Context recall measures how much of the ground truth
        information is captured in the retrieved context.

        Args:
            question: The user's question
            contexts: List of retrieved context strings
            ground_truth: Ground truth answer

        Returns:
            Context recall score (0.0 to 1.0) or None if evaluation fails
        """
        try:
            sample = SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )

            results = await asyncio.to_thread(
                evaluate,
                dataset=EvaluationDataset(samples=[sample]),
                metrics=[context_recall],
                llm=self.llm,
                embeddings=self.embeddings,
            )

            return results.to_pandas().iloc[0].get("context_recall")
        except Exception:
            return None
