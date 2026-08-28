"""Experiment runner for executing RAG experiments."""
import time
import uuid
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.evaluation.dataset import BenchmarkDataset, load_benchmark_dataset
from app.evaluation.grounding import GroundingChecker
from app.evaluation.ragas_eval import RAGASEvaluator
from app.models.schemas import Experiment, ExperimentResult, ExperimentConfig
from app.retrieval.factory import RetrievalFactory, RetrievalMode
from app.retrieval.reranker import CrossEncoderReranker, RerankedChunk
from app.retrieval.vector import RetrievedChunk


class ExperimentRunner:
    """Runs experiments with specific configurations."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self._retrieval_factory = None
        self._ragas_evaluator = None
        self._grounding_checker = None

    @property
    def retrieval_factory(self) -> RetrievalFactory:
        if self._retrieval_factory is None:
            self._retrieval_factory = RetrievalFactory()
        return self._retrieval_factory

    @property
    def ragas_evaluator(self) -> RAGASEvaluator | None:
        if self._ragas_evaluator is None and settings.openai_api_key:
            self._ragas_evaluator = RAGASEvaluator()
        return self._ragas_evaluator

    @property
    def grounding_checker(self) -> GroundingChecker | None:
        if self._grounding_checker is None and settings.openai_api_key:
            self._grounding_checker = GroundingChecker()
        return self._grounding_checker

    async def run_experiment(self, experiment_id: str) -> None:
        """Run an experiment with the given ID."""
        result = await self.session.execute(
            select(Experiment).where(Experiment.id == experiment_id)
        )
        experiment = result.scalar_one_or_none()

        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        await self.session.execute(
            update(Experiment)
            .where(Experiment.id == experiment_id)
            .values(status="running")
        )
        await self.session.commit()

        try:
            dataset: BenchmarkDataset = load_benchmark_dataset(experiment.dataset_path)
            config = ExperimentConfig(**experiment.config)

            total_latency = 0
            faithfulness_scores: list[float] = []
            relevancy_scores: list[float] = []
            precision_scores: list[float] = []
            recall_scores: list[float] = []
            grounding_scores: list[float] = []

            completed = 0
            for query in dataset.questions:
                result_data = await self._run_query(experiment_id, query, config)

                total_latency += result_data["latency_ms"]

                if result_data.get("faithfulness_score") is not None:
                    faithfulness_scores.append(result_data["faithfulness_score"])
                if result_data.get("relevancy_score") is not None:
                    relevancy_scores.append(result_data["relevancy_score"])
                if result_data.get("precision_score") is not None:
                    precision_scores.append(result_data["precision_score"])
                if result_data.get("recall_score") is not None:
                    recall_scores.append(result_data["recall_score"])
                if result_data.get("grounding_score") is not None:
                    grounding_scores.append(result_data["grounding_score"])

                completed += 1
                await self.session.execute(
                    update(Experiment)
                    .where(Experiment.id == experiment_id)
                    .values(completed_queries=completed)
                )
                await self.session.commit()

            num_queries = len(dataset.questions)
            avg_latency = total_latency / num_queries if num_queries else 0
            avg_faithfulness = (
                sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None
            )
            avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else None
            avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else None
            avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else None
            avg_grounding = sum(grounding_scores) / len(grounding_scores) if grounding_scores else None

            await self.session.execute(
                update(Experiment)
                .where(Experiment.id == experiment_id)
                .values(
                    status="completed",
                    completed_at=datetime.utcnow(),
                    avg_latency_ms=avg_latency,
                    avg_faithfulness=avg_faithfulness,
                    avg_relevancy=avg_relevancy,
                    avg_precision=avg_precision,
                    avg_recall=avg_recall,
                    avg_grounding_score=avg_grounding,
                )
            )
            await self.session.commit()

        except Exception:
            await self.session.execute(
                update(Experiment)
                .where(Experiment.id == experiment_id)
                .values(status="failed")
            )
            await self.session.commit()
            raise

    async def _run_query(
        self,
        experiment_id: str,
        query,
        config: ExperimentConfig,
    ) -> dict:
        """Run a single query and store results."""
        start_time = time.time()

        # Step 1: Retrieval
        mode = RetrievalMode(config.retrieval_mode)
        retrieve_k = config.candidate_k if config.reranker_enabled else config.top_k
        chunks, _ = self.retrieval_factory.retrieve(query.question, mode, top_k=retrieve_k)

        # Step 2: Reranking (if enabled)
        final_chunks = chunks
        if config.reranker_enabled and chunks:
            reranker = CrossEncoderReranker()
            reranked = reranker.rerank(query.question, chunks, top_k=config.reranker_top_k)
            final_chunks = reranked  # type: ignore[assignment]

        # Convert to RetrievedChunk for generation
        retrieved_chunks = [_to_retrieved_chunk(c) for c in final_chunks]

        # Step 3: Generation
        from app.generation.generator import AnswerGenerator

        generator = AnswerGenerator()
        answer, _citations = generator.generate(query.question, retrieved_chunks)

        # Step 4: Evaluation
        faithfulness_score = None
        relevancy_score = None
        precision_score = None
        recall_score = None

        if config.evaluation_enabled and self.ragas_evaluator:
            contexts = [chunk.text for chunk in retrieved_chunks]
            eval_result = await self.ragas_evaluator.evaluate(
                question=query.question,
                answer=answer,
                contexts=contexts,
                ground_truth=query.expected_answer,
            )
            faithfulness_score = eval_result.faithfulness_score
            relevancy_score = eval_result.answer_relevancy_score
            precision_score = eval_result.context_precision_score
            recall_score = eval_result.context_recall_score

        # Step 5: Grounding check
        grounding_score = None
        is_grounded = None

        if config.grounding_check_enabled and self.grounding_checker:
            contexts = [chunk.text for chunk in retrieved_chunks]
            grounding_result = self.grounding_checker.check_grounding(
                question=query.question,
                answer=answer,
                contexts=contexts,
            )
            grounding_score = grounding_result.grounding_score
            is_grounded = grounding_result.is_grounded

        latency_ms = int((time.time() - start_time) * 1000)

        retrieved_sources = [
            {
                "chunk_id": chunk.chunk_id,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "score": chunk.similarity_score,
            }
            for chunk in retrieved_chunks
        ]

        result_id = str(uuid.uuid4())
        experiment_result = ExperimentResult(
            id=result_id,
            experiment_id=experiment_id,
            query_id=query.id,
            question=query.question,
            expected_answer=query.expected_answer,
            generated_answer=answer,
            expected_sources=query.expected_sources,
            retrieved_sources=retrieved_sources,
            latency_ms=latency_ms,
            faithfulness_score=faithfulness_score,
            relevancy_score=relevancy_score,
            precision_score=precision_score,
            recall_score=recall_score,
            grounding_score=grounding_score,
            is_grounded=is_grounded,
            failure_type=None,
            failure_details={},
            trace_id=None,
        )

        self.session.add(experiment_result)
        await self.session.commit()

        return {
            "result_id": result_id,
            "latency_ms": latency_ms,
            "faithfulness_score": faithfulness_score,
            "relevancy_score": relevancy_score,
            "precision_score": precision_score,
            "recall_score": recall_score,
            "grounding_score": grounding_score,
        }


def _to_retrieved_chunk(chunk) -> RetrievedChunk:
    """Convert a retrieval result (any type) into a RetrievedChunk."""
    if isinstance(chunk, RetrievedChunk):
        return chunk
    if isinstance(chunk, RerankedChunk):
        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            document_name=chunk.document_name,
            page_number=chunk.page_number,
            similarity_score=chunk.reranker_score,
        )
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        document_name=chunk.document_name,
        page_number=chunk.page_number,
        similarity_score=getattr(chunk, "similarity_score", 0.0),
    )