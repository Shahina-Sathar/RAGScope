"""Failure analyzer for identifying RAG failure patterns."""
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import Experiment, ExperimentResult


class FailureAnalyzer:
    """Analyzes experiment results to identify failure patterns."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def analyze_experiment(self, experiment_id: str) -> dict:
        """Analyze all results for an experiment and identify failures."""
        # Load experiment results
        result = await self.session.execute(
            select(ExperimentResult).where(ExperimentResult.experiment_id == experiment_id)
        )
        results = result.scalars().all()

        if not results:
            return {"total_queries": 0, "failures": [], "failure_types": {}}

        # Analyze each result
        failures = []
        failure_types = {}

        for exp_result in results:
            failure_analysis = self._analyze_result(exp_result)
            
            if failure_analysis["failure_type"]:
                failures.append(
                    {
                        "query_id": exp_result.query_id,
                        "question": exp_result.question,
                        "failure_type": failure_analysis["failure_type"],
                        "details": failure_analysis["details"],
                        "recommendation": failure_analysis["recommendation"],
                    }
                )
                
                # Count failure types
                failure_type = failure_analysis["failure_type"]
                failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
                
                # Update result with failure info
                exp_result.failure_type = failure_analysis["failure_type"]
                exp_result.failure_details = failure_analysis["details"]

        await self.session.commit()

        # Update experiment with failure analysis
        exp_result_query = await self.session.execute(
            select(Experiment).where(Experiment.id == experiment_id)
        )
        experiment = exp_result_query.scalar_one()
        experiment.failure_analysis = {
            "total_failures": len(failures),
            "failure_types": failure_types,
            "failure_rate": len(failures) / len(results) if results else 0,
        }
        await self.session.commit()

        return {
            "total_queries": len(results),
            "total_failures": len(failures),
            "failure_rate": len(failures) / len(results) if results else 0,
            "failure_types": failure_types,
            "failures": failures,
        }

    def _analyze_result(self, result: ExperimentResult) -> dict:
        """Analyze a single experiment result for failures."""
        failure_type = None
        details = {}
        recommendation = None

        # Check for retrieval failures
        if not result.retrieved_sources:
            failure_type = "RETRIEVAL_FAILURE"
            details = {
                "reason": "No documents retrieved",
                "question": result.question,
            }
            recommendation = "Check if documents are properly indexed or adjust retrieval parameters"

        # Check for low retrieval scores
        elif result.retrieved_sources:
            avg_score = sum(s.get("score", 0) for s in result.retrieved_sources) / len(
                result.retrieved_sources
            )
            if avg_score < 0.5:
                failure_type = "LOW_RETRIEVAL_QUALITY"
                details = {
                    "reason": "Low retrieval scores",
                    "average_score": avg_score,
                    "threshold": 0.5,
                }
                recommendation = "Consider using hybrid retrieval or adjusting chunk size"

        # Check for grounding failures
        if result.is_grounded is False:
            if failure_type:
                # Multiple failures
                failure_type = f"{failure_type}+GROUNDING_FAILURE"
            else:
                failure_type = "GROUNDING_FAILURE"
            
            details["grounding_score"] = result.grounding_score
            details["grounding_threshold"] = 0.7
            recommendation = "Answer contains information not supported by context. Consider stricter grounding or better retrieval"

        # Check for low evaluation metrics
        if result.faithfulness_score is not None and result.faithfulness_score < 0.7:
            if failure_type:
                failure_type = f"{failure_type}+FAITHFULNESS_FAILURE"
            else:
                failure_type = "FAITHFULNESS_FAILURE"
            
            details["faithfulness_score"] = result.faithfulness_score
            details["threshold"] = 0.7
            recommendation = "Answer is not faithful to context. Check retrieval quality or generation prompt"

        if result.relevancy_score is not None and result.relevancy_score < 0.7:
            if failure_type:
                failure_type = f"{failure_type}+RELEVANCY_FAILURE"
            else:
                failure_type = "RELEVANCY_FAILURE"
            
            details["relevancy_score"] = result.relevancy_score
            details["threshold"] = 0.7
            recommendation = "Answer is not relevant to question. Check if retrieval is finding relevant documents"

        if result.precision_score is not None and result.precision_score < 0.7:
            if failure_type:
                failure_type = f"{failure_type}+PRECISION_FAILURE"
            else:
                failure_type = "PRECISION_FAILURE"
            
            details["precision_score"] = result.precision_score
            details["threshold"] = 0.7
            recommendation = "Retrieved context has low precision. Consider using reranking or hybrid retrieval"

        if result.recall_score is not None and result.recall_score < 0.7:
            if failure_type:
                failure_type = f"{failure_type}+RECALL_FAILURE"
            else:
                failure_type = "RECALL_FAILURE"
            
            details["recall_score"] = result.recall_score
            details["threshold"] = 0.7
            recommendation = "Retrieved context has low recall. Consider increasing top_k or using different retrieval strategy"

        # Check for high latency
        if result.latency_ms > 5000:  # 5 seconds
            if failure_type:
                failure_type = f"{failure_type}+HIGH_LATENCY"
            else:
                failure_type = "HIGH_LATENCY"
            
            details["latency_ms"] = result.latency_ms
            details["threshold_ms"] = 5000
            recommendation = "Query has high latency. Consider optimizing retrieval or reducing top_k"

        # If no failures detected
        if not failure_type:
            return {
                "failure_type": None,
                "details": {},
                "recommendation": None,
            }

        return {
            "failure_type": failure_type,
            "details": details,
            "recommendation": recommendation,
        }

    async def get_failure_summary(self, experiment_id: str) -> dict:
        """Get a summary of failures for an experiment."""
        result = await self.session.execute(
            select(ExperimentResult).where(ExperimentResult.experiment_id == experiment_id)
        )
        results = result.scalars().all()

        if not results:
            return {"total_queries": 0, "failure_summary": {}}

        # Count failures by type
        failure_counts = {}
        for exp_result in results:
            if exp_result.failure_type:
                # Handle compound failures
                failure_types = exp_result.failure_type.split("+")
                for ft in failure_types:
                    failure_counts[ft] = failure_counts.get(ft, 0) + 1

        # Calculate metrics
        total_queries = len(results)
        queries_with_failures = sum(1 for r in results if r.failure_type)

        return {
            "total_queries": total_queries,
            "queries_with_failures": queries_with_failures,
            "failure_rate": queries_with_failures / total_queries if total_queries > 0 else 0,
            "failure_counts": failure_counts,
            "most_common_failure": max(failure_counts, key=failure_counts.get)
            if failure_counts
            else None,
        }
