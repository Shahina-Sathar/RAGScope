"""Experiment comparison utilities."""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import Experiment, ExperimentResult


class ExperimentComparator:
    """Compares multiple experiments and provides insights."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def compare_experiments(self, experiment_ids: list[str]) -> dict:
        """Compare multiple experiments and provide insights."""
        if len(experiment_ids) < 2:
            raise ValueError("At least 2 experiments are required for comparison")

        # Load experiments
        experiments = []
        for exp_id in experiment_ids:
            result = await self.session.execute(
                select(Experiment).where(Experiment.id == exp_id)
            )
            exp = result.scalar_one_or_none()
            if exp:
                experiments.append(exp)

        if len(experiments) < 2:
            raise ValueError("Could not load enough experiments for comparison")

        # Compare metrics
        metrics_comparison = self._compare_metrics(experiments)

        # Compare failures
        failures_comparison = await self._compare_failures(experiment_ids)

        # Generate recommendations
        recommendations = self._generate_recommendations(experiments, metrics_comparison, failures_comparison)

        return {
            "experiment_ids": experiment_ids,
            "experiments": [
                {
                    "id": exp.id,
                    "name": exp.name,
                    "config": exp.config,
                    "avg_latency_ms": exp.avg_latency_ms,
                    "avg_faithfulness": exp.avg_faithfulness,
                    "avg_relevancy": exp.avg_relevancy,
                    "avg_precision": exp.avg_precision,
                    "avg_recall": exp.avg_recall,
                    "avg_grounding_score": exp.avg_grounding_score,
                }
                for exp in experiments
            ],
            "metrics_comparison": metrics_comparison,
            "failures_comparison": failures_comparison,
            "recommendations": recommendations,
        }

    def _compare_metrics(self, experiments: list[Experiment]) -> dict:
        """Compare metrics across experiments."""
        comparison = {}

        # Compare latency
        latencies = [exp.avg_latency_ms for exp in experiments if exp.avg_latency_ms is not None]
        if latencies:
            comparison["latency"] = {
                "min": min(latencies),
                "max": max(latencies),
                "avg": sum(latencies) / len(latencies),
                "best_experiment": experiments[latencies.index(min(latencies))].id,
            }

        # Compare faithfulness
        faithfulness = [exp.avg_faithfulness for exp in experiments if exp.avg_faithfulness is not None]
        if faithfulness:
            comparison["faithfulness"] = {
                "min": min(faithfulness),
                "max": max(faithfulness),
                "avg": sum(faithfulness) / len(faithfulness),
                "best_experiment": experiments[faithfulness.index(max(faithfulness))].id,
            }

        # Compare relevancy
        relevancy = [exp.avg_relevancy for exp in experiments if exp.avg_relevancy is not None]
        if relevancy:
            comparison["relevancy"] = {
                "min": min(relevancy),
                "max": max(relevancy),
                "avg": sum(relevancy) / len(relevancy),
                "best_experiment": experiments[relevancy.index(max(relevancy))].id,
            }

        # Compare precision
        precision = [exp.avg_precision for exp in experiments if exp.avg_precision is not None]
        if precision:
            comparison["precision"] = {
                "min": min(precision),
                "max": max(precision),
                "avg": sum(precision) / len(precision),
                "best_experiment": experiments[precision.index(max(precision))].id,
            }

        # Compare recall
        recall = [exp.avg_recall for exp in experiments if exp.avg_recall is not None]
        if recall:
            comparison["recall"] = {
                "min": min(recall),
                "max": max(recall),
                "avg": sum(recall) / len(recall),
                "best_experiment": experiments[recall.index(max(recall))].id,
            }

        # Compare grounding
        grounding = [exp.avg_grounding_score for exp in experiments if exp.avg_grounding_score is not None]
        if grounding:
            comparison["grounding"] = {
                "min": min(grounding),
                "max": max(grounding),
                "avg": sum(grounding) / len(grounding),
                "best_experiment": experiments[grounding.index(max(grounding))].id,
            }

        return comparison

    async def _compare_failures(self, experiment_ids: list[str]) -> dict:
        """Compare failure patterns across experiments."""
        comparison = {}

        for exp_id in experiment_ids:
            result = await self.session.execute(
                select(ExperimentResult).where(ExperimentResult.experiment_id == exp_id)
            )
            results = result.scalars().all()

            # Count failures by type
            failure_counts = {}
            for exp_result in results:
                if exp_result.failure_type:
                    failure_types = exp_result.failure_type.split("+")
                    for ft in failure_types:
                        failure_counts[ft] = failure_counts.get(ft, 0) + 1

            comparison[exp_id] = {
                "total_queries": len(results),
                "queries_with_failures": sum(1 for r in results if r.failure_type),
                "failure_counts": failure_counts,
            }

        return comparison

    def _generate_recommendations(
        self, experiments: list[Experiment], metrics_comparison: dict, failures_comparison: dict
    ) -> list[str]:
        """Generate recommendations based on comparison."""
        recommendations = []

        # Find best performing experiment
        if "faithfulness" in metrics_comparison:
            best_exp_id = metrics_comparison["faithfulness"]["best_experiment"]
            best_exp = next((exp for exp in experiments if exp.id == best_exp_id), None)
            if best_exp:
                recommendations.append(
                    f"Experiment '{best_exp.name}' has the best faithfulness score. "
                    f"Consider using its configuration: {best_exp.config}"
                )

        # Check for high latency
        if "latency" in metrics_comparison:
            max_latency = metrics_comparison["latency"]["max"]
            if max_latency > 5000:
                recommendations.append(
                    f"Some experiments have high latency (>5s). "
                    f"Consider reducing top_k or disabling expensive features like reranking"
                )

        # Check for common failure patterns
        all_failure_types = {}
        for exp_id, failure_data in failures_comparison.items():
            for failure_type, count in failure_data["failure_counts"].items():
                all_failure_types[failure_type] = all_failure_types.get(failure_type, 0) + count

        if all_failure_types:
            most_common = max(all_failure_types, key=all_failure_types.get)
            recommendations.append(
                f"Most common failure type is '{most_common}'. "
                f"Focus on addressing this issue to improve overall performance"
            )

        # Check for low grounding scores
        if "grounding" in metrics_comparison:
            avg_grounding = metrics_comparison["grounding"]["avg"]
            if avg_grounding < 0.7:
                recommendations.append(
                    f"Average grounding score is low ({avg_grounding:.2f}). "
                    f"Consider enabling stricter grounding checks or improving retrieval quality"
                )

        # Check for low precision
        if "precision" in metrics_comparison:
            avg_precision = metrics_comparison["precision"]["avg"]
            if avg_precision < 0.7:
                recommendations.append(
                    f"Average precision is low ({avg_precision:.2f}). "
                    f"Consider using reranking or hybrid retrieval to improve precision"
                )

        # Check for low recall
        if "recall" in metrics_comparison:
            avg_recall = metrics_comparison["recall"]["avg"]
            if avg_recall < 0.7:
                recommendations.append(
                    f"Average recall is low ({avg_recall:.2f}). "
                    f"Consider increasing top_k or using different retrieval strategies"
                )

        return recommendations
