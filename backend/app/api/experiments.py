"""Experiment API endpoints."""
import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_session, async_session_factory
from app.experiments.analyzer import FailureAnalyzer
from app.experiments.comparison import ExperimentComparator
from app.experiments.runner import ExperimentRunner
from app.models.schemas import (
    Experiment,
    ExperimentResult,
    ExperimentCreateRequest,
    ExperimentResponse,
    ExperimentListResponse,
    ExperimentResultResponse,
    ExperimentComparisonResponse,
)

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentCreateRequest,
    session: AsyncSession = Depends(get_session),
):
    """Create a new experiment."""
    experiment = Experiment(
        name=request.name,
        description=request.description,
        config=request.config.model_dump(),
        dataset_path=request.dataset_path,
        status="created",
    )
    
    # Count queries in dataset
    from app.evaluation.dataset import load_benchmark_dataset
    try:
        dataset = load_benchmark_dataset(request.dataset_path)
        experiment.total_queries = len(dataset.questions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {str(e)}")
    
    session.add(experiment)
    await session.commit()
    await session.refresh(experiment)
    
    return experiment


@router.get("", response_model=list[ExperimentListResponse])
async def list_experiments(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    """List all experiments."""
    query = select(Experiment)
    
    if status:
        query = query.where(Experiment.status == status)
    
    query = query.offset(skip).limit(limit).order_by(Experiment.created_at.desc())
    
    result = await session.execute(query)
    experiments = result.scalars().all()
    
    return experiments


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get a specific experiment."""
    result = await session.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return experiment


@router.post("/{experiment_id}/run")
async def run_experiment(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    """Run an experiment in the background."""
    # Check if experiment exists
    result = await session.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.status == "running":
        raise HTTPException(status_code=400, detail="Experiment is already running")
    
    # Run experiment in background
    async def run_experiment_task():
        async with async_session_factory() as task_session:
            runner = ExperimentRunner(task_session)
            await runner.run_experiment(experiment_id)
    
    background_tasks.add_task(run_experiment_task)
    
    return {"message": "Experiment started", "experiment_id": experiment_id}


@router.get("/{experiment_id}/results", response_model=list[ExperimentResultResponse])
async def get_experiment_results(
    experiment_id: str,
    skip: int = 0,
    limit: int = 100,
    failure_type: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    """Get results for an experiment."""
    query = select(ExperimentResult).where(ExperimentResult.experiment_id == experiment_id)
    
    if failure_type:
        query = query.where(ExperimentResult.failure_type.contains(failure_type))
    
    query = query.offset(skip).limit(limit).order_by(ExperimentResult.created_at.desc())
    
    result = await session.execute(query)
    results = result.scalars().all()
    
    return results


@router.post("/{experiment_id}/analyze")
async def analyze_experiment(
    experiment_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Analyze experiment results for failures."""
    # Check if experiment exists
    result = await session.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.status != "completed":
        raise HTTPException(status_code=400, detail="Experiment must be completed before analysis")
    
    # Run analysis
    analyzer = FailureAnalyzer(session)
    analysis = await analyzer.analyze_experiment(experiment_id)
    
    return analysis


@router.get("/{experiment_id}/failure-summary")
async def get_failure_summary(
    experiment_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get failure summary for an experiment."""
    analyzer = FailureAnalyzer(session)
    summary = await analyzer.get_failure_summary(experiment_id)
    
    return summary


@router.post("/compare", response_model=ExperimentComparisonResponse)
async def compare_experiments(
    experiment_ids: list[str],
    session: AsyncSession = Depends(get_session),
):
    """Compare multiple experiments."""
    if len(experiment_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 experiments are required for comparison")
    
    comparator = ExperimentComparator(session)
    comparison = await comparator.compare_experiments(experiment_ids)
    
    return comparison


@router.delete("/{experiment_id}")
async def delete_experiment(
    experiment_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Delete an experiment and its results."""
    # Check if experiment exists
    result = await session.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()
    
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.status == "running":
        raise HTTPException(status_code=400, detail="Cannot delete a running experiment")
    
    # Delete experiment results
    await session.execute(
        delete(ExperimentResult).where(ExperimentResult.experiment_id == experiment_id)
    )
    
    # Delete experiment
    await session.delete(experiment)
    await session.commit()
    
    return {"message": "Experiment deleted", "experiment_id": experiment_id}
