"""API-level tests for experiment endpoints (create + background run)."""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks

from app.api.experiments import create_experiment, run_experiment
from app.models.schemas import ExperimentConfig, ExperimentCreateRequest

DATASET = Path(__file__).resolve().parents[2] / "evaluation" / "questions_mini.json"


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.add = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def create_request():
    return ExperimentCreateRequest(
        name="test-experiment",
        description="api test",
        config=ExperimentConfig(),
        dataset_path=str(DATASET),
    )


class TestCreateExperiment:
    async def test_creates_experiment_with_question_count(self, mock_session, create_request):
        experiment = await create_experiment(create_request, mock_session)
        assert experiment.status == "created"
        assert experiment.total_queries == 5
        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()

    async def test_rejects_missing_dataset(self, mock_session):
        request = ExperimentCreateRequest(
            name="bad",
            config=ExperimentConfig(),
            dataset_path="/nonexistent/benchmark.json",
        )
        with pytest.raises(Exception) as exc_info:
            await create_experiment(request, mock_session)
        assert exc_info.value.status_code == 400


class TestRunExperiment:
    async def test_schedules_background_task_with_fresh_session(self):
        session = AsyncMock()
        session.execute = AsyncMock(
            return_value=AsyncMock(
                scalar_one_or_none=MagicMock(return_value=MagicMock(status="created"))
            )
        )

        background_tasks = BackgroundTasks()
        response = await run_experiment("exp-123", background_tasks, session)

        assert response["message"] == "Experiment started"
        assert len(background_tasks.tasks) == 1

        task = background_tasks.tasks[0]
        task_func, task_args, task_kwargs = task.func, task.args, task.kwargs
        assert task_args == ()

        mock_cm = AsyncMock()
        mock_task_session = AsyncMock()
        mock_cm.__aenter__.return_value = mock_task_session
        mock_runner = AsyncMock()
        mock_runner.run_experiment = AsyncMock()

        with patch(
            "app.api.experiments.async_session_factory", return_value=mock_cm
        ), patch(
            "app.api.experiments.ExperimentRunner", return_value=mock_runner
        ):
            await task_func(*task_args, **task_kwargs)

        mock_runner.run_experiment.assert_awaited_once_with("exp-123")
        mock_cm.__aexit__.assert_awaited_once()

    async def test_rejects_running_experiment(self):
        session = AsyncMock()
        session.execute = AsyncMock(
            return_value=AsyncMock(
                scalar_one_or_none=MagicMock(return_value=MagicMock(status="running"))
            )
        )
        with pytest.raises(Exception) as exc_info:
            await run_experiment("exp-123", BackgroundTasks(), session)
        assert exc_info.value.status_code == 400

    async def test_404_for_missing_experiment(self):
        session = AsyncMock()
        session.execute = AsyncMock(
            return_value=AsyncMock(scalar_one_or_none=MagicMock(return_value=None))
        )
        with pytest.raises(Exception) as exc_info:
            await run_experiment("exp-123", BackgroundTasks(), session)
        assert exc_info.value.status_code == 404