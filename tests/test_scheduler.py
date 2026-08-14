"""Unit tests for the background scheduler module."""

from unittest.mock import MagicMock, patch

from demandiq.monitoring import scheduler as sched_module
from demandiq.monitoring.scheduler import _retrain_job, schedule_retrain


def test_retrain_job_success() -> None:
    """Should call run_pipeline and log success."""
    with patch("demandiq.monitoring.scheduler.run_pipeline") as mock_pipeline:
        _retrain_job()
        mock_pipeline.assert_called_once()


def test_retrain_job_logs_error_on_exception() -> None:
    """Should catch exceptions and log an error without re-raising."""
    with patch("demandiq.monitoring.scheduler.run_pipeline", side_effect=RuntimeError("boom")):
        # Must not raise
        _retrain_job()


def test_schedule_retrain_starts_scheduler_and_adds_job() -> None:
    """Should start the scheduler if not running and add the cron job."""
    mock_scheduler = MagicMock()
    mock_scheduler.running = False
    mock_scheduler.get_jobs.return_value = []

    with patch.object(sched_module, "_scheduler", mock_scheduler):
        schedule_retrain("0 2 * * 0")

    mock_scheduler.start.assert_called_once()
    mock_scheduler.add_job.assert_called_once()


def test_schedule_retrain_skips_start_when_already_running() -> None:
    """Should not restart the scheduler if it is already running."""
    mock_scheduler = MagicMock()
    mock_scheduler.running = True
    mock_scheduler.get_jobs.return_value = []

    with patch.object(sched_module, "_scheduler", mock_scheduler):
        schedule_retrain("0 3 * * 1")

    mock_scheduler.start.assert_not_called()
    mock_scheduler.add_job.assert_called_once()


def test_schedule_retrain_removes_existing_jobs() -> None:
    """Should remove any pre-existing jobs before scheduling the new one."""
    existing_job = MagicMock()
    mock_scheduler = MagicMock()
    mock_scheduler.running = True
    mock_scheduler.get_jobs.return_value = [existing_job]

    with patch.object(sched_module, "_scheduler", mock_scheduler):
        schedule_retrain("0 2 * * 0")

    existing_job.remove.assert_called_once()


def test_schedule_retrain_invalid_cron_logs_error() -> None:
    """Should log an error on an invalid cron string without raising."""
    mock_scheduler = MagicMock()
    mock_scheduler.running = True
    mock_scheduler.get_jobs.return_value = []

    with (
        patch.object(sched_module, "_scheduler", mock_scheduler),
        patch(
            "demandiq.monitoring.scheduler.CronTrigger.from_crontab",
            side_effect=ValueError("bad cron"),
        ),
    ):
        # Must not raise
        schedule_retrain("bad-cron-expression")
