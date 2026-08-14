"""APScheduler configuration for automated pipeline retraining."""

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from demandiq.pipeline import run_pipeline

logger = logging.getLogger(__name__)

# Global scheduler instance
_scheduler = BackgroundScheduler()


def _retrain_job() -> None:
    """Wrapper function to execute the full pipeline during scheduled runs."""
    logger.info("Executing scheduled model retraining...")
    try:
        run_pipeline()
        logger.info("Scheduled model retraining completed successfully.")
    except Exception as e:
        logger.error("Scheduled model retraining failed: %s", e)


def schedule_retrain(cron_expr: str = "0 2 * * 0") -> None:
    """Schedule the model to retrain automatically based on a cron expression.

    Default is every Sunday at 2 AM.

    Args:
        cron_expr: Standard cron string (e.g., "0 2 * * 0").
    """
    if not _scheduler.running:
        _scheduler.start()
        logger.info("Background scheduler started.")

    # Clear existing jobs
    for job in _scheduler.get_jobs():
        job.remove()

    try:
        trigger = CronTrigger.from_crontab(cron_expr)
        _scheduler.add_job(
            _retrain_job, trigger=trigger, id="retrain_pipeline", replace_existing=True
        )
        logger.info("Model retraining scheduled with cron: '%s'", cron_expr)
    except ValueError as e:
        logger.error("Failed to parse cron expression '%s': %s", cron_expr, e)
