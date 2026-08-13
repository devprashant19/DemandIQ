"""Slack notification module for anomaly alerts."""

import logging
import urllib.request
import urllib.error
import json

from demandiq.config import settings

logger = logging.getLogger(__name__)


def send_slack_alert(city: str, markdown_digest: str) -> bool:
    """Send an anomaly digest to Slack via webhook.

    Args:
        city: City where anomalies were detected.
        markdown_digest: The markdown formatted digest content.

    Returns:
        bool: True if sent successfully, False otherwise.
    """
    if not settings.alert_enabled:
        logger.info("Alerts are disabled in configuration. Skipping Slack alert.")
        return False

    if not settings.slack_webhook_url:
        logger.warning("Slack webhook URL is not configured. Skipping Slack alert.")
        return False

    payload = {
        "text": f"*DemandIQ Anomaly Alert: {city}*\n\n{markdown_digest}"
    }

    try:
        req = urllib.request.Request(
            settings.slack_webhook_url.get_secret_value() if hasattr(settings.slack_webhook_url, 'get_secret_value') else settings.slack_webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                logger.info("Successfully sent anomaly Slack alert.")
                return True
            else:
                logger.error("Failed to send Slack alert. Status: %s", response.status)
                return False
    except Exception as e:
        logger.error("Exception while sending Slack alert: %s", e)
        return False
