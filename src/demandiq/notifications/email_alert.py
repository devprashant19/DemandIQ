"""Email notification module for anomaly alerts."""

import logging
import smtplib
from email.message import EmailMessage

from demandiq.config import settings

logger = logging.getLogger(__name__)


def send_email_alert(city: str, markdown_digest: str) -> bool:
    """Send an anomaly digest via email.

    Args:
        city: City where anomalies were detected.
        markdown_digest: The markdown formatted digest content.

    Returns:
        bool: True if sent successfully, False otherwise.
    """
    if not settings.alert_enabled:
        logger.info("Alerts are disabled in configuration. Skipping email alert.")
        return False

    if not all([settings.smtp_host, settings.smtp_port, settings.smtp_to]):
        logger.warning("SMTP configuration is incomplete. Skipping email alert.")
        return False

    try:
        msg = EmailMessage()
        msg.set_content(markdown_digest)
        msg["Subject"] = f"DemandIQ Anomaly Alert: {city}"
        msg["From"] = settings.smtp_user or "demandiq@localhost"
        msg["To"] = settings.smtp_to

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            if settings.smtp_user and settings.smtp_password:
                server.starttls()
                server.login(settings.smtp_user, settings.smtp_password.get_secret_value() if hasattr(settings.smtp_password, 'get_secret_value') else settings.smtp_password)
            server.send_message(msg)
        
        logger.info("Successfully sent anomaly email alert to %s", settings.smtp_to)
        return True
    except Exception as e:
        logger.error("Failed to send email alert: %s", e)
        return False
