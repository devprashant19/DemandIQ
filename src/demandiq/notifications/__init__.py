"""Notifications sub-package for DemandIQ anomaly push alerts."""

from demandiq.notifications.email_alert import send_email_alert
from demandiq.notifications.slack_alert import send_slack_alert

__all__ = ["send_email_alert", "send_slack_alert"]
