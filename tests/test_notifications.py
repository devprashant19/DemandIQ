"""Unit tests for the notifications module."""

from unittest.mock import MagicMock, patch

from demandiq.notifications.email_alert import send_email_alert
from demandiq.notifications.slack_alert import send_slack_alert


class TestEmailAlert:
    """Tests for the email alert module."""

    def test_alert_disabled_returns_false(self) -> None:
        """Should return False when alerts are disabled."""
        with patch("demandiq.notifications.email_alert.settings") as mock_settings:
            mock_settings.alert_enabled = False
            result = send_email_alert("NYC", "Some digest")
        assert result is False

    def test_incomplete_smtp_config_returns_false(self) -> None:
        """Should return False when SMTP config is incomplete."""
        with patch("demandiq.notifications.email_alert.settings") as mock_settings:
            mock_settings.alert_enabled = True
            mock_settings.smtp_host = ""
            mock_settings.smtp_port = None
            mock_settings.smtp_to = ""
            result = send_email_alert("NYC", "Some digest")
        assert result is False

    def test_smtp_exception_returns_false(self) -> None:
        """Should return False when SMTP raises an exception."""
        with (
            patch("demandiq.notifications.email_alert.settings") as mock_settings,
            patch("smtplib.SMTP") as mock_smtp,
        ):
            mock_settings.alert_enabled = True
            mock_settings.smtp_host = "smtp.example.com"
            mock_settings.smtp_port = 587
            mock_settings.smtp_to = "test@example.com"
            mock_settings.smtp_user = None
            mock_settings.smtp_password = None
            mock_smtp.side_effect = Exception("Connection refused")
            result = send_email_alert("NYC", "Some digest")
        assert result is False


class TestSlackAlert:
    """Tests for the Slack alert module."""

    def test_alert_disabled_returns_false(self) -> None:
        """Should return False when alerts are disabled."""
        with patch("demandiq.notifications.slack_alert.settings") as mock_settings:
            mock_settings.alert_enabled = False
            result = send_slack_alert("NYC", "Some digest")
        assert result is False

    def test_no_webhook_returns_false(self) -> None:
        """Should return False when webhook URL is not configured."""
        with patch("demandiq.notifications.slack_alert.settings") as mock_settings:
            mock_settings.alert_enabled = True
            mock_settings.slack_webhook_url = ""
            result = send_slack_alert("NYC", "Some digest")
        assert result is False

    def test_slack_exception_returns_false(self) -> None:
        """Should return False when HTTP request fails."""
        with (
            patch("demandiq.notifications.slack_alert.settings") as mock_settings,
            patch("urllib.request.urlopen") as mock_urlopen,
        ):
            mock_settings.alert_enabled = True
            mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
            mock_urlopen.side_effect = Exception("Network error")
            result = send_slack_alert("NYC", "Some digest")
        assert result is False

    def test_slack_success(self) -> None:
        """Should return True when Slack HTTP request succeeds."""
        with (
            patch("demandiq.notifications.slack_alert.settings") as mock_settings,
            patch("urllib.request.urlopen") as mock_urlopen,
        ):
            mock_settings.alert_enabled = True
            mock_settings.slack_webhook_url = "https://hooks.slack.com/test"
            mock_response = MagicMock()
            mock_response.status = 200
            mock_urlopen.return_value.__enter__ = MagicMock(return_value=mock_response)
            mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
            result = send_slack_alert("NYC", "Some digest")
        assert result is True
