"""Tests for notification system."""

import json
from unittest.mock import patch, MagicMock
from urllib.error import URLError

import pytest

from cents.models import Alert, AlertType
from cents.notify import send_webhook, notify


class TestSendWebhook:
    """Tests for send_webhook function."""

    def test_no_webhook_url_returns_false(self):
        """Returns False when no webhook URL configured."""
        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONVICTION_CHANGE,
            message="Test alert",
        )
        with patch("cents.notify.get_settings") as mock_settings:
            mock_settings.return_value.default_webhook = None
            result = send_webhook(alert, None)
            assert result is False

    @patch("cents.notify.urlopen")
    @patch("cents.notify.get_settings")
    def test_webhook_success(self, mock_settings, mock_urlopen):
        """Successfully sends webhook."""
        mock_settings.return_value.default_webhook = None
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONVICTION_CHANGE,
            message="Significant bullish signal",
        )
        result = send_webhook(alert, "https://hooks.example.com/webhook")
        assert result is True

    @patch("cents.notify.urlopen")
    @patch("cents.notify.get_settings")
    def test_webhook_uses_default_from_settings(self, mock_settings, mock_urlopen):
        """Uses default webhook from settings if not provided."""
        mock_settings.return_value.default_webhook = "https://default.webhook.com"
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONVICTION_CHANGE,
            message="Test alert",
        )
        result = send_webhook(alert, None)
        assert result is True

    @patch("cents.notify.urlopen")
    @patch("cents.notify.get_settings")
    def test_webhook_failure_returns_false(self, mock_settings, mock_urlopen):
        """Returns False on URLError."""
        mock_settings.return_value.default_webhook = None
        mock_urlopen.side_effect = URLError("Connection refused")

        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONVICTION_CHANGE,
            message="Test alert",
        )
        result = send_webhook(alert, "https://hooks.example.com/webhook")
        assert result is False

    @patch("cents.notify.urlopen")
    @patch("cents.notify.get_settings")
    def test_webhook_non_200_returns_false(self, mock_settings, mock_urlopen):
        """Returns False on non-200 response."""
        mock_settings.return_value.default_webhook = None
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONVICTION_CHANGE,
            message="Test alert",
        )
        result = send_webhook(alert, "https://hooks.example.com/webhook")
        assert result is False

    @patch("cents.notify.urlopen")
    @patch("cents.notify.get_settings")
    def test_webhook_payload_format(self, mock_settings, mock_urlopen):
        """Verifies webhook payload format."""
        mock_settings.return_value.default_webhook = None
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        alert = Alert(
            symbol="TSLA",
            alert_type=AlertType.PRICE_TRIGGER,
            message="Price target hit (bullish)",
        )
        send_webhook(alert, "https://hooks.example.com/webhook")

        # Get the Request object passed to urlopen
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        payload = json.loads(request.data.decode("utf-8"))

        assert payload["text"] == "*TSLA*: Price target hit (bullish)"
        assert payload["attachments"][0]["color"] == "#36a64f"  # bullish = green
        assert payload["attachments"][0]["fields"][0]["value"] == "price_trigger"

    @patch("cents.notify.urlopen")
    @patch("cents.notify.get_settings")
    def test_webhook_bearish_color(self, mock_settings, mock_urlopen):
        """Bearish alerts get red color."""
        mock_settings.return_value.default_webhook = None
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        alert = Alert(
            symbol="TSLA",
            alert_type=AlertType.CONVICTION_CHANGE,
            message="Significant bearish signal",
        )
        send_webhook(alert, "https://hooks.example.com/webhook")

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        payload = json.loads(request.data.decode("utf-8"))

        assert payload["attachments"][0]["color"] == "#d00000"  # bearish = red


class TestNotify:
    """Tests for notify function."""

    @patch("cents.notify.send_webhook")
    @patch("cents.notify.get_settings")
    def test_notify_prints_alert(self, mock_settings, mock_send, capsys):
        """Prints alert to stdout."""
        mock_settings.return_value.default_webhook = None

        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONVICTION_CHANGE,
            message="Test message",
        )
        notify(alert, quiet=False)

        captured = capsys.readouterr()
        assert "[ALERT] AAPL: Test message" in captured.out

    @patch("cents.notify.send_webhook")
    @patch("cents.notify.get_settings")
    def test_notify_quiet_mode(self, mock_settings, mock_send, capsys):
        """Quiet mode suppresses output."""
        mock_settings.return_value.default_webhook = None

        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONVICTION_CHANGE,
            message="Test message",
        )
        notify(alert, quiet=True)

        captured = capsys.readouterr()
        assert captured.out == ""

    @patch("cents.notify.send_webhook")
    @patch("cents.notify.get_settings")
    def test_notify_sends_webhook_when_configured(self, mock_settings, mock_send, capsys):
        """Sends webhook when URL is configured."""
        mock_settings.return_value.default_webhook = "https://webhook.example.com"
        mock_send.return_value = True

        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONVICTION_CHANGE,
            message="Test message",
        )
        notify(alert, quiet=False)

        mock_send.assert_called_once()
        captured = capsys.readouterr()
        assert "(webhook sent)" in captured.out

    @patch("cents.notify.send_webhook")
    @patch("cents.notify.get_settings")
    def test_notify_shows_webhook_failure(self, mock_settings, mock_send, capsys):
        """Shows webhook failure message."""
        mock_settings.return_value.default_webhook = "https://webhook.example.com"
        mock_send.return_value = False

        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONVICTION_CHANGE,
            message="Test message",
        )
        notify(alert, quiet=False)

        captured = capsys.readouterr()
        assert "(webhook failed)" in captured.out

    @patch("cents.notify.send_webhook")
    @patch("cents.notify.get_settings")
    def test_notify_uses_provided_webhook(self, mock_settings, mock_send):
        """Uses provided webhook URL over default."""
        mock_settings.return_value.default_webhook = "https://default.webhook.com"
        mock_send.return_value = True

        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONVICTION_CHANGE,
            message="Test message",
        )
        notify(alert, webhook_url="https://custom.webhook.com", quiet=True)

        mock_send.assert_called_once_with(alert, "https://custom.webhook.com")

    @patch("cents.notify.send_webhook")
    @patch("cents.notify.get_settings")
    def test_notify_no_webhook_no_call(self, mock_settings, mock_send):
        """Doesn't call send_webhook when no URL configured."""
        mock_settings.return_value.default_webhook = None

        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONVICTION_CHANGE,
            message="Test message",
        )
        notify(alert, quiet=True)

        mock_send.assert_not_called()
