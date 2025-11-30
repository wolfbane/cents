"""Notification system for alerts."""

import json
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

from cents.models import Alert
from cents.config import get_settings


def send_webhook(alert: Alert, webhook_url: Optional[str] = None) -> bool:
    """Send alert to webhook URL (Slack/Discord compatible)."""
    settings = get_settings()
    url = webhook_url or settings.default_webhook
    if not url:
        return False

    # Format for Slack/Discord
    payload = {
        "text": f"*{alert.symbol}*: {alert.message}",
        "attachments": [
            {
                "color": "#36a64f" if "bullish" in alert.message.lower() else "#d00000",
                "fields": [
                    {"title": "Type", "value": alert.alert_type.value, "short": True},
                    {"title": "Time", "value": alert.created_at.strftime("%Y-%m-%d %H:%M"), "short": True},
                ],
            }
        ],
    }

    try:
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=10) as response:
            return response.status == 200
    except (URLError, Exception):
        return False


def notify(alert: Alert, webhook_url: Optional[str] = None, quiet: bool = False) -> None:
    """Send notification for an alert."""

    # Try webhook if configured
    settings = get_settings()
    destination = webhook_url or settings.default_webhook

    if not quiet:
        print(f"[ALERT] {alert.symbol}: {alert.message}")

    if destination:
        success = send_webhook(alert, destination)
        if not quiet:
            print("  (webhook sent)" if success else "  (webhook failed)")
