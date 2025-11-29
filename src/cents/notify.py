"""Notification system for alerts."""

import json
import os
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

from cents.models import Alert


def send_webhook(alert: Alert, webhook_url: Optional[str] = None) -> bool:
    """Send alert to webhook URL (Slack/Discord compatible)."""
    url = webhook_url or os.environ.get("CENTS_WEBHOOK_URL")
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


def notify(alert: Alert, webhook_url: Optional[str] = None) -> None:
    """Send notification for an alert."""
    # Always print to terminal
    print(f"[ALERT] {alert.symbol}: {alert.message}")

    # Try webhook if configured
    if webhook_url or os.environ.get("CENTS_WEBHOOK_URL"):
        if send_webhook(alert, webhook_url):
            print("  (webhook sent)")
        else:
            print("  (webhook failed)")
