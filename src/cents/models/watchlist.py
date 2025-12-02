"""Watchlist domain model."""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass
class WatchlistItem:
    """A symbol being watched for research opportunities."""

    symbol: str
    notes: str = ""
    thesis_id: str | None = None  # Optional link to thesis
    threshold: float | None = None  # Custom conviction delta threshold
    alert_destination: str | None = None  # Custom webhook/alert target
    last_scanned: datetime | None = None
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
