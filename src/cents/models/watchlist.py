"""Watchlist domain model."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import uuid4


@dataclass
class WatchlistItem:
    """A symbol being watched for research opportunities."""

    symbol: str
    notes: str = ""
    thesis_id: Optional[str] = None  # Optional link to thesis
    threshold: Optional[float] = None  # Custom conviction delta threshold
    alert_destination: Optional[str] = None  # Custom webhook/alert target
    last_scanned: Optional[datetime] = None
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
