"""Delisting domain model — symbols that have left listed markets.

Tracked so that point-in-time universe resolution can include names that
were members of a screened index as of a past date but have since been
delisted. Without this, every screener universe is silently
survivorship-biased toward survivors.
"""

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class Delisting:
    """A symbol that has been delisted from public markets."""

    symbol: str
    delisted_on: date
    last_close: float | None = None
    source: str = "fmp"
    ingested_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("Delisting.symbol must be non-empty")
        self.symbol = self.symbol.strip().upper()
