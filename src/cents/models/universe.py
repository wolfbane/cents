"""Universe domain model — named collection of symbols for the factory."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4


class UniverseSource(str, Enum):
    STATIC = "static"
    WATCHLIST = "watchlist"
    FMP_INDEX = "fmp_index"
    SCREENER = "screener"


@dataclass
class Universe:
    """A named collection of symbols the factory walks."""

    name: str
    description: str = ""
    source: UniverseSource = UniverseSource.STATIC
    source_config: dict = field(default_factory=dict)
    symbols: list[str] = field(default_factory=list)
    is_default: bool = False
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Universe name must be non-empty")
        self.name = self.name.strip().lower()
        if self.source == UniverseSource.FMP_INDEX and not self.source_config.get("index"):
            raise ValueError("FMP_INDEX universe requires source_config['index']")
        if self.source == UniverseSource.SCREENER and not self.source_config.get("strategy"):
            raise ValueError("SCREENER universe requires source_config['strategy']")
        self.symbols = [s.strip().upper() for s in self.symbols if s and s.strip()]
