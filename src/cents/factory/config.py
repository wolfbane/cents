"""Factory configuration — TOML file at ~/.cents/factory.toml or CENTS_FACTORY_CONFIG."""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


DEFAULT_TOML = """\
# cents factory configuration

universe = "default"           # name of universe to walk (or "default" to use the marked-default)
budget_usd = 100000.0          # max gross notional across open positions
target_positions = 30          # informs default per-position sizing (budget / target_positions)
entry_threshold = 7.0          # |conviction_delta| required to open a new thesis
preemption_margin = 5.0        # new thesis must beat lowest open conviction by this margin to preempt
cohort_mode = "directional_only"  # "paired" (long + sector ETF short twin) or "directional_only"
default_horizon_days = 30
default_stop_pct = -5.0        # close-position trigger as % off entry (negative)
default_target_pct = 10.0      # close-position trigger as % off entry (positive)
max_new_per_run = 5            # rate-limit on new theses opened per run
max_per_premise_tag = 2        # max open theses sharing any single premise tag (0 disables)
"""


@dataclass
class FactoryConfig:
    """Resolved factory configuration."""

    universe: str = "default"
    budget_usd: float = 100000.0
    target_positions: int = 30
    entry_threshold: float = 7.0
    preemption_margin: float = 5.0
    cohort_mode: str = "directional_only"
    default_horizon_days: int = 30
    default_stop_pct: float = -5.0
    default_target_pct: float = 10.0
    max_new_per_run: int = 5
    max_per_premise_tag: int = 2

    def __post_init__(self) -> None:
        if self.cohort_mode not in {"paired", "directional_only"}:
            raise ValueError(
                f"cohort_mode must be 'paired' or 'directional_only', got {self.cohort_mode!r}"
            )
        if self.budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        if self.target_positions <= 0:
            raise ValueError("target_positions must be positive")
        if self.entry_threshold < 0:
            raise ValueError("entry_threshold must be non-negative")
        if self.max_new_per_run < 0:
            raise ValueError("max_new_per_run must be non-negative")
        if self.max_per_premise_tag < 0:
            raise ValueError("max_per_premise_tag must be non-negative")

    @property
    def position_size_usd(self) -> float:
        """Per-position dollar sizing derived from budget / target_positions."""
        return self.budget_usd / self.target_positions


def get_factory_config_path() -> Path:
    """Resolve the factory config path.

    Priority:
    1. CENTS_FACTORY_CONFIG env var
    2. ~/.cents/factory.toml
    """
    env_path = os.environ.get("CENTS_FACTORY_CONFIG")
    if env_path:
        return Path(env_path)
    return Path.home() / ".cents" / "factory.toml"


def load_factory_config(path: Path | None = None) -> FactoryConfig:
    """Load factory config from TOML, falling back to defaults if missing."""
    target = path or get_factory_config_path()
    if not target.exists():
        return FactoryConfig()

    try:
        data = tomllib.loads(target.read_text())
    except tomllib.TOMLDecodeError as exc:
        logger.warning("Failed to parse factory config %s: %s", target, exc)
        return FactoryConfig()
    except OSError as exc:
        logger.warning("Failed to read factory config %s: %s", target, exc)
        return FactoryConfig()

    fields = {
        "universe": data.get("universe", "default"),
        "budget_usd": float(data.get("budget_usd", 100000.0)),
        "target_positions": int(data.get("target_positions", 30)),
        "entry_threshold": float(data.get("entry_threshold", 7.0)),
        "preemption_margin": float(data.get("preemption_margin", 5.0)),
        "cohort_mode": str(data.get("cohort_mode", "directional_only")),
        "default_horizon_days": int(data.get("default_horizon_days", 30)),
        "default_stop_pct": float(data.get("default_stop_pct", -5.0)),
        "default_target_pct": float(data.get("default_target_pct", 10.0)),
        "max_new_per_run": int(data.get("max_new_per_run", 5)),
        "max_per_premise_tag": int(data.get("max_per_premise_tag", 2)),
    }
    return FactoryConfig(**fields)


def scaffold_factory_config(path: Path | None = None, force: bool = False) -> Path:
    """Write the default TOML at the resolved path. Returns the path written."""
    target = path or get_factory_config_path()
    if target.exists() and not force:
        raise FileExistsError(f"Factory config already exists at {target}. Use --force to overwrite.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(DEFAULT_TOML)
    return target
