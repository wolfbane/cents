"""Factory package — autonomous loop that walks a universe, opens/closes theses, and writes run logs."""

from cents.factory.config import FactoryConfig, load_factory_config, scaffold_factory_config
from cents.factory.engine import FactoryEngine
from cents.factory.universe_resolver import resolve_symbols
from cents.factory.sector_map import hedge_etf_for, SECTOR_ETF_MAP

__all__ = [
    "FactoryConfig",
    "FactoryEngine",
    "load_factory_config",
    "scaffold_factory_config",
    "resolve_symbols",
    "hedge_etf_for",
    "SECTOR_ETF_MAP",
]
