"""Screeners — the discovery layer for the factory.

A screener turns a (possibly very large) candidate universe into a ranked,
truncated list of symbols that pass a quant filter. Different screeners
encode different discovery strategies (value, growth, momentum, etc.); the
factory eventually labels every thesis with the screener that surfaced it so
downstream analytics can ask which discovery strategies actually pay off.

Protocol
--------
A screener is any object satisfying:

    class Screener(Protocol):
        name: str
        def describe(self) -> dict:
            '''Human-readable rule set: {"description": ..., "rules": [...]}.'''
        def screen(self, candidate_symbols: list[str] | None = None) -> list[str]:
            '''Return symbols passing the screen, strongest signal first.

            If ``candidate_symbols`` is provided, only those are evaluated.
            If ``None``, the screener may consult a full-universe source — but
            callers should gate that path explicitly.'''

Adding a screener
-----------------
1. Create ``src/cents/screeners/<name>.py`` with a class implementing the
   protocol. Reuse FMP/Alpaca singletons via the lazy ``_get_*_provider``
   helpers in the existing screeners. Each per-symbol fetch should be wrapped
   so a single failure skips that symbol rather than aborting the screen.
2. Register it in :data:`SCREENERS` below.
3. Add a test in ``tests/test_screeners.py`` that mocks the data provider.

Output ordering
---------------
Each screener returns symbols sorted by signal strength descending so that
``--limit N`` returns the strongest N. Ties are broken by symbol for
determinism.
"""

from typing import Protocol

from cents.screeners.growth import GrowthScreener
from cents.screeners.insider_cluster import InsiderClusterScreener
from cents.screeners.mean_reversion import MeanReversionScreener
from cents.screeners.momentum import MomentumScreener
from cents.screeners.value import ValueScreener


class Screener(Protocol):
    name: str

    def describe(self) -> dict: ...

    def screen(
        self,
        candidate_symbols: list[str] | None = None,
    ) -> list[str]: ...


SCREENERS: dict[str, Screener] = {
    "value": ValueScreener(),
    "growth": GrowthScreener(),
    "momentum": MomentumScreener(),
    "mean_reversion": MeanReversionScreener(),
    "insider_cluster": InsiderClusterScreener(),
}


def get_screener(name: str) -> Screener:
    """Return the screener registered under ``name`` or raise KeyError."""
    if name not in SCREENERS:
        raise KeyError(
            f"Unknown screener '{name}'. Available: {', '.join(sorted(SCREENERS))}"
        )
    return SCREENERS[name]


__all__ = [
    "Screener",
    "SCREENERS",
    "get_screener",
    "ValueScreener",
    "GrowthScreener",
    "MomentumScreener",
    "MeanReversionScreener",
    "InsiderClusterScreener",
]
