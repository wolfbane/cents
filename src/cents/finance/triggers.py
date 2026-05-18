"""Direction-aware target/stop trigger predicates.

Both the factory engine's close phase and the CLI ``cents recommend`` rule
engine need to ask "did this price hit the target?" / "did this price hit the
stop?" — with semantics that flip for SHORT theses (target sits below entry,
stop above). Keeping the predicates in one place stops the two consumers from
drifting.
"""

from __future__ import annotations

from cents.models import PositionSide


def target_hit(side: PositionSide, price: float, target: float | None) -> bool:
    """Did ``price`` cross ``target`` in the winning direction for ``side``?"""
    if target is None:
        return False
    if side == PositionSide.SHORT:
        return price <= target
    return price >= target


def stop_hit(side: PositionSide, price: float, stop: float | None) -> bool:
    """Did ``price`` cross ``stop`` in the losing direction for ``side``?"""
    if stop is None:
        return False
    if side == PositionSide.SHORT:
        return price >= stop
    return price <= stop
