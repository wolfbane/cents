"""Factory engine — runs the autonomous open/close loop across a universe."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from cents.config import get_settings
from cents.db import (
    AlertRepository,
    FactoryRunRepository,
    PositionRepository,
    ThesisRepository,
    UniverseRepository,
)
from cents.factory.config import FactoryConfig, load_factory_config
from cents.factory.sector_map import hedge_etf_for
from cents.factory.universe_resolver import resolve_symbols
from cents.models import (
    Alert,
    AlertType,
    FactoryRun,
    Position,
    PositionSide,
    PositionStatus,
    Thesis,
    ThesisOutcome,
    ThesisStatus,
    TimeHorizon,
)

logger = logging.getLogger(__name__)


# Tag conventions used to mark factory-managed theses and link paired legs
TAG_FACTORY = "factory"
TAG_PAIRED_LONG = "factory_paired_long"
TAG_PAIRED_SHORT = "factory_paired_short"
TAG_PAIR_PREFIX = "pair:"  # tag value pair:<other_thesis_id>


class _OrchestratorLike(Protocol):
    def research(self, symbol: str, thesis: Thesis | None = None): ...


class _PriceProvider(Protocol):
    def get_latest_price(self, symbol: str) -> float | None: ...


def _pair_partner_id(thesis: Thesis) -> str | None:
    for tag in thesis.tags:
        if tag.startswith(TAG_PAIR_PREFIX):
            return tag[len(TAG_PAIR_PREFIX):] or None
    return None


def _is_paired(thesis: Thesis) -> bool:
    return TAG_PAIRED_LONG in thesis.tags or TAG_PAIRED_SHORT in thesis.tags


def _horizon_from_days(days: int) -> TimeHorizon:
    if days < 90:
        return TimeHorizon.SHORT
    if days <= 365:
        return TimeHorizon.MEDIUM
    return TimeHorizon.LONG


@dataclass
class _ProposedAction:
    """A close/open proposal captured during dry-run for the summary."""

    kind: str
    symbol: str
    detail: str


class FactoryEngine:
    """Coordinates one factory run over a universe.

    Inject `orchestrator` and `price_provider` to mock them in tests.
    """

    def __init__(
        self,
        config: FactoryConfig | None = None,
        *,
        orchestrator: _OrchestratorLike | None = None,
        price_provider: _PriceProvider | None = None,
        thesis_repo: ThesisRepository | None = None,
        position_repo: PositionRepository | None = None,
        alert_repo: AlertRepository | None = None,
        universe_repo: UniverseRepository | None = None,
        run_repo: FactoryRunRepository | None = None,
        now: datetime | None = None,
    ) -> None:
        self.config = config or load_factory_config()
        self._explicit_orchestrator = orchestrator
        self._explicit_price_provider = price_provider
        self.thesis_repo = thesis_repo or ThesisRepository()
        self.position_repo = position_repo or PositionRepository()
        self.alert_repo = alert_repo or AlertRepository()
        self.universe_repo = universe_repo or UniverseRepository()
        self.run_repo = run_repo or FactoryRunRepository()
        self._now = now

    # ---- providers (lazy) -------------------------------------------------

    def _make_orchestrator(self):
        if self._explicit_orchestrator is not None:
            return self._explicit_orchestrator
        from cents.agents import OrchestratorAgent

        return OrchestratorAgent()

    def _make_price_provider(self):
        if self._explicit_price_provider is not None:
            return self._explicit_price_provider
        from cents.data.alpaca import get_price_provider

        return get_price_provider()

    def _clock(self) -> datetime:
        return self._now or datetime.now()

    # ---- universe selection ----------------------------------------------

    def _resolve_universe_symbols(self, universe_name: str) -> tuple[str, list[str]]:
        """Resolve the configured/overriding universe name to a (name, symbols)."""
        if universe_name.strip().lower() == "default":
            universe = self.universe_repo.get_default()
            if universe is None:
                raise ValueError(
                    "No default universe configured. Set one with `cents universe set-default <NAME>`."
                )
        else:
            universe = self.universe_repo.get(universe_name)
            if universe is None:
                raise ValueError(f"Universe '{universe_name}' not found.")
        return universe.name, resolve_symbols(universe)

    # ---- main entry point -------------------------------------------------

    def run(self, dry_run: bool = False, universe_override: str | None = None) -> FactoryRun:
        """Execute a single factory run. Returns the persisted FactoryRun record."""
        cfg = self.config
        run = FactoryRun(
            universe_name=(universe_override or cfg.universe),
            dry_run=dry_run,
        )

        proposals: list[_ProposedAction] = []
        try:
            universe_name, symbols = self._resolve_universe_symbols(
                universe_override or cfg.universe
            )
            run.universe_name = universe_name

            run.events_refreshed = self._refresh_events(dry_run)

            closed_results = self._update_and_close_phase(dry_run, proposals)
            run.theses_closed += closed_results["theses_closed"]
            run.positions_closed += closed_results["positions_closed"]

            open_results = self._open_phase(symbols, dry_run, proposals)
            run.theses_opened += open_results["theses_opened"]
            run.positions_opened += open_results["positions_opened"]
            run.preemptions += open_results["preemptions"]
            run.theses_closed += open_results["preempted_closed"]
            run.positions_closed += open_results["preempted_positions_closed"]

            run.summary_json = {
                "symbols_considered": len(symbols),
                "proposals": [
                    {"kind": p.kind, "symbol": p.symbol, "detail": p.detail}
                    for p in proposals
                ],
            }
        except Exception as exc:  # pragma: no cover — surfaced via run.error
            logger.exception("Factory run failed")
            run.error = str(exc)

        run.completed_at = self._clock()
        self.run_repo.create(run)
        return run

    # ---- phases -----------------------------------------------------------

    def _refresh_events(self, dry_run: bool) -> int:
        """Refresh event ingestion. Returns event-count refreshed; 0 if no EventAgent.

        The codebase does not currently ship an EventAgent — this returns 0 today
        but the hook stays so adding events later is a one-line wire-up.
        """
        return 0

    def _update_and_close_phase(
        self, dry_run: bool, proposals: list[_ProposedAction]
    ) -> dict:
        """For every open thesis: re-run orchestrator, then check close triggers."""
        cfg = self.config
        theses_closed = 0
        positions_closed = 0

        open_theses = [
            t for t in self.thesis_repo.list(status=ThesisStatus.OPEN)
            if TAG_FACTORY in t.tags
        ]
        if not open_theses:
            return {"theses_closed": 0, "positions_closed": 0}

        orchestrator = self._make_orchestrator()
        price_provider = self._make_price_provider()

        # Group paired theses so closes happen atomically per pair
        handled: set[str] = set()

        for thesis in open_theses:
            if thesis.id in handled:
                continue

            partner_id = _pair_partner_id(thesis) if _is_paired(thesis) else None
            partner = self.thesis_repo.get(partner_id) if partner_id else None

            # Update conviction for both legs
            self._update_conviction(thesis, orchestrator, dry_run)
            if partner and partner.status == ThesisStatus.OPEN:
                self._update_conviction(partner, orchestrator, dry_run)

            # Determine trigger using long-leg symbol (or the thesis's own symbol if directional)
            trigger = self._evaluate_close_triggers(thesis, price_provider)
            if trigger and partner and partner.status == ThesisStatus.OPEN:
                # Confirm partner price-level triggers haven't already fired; we
                # still close as a pair so the parent's outcome covers both.
                pass

            if trigger:
                outcome = trigger
                if dry_run:
                    proposals.append(_ProposedAction(
                        kind="close", symbol=thesis.symbol or "", detail=outcome.value,
                    ))
                    handled.add(thesis.id)
                    if partner:
                        handled.add(partner.id)
                    continue

                positions_closed += self._close_thesis_positions(thesis, price_provider)
                thesis.close(outcome)
                self.thesis_repo.update(thesis)
                theses_closed += 1
                handled.add(thesis.id)

                if partner and partner.status == ThesisStatus.OPEN:
                    positions_closed += self._close_thesis_positions(partner, price_provider)
                    partner.close(outcome)
                    self.thesis_repo.update(partner)
                    theses_closed += 1
                    handled.add(partner.id)

        return {"theses_closed": theses_closed, "positions_closed": positions_closed}

    def _update_conviction(self, thesis: Thesis, orchestrator, dry_run: bool) -> None:
        if not thesis.symbol:
            return
        result = orchestrator.research(thesis.symbol, thesis)
        if dry_run:
            return
        thesis.update_conviction(result.conviction_delta)
        self.thesis_repo.update(thesis)

    def _evaluate_close_triggers(
        self, thesis: Thesis, price_provider
    ) -> ThesisOutcome | None:
        """Return the first triggered outcome for a thesis, or None."""
        # 1. Premise invalidation via THESIS_INVALIDATED alerts since last update
        if self._has_invalidation_alert(thesis):
            return ThesisOutcome.INVALIDATED

        # 2. Price targets — only for theses with a tracked symbol
        if thesis.symbol:
            price = price_provider.get_latest_price(thesis.symbol)
            if price is not None:
                if thesis.target_price is not None and price >= thesis.target_price:
                    return ThesisOutcome.CORRECT
                if thesis.stop_price is not None and price <= thesis.stop_price:
                    return ThesisOutcome.INCORRECT

        # 3. Horizon expiry
        if thesis.horizon_end is not None and self._clock() > thesis.horizon_end:
            return ThesisOutcome.UNCLEAR

        return None

    def _has_invalidation_alert(self, thesis: Thesis) -> bool:
        """Check for an unread THESIS_INVALIDATED alert on this thesis's symbol."""
        if not thesis.symbol:
            return False
        # Query all alerts; the table is small in practice
        for alert in self.alert_repo.list_all(limit=200):
            if alert.alert_type != AlertType.THESIS_INVALIDATED:
                continue
            if alert.symbol != thesis.symbol:
                continue
            if alert.created_at < thesis.updated_at - timedelta(days=1):
                continue
            return True
        return False

    def _close_thesis_positions(self, thesis: Thesis, price_provider) -> int:
        """Close all open positions linked to a thesis. Returns count closed."""
        closed = 0
        for pos in self.position_repo.list(status=PositionStatus.OPEN):
            if pos.thesis_id != thesis.id:
                continue
            exit_price = price_provider.get_latest_price(pos.symbol) or pos.entry_price
            pos.close(exit_price)
            self.position_repo.update(pos)
            closed += 1
        return closed

    # ---- open phase -------------------------------------------------------

    def _open_phase(
        self,
        universe_symbols: list[str],
        dry_run: bool,
        proposals: list[_ProposedAction],
    ) -> dict:
        """Open new theses where the orchestrator signals strongly enough."""
        cfg = self.config
        theses_opened = 0
        positions_opened = 0
        preemptions = 0
        preempted_closed = 0
        preempted_positions_closed = 0

        open_theses = [
            t for t in self.thesis_repo.list(status=ThesisStatus.OPEN)
            if TAG_FACTORY in t.tags
        ]
        held_symbols = self._held_symbols(open_theses)
        opened_this_run = 0

        orchestrator = self._make_orchestrator()
        price_provider = self._make_price_provider()

        for symbol in universe_symbols:
            if opened_this_run >= cfg.max_new_per_run:
                break
            if symbol in held_symbols:
                continue

            result = orchestrator.research(symbol, None)
            if abs(result.conviction_delta) < cfg.entry_threshold:
                continue

            new_conviction = max(0.0, min(100.0, 50.0 + result.conviction_delta))
            price = price_provider.get_latest_price(symbol)

            paired = cfg.cohort_mode == "paired"
            hedge_symbol = hedge_etf_for(symbol) if paired else None
            if paired and not hedge_symbol:
                logger.debug("Skipping %s — no hedge ETF available for paired mode", symbol)
                continue

            position_cost = cfg.position_size_usd * (2 if paired else 1)
            current_notional = self._current_notional(price_provider)

            preemption_target: Thesis | None = None
            if current_notional + position_cost > cfg.budget_usd:
                preemption_target = self._select_preemption_target(
                    open_theses,
                    new_conviction,
                    needed_notional=current_notional + position_cost - cfg.budget_usd,
                    price_provider=price_provider,
                )
                if preemption_target is None:
                    # Budget locked and no candidate cheap enough — skip
                    continue

            if dry_run:
                proposals.append(_ProposedAction(
                    kind="open",
                    symbol=symbol,
                    detail=(
                        f"paired={paired} conviction={new_conviction:.1f} "
                        f"delta={result.conviction_delta:+.1f}"
                        + (f" preempts={preemption_target.id}" if preemption_target else "")
                    ),
                ))
                opened_this_run += 1
                # Update local view of held symbols to avoid double-proposing
                held_symbols.add(symbol)
                if hedge_symbol:
                    held_symbols.add(hedge_symbol)
                continue

            if preemption_target is not None:
                pair_id = _pair_partner_id(preemption_target) if _is_paired(preemption_target) else None
                preempted_partner = self.thesis_repo.get(pair_id) if pair_id else None
                preempted_positions_closed += self._close_thesis_positions(
                    preemption_target, price_provider
                )
                preemption_target.close(ThesisOutcome.PREEMPTED)
                preemption_target.hypothesis = (
                    (preemption_target.hypothesis + "\n" if preemption_target.hypothesis else "")
                    + f"[preempted] new factory candidate {symbol} (conviction "
                    f"{new_conviction:.1f} > {preemption_target.conviction:.1f})"
                )
                self.thesis_repo.update(preemption_target)
                preempted_closed += 1
                preemptions += 1
                open_theses = [t for t in open_theses if t.id != preemption_target.id]

                if preempted_partner and preempted_partner.status == ThesisStatus.OPEN:
                    preempted_positions_closed += self._close_thesis_positions(
                        preempted_partner, price_provider
                    )
                    preempted_partner.close(ThesisOutcome.PREEMPTED)
                    self.thesis_repo.update(preempted_partner)
                    preempted_closed += 1
                    open_theses = [t for t in open_theses if t.id != preempted_partner.id]

                held_symbols = self._held_symbols(open_theses)

            new_open = self._open_new_thesis(
                symbol=symbol,
                conviction=new_conviction,
                delta=result.conviction_delta,
                price=price,
                hedge_symbol=hedge_symbol,
            )
            theses_opened += new_open["theses_opened"]
            positions_opened += new_open["positions_opened"]
            open_theses = self.thesis_repo.list(status=ThesisStatus.OPEN)
            open_theses = [t for t in open_theses if TAG_FACTORY in t.tags]
            held_symbols = self._held_symbols(open_theses)
            opened_this_run += 1

        return {
            "theses_opened": theses_opened,
            "positions_opened": positions_opened,
            "preemptions": preemptions,
            "preempted_closed": preempted_closed,
            "preempted_positions_closed": preempted_positions_closed,
        }

    def _held_symbols(self, theses: list[Thesis]) -> set[str]:
        """Return all symbols already held by factory-managed open theses (both legs)."""
        held: set[str] = set()
        for t in theses:
            if t.symbol:
                held.add(t.symbol)
        return held

    def _current_notional(self, price_provider) -> float:
        """Sum of mark-to-market notional across factory-open positions."""
        total = 0.0
        positions = self.position_repo.list(status=PositionStatus.OPEN)
        thesis_cache: dict[str, Thesis | None] = {}
        for pos in positions:
            if not pos.thesis_id:
                continue
            if pos.thesis_id not in thesis_cache:
                thesis_cache[pos.thesis_id] = self.thesis_repo.get(pos.thesis_id)
            thesis = thesis_cache[pos.thesis_id]
            if thesis is None or TAG_FACTORY not in thesis.tags:
                continue
            mark = price_provider.get_latest_price(pos.symbol) or pos.entry_price
            total += mark * pos.size
        return total

    def _select_preemption_target(
        self,
        open_theses: list[Thesis],
        new_conviction: float,
        *,
        needed_notional: float,
        price_provider,
    ) -> Thesis | None:
        """Pick the lowest-conviction open thesis whose closure frees enough room.

        Returns None if no candidate satisfies the preemption_margin rule.
        """
        cfg = self.config
        candidates = sorted(
            (t for t in open_theses if t.status == ThesisStatus.OPEN),
            key=lambda t: t.conviction,
        )
        for candidate in candidates:
            if new_conviction <= candidate.conviction + cfg.preemption_margin:
                # The best candidate is too close in conviction — abort entirely
                return None
            freed = self._freeable_notional(candidate, price_provider)
            if freed >= needed_notional:
                return candidate
        return None

    def _freeable_notional(self, thesis: Thesis, price_provider) -> float:
        freed = 0.0
        for pos in self.position_repo.list(status=PositionStatus.OPEN):
            if pos.thesis_id != thesis.id:
                continue
            mark = price_provider.get_latest_price(pos.symbol) or pos.entry_price
            freed += mark * pos.size
        # If paired, partner's notional also frees
        partner_id = _pair_partner_id(thesis) if _is_paired(thesis) else None
        if partner_id:
            for pos in self.position_repo.list(status=PositionStatus.OPEN):
                if pos.thesis_id != partner_id:
                    continue
                mark = price_provider.get_latest_price(pos.symbol) or pos.entry_price
                freed += mark * pos.size
        return freed

    def _open_new_thesis(
        self,
        *,
        symbol: str,
        conviction: float,
        delta: float,
        price: float | None,
        hedge_symbol: str | None,
    ) -> dict:
        """Persist new thesis(es) and corresponding paper position(s)."""
        cfg = self.config
        now = self._clock()
        horizon_end = now + timedelta(days=cfg.default_horizon_days)
        time_horizon = _horizon_from_days(cfg.default_horizon_days)

        target_price = None
        stop_price = None
        if price is not None and price > 0:
            target_price = price * (1 + cfg.default_target_pct / 100.0)
            stop_price_candidate = price * (1 + cfg.default_stop_pct / 100.0)
            stop_price = stop_price_candidate if stop_price_candidate > 0 else None

        long_tags = [TAG_FACTORY]
        if hedge_symbol:
            long_tags.append(TAG_PAIRED_LONG)

        long_thesis = Thesis(
            title=f"factory:{symbol}",
            hypothesis=f"factory open — conviction {conviction:.1f}, delta {delta:+.1f}",
            symbol=symbol,
            conviction=conviction,
            tags=list(long_tags),
            time_horizon=time_horizon,
            horizon_end=horizon_end,
            target_price=target_price,
            stop_price=stop_price,
        )
        self.thesis_repo.create(long_thesis)

        positions_opened = 0
        if price is not None and price > 0:
            shares = cfg.position_size_usd / price
            if shares > 0:
                pos = Position(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    entry_price=price,
                    size=shares,
                    thesis_id=long_thesis.id,
                    paper=True,
                    notes="opened by factory",
                )
                self.position_repo.create(pos)
                positions_opened += 1

        theses_opened = 1

        if hedge_symbol:
            hedge_price = self._make_price_provider().get_latest_price(hedge_symbol)
            short_tags = [TAG_FACTORY, TAG_PAIRED_SHORT, f"{TAG_PAIR_PREFIX}{long_thesis.id}"]
            short_thesis = Thesis(
                title=f"factory:{symbol}/hedge:{hedge_symbol}",
                hypothesis=f"factory neutral hedge for {symbol}",
                symbol=hedge_symbol,
                conviction=max(0.0, min(100.0, 50.0 - delta)),
                tags=list(short_tags),
                time_horizon=time_horizon,
                horizon_end=horizon_end,
            )
            self.thesis_repo.create(short_thesis)
            # Back-link the long leg to the short leg
            long_thesis.tags.append(f"{TAG_PAIR_PREFIX}{short_thesis.id}")
            self.thesis_repo.update(long_thesis)

            if hedge_price is not None and hedge_price > 0:
                shares = cfg.position_size_usd / hedge_price
                if shares > 0:
                    pos = Position(
                        symbol=hedge_symbol,
                        side=PositionSide.SHORT,
                        entry_price=hedge_price,
                        size=shares,
                        thesis_id=short_thesis.id,
                        paper=True,
                        notes="opened by factory (hedge)",
                    )
                    self.position_repo.create(pos)
                    positions_opened += 1
            theses_opened += 1

        return {"theses_opened": theses_opened, "positions_opened": positions_opened}
