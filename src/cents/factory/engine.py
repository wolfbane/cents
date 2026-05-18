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
    LLMUsageRepository,
    PositionRepository,
    ThesisRepository,
    UniverseRepository,
)
from cents.factory.config import FactoryConfig, load_factory_config
from cents.factory.premise import capture_regime_snapshot, classify_premise_tags
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
    ThesisCohort,
    ThesisOutcome,
    ThesisStatus,
    TimeHorizon,
)
from cents.pricing import estimate_cost_usd

logger = logging.getLogger(__name__)


# Tag marking factory-managed theses so user-created theses aren't touched.
TAG_FACTORY = "factory"


class _OrchestratorLike(Protocol):
    def research(self, symbol: str, thesis: Thesis | None = None): ...


class _PriceProvider(Protocol):
    def get_latest_price(self, symbol: str) -> float | None: ...


def _is_paired(thesis: Thesis) -> bool:
    return thesis.cohort == ThesisCohort.NEUTRAL


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
        event_agent=None,
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
        self._explicit_event_agent = event_agent
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

            open_results = self._open_phase(
                symbols,
                dry_run,
                proposals,
                skip_symbols=closed_results["invalidated_symbols"],
                discovery_source=universe_name,
            )
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
        self._accumulate_llm_cost(run)
        self.run_repo.create(run)
        return run

    # ---- phases -----------------------------------------------------------

    def _accumulate_llm_cost(self, run: FactoryRun) -> None:
        """Diff llm_usage rows in [started_at, completed_at] into the run record."""
        usage_repo = LLMUsageRepository()
        rows = usage_repo.list_recent(since=run.started_at, limit=10000)
        cost = 0.0
        any_priced = False
        for row in rows:
            if run.completed_at is not None and row.called_at > run.completed_at:
                continue
            run.llm_input_tokens += row.input_tokens or 0
            run.llm_output_tokens += row.output_tokens or 0
            row_cost = estimate_cost_usd(
                row.model,
                row.input_tokens or 0,
                row.output_tokens or 0,
                cache_read=row.cache_read_input_tokens or 0,
                cache_write=row.cache_creation_input_tokens or 0,
            )
            if row_cost is not None:
                cost += row_cost
                any_priced = True
        run.llm_cost_usd = cost if any_priced else None

    def _refresh_events(self, dry_run: bool) -> int:
        """Pull new policy/macro events and fire premise-invalidation alerts."""
        if dry_run:
            return 0
        agent = self._explicit_event_agent
        if agent is None:
            from cents.agents import EventAgent
            agent = EventAgent()
        summary = agent.refresh()
        return int(summary.get("new", 0))

    def _update_and_close_phase(
        self, dry_run: bool, proposals: list[_ProposedAction]
    ) -> dict:
        """For every open thesis: re-run orchestrator, then check close triggers.

        Neutral-cohort theses own both legs (long on `symbol`, short on
        `hedge_symbol`) via two positions tied to the same thesis_id, so
        closing the thesis closes both legs naturally.
        """
        theses_closed = 0
        positions_closed = 0
        # Symbols (and hedge symbols) invalidated this run — the open phase
        # must skip them so the invalidating event can't reopen them in the
        # same cycle before it has aged out.
        invalidated_symbols: set[str] = set()

        open_theses = [
            t for t in self.thesis_repo.list(status=ThesisStatus.OPEN)
            if TAG_FACTORY in t.tags
        ]
        if not open_theses:
            return {"theses_closed": 0, "positions_closed": 0, "invalidated_symbols": invalidated_symbols}

        orchestrator = self._make_orchestrator()
        price_provider = self._make_price_provider()

        for thesis in open_theses:
            self._update_conviction(thesis, orchestrator, dry_run)
            trigger = self._evaluate_close_triggers(thesis, price_provider)
            if not trigger:
                continue

            if dry_run:
                proposals.append(_ProposedAction(
                    kind="close", symbol=thesis.symbol or "", detail=trigger.value,
                ))
                if trigger == ThesisOutcome.INVALIDATED:
                    if thesis.symbol:
                        invalidated_symbols.add(thesis.symbol)
                    if thesis.hedge_symbol:
                        invalidated_symbols.add(thesis.hedge_symbol)
                continue

            positions_closed += self._close_thesis_positions(thesis, price_provider)
            thesis.close(trigger)
            self.thesis_repo.update(thesis)
            theses_closed += 1
            if trigger == ThesisOutcome.INVALIDATED:
                if thesis.symbol:
                    invalidated_symbols.add(thesis.symbol)
                if thesis.hedge_symbol:
                    invalidated_symbols.add(thesis.hedge_symbol)

        return {
            "theses_closed": theses_closed,
            "positions_closed": positions_closed,
            "invalidated_symbols": invalidated_symbols,
        }

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
        # 1. Premise invalidation: EventAgent fires PREMISE_INVALIDATION alerts
        #    keyed by thesis_id when a policy event hits a thesis's premise_tags.
        if self._has_invalidation_alert(thesis):
            return ThesisOutcome.INVALIDATED

        # 2. Price targets — direction-aware on the primary (underlying) leg.
        if thesis.symbol:
            price = price_provider.get_latest_price(thesis.symbol)
            if price is not None:
                direction = self._primary_direction(thesis)
                if direction == PositionSide.SHORT:
                    # Short thesis wins when price drops, loses when it rises.
                    if thesis.target_price is not None and price <= thesis.target_price:
                        return ThesisOutcome.CORRECT
                    if thesis.stop_price is not None and price >= thesis.stop_price:
                        return ThesisOutcome.INCORRECT
                else:
                    if thesis.target_price is not None and price >= thesis.target_price:
                        return ThesisOutcome.CORRECT
                    if thesis.stop_price is not None and price <= thesis.stop_price:
                        return ThesisOutcome.INCORRECT

        # 3. Horizon expiry
        if thesis.horizon_end is not None and self._clock() > thesis.horizon_end:
            return ThesisOutcome.UNCLEAR

        return None

    def _primary_direction(self, thesis: Thesis) -> PositionSide:
        """Return the position side that represents the thesis's directional bet.

        The primary leg is the one whose symbol matches `thesis.symbol`; for a
        neutral-cohort thesis the hedge leg sits opposite. Falls back to LONG
        when no positions exist yet (e.g. dry-run lookup pre-creation).
        """
        for pos in self.position_repo.list(status=PositionStatus.OPEN):
            if pos.thesis_id == thesis.id and pos.symbol == thesis.symbol:
                return pos.side
        return PositionSide.LONG

    def _has_invalidation_alert(self, thesis: Thesis) -> bool:
        """Check for a PREMISE_INVALIDATION alert targeting this thesis."""
        for alert in self.alert_repo.list_all(limit=200):
            if alert.alert_type != AlertType.PREMISE_INVALIDATION:
                continue
            if alert.data.get("thesis_id") != thesis.id:
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
        *,
        skip_symbols: set[str] | None = None,
        discovery_source: str | None = None,
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
        skip_symbols = set(skip_symbols or set())
        held_symbols = self._held_symbols(open_theses) | skip_symbols
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

            # Classify premise tags now (one LLM call) so we can gate on
            # per-tag concentration before paying any further setup cost.
            premise_tags = classify_premise_tags(
                symbol,
                result.summary,
                [getattr(e, "content", "") for e in (result.evidence or [])],
            )
            if cfg.max_per_premise_tag > 0 and self._exceeds_premise_concentration(
                premise_tags, open_theses, cfg.max_per_premise_tag
            ):
                logger.debug(
                    "Skipping %s — premise tags %s already at concentration cap",
                    symbol, premise_tags,
                )
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
                held_symbols = self._held_symbols(open_theses) | skip_symbols

            new_open = self._open_new_thesis(
                symbol=symbol,
                conviction=new_conviction,
                delta=result.conviction_delta,
                price=price,
                hedge_symbol=hedge_symbol,
                premise_tags=premise_tags,
                discovery_source=discovery_source,
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
            if t.hedge_symbol:
                held.add(t.hedge_symbol)
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
        """Sum mark-to-market notional across all positions tied to this thesis.

        For NEUTRAL theses this naturally includes both long and short legs since
        both are linked via thesis_id.
        """
        freed = 0.0
        for pos in self.position_repo.list(status=PositionStatus.OPEN):
            if pos.thesis_id != thesis.id:
                continue
            mark = price_provider.get_latest_price(pos.symbol) or pos.entry_price
            freed += mark * pos.size
        return freed

    def _exceeds_premise_concentration(
        self,
        candidate_tags: list[str],
        open_theses: list[Thesis],
        cap: int,
    ) -> bool:
        """True if any candidate tag is already at the per-tag cap across open theses."""
        if not candidate_tags:
            return False
        counts: dict[str, int] = {}
        for t in open_theses:
            for tag in t.premise_tags:
                counts[tag] = counts.get(tag, 0) + 1
        return any(counts.get(tag, 0) >= cap for tag in candidate_tags)

    def _open_new_thesis(
        self,
        *,
        symbol: str,
        conviction: float,
        delta: float,
        price: float | None,
        hedge_symbol: str | None,
        premise_tags: list[str] | None = None,
        discovery_source: str | None = None,
    ) -> dict:
        """Persist a new factory thesis and its position(s), oriented by signal sign.

        Bullish signal (delta > 0):
          - Directional: LONG underlying.
          - Paired: LONG underlying + SHORT hedge ETF.
          - target_price above entry, stop_price below.
        Bearish signal (delta < 0):
          - Directional: SHORT underlying.
          - Paired: SHORT underlying + LONG hedge ETF.
          - target_price below entry (price has to drop to win), stop_price above.
        """
        cfg = self.config
        now = self._clock()
        horizon_end = now + timedelta(days=cfg.default_horizon_days)
        time_horizon = _horizon_from_days(cfg.default_horizon_days)

        is_short = delta < 0
        primary_side = PositionSide.SHORT if is_short else PositionSide.LONG
        hedge_side = PositionSide.LONG if is_short else PositionSide.SHORT
        direction_label = "short" if is_short else "long"

        target_price = None
        stop_price = None
        if price is not None and price > 0:
            tgt_mult = (1 - cfg.default_target_pct / 100.0) if is_short else (1 + cfg.default_target_pct / 100.0)
            stp_mult = (1 - cfg.default_stop_pct / 100.0) if is_short else (1 + cfg.default_stop_pct / 100.0)
            target_candidate = price * tgt_mult
            stop_candidate = price * stp_mult
            target_price = target_candidate if target_candidate > 0 else None
            stop_price = stop_candidate if stop_candidate > 0 else None

        cohort = ThesisCohort.NEUTRAL if hedge_symbol else ThesisCohort.DIRECTIONAL
        title = (
            f"factory:{direction_label} {symbol}/hedge:{hedge_symbol}"
            if hedge_symbol else f"factory:{direction_label} {symbol}"
        )
        hypothesis = (
            f"factory open — {direction_label} conviction {conviction:.1f}, delta {delta:+.1f}"
            + (f" (paired-neutral vs {hedge_symbol})" if hedge_symbol else "")
        )

        regime_snapshot = capture_regime_snapshot(now=now)

        thesis = Thesis(
            title=title,
            hypothesis=hypothesis,
            symbol=symbol,
            conviction=conviction,
            tags=[TAG_FACTORY],
            time_horizon=time_horizon,
            horizon_end=horizon_end,
            target_price=target_price,
            stop_price=stop_price,
            cohort=cohort,
            hedge_symbol=hedge_symbol,
            premise_tags=premise_tags or [],
            regime_snapshot=regime_snapshot,
            discovery_source=discovery_source,
        )
        self.thesis_repo.create(thesis)

        positions_opened = 0
        if price is not None and price > 0:
            shares = cfg.position_size_usd / price
            if shares > 0:
                self.position_repo.create(Position(
                    symbol=symbol,
                    side=primary_side,
                    entry_price=price,
                    size=shares,
                    thesis_id=thesis.id,
                    paper=True,
                    notes=f"opened by factory ({direction_label})",
                ))
                positions_opened += 1

        if hedge_symbol:
            hedge_price = self._make_price_provider().get_latest_price(hedge_symbol)
            if hedge_price is not None and hedge_price > 0:
                shares = cfg.position_size_usd / hedge_price
                if shares > 0:
                    self.position_repo.create(Position(
                        symbol=hedge_symbol,
                        side=hedge_side,
                        entry_price=hedge_price,
                        size=shares,
                        thesis_id=thesis.id,
                        paper=True,
                        notes=f"opened by factory ({direction_label} hedge leg)",
                    ))
                    positions_opened += 1

        return {"theses_opened": 1, "positions_opened": positions_opened}
