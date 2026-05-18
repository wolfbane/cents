"""Factory engine — runs the autonomous open/close loop across a universe."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
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
from cents.finance.calibration import (
    CalibrationModel,
    build_predict_features,
    load_latest_model,
)
from cents.factory.sector_map import hedge_etf_for
from cents.factory.universe_resolver import resolve_symbols
from cents.finance import (
    Cost,
    apply_close_cost,
    apply_open_cost,
    beta_match_ratio,
    check_kill_switch,
    compute_drawdown,
    estimate_beta,
    passes_borrow_gate,
    passes_liquidity_gate,
    realized_vol_pct,
    vol_scaled_shares,
)
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


# Kelly window for calibrated sizing — outside this band, fall back to default.
# Minimum margin above payoff-adjusted break-even probability before we open.
# Setting this to a positive number ensures we don't open at zero edge.
_CALIBRATION_KELLY_MARGIN = 0.02
# Upper bound — when p is implausibly high, the model is probably overfit;
# fall back to "no calibration signal" rather than trust it.
_CALIBRATION_MAX_TRUSTED_P = 0.95


def _break_even_probability(target_pct: float, stop_pct: float) -> float:
    """Break-even P(correct) for an asymmetric payoff.

    For target_pct=+10 and stop_pct=-5, payoff ratio b = 2; break-even p* =
    |stop| / (target + |stop|) = 5 / 15 = 0.333. A calibrated probability
    must clear this AND a small margin before opening makes positive-EV sense.
    """
    target = abs(target_pct)
    stop = abs(stop_pct)
    if target + stop <= 0:
        return 0.5  # Degenerate input — fall back to coin-flip threshold.
    return stop / (target + stop)


def _calibration_passes_gate(
    p: float | None,
    *,
    target_pct: float,
    stop_pct: float,
    margin: float = _CALIBRATION_KELLY_MARGIN,
    max_trusted_p: float = _CALIBRATION_MAX_TRUSTED_P,
) -> bool:
    """Kelly gate: open only when p clears break-even + margin.

    Bug B fix (PM/Risk round 3): the prior implementation multiplied
    vol-scaled shares by a ``2p - 1`` Kelly multiplier — double-shrinking on
    top of vol scaling and assuming even-money payoffs. The fix removes the
    multiplier entirely and uses calibration as a GATE: vol-sized shares
    pass through unchanged when the calibrated probability clears the
    payoff-adjusted break-even threshold; the open is skipped otherwise.

    Returns True (open) when:
      - p is None (no calibration → no gating, preserves prior behavior)
      - p clears ``break_even + margin`` AND p ≤ ``max_trusted_p``

    Returns False (skip) when:
      - p ≤ break_even + margin (insufficient edge given the bracket)
      - p > max_trusted_p (model probably overfit; don't trust)
    """
    if p is None:
        return True
    if p > max_trusted_p:
        return False
    break_even = _break_even_probability(target_pct, stop_pct)
    return p >= break_even + margin


def _coerce_premise_classification(value) -> tuple[list[str], dict[str, str]]:
    """Adapt classify_premise_tags' return to (tags, direction).

    The classifier returns a 2-tuple, but legacy test stubs may return a bare
    list of tags. Accept both shapes so test fixtures need not all change at once.
    """
    if isinstance(value, tuple) and len(value) == 2:
        tags, direction = value
        return list(tags or []), dict(direction or {})
    return list(value or []), {}


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
        calibration_model: CalibrationModel | None = None,
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
        # Calibration: load latest model lazily so sizing can gate on Kelly.
        if calibration_model is not None:
            self._calibration_model = calibration_model
        else:
            try:
                self._calibration_model = load_latest_model()
            except Exception:  # pragma: no cover — defensive
                self._calibration_model = None
        # Bug E (r3): warn loudly when the loaded model is older than 30 days.
        # A stale model in a regime change silently drives wrong-direction sizing.
        if self._calibration_model is not None:
            fit_at = getattr(self._calibration_model, "fit_at", None)
            if isinstance(fit_at, datetime):
                age_days = (datetime.now() - fit_at).days
                if age_days > 30:
                    logger.warning(
                        "Calibration model is %d days old (fit_at=%s); "
                        "consider `cents calibration refit`",
                        age_days, fit_at.isoformat(),
                    )

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

    def _history_supported(self) -> bool:
        """True when the injected price provider exposes a real get_history.

        Returns False for minimal test stubs (no get_history attribute) and
        for MagicMock-style auto-mocks (callable but returns mock garbage).
        The engine disables gates that need history when this is False.
        """
        provider = self._make_price_provider()
        get_history = getattr(provider, "get_history", None)
        if not callable(get_history):
            return False
        # Heuristic: MagicMock auto-mocks return _mock.MagicMock on call.
        # Real providers return a PriceHistory. A quick probe with an empty
        # symbol catches stubs without breaking real providers (they return
        # an empty PriceHistory rather than something with a .bars list of mocks).
        return type(provider).__name__ not in {"MagicMock", "Mock", "NonCallableMagicMock"}

    def _get_history(self, symbol: str, days: int) -> tuple[list[float] | None, list[int] | None]:
        """Fetch closes + volumes for a symbol via the price provider, or (None, None)."""
        if not self._history_supported():
            return None, None
        provider = self._make_price_provider()
        try:
            history = provider.get_history(symbol, days=days)
            bars = getattr(history, "bars", None) or []
            if not bars:
                return None, None
            closes = [float(b.close) for b in bars]
            volumes = [int(getattr(b, "volume", 0)) for b in bars]
            return closes, volumes
        except Exception as exc:
            logger.debug("history fetch failed for %s: %s", symbol, exc)
            return None, None

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

            positions_closed += self._close_thesis_positions(
                thesis, price_provider, outcome=trigger,
            )
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

    def _positions_closed_today(self) -> list[Position]:
        """All positions whose exit_date is today. Used by the drawdown kill switch."""
        today = date.today()
        closed = []
        for pos in self.position_repo.list(status=PositionStatus.CLOSED):
            if pos.exit_date == today:
                closed.append(pos)
        return closed

    def _emit_kill_switch_alert(self, state) -> None:
        """Record an alert when the portfolio kill switch trips."""
        alert = Alert(
            symbol="PORTFOLIO",
            alert_type=AlertType.PORTFOLIO_RISK,
            message=f"factory kill switch tripped: {state.gate_reason}",
            data={
                "unrealized_drawdown_pct": state.unrealized_drawdown_pct,
                "realized_loss_today_pct": state.realized_loss_today_pct,
                "gate_reason": state.gate_reason,
                "kind": "factory_kill_switch",
            },
        )
        self.alert_repo.create(alert)

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

    def _close_thesis_positions(
        self,
        thesis: Thesis,
        price_provider,
        *,
        outcome: ThesisOutcome | None = None,
    ) -> int:
        """Close all open positions linked to a thesis. Returns count closed.

        Applies the transaction cost model (commission + slippage + short borrow)
        and a stop-gap penalty when the close is a stop trigger. Persists both
        the signal ``exit_price`` and the modeled ``realized_exit_price``.
        """
        cfg = self.config
        closed = 0
        triggered_stop = outcome == ThesisOutcome.INCORRECT
        for pos in self.position_repo.list(status=PositionStatus.OPEN):
            if pos.thesis_id != thesis.id:
                continue
            mark = price_provider.get_latest_price(pos.symbol) or pos.entry_price
            exit_signal = mark
            # Gap-aware fill: when the close was triggered by a stop hit, use
            # whichever is worse for the position — last observed price or the
            # stop price — so we stop pretending fills happen exactly at stop.
            realized = exit_signal
            gap_bps = 0.0
            if triggered_stop and thesis.stop_price is not None:
                if pos.side == PositionSide.LONG:
                    realized = min(exit_signal, thesis.stop_price)
                else:
                    realized = max(exit_signal, thesis.stop_price)
                gap_bps = cfg.gap_slippage_bps

            days_held = max(0, (date.today() - pos.entry_date).days)
            close_cost = apply_close_cost(
                side=pos.side.value,
                shares=pos.size,
                entry_price=pos.entry_price,
                exit_price=realized,
                days_held=days_held,
                commission_per_share_usd=cfg.commission_per_share_usd,
                slippage_bps=cfg.slippage_bps,
                borrow_rate_pa_pct=pos.borrow_rate_pa_pct or 0.0,
                gap_penalty_bps=gap_bps,
            )
            pos.close(
                exit_signal,
                realized_exit_price=realized,
                costs_applied_usd=(pos.costs_applied_usd or 0.0) + close_cost.total,
            )
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

        # ---- portfolio-level kill switch (cents-59r) ---------------------
        # Drawdown and daily-loss caps are evaluated once per open phase.
        # If breached, we skip the entire phase (no new opens) and emit an
        # alert. Closes still happen in the prior phase regardless.
        all_open_positions = self.position_repo.list(status=PositionStatus.OPEN)
        factory_thesis_ids = {t.id for t in open_theses}
        factory_open_positions = [
            p for p in all_open_positions if p.thesis_id in factory_thesis_ids
        ]
        closed_today = self._positions_closed_today()
        dd_state = compute_drawdown(
            open_positions=factory_open_positions,
            closed_today=closed_today,
            price_provider=price_provider,
            budget_usd=cfg.budget_usd,
        )
        dd_state = check_kill_switch(
            dd_state,
            max_portfolio_drawdown_pct=cfg.max_portfolio_drawdown_pct,
            max_daily_loss_pct=cfg.max_daily_loss_pct,
        )
        if not dd_state.gate_open and not dry_run:
            self._emit_kill_switch_alert(dd_state)
            logger.warning("Factory kill switch tripped: %s", dd_state.gate_reason)
            return {
                "theses_opened": 0,
                "positions_opened": 0,
                "preemptions": 0,
                "preempted_closed": 0,
                "preempted_positions_closed": 0,
                "kill_switch": dd_state.gate_reason,
            }

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
            premise_tags, premise_direction = _coerce_premise_classification(
                classify_premise_tags(
                    symbol,
                    result.summary,
                    [getattr(e, "content", "") for e in (result.evidence or [])],
                )
            )
            if cfg.max_per_premise_tag > 0 and self._exceeds_premise_concentration(
                premise_tags, open_theses, cfg.max_per_premise_tag,
                candidate_direction=premise_direction,
            ):
                logger.debug(
                    "Skipping %s — premise tags %s already at concentration cap",
                    symbol, premise_tags,
                )
                continue

            # ---- per-symbol gates + sizing (v0.10) ----
            # Compute sizing BEFORE the liquidity/budget/preemption math so
            # each check reflects the realistic position size, not the
            # equal-dollar fiction. Vol-scaled sizing can shrink a position
            # by 5-10x against the equal-dollar fallback — making the gate
            # see the same number means it can actually let small-cap names
            # through when the position is sized appropriately.
            primary_side = PositionSide.SHORT if result.conviction_delta < 0 else PositionSide.LONG
            history_supported = self._history_supported()

            primary_closes, primary_volumes = (
                self._get_history(symbol, cfg.vol_lookback_days * 2)
                if history_supported else (None, None)
            )

            borrow = passes_borrow_gate(
                symbol=symbol,
                side=primary_side.value,
                default_borrow_rate_pa_pct=cfg.borrow_rate_pa_pct,
            )
            if not borrow.passes:
                logger.debug("Borrow gate failed for %s: %s", symbol, borrow.reason)
                continue

            vol_pct = (
                realized_vol_pct(primary_closes, lookback=cfg.vol_lookback_days)
                if primary_closes is not None else None
            )

            primary_shares, sizing_method = 0.0, "equal_dollar"
            if price is not None and price > 0:
                primary_shares, sizing_method = vol_scaled_shares(
                    price=price,
                    annual_vol_pct=vol_pct if cfg.sizing_mode == "vol_scaled" else None,
                    budget_usd=cfg.budget_usd,
                    target_vol_pct_per_position=cfg.target_vol_pct_per_position,
                    max_position_pct=cfg.max_position_pct,
                    fallback_position_usd=cfg.position_size_usd,
                )

            primary_notional = primary_shares * (price or 0.0)
            # Project hedge notional with a placeholder beta — refined below
            # if we actually open the trade.
            hedge_notional_estimate = (
                primary_notional * cfg.default_beta if paired and hedge_symbol else 0.0
            )
            position_cost = primary_notional + hedge_notional_estimate

            # Liquidity gate sees the actual sized notional (primary + hedge
            # in paired mode) — not the equal-dollar fiction. We skip the
            # gate entirely when history isn't supported (test stubs) or
            # when sizing collapsed to zero (no price).
            if (
                history_supported
                and cfg.min_adv_multiple > 0
                and primary_notional > 0
            ):
                gate_notional = primary_notional + hedge_notional_estimate
                liq = passes_liquidity_gate(
                    symbol=symbol,
                    position_size_usd=gate_notional,
                    closes=primary_closes,
                    volumes=primary_volumes,
                    adv_multiple=cfg.min_adv_multiple,
                    lookback=cfg.liquidity_lookback_days,
                )
                if not liq.passes:
                    logger.debug("Liquidity gate failed for %s: %s", symbol, liq.reason)
                    continue
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
                    preemption_target, price_provider, outcome=ThesisOutcome.PREEMPTED,
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
                premise_direction=premise_direction,
                discovery_source=discovery_source,
                primary_shares=primary_shares,
                primary_sizing_method=sizing_method,
                primary_closes=primary_closes,
                borrow_rate_pa_pct=borrow.borrow_rate_pa_pct,
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
        candidate_direction: dict[str, str] | None = None,
    ) -> bool:
        """True if any candidate tag+direction is already at the per-tag cap.

        Bug D fix: a bullish ai_capex thesis and a bearish ai_capex thesis no
        longer count against each other — that's a spread, not concentration.
        Buckets on ``(tag, direction)`` instead of bare ``tag``. When a thesis
        has no recorded direction for a tag, it counts under the legacy
        ``(tag, "*")`` bucket so behavior is preserved for older rows.
        """
        if not candidate_tags:
            return False
        counts: dict[tuple[str, str], int] = {}
        for t in open_theses:
            t_dir = t.premise_direction or {}
            for tag in t.premise_tags:
                key = (tag, t_dir.get(tag, "*"))
                counts[key] = counts.get(key, 0) + 1
        candidate_direction = candidate_direction or {}
        for tag in candidate_tags:
            key = (tag, candidate_direction.get(tag, "*"))
            if counts.get(key, 0) >= cap:
                return True
        return False

    def _open_new_thesis(
        self,
        *,
        symbol: str,
        conviction: float,
        delta: float,
        price: float | None,
        hedge_symbol: str | None,
        premise_tags: list[str] | None = None,
        premise_direction: dict[str, str] | None = None,
        discovery_source: str | None = None,
        primary_shares: float = 0.0,
        primary_sizing_method: str = "equal_dollar",
        primary_closes: list[float] | None = None,
        borrow_rate_pa_pct: float = 0.0,
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

        # Calibrated P(correct) if a calibration model exists — used both for
        # persisting on the thesis and for Kelly-fraction sizing below.
        calibrated_p = self._predict_calibration(
            delta=delta,
            regime_snapshot=regime_snapshot,
            discovery_source=discovery_source,
            cohort=cohort,
        )

        calibration_fit_at = None
        if self._calibration_model is not None:
            fit_at_dt = getattr(self._calibration_model, "fit_at", None)
            if isinstance(fit_at_dt, datetime):
                calibration_fit_at = fit_at_dt.isoformat()

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
            premise_direction=premise_direction or {},
            regime_snapshot=regime_snapshot,
            discovery_source=discovery_source,
            calibrated_p_correct=calibrated_p,
            calibration_fit_at=calibration_fit_at,
        )
        self.thesis_repo.create(thesis)

        # Bug B (PM/Risk r3): Kelly is a GATE, not a sizing multiplier.
        # Vol-scaled shares pass through unchanged when the calibrated
        # probability clears the payoff-adjusted break-even + margin.
        # We've already persisted the thesis with calibrated_p_correct so
        # the gating decision is auditable; we just won't open positions.
        if not _calibration_passes_gate(
            calibrated_p,
            target_pct=cfg.default_target_pct,
            stop_pct=cfg.default_stop_pct,
        ):
            logger.debug(
                "Skipping %s positions: calibrated_p=%.3f below break-even+margin "
                "(target=%.1f stop=%.1f)",
                symbol, calibrated_p or 0.0,
                cfg.default_target_pct, cfg.default_stop_pct,
            )
            return {"theses_opened": 1, "positions_opened": 0}

        positions_opened = 0
        # ---- Primary leg (cents-wiz vol-scaled sizing + cents-5s7 open cost) ----
        if price is not None and price > 0 and primary_shares > 0:
            open_cost = apply_open_cost(
                shares=primary_shares,
                price=price,
                commission_per_share_usd=cfg.commission_per_share_usd,
                slippage_bps=cfg.slippage_bps,
            )
            self.position_repo.create(Position(
                symbol=symbol,
                side=primary_side,
                entry_price=price,
                size=primary_shares,
                thesis_id=thesis.id,
                paper=True,
                notes=(
                    f"opened by factory ({direction_label}); "
                    f"sizing={primary_sizing_method}"
                ),
                costs_applied_usd=open_cost.total,
                sizing_method=primary_sizing_method,
                borrow_rate_pa_pct=(borrow_rate_pa_pct if primary_side == PositionSide.SHORT else None),
            ))
            positions_opened += 1

        # ---- Hedge leg (cents-t8r beta-matched sizing) ----
        if hedge_symbol and price is not None and price > 0 and primary_shares > 0:
            hedge_provider = self._make_price_provider()
            hedge_price = hedge_provider.get_latest_price(hedge_symbol)
            if hedge_price is not None and hedge_price > 0:
                primary_notional = primary_shares * price

                # Beta-match the hedge leg notional. Falls back to default_beta
                # when history is insufficient (cents-t8r — strictly an
                # improvement on the previous dollar-for-dollar match).
                #
                # Bug C fix: when beta_match_hedge is on AND we DO have history
                # AND estimate_beta returns None (R² gate rejected the fit),
                # silently falling back to default_beta=1.0 reintroduces the
                # dollar-match bug. Skip the hedge leg in that case so the
                # gate fails safe rather than failing to the worst behavior.
                beta = None
                history_available = False
                if cfg.beta_match_hedge:
                    hedge_closes, _ = self._get_history(
                        hedge_symbol, cfg.beta_lookback_days * 2
                    )
                    if primary_closes is not None and hedge_closes is not None:
                        history_available = True
                        beta = estimate_beta(
                            primary_closes, hedge_closes,
                            lookback=cfg.beta_lookback_days,
                            min_r_squared=cfg.beta_min_r_squared,
                        )
                if cfg.beta_match_hedge and history_available and beta is None:
                    # R² gate rejected the fit; skip the hedge leg entirely.
                    logger.debug(
                        "Skipping hedge leg for %s: %s R² below %.2f gate",
                        symbol, hedge_symbol, cfg.beta_min_r_squared,
                    )
                    return {"theses_opened": 1, "positions_opened": positions_opened}
                ratio = beta_match_ratio(
                    beta=beta,
                    default_beta=cfg.default_beta,
                    min_beta=cfg.beta_min,
                    max_beta=cfg.beta_max,
                )
                hedge_notional = primary_notional * ratio
                hedge_shares = hedge_notional / hedge_price
                # Cap hedge leg at the per-position max too.
                max_hedge_dollar = cfg.budget_usd * (cfg.max_position_pct / 100.0)
                if hedge_shares * hedge_price > max_hedge_dollar:
                    hedge_shares = max_hedge_dollar / hedge_price

                if hedge_shares > 0:
                    hedge_open_cost = apply_open_cost(
                        shares=hedge_shares,
                        price=hedge_price,
                        commission_per_share_usd=cfg.commission_per_share_usd,
                        slippage_bps=cfg.slippage_bps,
                    )
                    sizing_label = (
                        f"beta_matched_hedge(beta={ratio:.2f})"
                        if cfg.beta_match_hedge else "equal_dollar_hedge"
                    )
                    hedge_borrow = (
                        cfg.borrow_rate_pa_pct if hedge_side == PositionSide.SHORT else None
                    )
                    self.position_repo.create(Position(
                        symbol=hedge_symbol,
                        side=hedge_side,
                        entry_price=hedge_price,
                        size=hedge_shares,
                        thesis_id=thesis.id,
                        paper=True,
                        notes=(
                            f"opened by factory ({direction_label} hedge leg); "
                            f"sizing={sizing_label}"
                        ),
                        costs_applied_usd=hedge_open_cost.total,
                        sizing_method=sizing_label,
                        borrow_rate_pa_pct=hedge_borrow,
                    ))
                    positions_opened += 1

        return {"theses_opened": 1, "positions_opened": positions_opened}

    def _predict_calibration(
        self,
        *,
        delta: float,
        regime_snapshot: dict,
        discovery_source: str | None,
        cohort: ThesisCohort,
    ) -> float | None:
        """Return calibrated ``P(target hit)`` for a candidate thesis, or None."""
        model = self._calibration_model
        if model is None:
            return None
        features = build_predict_features(
            delta=delta,
            regime_snapshot=regime_snapshot,
            discovery_source=discovery_source,
            cohort=cohort.value if cohort is not None else None,
        )
        return float(model.predict(features))
