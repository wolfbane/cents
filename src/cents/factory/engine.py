"""Factory engine — runs the autonomous open/close loop across a universe."""

from __future__ import annotations

import json
import logging
import random
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol


class _PerSymbolTimeout(Exception):
    """Raised by _research_with_deadline when a symbol's research exceeds its budget."""


def _research_with_deadline(orchestrator, symbol: str, thesis, deadline_sec: float):
    """Run ``orchestrator.research(symbol, thesis)`` with a hard deadline.

    Catches hangs in ANY upstream (NewsAPI, FMP, Alpaca, Anthropic) without
    coupling the watchdog to which library happens to be on the call stack.
    The hung worker thread is daemonized and will be reaped by the OS when
    the underlying socket eventually times out — leaking a thread is much
    cheaper than burning 30+ minutes of wall-clock on a single symbol.
    """
    result_holder: list = [None]
    exc_holder: list = [None]

    def _worker():
        try:
            result_holder[0] = orchestrator.research(symbol, thesis)
        except BaseException as e:  # noqa: BLE001 — propagate to caller
            exc_holder[0] = e

    t = threading.Thread(target=_worker, daemon=True, name=f"research-{symbol}")
    t.start()
    t.join(timeout=deadline_sec)
    if t.is_alive():
        raise _PerSymbolTimeout(
            f"research({symbol}) exceeded {deadline_sec:.0f}s deadline"
        )
    if exc_holder[0] is not None:
        raise exc_holder[0]
    return result_holder[0]

from cents.config import get_settings
from cents.exceptions import CostCapExceeded
from cents.db import (
    AlertRepository,
    FactoryRunRepository,
    LLMUsageRepository,
    PositionRepository,
    ShadowOpenRepository,
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
    stop_hit,
    target_hit,
    vol_scaled_shares,
)
from cents.models import (
    Alert,
    AlertType,
    FactoryRun,
    Position,
    PositionSide,
    PositionStatus,
    ShadowOpen,
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
    def research(
        self, symbol: str, thesis: Thesis | None = None, as_of: date | None = None,
    ): ...


class _PriceProvider(Protocol):
    def get_latest_price(self, symbol: str) -> float | None: ...


def _is_paired(thesis: Thesis) -> bool:
    return thesis.cohort == ThesisCohort.NEUTRAL


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
        shadow_repo: ShadowOpenRepository | None = None,
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
        self.shadow_repo = shadow_repo or ShadowOpenRepository()
        self._now = now
        # Calibration: load latest model lazily so sizing can gate on Kelly.
        if calibration_model is not None:
            self._calibration_model = calibration_model
        else:
            try:
                self._calibration_model = load_latest_model()
            except Exception:  # pragma: no cover — defensive
                self._calibration_model = None
        # Bug E (r3) + cents-a1d: warn when the loaded model is >30d old AND
        # persist calibration_age_days so cohort analytics can stratify outcomes
        # by model freshness. A stale model in a regime change silently drives
        # wrong-direction sizing; the labeled dataset captures this if and only
        # if analytics can SEE the age.
        self._calibration_age_days: int | None = None
        if self._calibration_model is not None:
            fit_at = getattr(self._calibration_model, "fit_at", None)
            if isinstance(fit_at, datetime):
                self._calibration_age_days = (datetime.now() - fit_at).days
                if self._calibration_age_days > 30:
                    logger.warning(
                        "Calibration model is %d days old (fit_at=%s); "
                        "consider `cents calibration refit`",
                        self._calibration_age_days, fit_at.isoformat(),
                    )

        # Active experiment context (cents-hvz / cents-eat0). If an experiment is
        # registered, every thesis we open gets stamped with its id, and we
        # detect factory.toml drift from the frozen SHA. cents-eat0: drift now
        # ABORTS the run by default; operators must explicitly opt into a
        # discipline violation by setting allow_frozen_drift=True (CLI flag
        # --force-frozen-drift).
        # Lazy import to avoid an engine ↔ experiments cycle at module load.
        try:
            from cents.experiments import (
                compute_factory_config_sha,
                get_active_experiment,
            )
            self._active_experiment = get_active_experiment()
        except Exception:  # pragma: no cover — defensive
            self._active_experiment = None
        self._config_drift_detected: tuple[str, str] | None = None
        if self._active_experiment is not None:
            try:
                current_sha, _ = compute_factory_config_sha()
            except Exception:  # pragma: no cover — defensive
                current_sha = None
            if (
                current_sha is not None
                and current_sha != self._active_experiment.frozen_config_sha
            ):
                self._config_drift_detected = (
                    self._active_experiment.frozen_config_sha,
                    current_sha,
                )
                logger.warning(
                    "factory.toml SHA has drifted from experiment %r "
                    "(frozen %s, current %s). Treat config changes mid-experiment "
                    "as a discipline violation.",
                    self._active_experiment.name,
                    self._active_experiment.frozen_config_sha[:12],
                    current_sha[:12],
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
        """Resolve the configured/overriding universe name to a (name, symbols).

        When an active experiment carries a frozen_universe_json list, that
        list is used verbatim — guarantees the cohort population is constant
        across the experiment window. SCREENER universes otherwise drift
        daily and confound between-arm comparisons.
        """
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
        # Frozen-universe override (only when an experiment is active).
        if self._active_experiment is not None and self._active_experiment.frozen_universe_json:
            try:
                frozen = json.loads(self._active_experiment.frozen_universe_json)
                if isinstance(frozen, list) and frozen:
                    return universe.name, [str(s) for s in frozen]
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Experiment %s has unparseable frozen_universe_json; "
                    "falling back to live resolution",
                    self._active_experiment.name,
                )
        return universe.name, resolve_symbols(universe)

    # ---- main entry point -------------------------------------------------

    def run(
        self,
        dry_run: bool = False,
        universe_override: str | None = None,
        allow_frozen_drift: bool = False,
    ) -> FactoryRun:
        """Execute a single factory run. Returns the persisted FactoryRun record.

        ``allow_frozen_drift=False`` (default) refuses to run when factory.toml
        has drifted from an active experiment's frozen SHA — pre-registration
        discipline (cents-eat0). Set True (via CLI ``--force-frozen-drift``) to
        override; the drift is still persisted in the run's summary_json.
        """
        cfg = self.config

        # cents-eat0: enforce SHA freeze for active experiments. Detection
        # happens in __init__; here we either abort or proceed-with-warning.
        if self._config_drift_detected and not allow_frozen_drift:
            from cents.exceptions import ExperimentConfigDrift
            frozen, current = self._config_drift_detected
            raise ExperimentConfigDrift(
                experiment_name=self._active_experiment.name,
                frozen_sha=frozen,
                current_sha=current,
            )

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
                run_id=run.id,
            )
            run.theses_opened += open_results["theses_opened"]
            run.positions_opened += open_results["positions_opened"]
            run.preemptions += open_results["preemptions"]
            run.theses_closed += open_results["preempted_closed"]
            run.positions_closed += open_results["preempted_positions_closed"]

            run.summary_json = {
                "universe_size": len(symbols),
                "symbols_evaluated": open_results.get("symbols_evaluated", 0),
                "symbols_below_threshold": open_results.get("symbols_below_threshold", 0),
                "symbols_skipped_held": open_results.get("symbols_skipped_held", 0),
                "symbols_timed_out": open_results.get("symbols_timed_out", 0),
                "stop_reason": open_results.get("stop_reason", "end_of_universe"),
                "proposals": [
                    {"kind": p.kind, "symbol": p.symbol, "detail": p.detail}
                    for p in proposals
                ],
            }
            # cents-a1d: surface calibration model freshness so cohort analytics
            # can stratify outcomes by model age. None means no model loaded.
            if self._calibration_age_days is not None:
                run.summary_json["calibration_age_days"] = self._calibration_age_days
            # cents-eat0: persist drift acknowledgement so analytics can see
            # whether a pilot's data has any post-drift theses in it.
            if self._config_drift_detected and allow_frozen_drift:
                frozen, current = self._config_drift_detected
                run.summary_json["config_drift"] = {
                    "frozen_sha": frozen,
                    "current_sha": current,
                    "experiment": self._active_experiment.name,
                    "allowed_via_force_flag": True,
                }
        except CostCapExceeded:
            # Cost cap is an OPERATIONAL signal — the CLI / caller needs to
            # see the non-zero exit so cron supervisors can fire. Persist
            # the partial run first so analytics still see how far it got.
            logger.warning("Factory run aborted by cost cap")
            run.error = "cost_cap_exceeded"
            run.completed_at = self._clock()
            self._accumulate_llm_cost(run)
            self.run_repo.create(run)
            raise
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
        try:
            result = _research_with_deadline(
                orchestrator,
                thesis.symbol,
                thesis,
                get_settings().per_symbol_deadline_sec,
            )
        except _PerSymbolTimeout as exc:
            logger.warning(
                "Skipping conviction update for %s — %s", thesis.symbol, exc
            )
            return
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
                if target_hit(direction, price, thesis.target_price):
                    return ThesisOutcome.CORRECT
                if stop_hit(direction, price, thesis.stop_price):
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
        """Check for a PREMISE_INVALIDATION alert targeting this thesis.

        SQL-side filter — a Python-side scan of list_all(limit=N) silently
        produces false negatives once total alert volume exceeds N, leaving
        invalidated theses open. Correctness, not just perf.
        """
        since = thesis.updated_at - timedelta(days=1)
        return self.alert_repo.find_invalidation_for(thesis.id, since=since) is not None

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
        run_id: str | None = None,
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
        # Per-disposition counters for dry-run observability (cents-9yn).
        symbols_evaluated = 0
        symbols_below_threshold = 0
        symbols_skipped_held = 0
        symbols_timed_out = 0  # cents-87v: hard per-symbol deadline tripped
        stop_reason = "end_of_universe"
        per_symbol_deadline_sec = get_settings().per_symbol_deadline_sec

        # Shuffle universe order per run to eliminate systematic alphabetical bias
        # when max_new_per_run early-stops the loop. The seed is derived from
        # run_id so each run gets a different (but reproducible) order — both LLM
        # and random arms see the same shuffled order within a single run.
        # See research-design note in CLAUDE.md.
        if run_id is not None:
            shuffled_universe = list(universe_symbols)
            random.Random(run_id).shuffle(shuffled_universe)
        else:
            shuffled_universe = list(universe_symbols)

        orchestrator = self._make_orchestrator()
        # Each thesis is labeled with which orchestrator opened it so the
        # cohort analytics can compare LLM-arm vs random-arm outcomes.
        # Defensive against MagicMock fixtures (which auto-create attrs).
        _olabel = getattr(orchestrator, "orchestrator_label", "llm")
        orchestrator_label = _olabel if isinstance(_olabel, str) else "llm"
        price_provider = self._make_price_provider()

        # Batch-fetch latest prices for all symbols of currently-open factory
        # positions ONCE per open-phase. _current_notional is called per
        # candidate inside the universe loop; without this dict each call
        # fired N fresh latest-quote requests against Alpaca (30 positions ×
        # 100 candidates = 3000 quotes per run). The hedge ETF + underlying
        # symbols are both included so paired-neutral notionals stay accurate.
        open_position_symbols = sorted({
            pos.symbol
            for pos in self.position_repo.list(status=PositionStatus.OPEN)
            if pos.symbol
        })
        position_marks: dict[str, float] = {}
        if open_position_symbols and hasattr(price_provider, "get_latest_prices"):
            try:
                fetched = price_provider.get_latest_prices(open_position_symbols)
                if isinstance(fetched, dict):
                    position_marks = fetched
            except Exception as e:  # noqa: BLE001 — batch fetch is best-effort
                logger.debug("Batched price fetch failed; falling back per-symbol: %s", e)
                position_marks = {}

        # Factory-thesis ID set + regime snapshot, both computed once per phase.
        # Pre-Batch-I, _current_notional and _freeable_notional each did per-
        # position thesis_repo.get() lookups, and the open-phase called
        # capture_regime_snapshot per candidate — neither value changes mid-
        # phase, so doing them once removes O(candidates × positions) DB
        # round-trips and O(candidates) event-table scans.
        factory_thesis_ids = {t.id for t in open_theses}
        phase_regime_snapshot = capture_regime_snapshot(now=self._clock())
        # cents-a1d: stamp the calibration model's age into each thesis's regime
        # snapshot so post-experiment cohort analytics can stratify outcomes by
        # how stale the sizing model was. None means no model loaded.
        if self._calibration_age_days is not None:
            phase_regime_snapshot = {
                **phase_regime_snapshot,
                "calibration_age_days": self._calibration_age_days,
            }

        # Note (research-mode): we intentionally do NOT gate the open phase
        # on portfolio drawdown. The point of cents is to *record* what
        # happens to a labeled outcomes dataset, not to halt trading at
        # arbitrary thresholds. compute_drawdown / check_kill_switch in
        # cents/finance/portfolio.py remain available as analytic utilities
        # for callers who want to study drawdown behaviour, but the engine
        # itself never refuses to open on their account.

        for symbol in shuffled_universe:
            if opened_this_run >= cfg.max_new_per_run:
                stop_reason = "max_new_per_run"
                break
            if symbol in held_symbols:
                symbols_skipped_held += 1
                continue

            symbols_evaluated += 1
            try:
                result = _research_with_deadline(
                    orchestrator, symbol, None, per_symbol_deadline_sec
                )
            except _PerSymbolTimeout as exc:
                symbols_timed_out += 1
                symbols_evaluated -= 1  # didn't fully evaluate
                logger.warning("Skipping %s — %s", symbol, exc)
                continue
            if abs(result.conviction_delta) < cfg.entry_threshold:
                symbols_below_threshold += 1
                self._record_shadow(
                    dry_run=dry_run,
                    run_id=run_id,
                    symbol=symbol,
                    conviction_delta=result.conviction_delta,
                    reason="below_threshold",
                    price=price_provider.get_latest_price(symbol),
                    premise_tags=[],
                    premise_direction={},
                    discovery_source=discovery_source,
                    orchestrator_label=orchestrator_label,
                    regime_snapshot=phase_regime_snapshot,
                )
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
            # Pass side so the classifier can fall back to sector-derived
            # tags when the thesis text is too thin for the LLM to anchor
            # on (e.g. random-arm control theses) — without that fallback,
            # the random arm is silently un-invalidatable by events.
            side_hint = "short" if result.conviction_delta < 0 else "long"
            premise_tags, premise_direction = _coerce_premise_classification(
                classify_premise_tags(
                    symbol,
                    result.summary,
                    [getattr(e, "content", "") for e in (result.evidence or [])],
                    side=side_hint,
                )
            )
            # Random-arm theses skip the premise-concentration cap. The cap
            # exists to throttle clustering on the same regime-factor for the
            # LLM arm, where premise_tags come from the orchestrator's actual
            # summary (typically 1-3 thesis-specific tags). For the random
            # arm, _sector_fallback_tags emits ALL ~5 sector tags per open
            # because the summary is sparse ("random control: NVDA → delta=
            # +17.34"), so the cap kicks in after 2 sector-mates and gates
            # the random arm strictly tighter than the LLM arm — breaking
            # the "matched-cadence" claim. Skipping the cap for random keeps
            # the two arms' acceptance behaviour comparable. The cap is
            # research-purity infrastructure for the LLM arm; the random
            # arm has uniform-noise conviction by construction so per-tag
            # over-concentration is a non-problem there.
            apply_concentration_cap = (
                cfg.max_per_premise_tag > 0 and orchestrator_label == "llm"
            )
            if apply_concentration_cap and self._exceeds_premise_concentration(
                premise_tags, open_theses, cfg.max_per_premise_tag,
                candidate_direction=premise_direction,
            ):
                logger.debug(
                    "Skipping %s — premise tags %s already at concentration cap",
                    symbol, premise_tags,
                )
                self._record_shadow(
                    dry_run=dry_run,
                    run_id=run_id,
                    symbol=symbol,
                    conviction_delta=result.conviction_delta,
                    reason="concentration_cap",
                    price=price,
                    premise_tags=premise_tags,
                    premise_direction=premise_direction,
                    discovery_source=discovery_source,
                    orchestrator_label=orchestrator_label,
                    regime_snapshot=phase_regime_snapshot,
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

            # Record-only: borrow_rate_pa_pct is stamped on the Position below
            # so cohort analytics can see net-of-borrow returns. Per the doc
            # invariant ("engine doesn't filter on borrow"), we do NOT gate on
            # borrow.passes — leaving the gate in shape was a future-trap: the
            # moment passes_borrow_gate gets smarter (real Alpaca locate) the
            # engine would silently start skipping shorts.
            borrow = passes_borrow_gate(
                symbol=symbol,
                side=primary_side.value,
                default_borrow_rate_pa_pct=cfg.borrow_rate_pa_pct,
            )

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
            # Research-mode: we don't skip symbols on liquidity. ADV checks
            # in cents/finance/liquidity.py remain available as utilities for
            # callers who want to study illiquidity as a feature of the
            # outcomes dataset, but the engine doesn't filter on them.
            current_notional = self._current_notional(
                price_provider,
                marks=position_marks,
                factory_thesis_ids=factory_thesis_ids,
            )

            preemption_target: Thesis | None = None
            if current_notional + position_cost > cfg.budget_usd:
                preemption_target = self._select_preemption_target(
                    open_theses,
                    new_conviction,
                    needed_notional=current_notional + position_cost - cfg.budget_usd,
                    price_provider=price_provider,
                    marks=position_marks,
                )
                if preemption_target is None:
                    # Budget locked and no candidate cheap enough — skip
                    self._record_shadow(
                        dry_run=dry_run,
                        run_id=run_id,
                        symbol=symbol,
                        conviction_delta=result.conviction_delta,
                        reason="budget_locked",
                        price=price,
                        premise_tags=premise_tags,
                        premise_direction=premise_direction,
                        discovery_source=discovery_source,
                        orchestrator_label=orchestrator_label,
                        regime_snapshot=phase_regime_snapshot,
                    )
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
                orchestrator_label=orchestrator_label,
            )
            theses_opened += new_open["theses_opened"]
            positions_opened += new_open["positions_opened"]
            open_theses = self.thesis_repo.list(status=ThesisStatus.OPEN)
            open_theses = [t for t in open_theses if TAG_FACTORY in t.tags]
            # Reapply skip_symbols (invalidated-this-run symbols + their hedge
            # legs). Without this, a universe that contains the same symbol
            # twice (screener+watchlist overlap is common) could reopen the
            # invalidated symbol on its second appearance.
            held_symbols = self._held_symbols(open_theses) | skip_symbols
            opened_this_run += 1

        return {
            "theses_opened": theses_opened,
            "positions_opened": positions_opened,
            "preemptions": preemptions,
            "preempted_closed": preempted_closed,
            "preempted_positions_closed": preempted_positions_closed,
            "symbols_evaluated": symbols_evaluated,
            "symbols_below_threshold": symbols_below_threshold,
            "symbols_skipped_held": symbols_skipped_held,
            "symbols_timed_out": symbols_timed_out,
            "stop_reason": stop_reason,
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

    def _current_notional(
        self,
        price_provider,
        *,
        marks: dict[str, float] | None = None,
        factory_thesis_ids: set[str] | None = None,
    ) -> float:
        """Sum of mark-to-market notional across factory-open positions.

        ``marks`` (batched per-phase prices) and ``factory_thesis_ids`` (the
        set of factory-tagged open thesis IDs) are both per-phase fast paths.
        When either is omitted, fall back to per-call lookups so out-of-phase
        callers (analytics, ad-hoc inspection) keep working.
        """
        total = 0.0
        positions = self.position_repo.list(status=PositionStatus.OPEN)
        thesis_cache: dict[str, Thesis | None] = {}
        for pos in positions:
            if not pos.thesis_id:
                continue
            if factory_thesis_ids is not None:
                if pos.thesis_id not in factory_thesis_ids:
                    continue
            else:
                if pos.thesis_id not in thesis_cache:
                    thesis_cache[pos.thesis_id] = self.thesis_repo.get(pos.thesis_id)
                thesis = thesis_cache[pos.thesis_id]
                if thesis is None or TAG_FACTORY not in thesis.tags:
                    continue
            mark = (marks or {}).get(pos.symbol)
            if mark is None:
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
        marks: dict[str, float] | None = None,
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
            freed = self._freeable_notional(candidate, price_provider, marks=marks)
            if freed >= needed_notional:
                return candidate
        return None

    def _freeable_notional(
        self,
        thesis: Thesis,
        price_provider,
        *,
        marks: dict[str, float] | None = None,
    ) -> float:
        """Sum mark-to-market notional across all positions tied to this thesis.

        For NEUTRAL theses this naturally includes both long and short legs since
        both are linked via thesis_id. When ``marks`` is supplied, prices are
        sourced from the per-phase batched dict instead of per-position live
        quotes — same fast-path as _current_notional.
        """
        freed = 0.0
        for pos in self.position_repo.list(status=PositionStatus.OPEN):
            if pos.thesis_id != thesis.id:
                continue
            mark = (marks or {}).get(pos.symbol)
            if mark is None:
                mark = price_provider.get_latest_price(pos.symbol) or pos.entry_price
            freed += mark * pos.size
        return freed

    def _record_shadow(
        self,
        *,
        dry_run: bool,
        run_id: str | None,
        symbol: str,
        conviction_delta: float,
        reason: str,
        price: float | None,
        premise_tags: list[str],
        premise_direction: dict[str, str],
        discovery_source: str | None,
        orchestrator_label: str,
        regime_snapshot: dict | None = None,
    ) -> None:
        """Persist a rejected candidate to shadow_opens (cents-3mo).

        Skips writes during dry-run. Failures are logged but never raised —
        a broken shadow log must not abort a factory run.
        """
        if dry_run:
            return
        cfg = self.config
        side = "SHORT" if conviction_delta < 0 else "LONG"
        try:
            self.shadow_repo.create(ShadowOpen(
                run_id=run_id,
                symbol=symbol,
                conviction_delta=conviction_delta,
                reason=reason,
                would_be_entry_price=price,
                primary_side=side,
                premise_tags=premise_tags or [],
                premise_direction=premise_direction or {},
                regime_snapshot=regime_snapshot if regime_snapshot is not None else capture_regime_snapshot(now=self._clock()),
                orchestrator_label=orchestrator_label,
                experiment_id=self._active_experiment.id if self._active_experiment else None,
                discovery_source=discovery_source,
                horizon_days=cfg.default_horizon_days,
                created_at=self._clock(),
            ))
        except Exception:  # pragma: no cover — observability must not break runs
            logger.exception("Failed to record shadow_open for %s", symbol)

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

        Random-arm theses are excluded from the count. Their premise_tags come
        from ``_sector_fallback_tags`` (all 5 sector tags per open) and would
        otherwise saturate the cap after 2 sector-mates, gating subsequent
        LLM-arm opens on the same sector. The cap exists to throttle LLM-arm
        clustering on the same regime variable; random-arm sector tags carry
        no signal-driven clustering by construction.
        """
        if not candidate_tags:
            return False
        counts: dict[tuple[str, str], int] = {}
        for t in open_theses:
            if getattr(t, "orchestrator_label", "llm") != "llm":
                continue
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
        orchestrator_label: str = "llm",
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

        # Calibrated P(correct) if a calibration model exists. ONLY emitted
        # for the LLM arm — random-arm theses have a uniform-noise delta by
        # construction, so any predicted p_correct on them is meaningless.
        # calibrated_p_correct is RECORDED on the thesis row, never used to
        # gate the open (research-mode).
        calibrated_p = None
        if orchestrator_label == "llm":
            calibrated_p = self._predict_calibration(
                delta=delta,
                regime_snapshot=regime_snapshot,
                discovery_source=discovery_source,
                cohort=cohort,
                horizon_days=cfg.default_horizon_days,
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
            orchestrator_label=orchestrator_label,
            experiment_id=self._active_experiment.id if self._active_experiment else None,
        )
        self.thesis_repo.create(thesis)

        # Research-mode: calibrated_p_correct is RECORDED on the thesis row
        # (above) so analytics can stratify outcomes by predicted edge, but
        # we never skip an open on it. The cohort table will show what
        # happened at every p value, which is the research question.

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
        horizon_days: int | None = None,
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
            horizon_days=horizon_days,
        )
        return float(model.predict(features))
