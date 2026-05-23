"""Data layer for the visualization stack.

Every chart pulls from this module. The split is deliberate: rendering
code (ASCII / matplotlib / plotly) should never touch SQLite directly —
that way a fix to the LLM-cost-attribution rule lands in one place and
flows to every chart that depends on it.

The functions here return plain dataclasses and lists of dicts, NOT
pandas DataFrames. This keeps the data layer importable without pulling
matplotlib's transitive deps, and lets the test suite assert on shape
directly without DataFrame ceremony.

Attribution rules live next to the existing factory analyzer
(``cents/cli/factory.py:_attribute_llm_cost``) and are reused via
``cost_by_thesis()`` below so the dashboard, the report, and
``factory analyze --include-cost-per-outcome`` can never drift.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

from cents.db.repository import (
    AlertRepository,
    EventRepository,
    LLMUsageRepository,
    PositionRepository,
    ThesisRepository,
)
from cents.models.alert import AlertType
from cents.models.position import Position, PositionStatus
from cents.models.thesis import (
    HedgeBasis,
    PremiseSource,
    Thesis,
    ThesisCohort,
    ThesisOutcome,
    ThesisStatus,
)
from cents.pricing import estimate_cost_usd


# Mirror ``_OVERHEAD_OPERATIONS`` in cents/cli/factory.py — these run for
# both arms before a thesis is opened, so attributing them per-thesis
# would charge the random arm for LLM cost the random orchestrator
# never emitted. They land in unattributable_cost_usd instead.
_OVERHEAD_OPERATIONS = frozenset({"classify_premise", "tag_event"})

LOW_N_THRESHOLD = 30  # matches cents/cli/_disclosures.py


# ---------------------------------------------------------------------------
# Row shapes — one per chart family, kept narrow so callers don't grow
# accidental dependencies on incidental columns.
# ---------------------------------------------------------------------------


@dataclass
class ThesisRow:
    """Denormalized thesis + roll-up of its positions.

    ``pnl`` is the SUM of both legs' net pnl for neutral cohorts (one
    LONG underlying + one SHORT hedge on the same thesis_id), or the
    single leg's pnl for directional theses. Cohort analytics use this,
    never ``Position.pnl`` directly, so a thesis is treated as one unit
    of evidence regardless of leg count.
    """

    id: str
    symbol: str | None
    status: ThesisStatus
    outcome: ThesisOutcome | None
    cohort: ThesisCohort
    hedge_basis: HedgeBasis | None
    premise_classification_source: PremiseSource
    orchestrator_label: str
    experiment_id: str | None
    discovery_source: str | None
    premise_tags: list[str]
    regime_label: str | None
    created_at: datetime
    closed_at: datetime | None
    conviction: float
    calibrated_p_correct: float | None
    pnl: float | None
    held_days: float | None
    legs: int


@dataclass
class CohortMetrics:
    """Hit rate + cost for one cohort cell."""

    key: tuple[str, ...]              # e.g. ("llm", "directional")
    labels: tuple[str, ...]           # e.g. ("orchestrator", "cohort")
    opened: int
    judged: int
    correct: int
    invalidated: int
    win_rate: float | None
    win_rate_ci: tuple[float, float] | None
    avg_pnl: float | None
    llm_cost_usd: float


@dataclass
class CostDay:
    day: str          # YYYY-MM-DD
    cost_usd: float


@dataclass
class EvalPoint:
    date: str         # YYYY-MM-DD
    premise_f1: float | None
    sentiment_brier: float | None


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _regime_label(snapshot: dict | None) -> str | None:
    """Compact regime tag from a thesis ``regime_snapshot``.

    Falls back to ``None`` for legacy theses that never captured one.
    """
    if not snapshot:
        return None
    pol = snapshot.get("policy") or snapshot.get("regime")
    vol = snapshot.get("vol") or snapshot.get("volatility")
    if pol and vol:
        return f"{pol}:{vol}"
    return pol or vol or None


def _thesis_pnl(
    thesis: Thesis, positions: Sequence[Position]
) -> tuple[float | None, float | None, int]:
    """Aggregate legs into (pnl, held_days, leg_count).

    ``held_days`` is the max across legs (the close that actually
    resolved the thesis), not the average — for a paired-neutral thesis
    both legs close simultaneously anyway, so this only matters if one
    leg's exit_date is missing.
    """
    pnls = [p.pnl for p in positions if p.pnl is not None]
    pnl = sum(pnls) if pnls else None
    held_days: float | None = None
    for p in positions:
        if p.exit_date and p.entry_date:
            d = (p.exit_date - p.entry_date).days
            held_days = d if held_days is None else max(held_days, d)
    return pnl, held_days, len(positions)


def list_theses(
    *,
    experiment_id: str | None = None,
    since: datetime | None = None,
) -> list[ThesisRow]:
    """Return denormalized thesis rows for the viz layer.

    Joins ``Thesis`` with its ``Position`` rows so ``pnl`` and
    ``held_days`` come back already rolled up. This is the single join
    every cohort chart depends on; it gets a unit test in
    ``tests/test_viz_queries.py`` because a bug here corrupts every
    downstream chart silently.
    """
    theses_all = ThesisRepository().list()
    positions_all = PositionRepository().list()
    by_thesis: dict[str, list[Position]] = defaultdict(list)
    for p in positions_all:
        if p.thesis_id:
            by_thesis[p.thesis_id].append(p)

    out: list[ThesisRow] = []
    for t in theses_all:
        if experiment_id is not None and t.experiment_id != experiment_id:
            continue
        if since is not None and t.created_at < since:
            continue
        pnl, held_days, legs = _thesis_pnl(t, by_thesis.get(t.id, ()))
        out.append(
            ThesisRow(
                id=t.id,
                symbol=t.symbol,
                status=t.status,
                outcome=t.outcome,
                cohort=t.cohort,
                hedge_basis=t.hedge_basis,
                premise_classification_source=t.premise_classification_source,
                orchestrator_label=t.orchestrator_label,
                experiment_id=t.experiment_id,
                discovery_source=t.discovery_source,
                premise_tags=list(t.premise_tags or []),
                regime_label=_regime_label(t.regime_snapshot),
                created_at=t.created_at,
                closed_at=t.closed_at,
                conviction=t.conviction,
                calibrated_p_correct=t.calibrated_p_correct,
                pnl=pnl,
                held_days=held_days,
                legs=legs,
            )
        )
    return out


# ---------------------------------------------------------------------------
# LLM cost attribution — mirrors cents/cli/factory.py:_attribute_llm_cost
# ---------------------------------------------------------------------------


def cost_by_thesis(
    rows: Sequence[ThesisRow],
    *,
    since: datetime | None = None,
) -> tuple[dict[str, float], float]:
    """``({thesis_id: cost_usd}, unattributable_usd)``.

    Re-implemented over ``ThesisRow`` (rather than imported from
    factory.py) because the viz module also needs to call this on
    thesis snapshots that may pre-date the active experiment cutoff.
    Behaviour matches the factory analyzer line-for-line — when that
    rule changes, this function must change too.
    """
    usage_rows = LLMUsageRepository().list_recent(since=since, limit=None)
    by_id = {r.id: r for r in rows}
    by_symbol: dict[str, list[ThesisRow]] = defaultdict(list)
    for r in rows:
        if r.symbol:
            by_symbol[r.symbol].append(r)

    cost: dict[str, float] = {}
    unattributable = 0.0
    for u in usage_rows:
        c = estimate_cost_usd(
            u.model,
            u.input_tokens,
            u.output_tokens,
            cache_read=u.cache_read_input_tokens,
            cache_write=u.cache_creation_input_tokens,
        )
        if c is None:
            continue
        if getattr(u, "operation", None) in _OVERHEAD_OPERATIONS:
            unattributable += c
            continue
        ctx = u.context
        matched = False
        if ctx:
            direct = by_id.get(ctx)
            if direct is not None:
                cost[direct.id] = cost.get(direct.id, 0.0) + c
                matched = True
            else:
                for candidate in by_symbol.get(ctx, ()):
                    end = candidate.closed_at or datetime.now()
                    if candidate.created_at <= u.called_at <= end:
                        cost[candidate.id] = cost.get(candidate.id, 0.0) + c
                        matched = True
                        break
        if not matched:
            unattributable += c
    return cost, unattributable


def daily_llm_costs(*, window_days: int = 14) -> list[CostDay]:
    """Total LLM spend per day for the last ``window_days``.

    Includes overhead operations — this is the spend-against-cap view,
    not the per-thesis attribution view.
    """
    cutoff = datetime.now() - timedelta(days=window_days)
    rows = LLMUsageRepository().list_recent(since=cutoff, limit=None)
    by_day: dict[str, float] = defaultdict(float)
    for u in rows:
        c = estimate_cost_usd(
            u.model,
            u.input_tokens,
            u.output_tokens,
            cache_read=u.cache_read_input_tokens,
            cache_write=u.cache_creation_input_tokens,
        )
        if c is None:
            continue
        by_day[u.called_at.date().isoformat()] += c

    out: list[CostDay] = []
    today = datetime.now().date()
    for i in range(window_days - 1, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        out.append(CostDay(day=d, cost_usd=round(by_day.get(d, 0.0), 4)))
    return out


# ---------------------------------------------------------------------------
# Eval history (for the drift strip on the dashboard)
# ---------------------------------------------------------------------------


def eval_history(*, days: int = 14, root: Path | None = None) -> list[EvalPoint]:
    """Read the last ``days`` of ``~/.cents/data/eval_history/*.jsonl``.

    Each file is one date; one row per eval run on that date. Returns
    the latest row per date, oldest first. Empty list if the directory
    doesn't exist (fresh install — drift-check will tell the user to
    seed it).
    """
    base = root or Path.home() / ".cents" / "data" / "eval_history"
    if not base.is_dir():
        return []
    today = datetime.now().date()
    points: list[EvalPoint] = []
    for i in range(days - 1, -1, -1):
        d = today - timedelta(days=i)
        path = base / f"{d.isoformat()}.jsonl"
        if not path.exists():
            continue
        last: dict | None = None
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
        if last is None:
            continue
        points.append(
            EvalPoint(
                date=d.isoformat(),
                premise_f1=last.get("premise_f1"),
                sentiment_brier=last.get("sentiment_brier"),
            )
        )
    return points


# ---------------------------------------------------------------------------
# Cohort metrics — the same primitive backs charts 1, 3, 5, 11
# ---------------------------------------------------------------------------


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score CI for a binomial proportion.

    Used in preference to the normal-approximation CI because cohort N
    can be small (50–200 per arm in the pilot) and Wilson's coverage
    holds up at the tails. Returns ``None`` for n == 0.
    """
    if n == 0:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohort_metrics(
    rows: Sequence[ThesisRow],
    *,
    by: Sequence[str],
    cost: dict[str, float] | None = None,
) -> list[CohortMetrics]:
    """Group ``rows`` by the named axes and roll up the per-cell stats.

    Supported axes: ``orchestrator``, ``cohort``, ``regime``,
    ``discovery``, ``hedge_basis``, ``premise_classification_source``.

    Win rate is computed only over judged theses (closed and not
    PREEMPTED); INVALIDATED counts separately. Confidence intervals
    are Wilson at 95%.
    """
    axis_lookup = {
        "orchestrator": lambda r: r.orchestrator_label,
        "cohort": lambda r: r.cohort.value if r.cohort else "unknown",
        "regime": lambda r: r.regime_label or "unknown",
        "discovery": lambda r: r.discovery_source or "unknown",
        "hedge_basis": lambda r: (
            r.hedge_basis.value if r.hedge_basis else "directional"
        ),
        "premise_classification_source": lambda r: (
            r.premise_classification_source.value
            if r.premise_classification_source
            else "unknown"
        ),
    }
    extractors = []
    for axis in by:
        if axis not in axis_lookup:
            raise ValueError(
                f"unknown axis {axis!r}; supported: {sorted(axis_lookup)}"
            )
        extractors.append(axis_lookup[axis])

    groups: dict[tuple[str, ...], list[ThesisRow]] = defaultdict(list)
    for r in rows:
        key = tuple(str(fn(r)) for fn in extractors)
        groups[key].append(r)

    cost = cost or {}
    out: list[CohortMetrics] = []
    for key, members in sorted(groups.items()):
        opened = len(members)
        judged = [m for m in members if (
            m.status == ThesisStatus.CLOSED
            and m.outcome is not None
            and m.outcome != ThesisOutcome.PREEMPTED
        )]
        correct = sum(1 for m in judged if m.outcome == ThesisOutcome.CORRECT)
        invalidated = sum(
            1 for m in members if m.outcome == ThesisOutcome.INVALIDATED
        )
        win_rate = (correct / len(judged)) if judged else None
        ci = _wilson_ci(correct, len(judged)) if judged else None
        pnls = [m.pnl for m in members if m.pnl is not None]
        avg_pnl = sum(pnls) / len(pnls) if pnls else None
        llm_cost = sum(cost.get(m.id, 0.0) for m in members)
        out.append(
            CohortMetrics(
                key=key,
                labels=tuple(by),
                opened=opened,
                judged=len(judged),
                correct=correct,
                invalidated=invalidated,
                win_rate=win_rate,
                win_rate_ci=ci,
                avg_pnl=avg_pnl,
                llm_cost_usd=round(llm_cost, 4),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Calibration buckets (chart 4)
# ---------------------------------------------------------------------------


@dataclass
class CalibrationBucket:
    label: str           # e.g. "llm" or "random"
    bin_centre: float    # e.g. 0.55 for the [0.5, 0.6) bucket
    n: int
    realized: float      # observed hit rate in this bucket


def calibration_buckets(
    rows: Sequence[ThesisRow],
    *,
    split_by: str = "orchestrator_label",
    bins: int = 10,
) -> list[CalibrationBucket]:
    """Bucket judged theses by ``calibrated_p_correct`` and compute the
    observed hit rate per bucket, optionally split by another axis.

    Theses without a calibrated p are skipped — they have no place on a
    reliability plot.
    """
    judged = [
        r for r in rows
        if r.status == ThesisStatus.CLOSED
        and r.outcome in (ThesisOutcome.CORRECT, ThesisOutcome.INCORRECT, ThesisOutcome.PARTIAL)
        and r.calibrated_p_correct is not None
    ]
    edges = [i / bins for i in range(bins + 1)]
    by_group: dict[str, dict[int, list[ThesisRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in judged:
        p = r.calibrated_p_correct
        # Clamp p == 1.0 into the top bucket.
        idx = min(bins - 1, max(0, int(p * bins)))
        label = getattr(r, split_by, "all") or "unknown"
        by_group[str(label)][idx].append(r)

    out: list[CalibrationBucket] = []
    for label in sorted(by_group):
        for idx in sorted(by_group[label]):
            members = by_group[label][idx]
            if not members:
                continue
            wins = sum(1 for m in members if m.outcome == ThesisOutcome.CORRECT)
            out.append(
                CalibrationBucket(
                    label=label,
                    bin_centre=(edges[idx] + edges[idx + 1]) / 2,
                    n=len(members),
                    realized=wins / len(members),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Tag concentration over time (chart 6)
# ---------------------------------------------------------------------------


@dataclass
class TagSeriesPoint:
    day: str
    counts: dict[str, int]


def tag_concentration(
    rows: Sequence[ThesisRow],
    *,
    days: int = 30,
    top_n: int = 5,
) -> list[TagSeriesPoint]:
    """Open-thesis counts by dominant premise tag, one row per day.

    A thesis is "open on day D" if ``created_at <= D <= closed_at``
    (or still open). The top ``top_n`` tags by total active-day-count
    get their own series; everything else collapses into "other" so the
    chart stays legible.
    """
    today = datetime.now().date()
    span = [today - timedelta(days=i) for i in range(days - 1, -1, -1)]

    totals: dict[str, int] = defaultdict(int)
    for r in rows:
        dom = r.premise_tags[0] if r.premise_tags else None
        if dom is None:
            continue
        start = r.created_at.date()
        end = r.closed_at.date() if r.closed_at else today
        for d in span:
            if start <= d <= end:
                totals[dom] += 1
    top = {t for t, _ in sorted(totals.items(), key=lambda kv: -kv[1])[:top_n]}

    out: list[TagSeriesPoint] = []
    for d in span:
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            dom = r.premise_tags[0] if r.premise_tags else None
            if dom is None:
                continue
            start = r.created_at.date()
            end = r.closed_at.date() if r.closed_at else today
            if start <= d <= end:
                key = dom if dom in top else "other"
                counts[key] += 1
        out.append(TagSeriesPoint(day=d.isoformat(), counts=dict(counts)))
    return out


# ---------------------------------------------------------------------------
# Event fires (chart 6 overlay)
# ---------------------------------------------------------------------------


def invalidation_alerts(*, days: int = 30) -> list[tuple[str, str]]:
    """``(day, message)`` for PREMISE_INVALIDATION alerts in the window."""
    since = datetime.now() - timedelta(days=days)
    alerts = AlertRepository().list_all(since=since, limit=500)
    out: list[tuple[str, str]] = []
    for a in alerts:
        if a.alert_type == AlertType.PREMISE_INVALIDATION:
            out.append((a.created_at.date().isoformat(), a.message))
    return out


# ---------------------------------------------------------------------------
# Cumulative P&L curve (chart 7) — naive cumsum by close date.
# ---------------------------------------------------------------------------


@dataclass
class PnlPoint:
    day: str
    cum_pnl_by_label: dict[str, float]


def cumulative_pnl(
    rows: Sequence[ThesisRow],
    *,
    split_by: str = "orchestrator_label",
    days: int = 90,
) -> list[PnlPoint]:
    """Cumulative net P&L per group, indexed by close date.

    Uses ``ThesisRow.pnl`` which is already rolled up across legs and
    net of costs. Theses without a close date or pnl are skipped.
    """
    today = datetime.now().date()
    span = [today - timedelta(days=i) for i in range(days - 1, -1, -1)]
    grouped: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        if r.pnl is None or r.closed_at is None:
            continue
        d = r.closed_at.date()
        if d < span[0]:
            continue
        label = str(getattr(r, split_by, "all") or "unknown")
        grouped[label][d] = grouped[label].get(d, 0.0) + r.pnl

    out: list[PnlPoint] = []
    cum: dict[str, float] = defaultdict(float)
    for d in span:
        for label, by_day in grouped.items():
            cum[label] += by_day.get(d, 0.0)
        out.append(PnlPoint(day=d.isoformat(), cum_pnl_by_label=dict(cum)))
    return out


# ---------------------------------------------------------------------------
# Pinball scatter (chart 8) — one point per closed thesis.
# ---------------------------------------------------------------------------


@dataclass
class PinballPoint:
    label: str
    conviction_delta: float
    outcome: str
    pnl: float


def pinball_points(rows: Sequence[ThesisRow]) -> list[PinballPoint]:
    """Closed theses as (delta, outcome, pnl) tuples.

    ``conviction_delta`` is approximated from the thesis's ``conviction``
    centered at 50 — the orchestrator's per-symbol aggregate isn't
    persisted as a column, but ``conviction`` after the orchestrator's
    update IS, so ``conviction - 50`` is the recoverable proxy. Negative
    = bearish.
    """
    out: list[PinballPoint] = []
    for r in rows:
        if r.status != ThesisStatus.CLOSED or r.outcome is None or r.pnl is None:
            continue
        out.append(
            PinballPoint(
                label=r.orchestrator_label,
                conviction_delta=r.conviction - 50.0,
                outcome=r.outcome.value,
                pnl=r.pnl,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Tag × regime heatmap (chart 10)
# ---------------------------------------------------------------------------


@dataclass
class HeatCell:
    tag: str
    regime: str
    n: int
    win_rate: float | None


def tag_regime_heatmap(
    rows: Sequence[ThesisRow],
    *,
    min_n: int = 5,
    top_tags: int = 10,
) -> list[HeatCell]:
    """Hit rate per (dominant tag, regime) cell.

    Cells with fewer than ``min_n`` judged theses are returned with
    ``win_rate=None`` so the renderer can grey them out instead of
    quietly imputing 0%. Only the top ``top_tags`` by total N are
    kept — long-tail tags would dominate the row count and obscure
    the signal.
    """
    judged = [
        r for r in rows
        if r.status == ThesisStatus.CLOSED
        and r.outcome in (ThesisOutcome.CORRECT, ThesisOutcome.INCORRECT, ThesisOutcome.PARTIAL)
        and r.premise_tags
    ]
    by_tag_count: dict[str, int] = defaultdict(int)
    for r in judged:
        by_tag_count[r.premise_tags[0]] += 1
    tags = [t for t, _ in sorted(by_tag_count.items(), key=lambda kv: -kv[1])[:top_tags]]
    regimes = sorted({r.regime_label or "unknown" for r in judged})

    out: list[HeatCell] = []
    for tag in tags:
        for regime in regimes:
            members = [
                r for r in judged
                if r.premise_tags[0] == tag
                and (r.regime_label or "unknown") == regime
            ]
            if not members:
                out.append(HeatCell(tag=tag, regime=regime, n=0, win_rate=None))
                continue
            if len(members) < min_n:
                out.append(HeatCell(tag=tag, regime=regime, n=len(members), win_rate=None))
                continue
            wins = sum(1 for m in members if m.outcome == ThesisOutcome.CORRECT)
            out.append(
                HeatCell(
                    tag=tag,
                    regime=regime,
                    n=len(members),
                    win_rate=wins / len(members),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Bootstrap helper exposed for charts that want CIs on derived stats.
# ---------------------------------------------------------------------------


def bootstrap_diff_p(
    a: Sequence[int], b: Sequence[int], *, iters: int = 1000, seed: int = 17
) -> float:
    """Two-sided p-value for diff of means via bootstrap.

    ``a`` and ``b`` are 0/1 outcome lists (1 = win). Returns the
    fraction of resamples whose diff sign flips against the observed
    sign. Used by the 2×2 chart to footnote whether the LLM−random gap
    is significant.
    """
    if not a or not b:
        return 1.0
    rng = random.Random(seed)
    obs = (sum(a) / len(a)) - (sum(b) / len(b))
    if obs == 0:
        return 1.0
    sign = 1 if obs > 0 else -1
    against = 0
    n_a, n_b = len(a), len(b)
    for _ in range(iters):
        ra = [a[rng.randrange(n_a)] for _ in range(n_a)]
        rb = [b[rng.randrange(n_b)] for _ in range(n_b)]
        diff = (sum(ra) / n_a) - (sum(rb) / n_b)
        if (sign > 0 and diff <= 0) or (sign < 0 and diff >= 0):
            against += 1
    return against / iters
