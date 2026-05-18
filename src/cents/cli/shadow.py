"""Shadow-open CLI — log + analyze rejected factory candidates.

The factory's open phase rejects candidates that don't clear the entry
threshold, hit the per-tag concentration cap, or can't fit the budget.
Without the shadow-open log we have no way to ask the most basic question:
"do the rejected names systematically underperform the accepted ones?"

`cents shadow analyze` aggregates closed thesis returns against shadow-open
forward returns, grouped by rejection reason and orchestrator arm.
`cents shadow backfill` walks the shadow_opens table and fills forward
returns via the supplied price provider.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import click

from cents.db import (
    OutcomeRepository,
    PositionRepository,
    ShadowOpenRepository,
    ThesisRepository,
)
from cents.factory.engine import TAG_FACTORY
from cents.factory.shadow import backfill_forward_returns
from cents.models import PositionStatus, ShadowOpen, ThesisStatus

from ._shared import (
    default_subcommand,
    resolve_output_format,
    respond_with_output,
)


@default_subcommand("analyze")
def shadow(ctx):
    """Inspect shadow-opens — rejected factory candidates and their forward returns."""


# ---- analyze ---------------------------------------------------------------


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _hit_rate(deltas: list[float], returns: list[float]) -> float | None:
    """Fraction of (delta, return) pairs whose signs agree."""
    if not deltas:
        return None
    hits = sum(
        1 for d, r in zip(deltas, returns)
        if (d > 0 and r > 0) or (d < 0 and r < 0)
    )
    return hits / len(deltas)


def _accepted_returns(
    thesis_repo: ThesisRepository,
    position_repo: PositionRepository,
) -> list[dict[str, Any]]:
    """Realized PnL% per factory-managed thesis, primary-leg only.

    For paired theses the primary leg is the one whose symbol matches the
    thesis's symbol; the hedge leg lives in pnl through the cohort report.
    Here we want per-thesis direction-aware return so we can compare like-for-like
    with shadow forward returns.
    """
    rows: list[dict[str, Any]] = []
    positions = position_repo.list()
    by_thesis: dict[str, list] = defaultdict(list)
    for pos in positions:
        if pos.thesis_id:
            by_thesis[pos.thesis_id].append(pos)

    for thesis in thesis_repo.list():
        if TAG_FACTORY not in thesis.tags:
            continue
        legs = by_thesis.get(thesis.id, [])
        primary = next(
            (p for p in legs if p.symbol == thesis.symbol),
            None,
        )
        if primary is None:
            continue
        if primary.status != PositionStatus.CLOSED:
            continue
        if primary.exit_price is None or primary.entry_price in (None, 0):
            continue
        direction = 1.0 if primary.side.value.lower() == "long" else -1.0
        pnl_pct = direction * (primary.exit_price - primary.entry_price) / primary.entry_price
        rows.append({
            "symbol": thesis.symbol,
            "return": pnl_pct,
            "orchestrator_label": getattr(thesis, "orchestrator_label", "llm"),
        })
    return rows


def _shadow_rows(repo: ShadowOpenRepository) -> list[ShadowOpen]:
    """All shadow-opens, regardless of backfill state."""
    return repo.list()


def _summarize(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Mean / N / hit-rate summary for a list of (delta?, return) dicts."""
    returns = [r["return"] for r in rows if r.get("return") is not None]
    deltas = [r["conviction_delta"] for r in rows if "conviction_delta" in r]
    if deltas and len(deltas) == len(returns):
        hr = _hit_rate(deltas, returns)
    else:
        hr = None
    return {
        "n": len(returns),
        "mean_return": _mean(returns),
        "hit_rate": hr,
    }


def _shadow_returns(shadows: list[ShadowOpen]) -> list[dict[str, Any]]:
    """Flatten shadow rows into (conviction_delta, return) records."""
    rows: list[dict[str, Any]] = []
    for s in shadows:
        fr = s.forward_return_30d
        if fr is None:
            fr = s.forward_return_60d
        if fr is None:
            continue
        # Direction-aware: had we opened, a SHORT bet wins when return < 0.
        signed_return = (
            -fr if (s.primary_side or "").upper() == "SHORT" else fr
        )
        rows.append({
            "symbol": s.symbol,
            "return": signed_return,
            "conviction_delta": s.conviction_delta,
            "reason": s.reason,
            "orchestrator_label": s.orchestrator_label or "llm",
        })
    return rows


def _bucket_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[r.get(key) or "unknown"].append(r)
    return out


@shadow.command("analyze")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def shadow_analyze(output: str | None):
    """Compare accepted-vs-rejected forward returns by reason and arm."""
    output = resolve_output_format(output)

    thesis_repo = ThesisRepository()
    position_repo = PositionRepository()
    shadow_repo = ShadowOpenRepository()

    accepted = _accepted_returns(thesis_repo, position_repo)
    shadow_records = _shadow_returns(_shadow_rows(shadow_repo))

    payload: dict[str, Any] = {
        "accepted": _summarize(accepted),
        "rejected": _summarize(shadow_records),
        "by_reason": {
            reason: _summarize(bucket)
            for reason, bucket in _bucket_by(shadow_records, "reason").items()
        },
        "by_orchestrator": {
            "accepted": {
                arm: _summarize(bucket)
                for arm, bucket in _bucket_by(accepted, "orchestrator_label").items()
            },
            "rejected": {
                arm: _summarize(bucket)
                for arm, bucket in _bucket_by(shadow_records, "orchestrator_label").items()
            },
        },
    }

    respond_with_output(output, payload, lambda: _print_analyze(payload))


def _format_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:+.2f}%"


def _format_rate(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def _print_summary_line(label: str, summary: dict[str, Any]) -> None:
    click.echo(
        f"  {label:<22} n={summary['n']:>3}  "
        f"mean={_format_pct(summary['mean_return']):>9}  "
        f"hit_rate={_format_rate(summary['hit_rate']):>6}"
    )


def _print_analyze(payload: dict[str, Any]) -> None:
    click.echo("")
    click.echo("Shadow-open vs. accepted-thesis returns")
    click.echo("-" * 56)
    _print_summary_line("accepted", payload["accepted"])
    _print_summary_line("rejected (all)", payload["rejected"])
    by_reason = payload.get("by_reason") or {}
    if by_reason:
        click.echo("")
        click.echo("  By reason:")
        for reason in sorted(by_reason.keys()):
            _print_summary_line(f"  {reason}", by_reason[reason])
    by_arm = payload.get("by_orchestrator") or {}
    if by_arm.get("accepted") or by_arm.get("rejected"):
        click.echo("")
        click.echo("  By orchestrator arm:")
        arms = set(by_arm.get("accepted", {}).keys()) | set(by_arm.get("rejected", {}).keys())
        for arm in sorted(arms):
            click.echo(f"    arm '{arm}':")
            if arm in by_arm.get("accepted", {}):
                _print_summary_line(f"    accepted/{arm}", by_arm["accepted"][arm])
            if arm in by_arm.get("rejected", {}):
                _print_summary_line(f"    rejected/{arm}", by_arm["rejected"][arm])
    click.echo("")


# ---- backfill --------------------------------------------------------------


@shadow.command("backfill")
@click.option(
    "--horizon", "horizon_days", type=int, default=30,
    help="Forward-return horizon in days (30 or 60). Default 30.",
)
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def shadow_backfill(horizon_days: int, output: str | None):
    """Walk shadow_opens past the horizon and fill forward returns from price history."""
    output = resolve_output_format(output)

    # Lazy import so the CLI module doesn't pay the Alpaca import cost on every
    # invocation of `cents shadow`.
    from cents.data.alpaca import get_price_provider

    provider = get_price_provider()
    result = backfill_forward_returns(provider, horizon_days=horizon_days)

    payload = {
        "horizon_days": horizon_days,
        "scanned": result.scanned,
        "filled": result.filled,
        "skipped_no_history": result.skipped_no_history,
        "skipped_no_entry_price": result.skipped_no_entry_price,
        "skipped_too_young": result.skipped_too_young,
    }

    def _print_backfill() -> None:
        click.echo(
            f"Shadow backfill (horizon={horizon_days}d): "
            f"scanned={result.scanned} filled={result.filled} "
            f"too_young={result.skipped_too_young} "
            f"no_history={result.skipped_no_history} "
            f"no_entry_price={result.skipped_no_entry_price}"
        )

    respond_with_output(output, payload, _print_backfill)
