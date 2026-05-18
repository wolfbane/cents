"""CLI commands for cohort reporting (directional vs. policy-neutral)."""

from collections import defaultdict
from typing import Any

import click

from cents.db import PositionRepository, ThesisRepository
from cents.models import Position, PositionStatus, Thesis, ThesisCohort

from ._disclosures import LOW_N_THRESHOLD, disclosure_text, low_n_warning
from ._shared import resolve_output_format, respond_with_output


def _aggregate_cohort(
    theses: list[Thesis],
    positions_by_thesis: dict[str, list[Position]],
) -> dict[str, dict[str, Any]]:
    """Group theses by cohort and compute per-cohort aggregates.

    Spread P&L for a thesis is the sum across all its legs (long + short).
    Win rate = closed theses with positive realized P&L / closed theses.
    """
    buckets: dict[str, dict[str, Any]] = {
        cohort.value: {
            "cohort": cohort.value,
            "thesis_count": 0,
            "position_count": 0,
            "realized_pnl": 0.0,
            "closed_thesis_count": 0,
            "winning_theses": 0,
            "held_days_total": 0,
            "closed_position_count": 0,
        }
        for cohort in ThesisCohort
    }

    for thesis in theses:
        bucket = buckets[thesis.cohort.value]
        positions = positions_by_thesis.get(thesis.id, [])
        bucket["thesis_count"] += 1
        bucket["position_count"] += len(positions)

        thesis_realized = 0.0
        any_closed_leg = False
        all_closed = bool(positions)
        for pos in positions:
            if pos.status == PositionStatus.CLOSED and pos.pnl is not None:
                thesis_realized += pos.pnl
                any_closed_leg = True
                bucket["closed_position_count"] += 1
                if pos.exit_date and pos.entry_date:
                    bucket["held_days_total"] += (pos.exit_date - pos.entry_date).days
            else:
                all_closed = False

        bucket["realized_pnl"] += thesis_realized
        if positions and all_closed and any_closed_leg:
            bucket["closed_thesis_count"] += 1
            if thesis_realized > 0:
                bucket["winning_theses"] += 1

    for bucket in buckets.values():
        closed_n = bucket["closed_thesis_count"]
        bucket["win_rate"] = (bucket["winning_theses"] / closed_n) if closed_n else None
        closed_pos_n = bucket["closed_position_count"]
        bucket["avg_held_days"] = (bucket["held_days_total"] / closed_pos_n) if closed_pos_n else None
        # held_days_total/winning_theses are intermediate; drop from output
        del bucket["held_days_total"]
        del bucket["winning_theses"]

    return buckets


@click.command("cohort")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def cohort(output: str | None):
    """Report cohort-level P&L: directional vs. policy-neutral.

    Compares directional theses (implicit beta exposure) against neutral
    paired-hedge theses (control group) to isolate skill from macro beta.
    """
    output = resolve_output_format(output)

    thesis_repo = ThesisRepository()
    pos_repo = PositionRepository()
    all_theses = thesis_repo.list()
    all_positions = pos_repo.list()
    positions_by_thesis: dict[str, list[Position]] = defaultdict(list)
    for pos in all_positions:
        if pos.thesis_id:
            positions_by_thesis[pos.thesis_id].append(pos)

    buckets = _aggregate_cohort(all_theses, positions_by_thesis)
    ordered = [buckets[c.value] for c in ThesisCohort]

    # Low-N is judged on closed_thesis_count: an unclosed bucket has no
    # realized P&L to defend, so the warning is about anything that has
    # actually printed numbers.
    any_low_n = any(b["closed_thesis_count"] < LOW_N_THRESHOLD for b in ordered)

    payload = {
        "cohorts": ordered,
        "_disclosure": disclosure_text(),
        "_low_n": any_low_n,
    }

    respond_with_output(
        output,
        payload,
        lambda: _print_cohorts(ordered),
    )


def _print_cohorts(buckets: list[dict[str, Any]]) -> None:
    """Render cohort report in text form."""
    click.echo("")
    click.echo(f"{'Cohort':<12} {'Theses':>7} {'Positions':>10} {'Realized P&L':>14} {'Win Rate':>10} {'Avg Days':>10}")
    click.echo("-" * 66)
    for b in buckets:
        win = f"{b['win_rate'] * 100:.1f}%" if b["win_rate"] is not None else "—"
        held = f"{b['avg_held_days']:.1f}" if b["avg_held_days"] is not None else "—"
        sign = "+" if b["realized_pnl"] >= 0 else ""
        click.echo(
            f"{b['cohort']:<12} {b['thesis_count']:>7} {b['position_count']:>10} "
            f"{sign}${b['realized_pnl']:>12.2f} {win:>10} {held:>10}"
        )
    for b in buckets:
        warning = low_n_warning(b["closed_thesis_count"])
        if warning:
            click.echo(f"  [{b['cohort']}] {warning}")
    click.echo("")
    click.echo(disclosure_text())
    click.echo("")
