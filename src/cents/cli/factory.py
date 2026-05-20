"""Factory CLI — the autonomous open/close loop over a symbol universe."""

from __future__ import annotations

from datetime import datetime, timedelta

import click

from cents.db import (
    FactoryRunRepository,
    LLMUsageRepository,
    PositionRepository,
    ThesisRepository,
    UniverseRepository,
)
from cents.exceptions import CostCapExceeded, ExperimentConfigDrift
from cents.factory.config import (
    get_factory_config_path,
    load_factory_config,
    scaffold_factory_config,
)
from cents.factory.engine import FactoryEngine, TAG_FACTORY
from cents.llm_usage import cost_cap, current_run_cap_usd, current_run_spend_usd
from cents.models import PositionStatus, ThesisCohort, ThesisOutcome, ThesisStatus
from cents.pricing import estimate_cost_usd
from cents.serialization import serialize

from ._disclosures import LOW_N_THRESHOLD, disclosure_text, low_n_warning
from ._shared import (
    default_subcommand,
    exit_with_error,
    resolve_output_format,
    respond_with_output,
)


@default_subcommand("status")
def factory(ctx):
    """Run and inspect the autonomous factory loop."""


@factory.command("init")
@click.option("--force", is_flag=True, help="Overwrite existing config")
def factory_init(force: bool):
    """Scaffold ~/.cents/factory.toml with sensible defaults."""
    try:
        path = scaffold_factory_config(force=force)
    except FileExistsError as exc:
        exit_with_error(str(exc))
    click.echo(f"Wrote factory config to {path}")


@factory.command("run")
@click.option("--dry-run", is_flag=True, help="Plan actions without mutating state")
@click.option("--universe", "universe_name", help="Universe name (defaults to config / default)")
@click.option(
    "--max-cost-usd",
    "max_cost_usd",
    type=float,
    default=None,
    help=(
        "Abort the run if cumulative LLM spend would exceed this many USD. "
        "Checked PRE-call against a token estimate so the offending call is "
        "never made."
    ),
)
@click.option(
    "--orchestrator",
    type=click.Choice(["llm", "random"]),
    default="llm",
    show_default=True,
    help=(
        "Which orchestrator to use. 'llm' = the real multi-agent stack. "
        "'random' = the control arm (uniform conviction_delta, no LLM calls); "
        "use this to generate the baseline cohort that the LLM arm must beat."
    ),
)
@click.option(
    "--orchestrator-seed",
    type=int,
    default=None,
    help="RNG seed for --orchestrator random (reproducibility).",
)
@click.option(
    "--force-frozen-drift",
    is_flag=True,
    default=False,
    help=(
        "Acknowledge factory.toml drift from the active experiment's frozen "
        "SHA and run anyway. cents-eat0: by default a drifted config aborts "
        "the run to preserve pre-registration discipline. Use only when you've "
        "decided the discipline violation is acceptable (e.g. unblocking a "
        "non-pilot ad-hoc run while an experiment is active). The drift is "
        "still persisted in summary_json so analytics can see it."
    ),
)
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def factory_run(
    dry_run: bool,
    universe_name: str | None,
    max_cost_usd: float | None,
    orchestrator: str,
    orchestrator_seed: int | None,
    force_frozen_drift: bool,
    output: str | None,
):
    """Run the factory engine once."""
    output = resolve_output_format(output)
    config = load_factory_config()
    if orchestrator == "random":
        from cents.agents.random_orchestrator import RandomOrchestrator

        engine = FactoryEngine(
            config=config,
            orchestrator=RandomOrchestrator(seed=orchestrator_seed),
        )
    else:
        engine = FactoryEngine(config=config)

    try:
        with cost_cap(max_cost_usd):
            run = engine.run(
                dry_run=dry_run,
                universe_override=universe_name,
                allow_frozen_drift=force_frozen_drift,
            )
            spend = current_run_spend_usd()
            cap = current_run_cap_usd()
    except CostCapExceeded as exc:
        exit_with_error(str(exc))
        return  # pragma: no cover — exit_with_error raises SystemExit
    except ExperimentConfigDrift as exc:
        exit_with_error(str(exc))
        return  # pragma: no cover — exit_with_error raises SystemExit

    if output == "text":
        if cap is not None:
            click.echo(f"LLM spend so far: ${spend:.4f} of ${cap:.4f}")
        elif spend > 0:
            click.echo(f"LLM spend so far: ${spend:.4f} (no cap set)")

    respond_with_output(
        output,
        serialize(run),
        lambda: _print_run(run, dry_run=dry_run),
    )


def _print_run(run, *, dry_run: bool) -> None:
    label = "[dry-run] " if dry_run else ""
    click.echo(f"{label}Factory run {run.id} on universe '{run.universe_name}'")
    click.echo(f"  Theses opened:   {run.theses_opened}")
    click.echo(f"  Theses closed:   {run.theses_closed}")
    click.echo(f"  Preemptions:     {run.preemptions}")
    click.echo(f"  Positions:       {run.positions_opened} opened, {run.positions_closed} closed")
    summary = run.summary_json or {}
    universe_size = summary.get("universe_size")
    evaluated = summary.get("symbols_evaluated")
    if universe_size is not None and evaluated is not None:
        below = summary.get("symbols_below_threshold", 0)
        held = summary.get("symbols_skipped_held", 0)
        timed_out = summary.get("symbols_timed_out", 0)
        stop_reason = summary.get("stop_reason", "end_of_universe")
        timed_out_str = f", {timed_out} timed out" if timed_out else ""
        click.echo(
            f"  Symbols:         {evaluated} evaluated / {universe_size} in universe "
            f"({below} below threshold, {held} held{timed_out_str}); stop: {stop_reason}"
        )
    if run.error:
        click.echo(f"  Error: {run.error}")
    proposals = summary.get("proposals", [])
    if proposals:
        click.echo("  Proposals:")
        for p in proposals:
            click.echo(f"    - {p['kind']}: {p['symbol']} ({p['detail']})")


@factory.command("status")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def factory_status(output: str | None):
    """Summarize the factory's current state."""
    output = resolve_output_format(output)
    run_repo = FactoryRunRepository()
    thesis_repo = ThesisRepository()
    position_repo = PositionRepository()
    config = load_factory_config()

    open_theses = [t for t in thesis_repo.list(status=ThesisStatus.OPEN) if TAG_FACTORY in t.tags]
    open_positions = position_repo.list(status=PositionStatus.OPEN)
    factory_thesis_ids = {t.id for t in open_theses}
    factory_positions = [p for p in open_positions if p.thesis_id in factory_thesis_ids]
    notional = sum(p.entry_price * p.size for p in factory_positions)

    paired = sum(1 for t in open_theses if t.cohort == ThesisCohort.NEUTRAL)
    directional = len(open_theses) - paired

    latest = run_repo.latest()
    recent_runs = run_repo.list(limit=5)

    payload = {
        "config_path": str(get_factory_config_path()),
        "universe": config.universe,
        "open_theses_total": len(open_theses),
        "open_theses_directional": directional,
        "open_theses_paired": paired,
        "open_positions": len(factory_positions),
        "current_notional_usd": notional,
        "budget_usd": config.budget_usd,
        "latest_run": serialize(latest) if latest else None,
        "recent_runs": [serialize(r) for r in recent_runs],
    }

    respond_with_output(
        output,
        payload,
        lambda: _print_status(payload),
    )


def _print_status(payload: dict) -> None:
    click.echo(f"Config:        {payload['config_path']}")
    click.echo(f"Universe:      {payload['universe']}")
    click.echo(
        f"Open theses:   {payload['open_theses_total']} "
        f"(directional={payload['open_theses_directional']}, paired={payload['open_theses_paired']})"
    )
    click.echo(
        f"Notional:      ${payload['current_notional_usd']:,.2f} / "
        f"${payload['budget_usd']:,.2f}"
    )
    if payload["latest_run"]:
        run = payload["latest_run"]
        click.echo(f"Last run:      {run['id']} at {run['started_at']} (dry_run={run['dry_run']})")
    click.echo(f"Recent runs:   {len(payload['recent_runs'])}")


from enum import Enum


class AnalyzeAxis(str, Enum):
    COHORT = "cohort"
    DISCOVERY = "discovery"
    REGIME = "regime"
    ORCHESTRATOR = "orchestrator"
    PREMISE_TAGS_COUNT = "premise_tags_count"


# Per-axis bucketing — extending the analyze surface = add a case here.
_AXIS_BUCKET = {
    AnalyzeAxis.COHORT: lambda t: t.cohort.value,
    AnalyzeAxis.DISCOVERY: lambda t: t.discovery_source or "unspecified",
    AnalyzeAxis.REGIME: lambda t: _regime_bucket(t.regime_snapshot),
    AnalyzeAxis.ORCHESTRATOR: lambda t: t.orchestrator_label or "unspecified",
    # Stratify by recorded tag count (cents-2xd4). Lets the analyst verify
    # the two arms have comparable tag-set distributions post-cap.
    AnalyzeAxis.PREMISE_TAGS_COUNT: lambda t: str(getattr(t, "premise_tags_count", 0) or 0),
}


@factory.command("analyze")
@click.option("--since-days", type=int, default=90, help="Look-back window in days")
@click.option(
    "--by",
    "by_axes",
    default="cohort",
    help=(
        "Comma-separated grouping axes "
        "(cohort,discovery,regime,orchestrator,premise_tags_count). "
        "Multiple axes produce a cross-tab. Default: cohort."
    ),
)
@click.option(
    "--include-cost-per-outcome",
    "include_cost",
    is_flag=True,
    default=False,
    help=(
        "Augment each cell with LLM cost per opened / judged / correct thesis. "
        "Lets the operator see whether spend per outcome exceeds the average "
        "P&L per outcome (negative-EV pipeline check). Adds an "
        "`unattributable_cost_usd` top-level field for LLM spend that could "
        "not be attributed to any thesis in the window."
    ),
)
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def factory_analyze(
    since_days: int,
    by_axes: str,
    include_cost: bool,
    output: str | None,
):
    """Outcomes stratified by one or more discovery / cohort / regime axes."""
    output = resolve_output_format(output)
    axis_strs = [a.strip() for a in by_axes.split(",") if a.strip()]
    if not axis_strs:
        exit_with_error("--by requires at least one axis")
    try:
        axes = [AnalyzeAxis(a) for a in axis_strs]
    except ValueError as exc:
        valid = ", ".join(a.value for a in AnalyzeAxis)
        exit_with_error(f"{exc}. Valid axes: {valid}")

    thesis_repo = ThesisRepository()
    cutoff = datetime.now() - timedelta(days=since_days)
    position_repo = PositionRepository()

    factory_theses = [t for t in thesis_repo.list() if TAG_FACTORY in t.tags]
    factory_theses = [t for t in factory_theses if t.created_at >= cutoff]

    # Pre-build positions index once (was: O(G × P) scan per cell).
    positions_by_thesis: dict[str, list] = {}
    for pos in position_repo.list():
        if pos.thesis_id:
            positions_by_thesis.setdefault(pos.thesis_id, []).append(pos)

    # Per-thesis LLM cost attribution (only when the operator opts in — the
    # join is cheap but we don't want to alter the default wire shape).
    cost_by_thesis: dict[str, float] = {}
    unattributable_cost_usd = 0.0
    if include_cost:
        cost_by_thesis, unattributable_cost_usd = _attribute_llm_cost(
            cutoff=cutoff, factory_theses=factory_theses
        )

    if axes == [AnalyzeAxis.COHORT]:
        directional = [t for t in factory_theses if t.cohort == ThesisCohort.DIRECTIONAL]
        paired = [t for t in factory_theses if t.cohort == ThesisCohort.NEUTRAL]
        d_metrics = _cohort_metrics(directional, positions_by_thesis)
        n_metrics = _cohort_metrics(paired, positions_by_thesis)
        d_metrics["low_n"] = d_metrics["judged"] < LOW_N_THRESHOLD
        n_metrics["low_n"] = n_metrics["judged"] < LOW_N_THRESHOLD
        if include_cost:
            _augment_with_cost(d_metrics, directional, cost_by_thesis)
            _augment_with_cost(n_metrics, paired, cost_by_thesis)
        any_low_n = d_metrics["low_n"] or n_metrics["low_n"]
        payload = {
            "since_days": since_days,
            "by": ["cohort"],
            "directional": d_metrics,
            "neutral": n_metrics,
            "_disclosure": disclosure_text(),
            "_low_n": any_low_n,
        }
        if include_cost:
            payload["unattributable_cost_usd"] = round(unattributable_cost_usd, 6)
        respond_with_output(
            output,
            payload,
            lambda: _print_analyze_legacy(payload, include_cost=include_cost),
        )
        return

    groups: dict[tuple[str, ...], list] = {}
    for t in factory_theses:
        key = tuple(_AXIS_BUCKET[axis](t) for axis in axes)
        groups.setdefault(key, []).append(t)

    cells: list[dict] = []
    any_low_n = False
    for key, theses in sorted(groups.items(), key=lambda kv: kv[0]):
        cell = {axis.value: key[i] for i, axis in enumerate(axes)}
        metrics = _cohort_metrics(theses, positions_by_thesis)
        metrics["low_n"] = metrics["judged"] < LOW_N_THRESHOLD
        if include_cost:
            _augment_with_cost(metrics, theses, cost_by_thesis)
        any_low_n = any_low_n or metrics["low_n"]
        cell["metrics"] = metrics
        cells.append(cell)

    payload = {
        "since_days": since_days,
        "by": [a.value for a in axes],
        "cells": cells,
        "_disclosure": disclosure_text(),
        "_low_n": any_low_n,
    }
    if include_cost:
        payload["unattributable_cost_usd"] = round(unattributable_cost_usd, 6)
    respond_with_output(
        output,
        payload,
        lambda: _print_analyze_crosstab(payload, include_cost=include_cost),
    )


def _regime_bucket(snapshot: dict) -> str:
    """Bucket a regime snapshot into a stable, interpretable label.

    Polarity bucket: derived from `polarity_score` ∈ {neg, zero, pos}.
    Volume bucket: derived from `event_count` (≤10 low, 11-30 med, >30 high).
    Result form is `polarity:volume` so it's grep-able in JSON output.
    """
    polarity = snapshot.get("polarity_score")
    if polarity is None:
        pol_label = "unknown"
    elif polarity < -0.05:
        pol_label = "neg"
    elif polarity > 0.05:
        pol_label = "pos"
    else:
        pol_label = "zero"

    count = snapshot.get("event_count")
    if count is None:
        vol_label = "unknown"
    elif count < 10:
        vol_label = "low"
    elif count <= 30:
        vol_label = "med"
    else:
        vol_label = "high"

    return f"{pol_label}:{vol_label}"


def _cohort_metrics(theses, positions_by_thesis: dict) -> dict:
    opened = len(theses)
    closed = [t for t in theses if t.status == ThesisStatus.CLOSED]
    preempted = [t for t in closed if t.outcome == ThesisOutcome.PREEMPTED]
    judged = [t for t in closed if t.outcome != ThesisOutcome.PREEMPTED]
    wins = [t for t in judged if t.outcome == ThesisOutcome.CORRECT]
    win_rate = (len(wins) / len(judged)) if judged else None

    pnl_values: list[float] = []
    held_days_values: list[float] = []
    for t in theses:
        for pos in positions_by_thesis.get(t.id, ()):
            if pos.pnl is not None:
                pnl_values.append(pos.pnl)
            if pos.exit_date and pos.entry_date:
                held_days_values.append((pos.exit_date - pos.entry_date).days)

    avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else None
    avg_held_days = sum(held_days_values) / len(held_days_values) if held_days_values else None

    return {
        "opened": opened,
        "closed": len(closed),
        "preempted": len(preempted),
        "judged": len(judged),
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "avg_held_days": avg_held_days,
    }


def _attribute_llm_cost(
    *,
    cutoff: datetime,
    factory_theses: list,
) -> tuple[dict[str, float], float]:
    """Build a `{thesis_id: total_cost_usd}` mapping from llm_usage rows.

    Attribution rules:
    1. If a row's `context` matches a known `thesis.id` directly, attribute
       to that thesis (future-proof for explicit thesis_id contexts).
    2. Else if `context` matches a `thesis.symbol` AND the call's `called_at`
       falls within the thesis's open lifetime, attribute to that thesis.
       Lifetime = ``thesis.created_at <= called_at <= (thesis.closed_at or now)``.
    3. Otherwise the row contributes to ``unattributable_cost_usd``
       (ad-hoc research calls, scan sweeps, anything not tied to a thesis).

    Premise-classification calls (operation == "classify_premise") run for
    every arm — they happen BEFORE a thesis is opened, as part of the
    common-overhead per-candidate evaluation. Attributing them to whatever
    thesis later opens on the same symbol charges the random arm for LLM
    cost the random orchestrator never emitted. Route them to
    ``unattributable`` instead so the per-arm cost-of-signal comparison
    isn't biased.

    Returns (cost_by_thesis, unattributable_cost_usd).
    """
    usage_rows = LLMUsageRepository().list_recent(since=cutoff, limit=None)
    by_id = {t.id: t for t in factory_theses}
    by_symbol: dict[str, list] = {}
    for t in factory_theses:
        if t.symbol:
            by_symbol.setdefault(t.symbol, []).append(t)

    # Operations that are experiment-wide overhead, not per-thesis cost.
    # Attributing them to a thesis would charge the wrong arm.
    _OVERHEAD_OPERATIONS = frozenset({"classify_premise", "tag_event"})

    cost_by_thesis: dict[str, float] = {}
    unattributable = 0.0
    for row in usage_rows:
        cost = estimate_cost_usd(
            row.model,
            row.input_tokens,
            row.output_tokens,
            cache_read=row.cache_read_input_tokens,
            cache_write=row.cache_creation_input_tokens,
        )
        if cost is None:
            # Unknown model — surface as unattributable rather than silently
            # treating as zero so we don't underreport spend.
            continue
        if getattr(row, "operation", None) in _OVERHEAD_OPERATIONS:
            unattributable += cost
            continue
        attributed = False
        ctx = row.context
        if ctx:
            # Rule 1: exact thesis_id match.
            thesis = by_id.get(ctx)
            if thesis is not None:
                cost_by_thesis[thesis.id] = cost_by_thesis.get(thesis.id, 0.0) + cost
                attributed = True
            else:
                # Rule 2: symbol + lifetime match.
                candidates = by_symbol.get(ctx, [])
                for t in candidates:
                    end = t.closed_at or datetime.now()
                    if t.created_at <= row.called_at <= end:
                        cost_by_thesis[t.id] = cost_by_thesis.get(t.id, 0.0) + cost
                        attributed = True
                        break
        if not attributed:
            unattributable += cost
    return cost_by_thesis, unattributable


def _augment_with_cost(
    metrics: dict,
    theses: list,
    cost_by_thesis: dict[str, float],
) -> None:
    """Mutate `metrics` to add cost-per-outcome fields for the given theses."""
    total = sum(cost_by_thesis.get(t.id, 0.0) for t in theses)
    opened = metrics["opened"]
    judged = metrics["judged"]
    correct = sum(
        1
        for t in theses
        if t.status == ThesisStatus.CLOSED and t.outcome == ThesisOutcome.CORRECT
    )
    metrics["llm_cost_total_usd"] = round(total, 6)
    metrics["llm_cost_per_opened"] = round(total / opened, 6) if opened else None
    metrics["llm_cost_per_judged"] = round(total / judged, 6) if judged else None
    metrics["llm_cost_per_correct"] = round(total / correct, 6) if correct else None


def _print_analyze_legacy(payload: dict, *, include_cost: bool = False) -> None:
    click.echo(f"Cohort analysis (last {payload['since_days']} days)")
    for cohort_name in ("directional", "neutral"):
        m = payload[cohort_name]
        click.echo(f"  {cohort_name}:")
        _print_metrics(m, include_cost=include_cost)
        warning = low_n_warning(m["judged"])
        if warning:
            click.echo(f"    {warning}")
    if include_cost:
        _print_unattributable_footer(payload)
    _print_disclosure_footer()


def _print_analyze_crosstab(payload: dict, *, include_cost: bool = False) -> None:
    axes = payload["by"]
    click.echo(
        f"Analysis (last {payload['since_days']} days) by " + ", ".join(axes)
    )
    if not payload["cells"]:
        click.echo("  (no factory theses in window)")
        if include_cost:
            _print_unattributable_footer(payload)
        _print_disclosure_footer()
        return
    for cell in payload["cells"]:
        key = " / ".join(f"{a}={cell[a]}" for a in axes)
        marker = " [low N]" if cell["metrics"].get("low_n") else ""
        click.echo(f"  {key}{marker}:")
        _print_metrics(cell["metrics"], include_cost=include_cost)
        warning = low_n_warning(cell["metrics"]["judged"])
        if warning:
            click.echo(f"    {warning}")
    if include_cost:
        _print_unattributable_footer(payload)
    _print_disclosure_footer()


def _print_disclosure_footer() -> None:
    click.echo("")
    click.echo(disclosure_text())


def _print_unattributable_footer(payload: dict) -> None:
    """Render the unattributable-cost line shown only when --include-cost-per-outcome is set."""
    unattr = payload.get("unattributable_cost_usd", 0.0) or 0.0
    click.echo(f"  unattributable LLM spend (no thesis): ${unattr:.4f}")


def _format_cost_field(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:.4f}"


def _print_metrics(m: dict, *, include_cost: bool = False) -> None:
    click.echo(f"    opened:    {m['opened']}")
    click.echo(f"    closed:    {m['closed']} (preempted: {m['preempted']})")
    win = "n/a" if m["win_rate"] is None else f"{m['win_rate'] * 100:.1f}%"
    avg_pnl = "n/a" if m["avg_pnl"] is None else f"${m['avg_pnl']:.2f}"
    avg_held = "n/a" if m["avg_held_days"] is None else f"{m['avg_held_days']:.1f}d"
    click.echo(f"    win_rate:  {win}")
    click.echo(f"    avg_pnl:   {avg_pnl}")
    click.echo(f"    avg_held:  {avg_held}")
    if include_cost:
        cpo = _format_cost_field(m.get("llm_cost_per_opened"))
        cpj = _format_cost_field(m.get("llm_cost_per_judged"))
        cpc = _format_cost_field(m.get("llm_cost_per_correct"))
        click.echo(
            f"    llm $/outcome: opened={cpo} judged={cpj} correct={cpc}"
        )
