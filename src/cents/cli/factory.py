"""Factory CLI — the autonomous open/close loop over a symbol universe."""

from __future__ import annotations

from datetime import datetime, timedelta

import click

from cents.config import get_settings
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
from cents.factory.universe_resolver import resolve_symbols
from cents.llm_usage import cost_cap, current_run_cap_usd, current_run_spend_usd
from cents.models import PositionStatus, PremiseSource, ThesisCohort, ThesisOutcome, ThesisStatus
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
    # Probe the configured default_universe so a stale/missing universe
    # doesn't silently produce 0 candidates once cron starts running.
    # Warnings only — operators set up cents in stages.
    _validate_default_universe()


def _validate_default_universe() -> None:
    """Probe the configured default universe and emit WARNINGs on issues.

    Checks:
      1. The universe named in factory.toml resolves to a Universe row.
      2. It has >=1 symbol after resolution.
      3. A spot-check ticker resolves at FMP (cheap profile lookup,
         absorbed by the cache on repeat calls).

    Never raises — this is an init-time visibility aid, not a gate.
    """
    try:
        cfg = load_factory_config()
    except Exception as exc:  # pragma: no cover — defensive
        click.echo(f"WARNING: could not load factory config to probe universe: {exc}")
        return

    universe_name = (cfg.universe or "").strip()
    repo = UniverseRepository()

    if universe_name.lower() == "default" or universe_name == "":
        universe = repo.get_default()
        display_name = "default"
        not_found_remediation = (
            "  - No universe is marked default. Create one and mark it default:\n"
            "      cents universe create <name> --source ...\n"
            "      cents universe set-default <name>"
        )
    else:
        universe = repo.get(universe_name)
        display_name = universe_name
        not_found_remediation = (
            f"  - Create one with: cents universe create {universe_name} --source ...\n"
            f"  - Or edit {get_factory_config_path()} to point at an existing universe\n"
            "  - Or set CENTS_FACTORY_CONFIG to point at a working config"
        )

    if universe is None:
        click.echo(
            f"WARNING: default_universe '{display_name}' is not registered in the cents DB."
        )
        click.echo(not_found_remediation)
        return

    try:
        symbols = resolve_symbols(universe)
    except Exception as exc:
        click.echo(
            f"WARNING: default_universe '{display_name}' failed to resolve symbols: {exc}"
        )
        click.echo(
            "  - Check the universe's source_config (e.g. screener parent, FMP index key)\n"
            f"  - Recreate it: cents universe create {display_name} --source ..."
        )
        return

    if not symbols:
        click.echo(
            f"WARNING: default_universe '{display_name}' resolved to 0 symbols."
        )
        click.echo(
            f"  - The universe exists but is empty. Recreate or extend it:\n"
            f"      cents universe create {display_name} --source ..."
        )
        return

    # FMP spot-check. Treat unconfigured key as a separate, softer warning
    # (the operator may be doing offline setup; an FMP_INDEX universe would
    # have already failed in resolve_symbols above with a clear error).
    settings = get_settings()
    if not settings.fmp_api_key:
        click.echo(
            f"NOTE: default_universe '{display_name}' has {len(symbols)} symbol(s); "
            "skipped FMP spot-check (FMP_API_KEY not configured)."
        )
        click.echo(
            "  - Set fmp_api_key in ~/.cents/config.toml (or FMP_API_KEY env var) "
            "before scheduling cents factory run."
        )
        return

    probe_symbol = symbols[0]
    try:
        from cents.data.fmp import FMPFundamentalsProvider

        provider = FMPFundamentalsProvider()
        profile = provider._fetch_json(
            "profile", symbol=probe_symbol, use_cache=True, daily_key=True
        )
    except Exception as exc:
        click.echo(
            f"WARNING: default_universe '{display_name}' has {len(symbols)} symbol(s) "
            f"but FMP probe of {probe_symbol} failed: {exc}"
        )
        click.echo(
            "  - Verify FMP_API_KEY is valid and the network reaches financialmodelingprep.com"
        )
        return

    if not profile:
        click.echo(
            f"WARNING: default_universe '{display_name}' has {len(symbols)} symbol(s) "
            f"but FMP probe of {probe_symbol} returned no data."
        )
        click.echo(
            "  - Verify the ticker is valid at FMP, or the FMP_API_KEY plan covers it\n"
            "  - The universe may contain stale / delisted symbols"
        )
        return

    click.echo(
        f"Probed default_universe '{display_name}': {len(symbols)} symbol(s); "
        f"FMP profile for {probe_symbol} OK."
    )


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


# ---- funnel -----------------------------------------------------------------
#
# Per-arm open-phase funnel: evaluated → rejected (by reason) → opened, plus
# cross-arm crowding attribution for concentration_cap rejections. Built for
# the pilot_v1 post-mortem question: "who is eating the LLM arm's opens —
# the threshold, the tag cap, or the other arm's book?"


def _funnel_signed_return(shadow) -> float | None:
    """Direction-aware forward return: had we opened, SHORT wins when fwd < 0."""
    fr = shadow.forward_return_30d
    if fr is None:
        fr = shadow.forward_return_60d
    if fr is None:
        return None
    return -fr if (shadow.primary_side or "").upper() == "SHORT" else fr


def _funnel_reason_cell(shadows: list) -> dict:
    """n / backfilled / mean signed fwd return / hit-rate for one (arm, reason)."""
    signed = [r for r in (_funnel_signed_return(s) for s in shadows) if r is not None]
    return {
        "n": len(shadows),
        "backfilled_n": len(signed),
        "mean_fwd_return": (sum(signed) / len(signed)) if signed else None,
        "hit_rate": (sum(1 for r in signed if r > 0) / len(signed)) if signed else None,
    }


def _funnel_crowding(shadows: list, factory_theses: list, cap: int) -> dict:
    """Attribute concentration_cap rejections to the arm(s) holding the bucket.

    For each rejected candidate, reconstructs which theses were open at
    rejection time and shared a capped (tag, direction) bucket with it, then
    counts the blockers by orchestrator arm. Uses the CURRENT cap value —
    if max_per_premise_tag changed inside the window, attribution is
    approximate.
    """
    out: dict[str, dict] = {}
    for s in shadows:
        if s.reason != "concentration_cap":
            continue
        arm = s.orchestrator_label or "llm"
        cell = out.setdefault(arm, {
            "blocked_n": 0,
            "blocked_with_other_arm_blocker": 0,
            "blocking_theses_by_arm": {},
        })
        cell["blocked_n"] += 1

        open_at_rejection = [
            t for t in factory_theses
            if t.created_at <= s.created_at
            and (t.closed_at is None or t.closed_at >= s.created_at)
        ]
        s_dir = s.premise_direction or {}
        blockers: dict[str, object] = {}
        for tag in s.premise_tags or []:
            want_dir = s_dir.get(tag, "*")
            sharing = [
                t for t in open_at_rejection
                if tag in (t.premise_tags or [])
                and (t.premise_direction or {}).get(tag, "*") == want_dir
            ]
            if len(sharing) >= cap:
                for t in sharing:
                    blockers[t.id] = t
        blocker_arms = [
            (getattr(t, "orchestrator_label", None) or "llm") for t in blockers.values()
        ]
        for blocker_arm in blocker_arms:
            counts = cell["blocking_theses_by_arm"]
            counts[blocker_arm] = counts.get(blocker_arm, 0) + 1
        if any(b != arm for b in blocker_arms):
            cell["blocked_with_other_arm_blocker"] += 1
    return out


@factory.command("funnel")
@click.option("--since-days", type=int, default=30, help="Look-back window in days")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def factory_funnel(since_days: int, output: str | None):
    """Per-arm open-phase funnel: evaluated → rejected (by reason) → opened.

    Run `cents shadow backfill` first so rejection rows carry forward
    returns — those tell you whether a rejection rule is leaving signal
    on the table. The crowding section attributes concentration_cap
    rejections to the arm(s) whose open theses held the bucket.
    """
    from cents.db import ShadowOpenRepository

    output = resolve_output_format(output)
    cfg = load_factory_config()
    cutoff = datetime.now() - timedelta(days=since_days)

    run_repo = FactoryRunRepository()
    thesis_repo = ThesisRepository()
    shadow_repo = ShadowOpenRepository()

    runs = [
        r for r in run_repo.list()
        if not r.dry_run and r.started_at >= cutoff
    ]
    shadows = [s for s in shadow_repo.list() if s.created_at >= cutoff]
    factory_theses = [t for t in thesis_repo.list() if TAG_FACTORY in t.tags]
    opened_in_window = [t for t in factory_theses if t.created_at >= cutoff]

    arms: dict[str, dict] = {}

    def _arm_cell(arm: str) -> dict:
        return arms.setdefault(arm, {
            "runs": 0,
            "evaluated": 0,
            "skipped_held": 0,
            "timed_out": 0,
            "opened": 0,
            "rejections": {},
        })

    for r in runs:
        summary = r.summary_json or {}
        cell = _arm_cell(summary.get("orchestrator") or "unknown")
        cell["runs"] += 1
        cell["evaluated"] += summary.get("symbols_evaluated", 0) or 0
        cell["skipped_held"] += summary.get("symbols_skipped_held", 0) or 0
        cell["timed_out"] += summary.get("symbols_timed_out", 0) or 0

    by_arm_reason: dict[tuple[str, str], list] = {}
    for s in shadows:
        by_arm_reason.setdefault(
            (s.orchestrator_label or "llm", s.reason), []
        ).append(s)
    for (arm, reason), rows in sorted(by_arm_reason.items()):
        _arm_cell(arm)["rejections"][reason] = _funnel_reason_cell(rows)

    for t in opened_in_window:
        _arm_cell(t.orchestrator_label or "llm")["opened"] += 1

    payload = {
        "since_days": since_days,
        "max_per_premise_tag": cfg.max_per_premise_tag,
        "arms": arms,
        "crowding": _funnel_crowding(shadows, factory_theses, cfg.max_per_premise_tag),
        "notes": [
            "Runs recorded before summary_json carried 'orchestrator' group under arm 'unknown'.",
            "Crowding attribution uses the CURRENT max_per_premise_tag; approximate if the cap changed inside the window.",
            "Forward-return columns require `cents shadow backfill` to have run past each row's horizon.",
        ],
    }
    respond_with_output(output, payload, lambda: _print_funnel(payload))


def _print_funnel(payload: dict) -> None:
    def _pct(v: float | None) -> str:
        return "—" if v is None else f"{v * 100:+.2f}%"

    def _rate(v: float | None) -> str:
        return "—" if v is None else f"{v * 100:.0f}%"

    click.echo("")
    click.echo(
        f"Open-phase funnel — last {payload['since_days']} days "
        f"(max_per_premise_tag={payload['max_per_premise_tag']})"
    )
    click.echo("-" * 72)
    for arm in sorted(payload["arms"].keys()):
        cell = payload["arms"][arm]
        click.echo(f"arm '{arm}' ({cell['runs']} runs):")
        click.echo(f"  evaluated            {cell['evaluated']:>5}")
        click.echo(f"  skipped (held)       {cell['skipped_held']:>5}")
        if cell["timed_out"]:
            click.echo(f"  timed out            {cell['timed_out']:>5}")
        for reason in sorted(cell["rejections"].keys()):
            rcell = cell["rejections"][reason]
            click.echo(
                f"  {reason:<20} {rcell['n']:>5}   "
                f"[fwd n={rcell['backfilled_n']:>3}  "
                f"mean={_pct(rcell['mean_fwd_return']):>8}  "
                f"hit={_rate(rcell['hit_rate']):>4}]"
            )
        click.echo(f"  opened               {cell['opened']:>5}")
        click.echo("")
    crowding = payload.get("crowding") or {}
    if crowding:
        click.echo("Cross-arm crowding (concentration_cap rejections):")
        for arm in sorted(crowding.keys()):
            c = crowding[arm]
            blockers = ", ".join(
                f"{a}={n}" for a, n in sorted(c["blocking_theses_by_arm"].items())
            ) or "n/a"
            click.echo(
                f"  {arm}: {c['blocked_n']} blocked; blocking theses by arm: {blockers}; "
                f"{c['blocked_with_other_arm_blocker']}/{c['blocked_n']} had ≥1 other-arm blocker"
            )
        click.echo("")
    for note in payload.get("notes") or []:
        click.echo(f"note: {note}")


from enum import Enum


class AnalyzeAxis(str, Enum):
    COHORT = "cohort"
    DISCOVERY = "discovery"
    REGIME = "regime"
    ORCHESTRATOR = "orchestrator"
    PREMISE_CLASSIFICATION_SOURCE = "premise_classification_source"
    PREMISE_TAGS_COUNT = "premise_tags_count"
    HEDGE_BASIS = "hedge_basis"


def _premise_source_bucket(t) -> str:
    raw = t.premise_classification_source
    return raw.value if hasattr(raw, "value") else (raw or PremiseSource.FALLBACK_EMPTY.value)


def _hedge_basis_bucket(t) -> str:
    raw = t.hedge_basis
    if raw is None:
        return "directional"
    return raw.value if hasattr(raw, "value") else raw


# Per-axis bucketing — extending the analyze surface = add a case here.
_AXIS_BUCKET = {
    AnalyzeAxis.COHORT: lambda t: t.cohort.value,
    AnalyzeAxis.DISCOVERY: lambda t: t.discovery_source or "unspecified",
    AnalyzeAxis.REGIME: lambda t: _regime_bucket(t.regime_snapshot),
    AnalyzeAxis.ORCHESTRATOR: lambda t: t.orchestrator_label or "unspecified",
    AnalyzeAxis.PREMISE_CLASSIFICATION_SOURCE: _premise_source_bucket,
    AnalyzeAxis.PREMISE_TAGS_COUNT: lambda t: str(t.premise_tags_count or 0),
    AnalyzeAxis.HEDGE_BASIS: _hedge_basis_bucket,
}


@factory.command("analyze")
@click.option("--since-days", type=int, default=90, help="Look-back window in days")
@click.option(
    "--by",
    "by_axes",
    default="cohort",
    help=(
        "Comma-separated grouping axes "
        "(cohort,discovery,regime,orchestrator,premise_classification_source,"
        "premise_tags_count,hedge_basis). "
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
