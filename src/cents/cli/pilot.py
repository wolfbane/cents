"""``cents pilot dashboard`` — operational view for a running experiment.

ASCII dashboard built from the three "ops" charts (experiment
progress, cost/drift strip, cost-per-correct leaderboard). Renders
once and exits with ``--once``; otherwise refreshes on key-press so a
solo developer can leave it open in a Tailscale-attached terminal on
the mini.

The renderers live in ``cents.viz.ascii``; data prep in
``cents.viz.queries``. This module only orchestrates.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import click

from ._shared import default_subcommand


@default_subcommand("dashboard")
def pilot(ctx):
    """Operational views for a running pilot experiment."""


@pilot.command("dashboard")
@click.option(
    "--experiment",
    "experiment_name",
    type=str,
    default=None,
    help="Experiment name. Defaults to the latest ACTIVE experiment.",
)
@click.option(
    "--once", is_flag=True, help="Render once and exit (for launchd / cron)."
)
@click.option(
    "--cost-window-days", type=int, default=14, show_default=True,
    help="Window for the cost-strip and eval-history mini-chart.",
)
def dashboard(experiment_name: str | None, once: bool, cost_window_days: int):
    """Render the live ops dashboard.

    Single-render mode by default — the loop / refresh behavior is
    explicitly opt-in for the hands-on case, because the launchd job
    just wants one PNG-equivalent pass.
    """
    try:
        from rich.console import Console
        from cents.viz import ascii as viz_ascii
        from cents.viz import queries as q
    except ImportError:
        click.echo(
            "cents pilot dashboard needs the [viz] extra:  pip install -e '.[viz]'",
            err=True,
        )
        sys.exit(2)

    from cents.config import get_settings
    from cents.db.repository import ExperimentRepository

    console = Console()

    exp_repo = ExperimentRepository()
    if experiment_name:
        exp = exp_repo.get_by_name(experiment_name)
        if exp is None:
            click.echo(f"experiment {experiment_name!r} not found", err=True)
            sys.exit(2)
    else:
        active = exp_repo.list_active()
        if not active:
            click.echo(
                "no active experiment. Use `cents experiment register ...` or pass --experiment.",
                err=True,
            )
            sys.exit(2)
        active.sort(key=lambda e: e.started_at or datetime.min, reverse=True)
        exp = active[0]

    rows = q.list_theses(experiment_id=exp.id)
    cost_map, _unattr = q.cost_by_thesis(rows)
    by_orchestrator = q.cohort_metrics(rows, by=["orchestrator"], cost=cost_map)
    metrics_by_arm = {m.key[0]: m for m in by_orchestrator}

    # Progress panel.
    minimum_n = getattr(exp, "minimum_n_per_arm", 0) or 0
    minimum_days = getattr(exp, "minimum_calendar_days", None) or 14
    progress_panel = viz_ascii.render_experiment_progress(
        experiment_name=exp.name,
        started_at=exp.started_at or datetime.now(),
        minimum_calendar_days=minimum_days,
        minimum_n_per_arm=minimum_n,
        metrics_by_arm=metrics_by_arm,
        factory_sha=getattr(exp, "frozen_config_sha", None),
    )

    # Cost + drift strip.
    settings = get_settings()
    daily_cap = settings.max_llm_spend_usd_per_day
    costs = q.daily_llm_costs(window_days=cost_window_days)
    evals = q.eval_history(days=cost_window_days)
    baseline_f1 = _load_baseline_f1()
    cost_panel = viz_ascii.render_cost_drift_strip(
        costs,
        evals,
        daily_cap_usd=daily_cap,
        baseline_f1=baseline_f1,
    )

    # Cost-per-correct leaderboard (3-axis grouping makes the table read
    # right — orchestrator × cohort × hedge_basis covers the headline
    # cells without exploding to cardinality > 12).
    cost_metrics = q.cohort_metrics(
        rows, by=["orchestrator", "cohort"], cost=cost_map
    )
    cost_panel_2 = viz_ascii.render_cost_per_correct(cost_metrics)

    dashboard_view = viz_ascii.assemble_dashboard(
        progress_panel, cost_panel, cost_panel_2
    )

    if once:
        console.print(dashboard_view)
        return

    # Refresh-on-keypress loop. Live-refresh would be possible with
    # rich.live but for a research dashboard, "press any key, re-pull"
    # is more honest about what just changed.
    while True:
        console.clear()
        console.print(dashboard_view)
        click.echo("\n[Enter] to refresh · Ctrl-C to exit")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            return
        rows = q.list_theses(experiment_id=exp.id)
        cost_map, _ = q.cost_by_thesis(rows)
        by_orchestrator = q.cohort_metrics(rows, by=["orchestrator"], cost=cost_map)
        metrics_by_arm = {m.key[0]: m for m in by_orchestrator}
        progress_panel = viz_ascii.render_experiment_progress(
            experiment_name=exp.name,
            started_at=exp.started_at or datetime.now(),
            minimum_calendar_days=minimum_days,
            minimum_n_per_arm=minimum_n,
            metrics_by_arm=metrics_by_arm,
            factory_sha=getattr(exp, "frozen_config_sha", None),
        )
        costs = q.daily_llm_costs(window_days=cost_window_days)
        evals = q.eval_history(days=cost_window_days)
        cost_panel = viz_ascii.render_cost_drift_strip(
            costs, evals,
            daily_cap_usd=daily_cap,
            baseline_f1=baseline_f1,
        )
        cost_metrics = q.cohort_metrics(
            rows, by=["orchestrator", "cohort"], cost=cost_map
        )
        cost_panel_2 = viz_ascii.render_cost_per_correct(cost_metrics)
        dashboard_view = viz_ascii.assemble_dashboard(
            progress_panel, cost_panel, cost_panel_2
        )


def _load_baseline_f1() -> float | None:
    """Read ``src/cents/eval/baseline.json`` for the locked F1.

    Returns ``None`` if the file is missing or malformed — the cost
    strip degrades gracefully (it just won't draw the green/red lines).
    """
    try:
        import cents.eval as _eval_pkg

        path = Path(_eval_pkg.__file__).parent / "baseline.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        v = data.get("premise_f1")
        return float(v) if v is not None else None
    except (ImportError, json.JSONDecodeError, ValueError, OSError):
        return None
