"""ASCII chart renderers for ``cents pilot dashboard``.

Pure functions: every renderer takes data (already prepared by
``cents.viz.queries``) and returns a rich Renderable. The CLI assembles
them into a Layout — that's where any interactive bits would live, not
here.

Third-party deps (``rich``, ``plotext``) are imported at module top
because anything that imports this module is already inside a CLI path
that opted into the ``[viz]`` extra. The CLI wrapper in
``cents.cli.pilot`` catches ImportError and prints the friendly hint.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Sequence

import plotext as plt
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cents.viz.queries import (
    CohortMetrics,
    CostDay,
    EvalPoint,
)


# ---------------------------------------------------------------------------
# Chart 1: experiment progress
# ---------------------------------------------------------------------------


def render_experiment_progress(
    *,
    experiment_name: str,
    started_at: datetime,
    minimum_calendar_days: int,
    minimum_n_per_arm: int,
    metrics_by_arm: dict[str, CohortMetrics],
    factory_sha: str | None = None,
) -> Panel:
    """Panel with two horizontal progress bars (N) + a calendar bar.

    ``metrics_by_arm`` is a dict like ``{"llm": ..., "random": ...}``
    where each value is the ``CohortMetrics`` cell for that arm.
    """
    elapsed = max(0, (datetime.now() - started_at).days)
    cal_pct = min(100.0, 100.0 * elapsed / minimum_calendar_days)
    cal_status = "✓" if elapsed >= minimum_calendar_days else f"need {minimum_calendar_days - elapsed}d"

    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column("arm", width=10)
    table.add_column("bar", ratio=2)
    table.add_column("count", width=14)
    table.add_column("hit", width=18)
    table.add_column("cost", justify="right", width=10)

    for arm in ("llm", "random"):
        m = metrics_by_arm.get(arm)
        opened = m.opened if m else 0
        bar = _hbar(opened, minimum_n_per_arm, width=22)
        count = f"{opened}/{minimum_n_per_arm}"
        if m and m.win_rate is not None:
            wr = f"{m.win_rate * 100:.1f}%"
            if m.win_rate_ci:
                lo, hi = m.win_rate_ci
                wr += f" [{lo*100:.0f}–{hi*100:.0f}]"
        else:
            wr = "—"
        cost = f"${(m.llm_cost_usd if m else 0):.2f}"
        table.add_row(arm, bar, count, wr, cost)

    cal_bar = _hbar(elapsed, minimum_calendar_days, width=22)
    table.add_row(
        Text("calendar", style="bold"),
        cal_bar,
        f"{elapsed}/{minimum_calendar_days}d",
        cal_status,
        "",
    )

    # Readiness is checked against the EXPECTED arms (llm + random) so
    # an absent arm reads as "0 opened", not "not enough info → ready".
    # Otherwise an experiment with only-LLM theses could falsely satisfy
    # the N-gate.
    expected_arms = ("llm", "random")
    n_ready = all(
        (metrics_by_arm.get(arm).opened if metrics_by_arm.get(arm) else 0)
        >= minimum_n_per_arm
        for arm in expected_arms
    )
    cal_ready = elapsed >= minimum_calendar_days
    if n_ready and cal_ready:
        verdict = Text("✓ ready to finalize", style="bold green")
    else:
        missing = []
        if not n_ready:
            missing.append("N")
        if not cal_ready:
            missing.append("calendar")
        verdict = Text(
            f"waiting on: {', '.join(missing)} (later-of stop)",
            style="yellow",
        )

    subtitle = factory_sha[:7] if factory_sha else ""
    title = f"{experiment_name} · {subtitle}".strip(" ·")
    return Panel(Group(table, Text(""), verdict), title=title, border_style="cyan")


# ---------------------------------------------------------------------------
# Chart 2: cost / drift strip
# ---------------------------------------------------------------------------


def render_cost_drift_strip(
    costs: Sequence[CostDay],
    evals: Sequence[EvalPoint],
    *,
    daily_cap_usd: float | None,
    baseline_f1: float | None,
    drift_threshold_pp: float = 5.0,
) -> Panel:
    """Two side-by-side bar/line plots — daily cost + F1 vs baseline.

    Renders to a multi-line string via plotext (which writes to stdout
    normally, but we capture via ``plt.build()``).
    """
    plt.clear_figure()
    plt.subplots(1, 2)

    # Left: daily spend. Use integer x to avoid plotext's date parser
    # auto-interpreting "MM-DD" strings as dd/mm/yyyy.
    plt.subplot(1, 1)
    plt.theme("clear")
    day_labels = [c.day[5:] for c in costs]  # MM-DD
    values = [c.cost_usd for c in costs]
    xs = list(range(len(values)))
    plt.bar(xs, values, marker="hd")
    plt.xticks(xs, day_labels)
    plt.title("LLM spend ($/day)")
    if daily_cap_usd is not None:
        plt.hline(daily_cap_usd, color="red")
        plt.ylim(0, max(daily_cap_usd * 1.1, max(values, default=0.0) * 1.1, 1.0))

    # Right: F1 over time.
    plt.subplot(1, 2)
    plt.theme("clear")
    if evals:
        eval_labels = [e.date[5:] for e in evals]
        f1 = [e.premise_f1 if e.premise_f1 is not None else float("nan") for e in evals]
        xs2 = list(range(len(f1)))
        plt.plot(xs2, f1, marker="hd")
        plt.xticks(xs2, eval_labels)
        plt.title("premise F1")
        if baseline_f1 is not None:
            plt.hline(baseline_f1, color="green")
            # Drift threshold = baseline − N pp.
            plt.hline(baseline_f1 - drift_threshold_pp / 100, color="red")
        plt.ylim(0.0, 1.0)
    else:
        plt.title("premise F1 (no history yet)")

    plt.plot_size(80, 12)
    chart = plt.build()
    plt.clear_figure()

    summary = Text()
    today_cost = costs[-1].cost_usd if costs else 0.0
    if daily_cap_usd is not None:
        summary.append(f"today ${today_cost:.2f} / cap ${daily_cap_usd:.2f}")
    else:
        summary.append(f"today ${today_cost:.2f}")
    if evals and baseline_f1 is not None and evals[-1].premise_f1 is not None:
        delta_pp = (evals[-1].premise_f1 - baseline_f1) * 100
        style = "red" if delta_pp < -drift_threshold_pp else "green"
        summary.append("  ·  ")
        summary.append(
            f"F1 {evals[-1].premise_f1:.3f} ({delta_pp:+.1f}pp vs baseline {baseline_f1:.3f})",
            style=style,
        )
    return Panel(Group(Text(chart), summary), title="cost & drift", border_style="cyan")


# ---------------------------------------------------------------------------
# Chart 11: cost-per-correct leaderboard
# ---------------------------------------------------------------------------


def render_cost_per_correct(
    metrics: Sequence[CohortMetrics],
    *,
    random_baseline: CohortMetrics | None = None,
) -> Panel:
    """Horizontal bar chart of $-per-correct per cohort.

    Cohorts with zero correct are shown as "—" rather than dropped, so
    a misbehaving arm doesn't disappear from the leaderboard.
    """
    rows: list[tuple[str, float | None, str]] = []
    for m in metrics:
        label = "·".join(m.key)
        per = (m.llm_cost_usd / m.correct) if m.correct else None
        rows.append((label, per, f"{m.correct}/{m.judged}"))
    rows.sort(key=lambda r: (r[1] is None, r[1] or 0.0))

    max_val = max((r[1] for r in rows if r[1] is not None), default=0.0) or 1.0
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column("cohort", width=32)
    table.add_column("bar", ratio=2)
    table.add_column("$ / correct", justify="right", width=14)
    table.add_column("correct/judged", justify="right", width=16)

    for label, per, ratio in rows:
        if per is None:
            table.add_row(label, "—", "—", ratio)
        else:
            table.add_row(label, _hbar(per, max_val, width=22), f"${per:.3f}", ratio)

    footer = Text(
        "lower is better — $ per correct thesis · random arm is the $0 baseline",
        style="dim",
    )
    return Panel(Group(table, footer), title="cost per correct", border_style="cyan")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hbar(value: float, cap: float, *, width: int = 20) -> str:
    """Unicode horizontal bar with eighths-resolution partial blocks.

    Returns ``"█"`` styled at exactly ``width`` characters when value
    >= cap. Empty string padded to width when cap is zero.
    """
    if cap <= 0:
        return " " * width
    frac = min(1.0, max(0.0, value / cap))
    full = math.floor(frac * width)
    remainder = (frac * width) - full
    partial_idx = math.floor(remainder * 8)
    partials = " ▏▎▍▌▋▊▉"
    out = "█" * full
    if full < width:
        out += partials[partial_idx]
    return out.ljust(width)


def assemble_dashboard(
    *panels: RenderableType,
) -> Group:
    """Stack a sequence of panels for the live dashboard.

    Trivially thin — exists so the CLI can stay declarative.
    """
    return Group(*panels)
