"""``cents report`` — static visualization exports.

Emits PNGs + JSON sidecars (+ one HTML sunburst) into a dated snapshot
directory under ``website/public/reports/YYYY-MM-DD/`` by default. The
``--publish`` flag also mirrors the snapshot into ``reports/latest/``
so the public site picks up the newest run.

Dated snapshots are honest — you can show how the chart looked
mid-pilot. The trade-off was discussed in the design conversation:
default to dated + gitignored locally, opt-in to publish.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import click


ALL_CHARTS = (
    "hit_rate_2x2",
    "calibration_reliability",
    "invalidation_funnel",
    "tag_concentration",
    "pnl_curve",
    "pinball",
    "tag_regime_heatmap",
    "agent_contribution",
)


@click.command("report")
@click.option(
    "--experiment", "experiment_name", type=str, default=None,
    help="Experiment to report on. Defaults to the latest active experiment.",
)
@click.option(
    "--out", "out_path", type=click.Path(path_type=Path), default=None,
    help="Output directory. Default: ./reports/YYYY-MM-DD/",
)
@click.option(
    "--chart", "charts", type=click.Choice(ALL_CHARTS), multiple=True,
    help="Render only this chart. Repeatable. Default: all.",
)
@click.option(
    "--publish", is_flag=True,
    help="Mirror the snapshot into <root>/latest/ for the public site.",
)
@click.option(
    "--regime-spans", type=click.Path(path_type=Path), default=None,
    help="JSON file with [[start_day, end_day, label], …] for the P&L overlay.",
)
def report(
    experiment_name: str | None,
    out_path: Path | None,
    charts: tuple[str, ...],
    publish: bool,
    regime_spans: Path | None,
):
    """Render the full static report for an experiment."""
    try:
        from cents.viz import queries as q
        from cents.viz import static as viz_static
        from cents.viz import sunburst as viz_sunburst
    except ImportError:
        click.echo(
            "cents report needs the [viz] extra:  pip install -e '.[viz]'",
            err=True,
        )
        sys.exit(2)

    from cents.db.repository import ExperimentRepository

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
                "no active experiment; pass --experiment NAME",
                err=True,
            )
            sys.exit(2)
        active.sort(key=lambda e: e.started_at or datetime.min, reverse=True)
        exp = active[0]

    today = datetime.now().date().isoformat()
    out_dir = out_path or (Path.cwd() / "reports" / today)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected: Iterable[str] = charts or ALL_CHARTS

    rows = q.list_theses(experiment_id=exp.id)
    cost_map, _unattr = q.cost_by_thesis(rows)

    written: list[Path] = []

    if "hit_rate_2x2" in selected:
        cells = q.cohort_metrics(
            rows, by=["orchestrator", "cohort"], cost=cost_map
        )
        cells_by_key = {(c.key[0], c.key[1]): c for c in cells}
        written.append(viz_static.render_hit_rate_2x2(
            cells_by_key, rows, out_dir=out_dir,
        ))

    if "calibration_reliability" in selected:
        buckets = q.calibration_buckets(rows)
        written.append(viz_static.render_calibration_reliability(
            buckets, out_dir=out_dir,
        ))

    if "invalidation_funnel" in selected:
        written.append(viz_static.render_invalidation_funnel(
            rows, out_dir=out_dir,
        ))

    if "tag_concentration" in selected:
        pts = q.tag_concentration(rows, days=30)
        invs = q.invalidation_alerts(days=30)
        written.append(viz_static.render_tag_concentration(
            pts, invs, out_dir=out_dir,
        ))

    if "pnl_curve" in selected:
        pts = q.cumulative_pnl(rows, days=90)
        spans = _load_regime_spans(regime_spans)
        written.append(viz_static.render_pnl_curve(
            pts, regime_spans=spans, out_dir=out_dir,
        ))

    if "pinball" in selected:
        pts = q.pinball_points(rows)
        written.append(viz_static.render_pinball(pts, out_dir=out_dir))

    if "tag_regime_heatmap" in selected:
        cells = q.tag_regime_heatmap(rows)
        written.append(viz_static.render_tag_regime_heatmap(
            cells, out_dir=out_dir,
        ))

    if "agent_contribution" in selected:
        written.append(viz_sunburst.render_agent_contribution_sunburst(
            rows, out_dir=out_dir,
        ))

    click.echo(f"wrote {len(written)} chart{'s' if len(written) != 1 else ''} to {out_dir}")
    for p in written:
        click.echo(f"  {p.name}")

    if publish:
        latest = out_dir.parent / "latest"
        if latest.exists():
            shutil.rmtree(latest)
        shutil.copytree(out_dir, latest)
        click.echo(f"published → {latest}")


def _load_regime_spans(path: Path | None) -> list[tuple[str, str, str]]:
    """Load (start_day, end_day, label) tuples for the P&L overlay.

    Optional — empty list means "draw no shaded regime bands". The
    spans file is small enough to hand-edit, so no schema beyond a
    flat JSON list of triples is enforced.
    """
    if path is None:
        return []
    import json
    raw = json.loads(path.read_text())
    return [(str(s), str(e), str(l)) for s, e, l in raw]
