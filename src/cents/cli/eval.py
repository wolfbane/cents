"""Eval CLI — run the LLM eval harness against golden sets.

The harness exercises the LIVE Anthropic API; if ANTHROPIC_API_KEY is not
configured, the runner returns a result with `skipped_reason` set and the
command exits non-zero so CI / cron can detect missing credentials.

Subcommands:

- ``cents eval run``: run premise + sentiment evals, optionally gating
  against a baseline, persisting today's metrics as the new baseline, or
  appending to the trailing-history JSONL for drift detection.
- ``cents eval golden show``: inspect packaged golden fixture sets.
- ``cents eval drift-check``: compare today's premise_f1 to the trailing
  median; fire a MODEL_DRIFT alert if the regression exceeds the threshold.
- ``cents eval calibrate-thresholds``: search the score-→-band threshold
  grid for the pair that maximises band accuracy on the sentiment golden
  set; write the result to ``src/cents/eval/thresholds.json``.
"""

from __future__ import annotations

import click

from cents.db import AlertRepository
from cents.eval.baseline import (
    detect_drift,
    evaluate_gate,
    load_baseline,
    load_history,
    persist_baseline as _persist_baseline_file,
    persist_history_row,
    persist_thresholds,
)
from cents.eval.calibrate import calibrate_thresholds
from cents.eval.runner import (
    load_premise_golden,
    load_sentiment_golden,
    run_eval,
    run_sentiment_eval,
)
from cents.eval.report import (
    print_eval,
    print_premise_golden,
    print_sentiment_golden,
)
from cents.models import Alert, AlertType

from ._shared import (
    default_subcommand,
    exit_with_error,
    resolve_output_format,
    respond_with_output,
)


_SET_CHOICES = ["premise", "sentiment", "all"]


@default_subcommand("run")
def eval_(ctx):
    """Run the LLM eval harness against golden sets."""


@eval_.command("run")
@click.option(
    "--set",
    "set_name",
    type=click.Choice(_SET_CHOICES),
    default="all",
    help="Which eval to run (default: all).",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Cap fixtures per set (handy for smoke-testing).",
)
@click.option(
    "--output", "-o",
    type=click.Choice(["text", "json"]),
    help="Output format.",
)
@click.option(
    "--gate",
    is_flag=True,
    help="Compare to baseline.json; exit non-zero on regression beyond tolerance.",
)
@click.option(
    "--baseline-f1",
    type=float,
    default=None,
    help="Override baseline F1 for one-off gating (otherwise reads baseline.json).",
)
@click.option(
    "--baseline-brier",
    type=float,
    default=None,
    help="Override baseline Brier for one-off gating (otherwise reads baseline.json).",
)
@click.option(
    "--tolerance-pp",
    type=float,
    default=5.0,
    help="Allowed metric drop in percentage points before --gate fails (default 5).",
)
@click.option(
    "--persist-baseline",
    is_flag=True,
    help="Write today's metrics to baseline.json and stamp locked_at.",
)
@click.option(
    "--persist-history",
    is_flag=True,
    help="Append today's metrics to ~/.cents/data/eval_history/YYYY-MM-DD.jsonl.",
)
def eval_run(
    set_name: str,
    limit: int | None,
    output: str | None,
    gate: bool,
    baseline_f1: float | None,
    baseline_brier: float | None,
    tolerance_pp: float,
    persist_baseline: bool,  # noqa: FBT001  click flag
    persist_history: bool,  # noqa: FBT001  click flag
):
    """Run the eval against the live Anthropic API."""
    output = resolve_output_format(output)
    result = run_eval(sets=set_name, limit=limit)

    # If both subevals are skipped (the only realistic skip path: missing key),
    # surface as a non-zero exit so CI / cron treat it as a failure.
    parts = [r for r in (result.premise, result.sentiment) if r is not None]
    all_skipped = parts and all(r.skipped_reason for r in parts)

    # Pull the metrics out of the result (None if that half was skipped).
    premise_f1 = result.premise.f1 if result.premise and not result.premise.skipped_reason else None
    sentiment_brier = (
        result.sentiment.brier_score
        if result.sentiment and not result.sentiment.skipped_reason
        else None
    )
    premise_f1_ci = (
        result.premise.f1_ci
        if result.premise and not result.premise.skipped_reason
        else None
    )
    sentiment_brier_ci = (
        result.sentiment.brier_ci
        if result.sentiment and not result.sentiment.skipped_reason
        else None
    )

    extra: dict = {}

    if persist_history and not all_skipped:
        n_premise = result.premise.fixtures_run if result.premise else 0
        n_sentiment = result.sentiment.fixtures_run if result.sentiment else 0
        persist_history_row(
            premise_f1=premise_f1,
            sentiment_brier=sentiment_brier,
            premise_f1_ci=premise_f1_ci,
            sentiment_brier_ci=sentiment_brier_ci,
            model_snapshot=result.model,
            n_fixtures_premise=n_premise,
            n_fixtures_sentiment=n_sentiment,
        )
        extra["history_persisted"] = True

    if persist_baseline and not all_skipped:
        new_baseline = _persist_baseline_file(
            premise_f1=premise_f1,
            sentiment_brier=sentiment_brier,
            model_snapshot=result.model,
        )
        extra["baseline_persisted"] = True
        extra["baseline"] = new_baseline

    gate_outcome: dict | None = None
    if gate:
        baseline = load_baseline()
        if baseline_f1 is not None:
            baseline = {**baseline, "premise_f1": baseline_f1, "locked_at": baseline.get("locked_at") or "override"}
        if baseline_brier is not None:
            baseline = {**baseline, "sentiment_brier": baseline_brier, "locked_at": baseline.get("locked_at") or "override"}
        gate_outcome = evaluate_gate(
            premise_f1, sentiment_brier, tolerance_pp=tolerance_pp, baseline=baseline
        )
        extra["gate"] = gate_outcome

    payload = result.to_dict()
    if extra:
        payload["extras"] = extra

    def _text_printer() -> None:
        print_eval(result)
        if gate_outcome is not None:
            click.echo("")
            click.echo("Baseline gate:")
            for msg in gate_outcome["messages"]:
                click.echo(f"  - {msg}")
            if gate_outcome["permissive"]:
                click.echo("  (gate is permissive; no failure)")
            elif gate_outcome["passed"]:
                click.echo("  PASS")
            else:
                click.echo("  FAIL")
        if extra.get("baseline_persisted"):
            click.echo("Baseline updated.")
        if extra.get("history_persisted"):
            click.echo("History row appended.")

    respond_with_output(output, payload, _text_printer)

    if all_skipped:
        raise SystemExit(1)
    if gate and gate_outcome is not None and not gate_outcome["passed"] and not gate_outcome["permissive"]:
        raise SystemExit(2)


_SET_CHOICES_GOLDEN = ["premise", "sentiment"]


@eval_.group("golden")
def golden():
    """Inspect the golden fixture sets."""


@golden.command("show")
@click.option(
    "--set",
    "set_name",
    type=click.Choice(_SET_CHOICES_GOLDEN),
    required=True,
    help="Which golden set to inspect.",
)
@click.option(
    "--output", "-o",
    type=click.Choice(["text", "json"]),
    help="Output format.",
)
def golden_show(set_name: str, output: str | None):
    """List fixtures in a golden set."""
    output = resolve_output_format(output)
    if set_name == "premise":
        fixtures = load_premise_golden()
        respond_with_output(
            output,
            {"set": "premise", "count": len(fixtures), "fixtures": fixtures},
            lambda: print_premise_golden(fixtures),
        )
    elif set_name == "sentiment":
        fixtures = load_sentiment_golden()
        respond_with_output(
            output,
            {"set": "sentiment", "count": len(fixtures), "fixtures": fixtures},
            lambda: print_sentiment_golden(fixtures),
        )
    else:
        exit_with_error(f"Unknown set: {set_name}")


@eval_.command("drift-check")
@click.option(
    "--threshold-pp",
    type=float,
    default=5.0,
    help="Drift threshold in percentage points (default: 5).",
)
@click.option(
    "--window",
    type=int,
    default=7,
    help="Trailing rows considered for the median (default: 7).",
)
@click.option(
    "--output", "-o",
    type=click.Choice(["text", "json"]),
    help="Output format.",
)
def eval_drift_check(threshold_pp: float, window: int, output: str | None):
    """Compare today's premise_f1 to the trailing-window median; fire a MODEL_DRIFT alert on regression."""
    output = resolve_output_format(output)
    outcome = detect_drift(threshold_pp=threshold_pp, window=window)

    if outcome["drift_detected"]:
        alert_repo = AlertRepository()
        alert = Alert(
            symbol="",
            alert_type=AlertType.MODEL_DRIFT,
            message=(
                f"Eval drift: premise_f1 {outcome['today_f1']:.3f} is "
                f"{abs(outcome['delta_pp']):.1f}pp below trailing-{outcome['window_size']} "
                f"median {outcome['median_f1']:.3f}."
            ),
            data={
                "today_f1": outcome["today_f1"],
                "median_f1": outcome["median_f1"],
                "delta_pp": outcome["delta_pp"],
                "window_size": outcome["window_size"],
                "threshold_pp": threshold_pp,
            },
        )
        alert_repo.create(alert)
        outcome["alert_id"] = alert.id

    def _text_printer() -> None:
        for msg in outcome["messages"]:
            click.echo(msg)
        if outcome["drift_detected"]:
            click.echo(f"MODEL_DRIFT alert fired (id={outcome.get('alert_id')}).")

    respond_with_output(output, outcome, _text_printer)
    if outcome["drift_detected"]:
        raise SystemExit(2)


@eval_.command("calibrate-thresholds")
@click.option(
    "--output", "-o",
    type=click.Choice(["text", "json"]),
    help="Output format.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Cap fixtures (handy for smoke-testing).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the recommended thresholds without writing thresholds.json.",
)
def eval_calibrate_thresholds(output: str | None, limit: int | None, dry_run: bool):
    """Search the threshold grid for the pair that maximises band-accuracy.

    Hits the live Anthropic API once to score every sentiment-golden fixture.
    Tests bypass this by injecting synthetic ``fixtures`` into
    ``calibrate_thresholds()`` directly.
    """
    output = resolve_output_format(output)
    sent_result = run_sentiment_eval(limit=limit)
    if sent_result.skipped_reason:
        exit_with_error(sent_result.skipped_reason)
    fixtures = [
        {"score": f.get("score", 0.0), "expected_score_band": f.get("expected_band")}
        for f in sent_result.fixtures
        if f.get("scoring_method") == "llm"
    ]
    calibration = calibrate_thresholds(fixtures)

    if not dry_run and calibration.n_fixtures > 0:
        persist_thresholds(
            positive_threshold=calibration.positive_threshold,
            negative_threshold=calibration.negative_threshold,
            accuracy=calibration.accuracy,
            model_snapshot=None,
        )

    payload = calibration.to_dict()
    payload["dry_run"] = dry_run

    def _text_printer() -> None:
        click.echo(
            f"Calibrated thresholds: positive={calibration.positive_threshold:.2f} "
            f"negative={calibration.negative_threshold:.2f}  "
            f"accuracy={calibration.accuracy:.3f}  "
            f"balance={calibration.balance:.3f}  "
            f"n={calibration.n_fixtures}  grid={calibration.grid_searched}"
        )
        if dry_run:
            click.echo("(dry-run: thresholds.json NOT updated)")

    respond_with_output(output, payload, _text_printer)
