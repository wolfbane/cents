"""Calibration CLI — fit and inspect the logistic-regression calibration model.

The model maps `(aggregate_conviction_delta, regime, discovery, cohort)` →
`P(target hit before stop)`. Once fitted, the factory engine uses the
prediction as a Kelly fraction on share size at open time.
"""

from __future__ import annotations

import click

from cents.db import ThesisRepository
from cents.finance.calibration import (
    DEFAULT_MODEL_DIR,
    fit_calibration,
    load_latest_model,
    reliability_buckets,
    save_model,
)
from cents.models import ThesisOutcome, ThesisStatus

from ._shared import (
    default_subcommand,
    exit_with_error,
    resolve_output_format,
    respond_with_output,
)


@default_subcommand("report")
def calibration(ctx):
    """Fit and inspect the conviction calibration model."""


_MIN_OBSERVATIONS = 30


def _closed_decided_theses(repo: ThesisRepository) -> list:
    """Return theses with a decided win/loss outcome (calibration-eligible)."""
    decided = (ThesisOutcome.CORRECT, ThesisOutcome.INCORRECT)
    return [
        t
        for t in repo.list()
        if t.status != ThesisStatus.OPEN and t.outcome in decided
    ]


@calibration.command("refit")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
@click.option(
    "--min-observations",
    type=int,
    default=_MIN_OBSERVATIONS,
    show_default=True,
    help="Minimum number of decided theses required to fit.",
)
@click.option(
    "--holdout-pct",
    type=float,
    default=0.2,
    show_default=True,
    help=(
        "Fraction of observations held out for honest generalisation metrics. "
        "0.0 disables; the report then carries only in-sample numbers."
    ),
)
def calibration_refit(output: str | None, min_observations: int, holdout_pct: float):
    """Fit a fresh calibration model against the current outcomes dataset."""
    output = resolve_output_format(output)
    repo = ThesisRepository()
    theses = _closed_decided_theses(repo)
    if len(theses) < min_observations:
        exit_with_error(
            f"Not enough decided theses to fit: have {len(theses)}, "
            f"need {min_observations}. Run the factory longer or lower --min-observations."
        )

    model = fit_calibration(
        theses, min_observations=min_observations, holdout_pct=holdout_pct,
    )
    if model is None:
        # `fit_calibration` returns None when train split falls below
        # min_observations after the holdout cut.
        exit_with_error(
            "Calibration fit returned no model — train split below threshold "
            "after holdout. Try a smaller --holdout-pct or run the factory longer."
        )

    path = save_model(model)
    payload = {
        "model_path": str(path),
        "n_observations": model.n_observations,
        "n_train": model.n_train,
        "n_holdout": model.n_holdout,
        "brier_score_in_sample": model.brier_score,
        "auc_in_sample": model.auc,
        "brier_score_holdout": model.brier_holdout,
        "auc_holdout": model.auc_holdout,
        "intercept": model.intercept,
        "coef": model.coef,
        "fit_at": model.fit_at.isoformat(),
        "_caveat": (
            "Targets/stops are chosen by the engine itself, so this model "
            "fits the asymmetry of a fixed bracket — Brier/AUC look better "
            "than out-of-sample edge will be. See /scope/."
        ),
    }
    respond_with_output(output, payload, lambda: _print_refit(payload))


def _print_refit(payload: dict) -> None:
    n = payload["n_observations"]
    n_train = payload.get("n_train") or n
    n_holdout = payload.get("n_holdout") or 0
    click.echo(f"Fitted calibration on {n} observations (train={n_train}, holdout={n_holdout}).")
    click.echo(f"  Brier (in-sample): {payload['brier_score_in_sample']:.4f}")
    click.echo(f"  AUC   (in-sample): {payload['auc_in_sample']:.4f}")
    if payload.get("brier_score_holdout") is not None:
        click.echo(f"  Brier (held-out):  {payload['brier_score_holdout']:.4f}  ← honest metric")
        click.echo(f"  AUC   (held-out):  {payload['auc_holdout']:.4f}")
    click.echo(f"  Saved to:          {payload['model_path']}")
    click.echo()
    click.echo(f"  Note: {payload['_caveat']}")


@calibration.command("report")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), help="Output format")
def calibration_report(output: str | None):
    """Inspect the latest persisted model and its reliability diagram."""
    output = resolve_output_format(output)
    model = load_latest_model()
    if model is None:
        exit_with_error(
            f"No calibration model found in {DEFAULT_MODEL_DIR}. "
            "Run `cents calibration refit` first."
        )

    repo = ThesisRepository()
    theses = _closed_decided_theses(repo)
    diagram = reliability_buckets(model, theses)

    payload = {
        "model_dir": str(DEFAULT_MODEL_DIR),
        "n_observations": model.n_observations,
        "brier_score": model.brier_score,
        "auc": model.auc,
        "intercept": model.intercept,
        "coef": model.coef,
        "feature_names": model.feature_names,
        "fit_at": model.fit_at.isoformat(),
        "reliability": diagram,
    }
    respond_with_output(output, payload, lambda: _print_report(payload))


def _print_report(payload: dict) -> None:
    click.echo(f"Calibration model (fit at {payload['fit_at']})")
    click.echo(f"  N:            {payload['n_observations']}")
    click.echo(f"  Brier score:  {payload['brier_score']:.4f}")
    click.echo(f"  AUC:          {payload['auc']:.4f}")
    click.echo(f"  Intercept:    {payload['intercept']:+.4f}")
    click.echo("  Coefficients:")
    for name in payload["feature_names"]:
        weight = payload["coef"].get(name, 0.0)
        click.echo(f"    {name:<35} {weight:+.4f}")
    if payload["reliability"]:
        click.echo("  Reliability diagram (bucket → predicted vs actual):")
        for row in payload["reliability"]:
            click.echo(
                f"    [{row['bucket_low']:.1f}-{row['bucket_high']:.1f}) "
                f"n={row['n']:<4} pred={row['avg_predicted']:.3f} "
                f"actual={row['avg_actual']:.3f}"
            )
    else:
        click.echo("  Reliability diagram: (no decided theses to bucket)")
