"""Pretty-print eval results for the CLI.

JSON output is handled by the CLI via the shared `respond_with_output` helper;
this module is text-only.
"""

from __future__ import annotations

from typing import Callable

import click

from cents.eval.runner import EvalResult, PremiseEvalResult, SentimentEvalResult


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_ci(ci: tuple[float, float]) -> str:
    lo, hi = ci
    return f"[{lo:.3f}, {hi:.3f}]"


def _print_premise_text(result: PremiseEvalResult) -> None:
    click.echo("Premise classifier eval")
    if result.skipped_reason:
        click.echo(f"  SKIPPED: {result.skipped_reason}")
        return
    click.echo(f"  Fixtures:   {result.fixtures_run}")
    click.echo(f"  TP / FP / FN: {result.tp} / {result.fp} / {result.fn}")
    click.echo(
        f"  Precision:  {_fmt_pct(result.precision)}    "
        f"Recall:  {_fmt_pct(result.recall)}    "
        f"F1:  {result.f1:.3f} {_fmt_ci(result.f1_ci)}"
    )
    misses = [f for f in result.fixtures if f["fp"] > 0 or f["fn"] > 0]
    if misses:
        click.echo(f"  Imperfect fixtures: {len(misses)}")
        for f in misses[:5]:
            click.echo(
                f"    - {f['id']} ({f['symbol']}): "
                f"expected={f['expected']} predicted={f['predicted']}"
            )
        if len(misses) > 5:
            click.echo(f"    ... ({len(misses) - 5} more — see JSON output)")


def _print_sentiment_text(result: SentimentEvalResult) -> None:
    click.echo("Sentiment scorer eval")
    if result.skipped_reason:
        click.echo(f"  SKIPPED: {result.skipped_reason}")
        return
    click.echo(f"  Fixtures:    {result.fixtures_run}")
    click.echo(
        f"  Correct band: {result.correct_band}/{result.fixtures_run} "
        f"({result.accuracy:.3f} {_fmt_ci(result.accuracy_ci)})"
    )
    click.echo(
        f"  Brier score: {result.brier_score:.4f} {_fmt_ci(result.brier_ci)}  "
        f"(lower is better)"
    )
    click.echo("  Confusion matrix (rows=expected, cols=predicted):")
    bands = ("bullish", "neutral", "bearish")
    header = "             " + "  ".join(f"{b:>8}" for b in bands)
    click.echo(header)
    for expected in bands:
        row = result.confusion_matrix.get(expected, {})
        cells = "  ".join(f"{row.get(p, 0):>8d}" for p in bands)
        click.echo(f"    {expected:>8}: {cells}")


def print_eval(result: EvalResult) -> None:
    """Print a full EvalResult to stdout."""
    if result.model:
        click.echo(f"Model: {result.model}")
        click.echo("")
    if result.premise is not None:
        _print_premise_text(result.premise)
        click.echo("")
    if result.sentiment is not None:
        _print_sentiment_text(result.sentiment)


def print_premise_golden(fixtures: list[dict]) -> None:
    click.echo(f"Premise golden set: {len(fixtures)} fixtures")
    for f in fixtures:
        tags = ", ".join(f["expected_tags"]) or "(none)"
        click.echo(f"  {f['id']:>8} {f['symbol']:>6}  tags=[{tags}]")
        summary = f.get("thesis_summary", "")
        if summary:
            click.echo(f"           {summary[:100]}")


def print_sentiment_golden(fixtures: list[dict]) -> None:
    click.echo(f"Sentiment golden set: {len(fixtures)} fixtures")
    for f in fixtures:
        click.echo(
            f"  {f['id']:>8} {f['symbol']:>6}  band={f['expected_score_band']:>8}  "
            f"{f.get('article_title', '')[:80]}"
        )
