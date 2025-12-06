"""Scan command for watchlist monitoring."""

import json
from datetime import datetime as dt

import click

from cents.agents import OrchestratorAgent
from cents.db import WatchlistRepository, AlertRepository, ThesisRepository, EvidenceRepository
from cents.models import Alert, AlertType, ThesisStatus
from cents.notify import notify

from ._shared import get_settings_lazy, generate_thesis_suggestion


@click.command("scan")
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=None,
    help="Default conviction change threshold for alerts (default: from config)",
)
@click.option("--webhook", "-w", help="Webhook URL for notifications")
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default=None,
    help="Output format for scan results (default: from config)",
)
@click.option("--quiet", is_flag=True, help="Suppress verbose logs for scripting")
@click.option("--expiry-days", type=int, default=7, help="Days before expiry to alert")
@click.option("--batch-suggest", is_flag=True, help="Generate thesis suggestions for all symbols")
@click.option("--apply", "apply_changes", is_flag=True, help="Save evidence and update thesis conviction")
def scan(threshold: float | None, webhook: str | None, output: str | None, quiet: bool, expiry_days: int, batch_suggest: bool, apply_changes: bool):
    """Scan watchlist and generate alerts for significant changes."""
    settings = get_settings_lazy()
    if threshold is None:
        threshold = settings.default_scan_threshold
    if output is None:
        output = settings.default_output
    verbose = output == "text" and not quiet

    watch_repo = WatchlistRepository()
    alert_repo = AlertRepository()
    thesis_repo = ThesisRepository()
    evidence_repo = EvidenceRepository() if apply_changes else None

    items = watch_repo.list()
    if not items:
        if output == "json":
            click.echo(json.dumps([], indent=2))
        else:
            click.echo("Watchlist is empty. Add symbols with: cents watch add <SYMBOL>")
        return

    if verbose:
        click.echo(f"Scanning {len(items)} symbols...\n")

    alerts_generated = 0
    scan_results: list[dict] = []

    for item in items:
        if verbose:
            click.echo(f"--- {item.symbol} ---")

        # Get linked thesis if any
        thesis = thesis_repo.get(item.thesis_id) if item.thesis_id else None

        # Run orchestrator
        agent = OrchestratorAgent()
        result = agent.research(item.symbol, thesis)

        # Save evidence and update conviction if --apply
        evidence_saved = 0
        evidence_skipped = 0
        if apply_changes and result.evidence:
            for e in result.evidence:
                e.symbol = item.symbol
                if thesis:
                    e.thesis_id = thesis.id
                if evidence_repo.create(e, dedupe=True):
                    evidence_saved += 1
                else:
                    evidence_skipped += 1

            # Only update conviction if we actually added new evidence
            if evidence_saved > 0 and thesis:
                # Scale delta by proportion of new evidence
                scale = evidence_saved / (evidence_saved + evidence_skipped)
                scaled_delta = result.conviction_delta * scale
                thesis.update_conviction(scaled_delta)
                thesis_repo.update(thesis)
                if verbose:
                    click.echo(f"  Applied: {evidence_saved} new evidence ({evidence_skipped} duplicates skipped), conviction now {thesis.conviction:.1f}%")
            elif evidence_saved > 0 and verbose:
                click.echo(f"  Saved {evidence_saved} new evidence ({evidence_skipped} duplicates skipped, no thesis linked)")
            elif verbose and evidence_skipped > 0:
                click.echo(f"  No new evidence ({evidence_skipped} duplicates skipped)")

        effective_threshold = item.threshold if item.threshold is not None else threshold
        destination = webhook or item.alert_destination or settings.default_webhook

        if verbose:
            click.echo(f"  {result.summary}")
            click.echo(f"  Threshold: {effective_threshold:+.1f}")

        triggered = False
        alert_message = None
        expiry_alert = None

        # Check for significant conviction change
        if abs(result.conviction_delta) >= effective_threshold:
            direction = "bullish" if result.conviction_delta > 0 else "bearish"
            alert_obj = Alert(
                symbol=item.symbol,
                alert_type=AlertType.CONVICTION_CHANGE,
                message=f"Significant {direction} signal: {result.conviction_delta:+.1f} conviction",
                data={
                    "conviction_delta": result.conviction_delta,
                    "evidence_count": len(result.evidence),
                },
            )
            alert_repo.create(alert_obj)
            alerts_generated += 1
            triggered = True
            alert_message = alert_obj.message
            if verbose:
                click.echo(f"  [!] Alert: {alert_obj.message}")
            notify(alert_obj, destination, quiet=quiet)

        # Check for thesis expiry
        if thesis and thesis.horizon_end and thesis.status == ThesisStatus.OPEN:
            days_until_expiry = (thesis.horizon_end - dt.now()).days
            if 0 <= days_until_expiry <= expiry_days:
                expiry_alert_msg = f"Thesis '{thesis.title}' expires in {days_until_expiry} days"
                expiry_alert = Alert(
                    symbol=item.symbol,
                    alert_type=AlertType.THESIS_EXPIRY,
                    message=expiry_alert_msg,
                    data={
                        "thesis_id": thesis.id,
                        "horizon_end": thesis.horizon_end.isoformat(),
                        "days_until_expiry": days_until_expiry,
                    },
                )
                alert_repo.create(expiry_alert)
                alerts_generated += 1
                if verbose:
                    click.echo(f"  [!] Expiry: {expiry_alert_msg}")
                notify(expiry_alert, destination, quiet=quiet)
            elif days_until_expiry < 0:
                expiry_alert_msg = f"Thesis '{thesis.title}' has expired ({-days_until_expiry} days ago)"
                expiry_alert = Alert(
                    symbol=item.symbol,
                    alert_type=AlertType.THESIS_EXPIRY,
                    message=expiry_alert_msg,
                    data={
                        "thesis_id": thesis.id,
                        "horizon_end": thesis.horizon_end.isoformat(),
                        "days_until_expiry": days_until_expiry,
                    },
                )
                alert_repo.create(expiry_alert)
                alerts_generated += 1
                if verbose:
                    click.echo(f"  [!] Expired: {expiry_alert_msg}")
                notify(expiry_alert, destination, quiet=quiet)

        # Update last_scanned
        watch_repo.update_scanned(item.symbol)
        if verbose:
            click.echo()

        scan_result = {
            "symbol": item.symbol,
            "summary": result.summary,
            "conviction_delta": result.conviction_delta,
            "dimension_scores": result.dimension_scores,
            "threshold": effective_threshold,
            "alerted": triggered,
            "alert_message": alert_message,
            "alert_destination": destination if triggered else None,
            "expiry_alert": expiry_alert.message if expiry_alert else None,
            "evidence_saved": evidence_saved,
            "conviction_updated": thesis.conviction if (apply_changes and thesis) else None,
        }

        # Generate thesis suggestion if requested
        if batch_suggest:
            # Collect evidence as dicts for suggestion generation
            evidence_dicts = []
            for ev in result.evidence:
                evidence_dicts.append({
                    "agent": ev.agent,
                    "type": ev.type.value,
                    "content": ev.content,
                    "dimension": ev.dimension.value if ev.dimension else None,
                    "metadata": ev.metadata,
                })
            agent_outputs = [{
                "agent": "orchestrator",
                "summary": result.summary,
                "conviction_delta": result.conviction_delta,
                "evidence": evidence_dicts,
            }]
            suggestion = generate_thesis_suggestion(
                item.symbol, agent_outputs, result.conviction_delta
            )
            suggestion["dimension_scores"] = result.dimension_scores
            scan_result["thesis_suggestion"] = suggestion

            if verbose:
                click.echo("  Thesis suggestion:")
                click.echo(f"    Valuation: {suggestion.get('valuation', 'unknown')}")
                click.echo(f"    Conviction: {suggestion.get('conviction', 50):.1f}")
                if result.dimension_scores:
                    dims = ", ".join(f"{k}: {v:+.0f}" for k, v in result.dimension_scores.items() if v != 0)
                    if dims:
                        click.echo(f"    Dimensions: {dims}")

        scan_results.append(scan_result)

    if output == "json":
        click.echo(json.dumps(scan_results, indent=2))
        return

    click.echo(f"Scan complete. Generated {alerts_generated} alerts.")
    if alerts_generated > 0 and not quiet:
        click.echo("View alerts with: cents alert list")

    if batch_suggest and not quiet:
        click.echo("\nUse --output=json to get full thesis suggestions for each symbol.")
