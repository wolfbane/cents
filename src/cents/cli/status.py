"""CLI command for showing system status."""

from pathlib import Path

import click

from cents.config import get_settings
from cents.datasets import get_active_dataset, get_datasets_path
from cents.db.repository import (
    ThesisRepository,
    PositionRepository,
    AlertRepository,
    WatchlistRepository,
)
from cents.models import ThesisStatus, PositionStatus


def _get_config_path() -> Path:
    """Get the config file path."""
    return Path.home() / ".cents" / "config.toml"


@click.command()
def status():
    """Show current configuration and database status.

    Displays the active dataset, config file locations, API key status,
    settings, and database statistics.
    """
    settings = get_settings()
    dataset_name, db_path = get_active_dataset()
    config_path = _get_config_path()
    datasets_path = get_datasets_path()

    # Dataset info
    click.echo()
    click.echo(click.style("Dataset", bold=True))
    click.echo(f"  Active:   {dataset_name}")
    click.echo(f"  Database: {db_path}")
    click.echo()

    # Config files
    click.echo(click.style("Config Files", bold=True))
    config_exists = "✓" if config_path.exists() else "✗"
    datasets_exists = "✓" if datasets_path.exists() else "✗"
    click.echo(f"  {config_exists} {config_path}")
    click.echo(f"  {datasets_exists} {datasets_path}")
    click.echo()

    # API Keys
    click.echo(click.style("API Keys", bold=True))
    keys = [
        ("FMP", settings.fmp_api_key),
        ("Alpaca", settings.alpaca_api_key),
        ("Anthropic", settings.anthropic_api_key),
        ("NewsAPI", settings.news_api_key),
        ("FRED", settings.fred_api_key),
    ]
    key_status = []
    for name, value in keys:
        marker = click.style("✓", fg="green") if value else click.style("✗", fg="red")
        key_status.append(f"{marker} {name}")

    # Display in two columns
    for i in range(0, len(key_status), 2):
        left = key_status[i]
        right = key_status[i + 1] if i + 1 < len(key_status) else ""
        click.echo(f"  {left:20} {right}")
    click.echo()

    # Settings
    click.echo(click.style("Settings", bold=True))
    click.echo(f"  Scan threshold: {settings.default_scan_threshold}")
    click.echo(f"  Output format:  {settings.default_output}")
    click.echo(f"  API timeout:    {settings.default_api_timeout}s")
    if settings.default_webhook:
        click.echo(f"  Webhook:        {settings.default_webhook}")
    click.echo()

    # Database stats
    click.echo(click.style("Database", bold=True))
    try:
        thesis_repo = ThesisRepository()
        position_repo = PositionRepository()
        alert_repo = AlertRepository()
        watchlist_repo = WatchlistRepository()

        all_theses = thesis_repo.list()
        open_theses = [t for t in all_theses if t.status == ThesisStatus.OPEN]

        all_positions = position_repo.list()
        open_positions = [p for p in all_positions if p.status == PositionStatus.OPEN]

        alerts = alert_repo.list_all(limit=1000)
        unread_alerts = alert_repo.list_unread()

        watchlist = watchlist_repo.list()

        click.echo(f"  Theses:    {len(open_theses)} open / {len(all_theses)} total")
        click.echo(f"  Positions: {len(open_positions)} open / {len(all_positions)} total")
        click.echo(f"  Alerts:    {len(unread_alerts)} unread / {len(alerts)} total")
        click.echo(f"  Watchlist: {len(watchlist)} symbols")
    except Exception as e:
        click.echo(f"  Error reading database: {e}")

    click.echo()
