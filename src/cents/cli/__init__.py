"""CLI entry point for cents.

This package contains all CLI commands organized into submodules:
- thesis: Investment thesis management
- position: Position tracking
- research: Agent-based research
- outcome: Outcome recording
- scan: Watchlist scanning
- watch: Watchlist management
- alert: Alert management
- broker: Alpaca broker integration
"""

import logging

import click
from importlib.metadata import version as pkg_version

from .thesis import thesis
from .position import position
from .research import research
from .outcome import outcome
from .scan import scan
from .watch import watch
from .alert import alert
from .broker import broker
from .evidence import evidence

# Re-export shared utilities for backwards compatibility with tests
from ._shared import (
    validate_symbol,
    generate_thesis_suggestion as _generate_thesis_suggestion,
    evidence_to_dict as _evidence_to_dict,
)


@click.group()
@click.version_option(version=pkg_version("cents"), package_name="cents")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool):
    """Cents: Agentic investing guidance."""
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


# Register command groups
cli.add_command(thesis)
cli.add_command(position)
cli.add_command(research)
cli.add_command(outcome)
cli.add_command(scan)
cli.add_command(watch)
cli.add_command(alert)
cli.add_command(broker)
cli.add_command(evidence)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
