"""CLI commands for portfolio management."""

import click

from cents.datasets import (
    add_dataset,
    get_active_dataset,
    list_datasets,
    remove_dataset,
    set_active_dataset,
)

from ._shared import default_subcommand, exit_with_error


@default_subcommand("list")
def portfolio(ctx):
    """Manage portfolios (separate database files).

    Switch between different portfolios for tracking separate accounts
    or users. Use 'cents portfolio add' to register a database file,
    then 'cents portfolio use' to switch between them.
    """


@portfolio.command("add")
@click.argument("name")
@click.argument("path", type=click.Path())
def add_cmd(name: str, path: str):
    """Register a new portfolio.

    NAME is a short identifier (e.g., 'personal', 'ira', 'friend').
    PATH is the path to the database file.

    Example: cents portfolio add friend ~/Downloads/friend.db
    """
    try:
        resolved = add_dataset(name, path)
        click.echo(f"Added portfolio '{name}' -> {resolved}")
    except ValueError as e:
        exit_with_error(str(e))


@portfolio.command("use")
@click.argument("name")
def use_cmd(name: str):
    """Switch to a portfolio.

    All subsequent cents commands will use this portfolio.

    Example: cents portfolio use friend
    """
    try:
        set_active_dataset(name)
        click.echo(f"Switched to portfolio '{name}'")
    except ValueError as e:
        exit_with_error(str(e))


@portfolio.command("list")
def list_cmd():
    """List all registered portfolios."""
    datasets = list_datasets()

    if not datasets:
        click.echo("No portfolios configured")
        return

    for name, (path, is_active) in sorted(datasets.items()):
        marker = " (active)" if is_active else ""
        click.echo(f"  {name:15} {path}{marker}")


@portfolio.command("remove")
@click.argument("name")
def remove_cmd(name: str):
    """Remove a portfolio (does not delete the database file).

    Example: cents portfolio remove friend
    """
    try:
        remove_dataset(name)
        click.echo(f"Removed portfolio '{name}'")
    except ValueError as e:
        exit_with_error(str(e))


@portfolio.command("current")
def current_cmd():
    """Show the currently active portfolio."""
    name, path = get_active_dataset()
    click.echo(f"{name} ({path})")
