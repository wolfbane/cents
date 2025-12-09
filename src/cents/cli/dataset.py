"""CLI commands for dataset management."""

import click

from cents.datasets import (
    add_dataset,
    get_active_dataset,
    list_datasets,
    remove_dataset,
    set_active_dataset,
)


@click.group()
def dataset():
    """Manage named datasets (database files).

    Datasets allow you to work with multiple portfolios. Use 'cents dataset add'
    to register a database file, then 'cents dataset use' to switch between them.
    """
    pass


@dataset.command("add")
@click.argument("name")
@click.argument("path", type=click.Path())
def add_cmd(name: str, path: str):
    """Register a new named dataset.

    NAME is a short identifier for the dataset (e.g., 'friend', 'backup').
    PATH is the path to the database file.

    Example: cents dataset add friend ~/Downloads/friend-portfolio.db
    """
    try:
        resolved = add_dataset(name, path)
        click.echo(f"Added dataset '{name}' -> {resolved}")
    except ValueError as e:
        raise click.ClickException(str(e))


@dataset.command("use")
@click.argument("name")
def use_cmd(name: str):
    """Switch to a named dataset.

    All subsequent cents commands will use this dataset.

    Example: cents dataset use friend
    """
    try:
        set_active_dataset(name)
        click.echo(f"Switched to dataset '{name}'")
    except ValueError as e:
        raise click.ClickException(str(e))


@dataset.command("list")
def list_cmd():
    """List all registered datasets."""
    datasets = list_datasets()

    if not datasets:
        click.echo("No datasets configured")
        return

    for name, (path, is_active) in sorted(datasets.items()):
        marker = " (active)" if is_active else ""
        click.echo(f"  {name:15} {path}{marker}")


@dataset.command("remove")
@click.argument("name")
def remove_cmd(name: str):
    """Remove a named dataset (does not delete the database file).

    Example: cents dataset remove friend
    """
    try:
        remove_dataset(name)
        click.echo(f"Removed dataset '{name}'")
    except ValueError as e:
        raise click.ClickException(str(e))


@dataset.command("current")
def current_cmd():
    """Show the currently active dataset."""
    name, path = get_active_dataset()
    click.echo(f"{name} ({path})")
