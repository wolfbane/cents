"""API cache management CLI."""

from __future__ import annotations

import json

import click

from cents.cache import get_cache

from ._shared import default_subcommand


@default_subcommand("stats")
def cache(ctx):
    """Inspect and maintain the api_cache table."""


@cache.command("stats")
@click.option(
    "--output", "-o",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
)
def cache_stats(output: str):
    """Per-(provider, endpoint) row count, size, age, and TTL policy."""
    rows = get_cache().detailed_stats()
    if output == "json":
        click.echo(json.dumps(rows, indent=2))
        return

    if not rows:
        click.echo("api_cache is empty.")
        return

    header = f"{'provider':10s}  {'endpoint':32s}  {'rows':>6s}  {'size':>8s}  {'TTL':>5s}  oldest → newest"
    click.echo(header)
    click.echo("-" * len(header))
    total_rows = 0
    total_bytes = 0
    for r in rows:
        size = _fmt_bytes(r["bytes"])
        ttl = "—" if r["ttl_days"] is None else f"{r['ttl_days']}d"
        # Truncate iso strings to date for compactness
        oldest = (r["oldest"] or "")[:10]
        newest = (r["newest"] or "")[:10]
        click.echo(
            f"{r['provider']:10s}  {r['endpoint']:32s}  {r['rows']:>6d}  {size:>8s}  {ttl:>5s}  {oldest} → {newest}"
        )
        total_rows += r["rows"]
        total_bytes += r["bytes"]
    click.echo("-" * len(header))
    click.echo(f"{'total':10s}  {'':32s}  {total_rows:>6d}  {_fmt_bytes(total_bytes):>8s}")


@cache.command("prune")
@click.option(
    "--output", "-o",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
)
def cache_prune(output: str):
    """Delete cache rows past their TTL (per the policy in cents/cache.py)."""
    deleted = get_cache().prune()
    payload = {
        f"{p}/{e}": n for (p, e), n in deleted.items()
    }
    total = sum(deleted.values())
    if output == "json":
        click.echo(json.dumps({"deleted_by_endpoint": payload, "total": total}, indent=2))
        return

    if not deleted:
        click.echo("Nothing to prune (all TTLs within policy).")
        return
    click.echo(f"Pruned {total} rows:")
    for key, n in sorted(payload.items(), key=lambda kv: -kv[1]):
        click.echo(f"  {key:42s}  {n:>6d}")


@cache.command("clear")
@click.option("--provider", type=str, default=None, help="Only clear this provider")
@click.confirmation_option(prompt="Wipe the api_cache table? This cannot be undone.")
def cache_clear(provider: str | None):
    """DELETE every row in api_cache (or for a single provider)."""
    n = get_cache().clear(provider=provider)
    scope = f"provider={provider}" if provider else "all providers"
    click.echo(f"Cleared {n} rows ({scope}).")


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 ** 2:
        return f"{n/1024:.1f}K"
    if n < 1024 ** 3:
        return f"{n/1024/1024:.1f}M"
    return f"{n/1024/1024/1024:.1f}G"
