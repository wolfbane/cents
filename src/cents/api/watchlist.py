"""Watchlist API routes."""

from flask import Blueprint, jsonify

from cents.db import WatchlistRepository
from cents.models import WatchlistItem
from cents.serialization import serialize

from .errors import ValidationError, NotFoundError, ConflictError, require_json, require_fields

watchlist_bp = Blueprint("watchlist", __name__)


@watchlist_bp.route("/watchlist", methods=["GET"])
def list_watchlist():
    """List all watchlist items."""
    repo = WatchlistRepository()
    items = repo.list()
    return jsonify([serialize(item) for item in items])


@watchlist_bp.route("/watchlist/<symbol>", methods=["GET"])
def get_watchlist_item(symbol: str):
    """Get a watchlist item by symbol."""
    repo = WatchlistRepository()
    item = repo.get(symbol.upper())
    if item is None:
        raise NotFoundError(f"Symbol {symbol.upper()} not in watchlist")
    return jsonify(serialize(item))


@watchlist_bp.route("/watchlist", methods=["POST"])
def add_to_watchlist():
    """Add a symbol to watchlist."""
    data = require_json()
    require_fields(data, "symbol")

    symbol = data["symbol"].upper()
    repo = WatchlistRepository()

    # Check if already exists
    if repo.get(symbol) is not None:
        raise ConflictError(f"Symbol {symbol} already in watchlist")

    item = WatchlistItem(
        symbol=symbol,
        thesis_id=data.get("thesis_id"),
        threshold=data.get("threshold"),
        alert_destination=data.get("alert_destination"),
    )

    repo.add(item)
    return jsonify(serialize(item)), 201


@watchlist_bp.route("/watchlist/<symbol>", methods=["DELETE"])
def remove_from_watchlist(symbol: str):
    """Remove a symbol from watchlist."""
    repo = WatchlistRepository()
    if not repo.remove(symbol.upper()):
        raise NotFoundError(f"Symbol {symbol.upper()} not in watchlist")
    return "", 204
