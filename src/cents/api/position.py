"""Position API routes."""

from flask import Blueprint, jsonify

from cents.db import PositionRepository
from cents.models import Position, PositionSide, PositionStatus
from cents.serialization import serialize

from .errors import ValidationError, NotFoundError, require_json, require_fields

position_bp = Blueprint("position", __name__)


@position_bp.route("/positions", methods=["GET"])
def list_positions():
    """List all positions."""
    repo = PositionRepository()
    positions = repo.list()
    return jsonify([serialize(p) for p in positions])


@position_bp.route("/positions/<position_id>", methods=["GET"])
def get_position(position_id: str):
    """Get a position by ID."""
    repo = PositionRepository()
    position = repo.get(position_id)
    if position is None:
        raise NotFoundError(f"Position {position_id} not found")
    return jsonify(serialize(position))


@position_bp.route("/positions", methods=["POST"])
def create_position():
    """Create a new position."""
    data = require_json()
    require_fields(data, "symbol", "size", "entry_price")

    # Parse side, default to long
    side_str = data.get("side", "long").lower()
    try:
        side = PositionSide(side_str)
    except ValueError:
        raise ValidationError(f"Invalid side: {side_str}. Must be 'long' or 'short'")

    try:
        position = Position(
            symbol=data["symbol"].upper(),
            side=side,
            size=float(data["size"]),
            entry_price=float(data["entry_price"]),
            thesis_id=data.get("thesis_id"),
        )
    except ValueError as e:
        raise ValidationError(str(e))

    repo = PositionRepository()
    repo.create(position)
    return jsonify(serialize(position)), 201


@position_bp.route("/positions/<position_id>", methods=["PATCH"])
def update_position(position_id: str):
    """Update a position."""
    repo = PositionRepository()
    position = repo.get(position_id)
    if position is None:
        raise NotFoundError(f"Position {position_id} not found")

    data = require_json()

    # Update allowed fields
    if "size" in data:
        size = float(data["size"])
        if size <= 0:
            raise ValidationError("size must be positive")
        position.size = size
    if "status" in data:
        try:
            position.status = PositionStatus(data["status"])
        except ValueError:
            raise ValidationError(f"Invalid status: {data['status']}")
    if "exit_price" in data:
        exit_price = float(data["exit_price"])
        if exit_price <= 0:
            raise ValidationError("exit_price must be positive")
        position.exit_price = exit_price
    if "notes" in data:
        position.notes = data["notes"]

    repo.update(position)
    return jsonify(serialize(position))


@position_bp.route("/positions/<position_id>", methods=["DELETE"])
def delete_position(position_id: str):
    """Delete a position."""
    repo = PositionRepository()
    if not repo.delete(position_id):
        raise NotFoundError(f"Position {position_id} not found")
    return "", 204
