"""Thesis API routes."""

from flask import Blueprint, jsonify

from cents.db import ThesisRepository
from cents.models import Thesis, ThesisStatus
from cents.serialization import serialize

from .errors import ValidationError, NotFoundError, require_json, require_fields

thesis_bp = Blueprint("thesis", __name__)


@thesis_bp.route("/theses", methods=["GET"])
def list_theses():
    """List all theses."""
    repo = ThesisRepository()
    theses = repo.list()
    return jsonify([serialize(t) for t in theses])


@thesis_bp.route("/theses/<thesis_id>", methods=["GET"])
def get_thesis(thesis_id: str):
    """Get a thesis by ID."""
    repo = ThesisRepository()
    thesis = repo.get(thesis_id)
    if thesis is None:
        raise NotFoundError(f"Thesis {thesis_id} not found")
    return jsonify(serialize(thesis))


@thesis_bp.route("/theses", methods=["POST"])
def create_thesis():
    """Create a new thesis."""
    data = require_json()
    require_fields(data, "symbol", "title", "hypothesis")

    thesis = Thesis(
        symbol=data["symbol"].upper(),
        title=data["title"],
        hypothesis=data["hypothesis"],
        conviction=data.get("conviction", 50.0),
        valuation=data.get("valuation"),
        business_quality=data.get("business_quality"),
        time_horizon=data.get("time_horizon"),
        target_price=data.get("target_price"),
        stop_price=data.get("stop_price"),
    )

    repo = ThesisRepository()
    repo.create(thesis)
    return jsonify(serialize(thesis)), 201


@thesis_bp.route("/theses/<thesis_id>", methods=["PATCH"])
def update_thesis(thesis_id: str):
    """Update a thesis."""
    repo = ThesisRepository()
    thesis = repo.get(thesis_id)
    if thesis is None:
        raise NotFoundError(f"Thesis {thesis_id} not found")

    data = require_json()

    # Update allowed fields
    if "conviction" in data:
        thesis.conviction = float(data["conviction"])
    if "status" in data:
        thesis.status = ThesisStatus(data["status"])
    if "valuation" in data:
        thesis.valuation = data["valuation"]
    if "business_quality" in data:
        thesis.business_quality = data["business_quality"]
    if "target_price" in data:
        thesis.target_price = data["target_price"]
    if "stop_price" in data:
        thesis.stop_price = data["stop_price"]

    repo.update(thesis)
    return jsonify(serialize(thesis))


@thesis_bp.route("/theses/<thesis_id>", methods=["DELETE"])
def delete_thesis(thesis_id: str):
    """Delete a thesis."""
    repo = ThesisRepository()
    if not repo.delete(thesis_id):
        raise NotFoundError(f"Thesis {thesis_id} not found")
    return "", 204
