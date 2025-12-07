"""API route blueprints."""

from flask import Blueprint, jsonify

api_bp = Blueprint("api", __name__)


@api_bp.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})
