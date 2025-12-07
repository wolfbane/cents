"""API error handling middleware."""

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException


class APIError(Exception):
    """Base API error with status code and optional details."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        """Convert error to JSON-serializable dict."""
        response = {
            "error": self.error_code,
            "message": self.message,
        }
        if self.details:
            response["details"] = self.details
        return response


class ValidationError(APIError):
    """Request validation failed (400)."""

    status_code = 400
    error_code = "validation_error"


class NotFoundError(APIError):
    """Resource not found (404)."""

    status_code = 404
    error_code = "not_found"


class ConflictError(APIError):
    """Resource conflict, e.g. duplicate (409)."""

    status_code = 409
    error_code = "conflict"


def require_json() -> dict:
    """Get JSON body from request, raising ValidationError if missing/invalid.

    Returns:
        Parsed JSON dict from request body

    Raises:
        ValidationError: If Content-Type is not JSON or body is invalid
    """
    if not request.is_json:
        raise ValidationError("Content-Type must be application/json")
    data = request.get_json(silent=True)
    if data is None:
        raise ValidationError("Invalid JSON body")
    return data


def require_fields(data: dict, *fields: str) -> None:
    """Validate that required fields are present in data.

    Args:
        data: Dict to validate
        fields: Required field names

    Raises:
        ValidationError: If any required field is missing
    """
    missing = [f for f in fields if f not in data or data[f] is None]
    if missing:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing)}",
            details={"missing_fields": missing},
        )


def register_error_handlers(app: Flask) -> None:
    """Register error handlers on the Flask app."""

    @app.errorhandler(APIError)
    def handle_api_error(error: APIError):
        """Handle custom API errors."""
        return jsonify(error.to_dict()), error.status_code

    @app.errorhandler(HTTPException)
    def handle_http_exception(error: HTTPException):
        """Handle Werkzeug HTTP exceptions."""
        return jsonify({
            "error": error.name.lower().replace(" ", "_"),
            "message": error.description,
        }), error.code

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        """Handle unexpected errors."""
        # Log the error in non-testing mode
        if not app.config.get("TESTING"):
            app.logger.exception("Unexpected error: %s", error)
        return jsonify({
            "error": "internal_error",
            "message": "An unexpected error occurred",
        }), 500
