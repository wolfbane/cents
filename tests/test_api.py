"""Tests for Flask API."""

import pytest

from cents.api import create_app
from cents.api.errors import (
    APIError,
    ValidationError,
    NotFoundError,
    ConflictError,
    require_json,
    require_fields,
)


@pytest.fixture
def app():
    """Create test app."""
    return create_app({"TESTING": True})


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


def test_health_check(client):
    """Test health endpoint returns ok."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_404_handler(client):
    """Test 404 returns JSON error."""
    response = client.get("/api/v1/nonexistent")
    assert response.status_code == 404
    assert response.json["error"] == "not_found"


# --- APIError class tests ---


def test_api_error_to_dict():
    """Test APIError serialization."""
    error = APIError("Something went wrong")
    result = error.to_dict()
    assert result == {"error": "internal_error", "message": "Something went wrong"}


def test_api_error_with_details():
    """Test APIError with details."""
    error = APIError("Failed", details={"field": "name"})
    result = error.to_dict()
    assert result["details"] == {"field": "name"}


def test_validation_error():
    """Test ValidationError has correct status and code."""
    error = ValidationError("Invalid input")
    assert error.status_code == 400
    assert error.error_code == "validation_error"


def test_not_found_error():
    """Test NotFoundError has correct status and code."""
    error = NotFoundError("Thesis not found")
    assert error.status_code == 404
    assert error.error_code == "not_found"


def test_conflict_error():
    """Test ConflictError has correct status and code."""
    error = ConflictError("Already exists")
    assert error.status_code == 409
    assert error.error_code == "conflict"


# --- Error handler integration tests ---


def test_validation_error_handler(app, client):
    """Test ValidationError returns 400 with JSON."""
    @app.route("/test-validation")
    def trigger_validation():
        raise ValidationError("Bad input", details={"field": "symbol"})

    response = client.get("/test-validation")
    assert response.status_code == 400
    assert response.json["error"] == "validation_error"
    assert response.json["message"] == "Bad input"
    assert response.json["details"]["field"] == "symbol"


def test_not_found_error_handler(app, client):
    """Test NotFoundError returns 404 with JSON."""
    @app.route("/test-notfound")
    def trigger_notfound():
        raise NotFoundError("Thesis abc123 not found")

    response = client.get("/test-notfound")
    assert response.status_code == 404
    assert response.json["error"] == "not_found"
    assert "abc123" in response.json["message"]


def test_unexpected_error_handler(app, client):
    """Test unexpected errors return 500 with generic message."""
    @app.route("/test-crash")
    def trigger_crash():
        raise RuntimeError("Oops")

    response = client.get("/test-crash")
    assert response.status_code == 500
    assert response.json["error"] == "internal_error"
    # Should not leak internal error details
    assert "Oops" not in response.json["message"]


# --- require_json and require_fields tests ---


def test_require_json_success(app, client):
    """Test require_json returns parsed JSON."""
    @app.route("/test-json", methods=["POST"])
    def test_json():
        data = require_json()
        return {"received": data}

    response = client.post(
        "/test-json",
        json={"symbol": "AAPL"},
    )
    assert response.status_code == 200
    assert response.json["received"]["symbol"] == "AAPL"


def test_require_json_missing_content_type(app, client):
    """Test require_json fails without JSON content type."""
    @app.route("/test-json-ct", methods=["POST"])
    def test_json_ct():
        require_json()
        return {"ok": True}

    response = client.post("/test-json-ct", data="not json")
    assert response.status_code == 400
    assert "application/json" in response.json["message"]


def test_require_json_invalid_body(app, client):
    """Test require_json fails with invalid JSON."""
    @app.route("/test-json-invalid", methods=["POST"])
    def test_json_invalid():
        require_json()
        return {"ok": True}

    response = client.post(
        "/test-json-invalid",
        data="not valid json",
        content_type="application/json",
    )
    assert response.status_code == 400
    assert "Invalid JSON" in response.json["message"]


def test_require_fields_success():
    """Test require_fields passes with all fields present."""
    data = {"symbol": "AAPL", "price": 150}
    require_fields(data, "symbol", "price")  # Should not raise


def test_require_fields_missing():
    """Test require_fields raises for missing fields."""
    data = {"symbol": "AAPL"}
    with pytest.raises(ValidationError) as exc_info:
        require_fields(data, "symbol", "price", "quantity")

    error = exc_info.value
    assert "price" in error.message
    assert "quantity" in error.message
    assert error.details["missing_fields"] == ["price", "quantity"]


def test_require_fields_none_value():
    """Test require_fields treats None as missing."""
    data = {"symbol": "AAPL", "price": None}
    with pytest.raises(ValidationError) as exc_info:
        require_fields(data, "symbol", "price")

    assert "price" in exc_info.value.message
