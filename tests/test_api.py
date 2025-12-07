"""Tests for Flask API."""

import os
import tempfile
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
from cents.db import close_connection


@pytest.fixture
def app(tmp_path):
    """Create test app with isolated database."""
    # Use a temporary database file for test isolation
    db_file = tmp_path / "test_cents.db"
    old_db_path = os.environ.get("CENTS_DB_PATH")
    os.environ["CENTS_DB_PATH"] = str(db_file)

    # Close any existing connection so it picks up the new path
    close_connection()

    app = create_app({"TESTING": True})

    yield app

    # Cleanup: close connection and restore original env
    close_connection()
    if old_db_path is not None:
        os.environ["CENTS_DB_PATH"] = old_db_path
    else:
        os.environ.pop("CENTS_DB_PATH", None)


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


# --- Watchlist API tests ---


def test_list_watchlist_empty(client):
    """Test listing empty watchlist."""
    response = client.get("/api/v1/watchlist")
    assert response.status_code == 200
    assert response.json == []


def test_add_to_watchlist(client):
    """Test adding a symbol to watchlist."""
    response = client.post(
        "/api/v1/watchlist",
        json={"symbol": "AAPL"},
    )
    assert response.status_code == 201
    assert response.json["symbol"] == "AAPL"
    assert "id" in response.json


def test_add_to_watchlist_with_optional_fields(client):
    """Test adding with threshold and alert_destination."""
    response = client.post(
        "/api/v1/watchlist",
        json={
            "symbol": "NVDA",
            "threshold": 5.0,
            "alert_destination": "https://hooks.example.com/alert",
        },
    )
    assert response.status_code == 201
    assert response.json["symbol"] == "NVDA"
    assert response.json["threshold"] == 5.0
    assert response.json["alert_destination"] == "https://hooks.example.com/alert"


def test_add_to_watchlist_missing_symbol(client):
    """Test adding without required symbol field."""
    response = client.post(
        "/api/v1/watchlist",
        json={},
    )
    assert response.status_code == 400
    assert response.json["error"] == "validation_error"
    assert "symbol" in response.json["message"]


def test_add_to_watchlist_duplicate(client):
    """Test adding duplicate symbol returns conflict."""
    client.post("/api/v1/watchlist", json={"symbol": "TSLA"})
    response = client.post("/api/v1/watchlist", json={"symbol": "TSLA"})
    assert response.status_code == 409
    assert response.json["error"] == "conflict"
    assert "TSLA" in response.json["message"]


def test_add_to_watchlist_lowercase_normalizes(client):
    """Test symbol is normalized to uppercase."""
    response = client.post(
        "/api/v1/watchlist",
        json={"symbol": "goog"},
    )
    assert response.status_code == 201
    assert response.json["symbol"] == "GOOG"


def test_get_watchlist_item(client):
    """Test getting a watchlist item by symbol."""
    client.post("/api/v1/watchlist", json={"symbol": "MSFT"})
    response = client.get("/api/v1/watchlist/MSFT")
    assert response.status_code == 200
    assert response.json["symbol"] == "MSFT"


def test_get_watchlist_item_lowercase(client):
    """Test getting watchlist item with lowercase symbol."""
    client.post("/api/v1/watchlist", json={"symbol": "AMZN"})
    response = client.get("/api/v1/watchlist/amzn")
    assert response.status_code == 200
    assert response.json["symbol"] == "AMZN"


def test_get_watchlist_item_not_found(client):
    """Test getting non-existent watchlist item returns 404."""
    response = client.get("/api/v1/watchlist/NOTEXIST")
    assert response.status_code == 404
    assert response.json["error"] == "not_found"
    assert "NOTEXIST" in response.json["message"]


def test_remove_from_watchlist(client):
    """Test removing a symbol from watchlist."""
    client.post("/api/v1/watchlist", json={"symbol": "META"})
    response = client.delete("/api/v1/watchlist/META")
    assert response.status_code == 204
    assert response.data == b""

    # Verify it's gone
    response = client.get("/api/v1/watchlist/META")
    assert response.status_code == 404


def test_remove_from_watchlist_lowercase(client):
    """Test removing with lowercase symbol."""
    client.post("/api/v1/watchlist", json={"symbol": "NFLX"})
    response = client.delete("/api/v1/watchlist/nflx")
    assert response.status_code == 204


def test_remove_from_watchlist_not_found(client):
    """Test removing non-existent symbol returns 404."""
    response = client.delete("/api/v1/watchlist/NOTEXIST")
    assert response.status_code == 404
    assert response.json["error"] == "not_found"


def test_list_watchlist_with_items(client):
    """Test listing watchlist with multiple items."""
    client.post("/api/v1/watchlist", json={"symbol": "AAPL"})
    client.post("/api/v1/watchlist", json={"symbol": "GOOGL"})
    client.post("/api/v1/watchlist", json={"symbol": "MSFT"})

    response = client.get("/api/v1/watchlist")
    assert response.status_code == 200
    symbols = [item["symbol"] for item in response.json]
    assert "AAPL" in symbols
    assert "GOOGL" in symbols
    assert "MSFT" in symbols


# --- Research API tests ---


def test_list_agents(client):
    """Test GET /api/v1/research/agents lists available agents."""
    response = client.get("/api/v1/research/agents")
    assert response.status_code == 200
    assert "agents" in response.json
    # Should include known agents
    assert "fundamentals" in response.json["agents"]
    assert "technical" in response.json["agents"]
    assert "orchestrator" in response.json["agents"]


def test_run_research_basic(client, mocker):
    """Test POST /api/v1/research/AAPL runs research with mocked agent."""
    from cents.agents.base import AgentResult
    from cents.models import Evidence, EvidenceType

    # Create mock result
    mock_evidence = Evidence(
        thesis_id=None,
        symbol="AAPL",
        agent="orchestrator",
        content="Test evidence",
        source="test",
        type=EvidenceType.SUPPORTING,
        confidence=0.8,
    )
    mock_result = AgentResult(
        evidence=[mock_evidence],
        conviction_delta=5.0,
        summary="Test summary",
        dimension_scores={"valuation": 2.0, "quality": 3.0},
    )

    # Mock the OrchestratorAgent
    mock_agent_class = mocker.patch("cents.api.research.OrchestratorAgent")
    mock_agent_instance = mock_agent_class.return_value
    mock_agent_instance.research.return_value = mock_result

    # Mock EvidenceRepository to avoid DB calls
    mock_repo = mocker.patch("cents.api.research.EvidenceRepository")
    mock_repo.return_value.create.return_value = mock_evidence

    response = client.post("/api/v1/research/aapl?save=false")
    assert response.status_code == 200

    data = response.json
    assert data["symbol"] == "AAPL"  # Should be uppercased
    assert data["conviction_delta"] == 5.0
    assert data["summary"] == "Test summary"
    assert data["dimension_scores"]["valuation"] == 2.0
    assert len(data["evidence"]) == 1
    assert data["evidence"][0]["content"] == "Test evidence"


def test_run_research_specific_agent(client, mocker):
    """Test POST /api/v1/research/AAPL?agent=fundamentals uses specific agent."""
    from cents.agents.base import AgentResult

    mock_result = AgentResult(
        evidence=[],
        conviction_delta=3.0,
        summary="Fundamentals analysis",
    )

    # Mock the AGENTS dict
    mock_agent_class = mocker.MagicMock()
    mock_agent_class.return_value.research.return_value = mock_result
    mocker.patch.dict("cents.api.research.AGENTS", {"fundamentals": mock_agent_class})

    response = client.post("/api/v1/research/AAPL?agent=fundamentals&save=false")
    assert response.status_code == 200

    data = response.json
    assert data["conviction_delta"] == 3.0
    assert data["summary"] == "Fundamentals analysis"

    # Verify the fundamentals agent was called
    mock_agent_class.assert_called_once()
    mock_agent_class.return_value.research.assert_called_once()


def test_run_research_invalid_agent(client):
    """Test POST /api/v1/research/AAPL?agent=invalid returns 400."""
    response = client.post("/api/v1/research/AAPL?agent=invalid_agent")
    assert response.status_code == 400
    assert response.json["error"] == "validation_error"
    assert "Unknown agent" in response.json["message"]
    assert "valid_agents" in response.json["details"]


def test_run_research_with_thesis(client, mocker):
    """Test POST /api/v1/research/AAPL?thesis_id=xxx uses thesis context."""
    from cents.agents.base import AgentResult
    from cents.models import Thesis, ThesisStatus

    # Create a mock thesis
    mock_thesis = Thesis(
        id="test-thesis-123",
        title="AAPL Bull Thesis",
        hypothesis="Apple will grow",
        status=ThesisStatus.OPEN,
        conviction=50.0,
        symbol="AAPL",
    )

    mock_result = AgentResult(
        evidence=[],
        conviction_delta=5.0,
        summary="Research with thesis",
    )

    # Mock ThesisRepository.get to return thesis
    mock_thesis_repo = mocker.patch("cents.api.research.ThesisRepository")
    mock_thesis_repo.return_value.get.return_value = mock_thesis

    # Mock the OrchestratorAgent
    mock_agent_class = mocker.patch("cents.api.research.OrchestratorAgent")
    mock_agent_instance = mock_agent_class.return_value
    mock_agent_instance.research.return_value = mock_result

    response = client.post("/api/v1/research/AAPL?thesis_id=test-thesis-123&save=false")
    assert response.status_code == 200

    data = response.json
    assert data["thesis_id"] == "test-thesis-123"

    # Verify research was called with thesis
    mock_agent_instance.research.assert_called_once_with("AAPL", mock_thesis)


def test_run_research_thesis_not_found(client, mocker):
    """Test POST /api/v1/research/AAPL?thesis_id=xxx returns 404 if thesis not found."""
    # Mock ThesisRepository.get to return None
    mock_thesis_repo = mocker.patch("cents.api.research.ThesisRepository")
    mock_thesis_repo.return_value.get.return_value = None

    response = client.post("/api/v1/research/AAPL?thesis_id=nonexistent")
    assert response.status_code == 404
    assert response.json["error"] == "not_found"
    assert "nonexistent" in response.json["message"]


# --- Thesis API tests ---


def test_list_theses_empty(client):
    """Test listing empty theses."""
    response = client.get("/api/v1/theses")
    assert response.status_code == 200
    assert response.json == []


def test_create_thesis(client):
    """Test creating a thesis."""
    response = client.post(
        "/api/v1/theses",
        json={
            "symbol": "NVDA",
            "title": "AI Dominance",
            "hypothesis": "NVIDIA will dominate AI compute for the next 5 years",
        },
    )
    assert response.status_code == 201
    assert response.json["symbol"] == "NVDA"
    assert response.json["title"] == "AI Dominance"
    assert response.json["hypothesis"] == "NVIDIA will dominate AI compute for the next 5 years"
    assert response.json["conviction"] == 50.0
    assert response.json["status"] == "open"
    assert "id" in response.json


def test_create_thesis_with_optional_fields(client):
    """Test creating a thesis with optional fields."""
    response = client.post(
        "/api/v1/theses",
        json={
            "symbol": "AAPL",
            "title": "Apple AI Play",
            "hypothesis": "Apple Intelligence will drive upgrade cycle",
            "conviction": 75.0,
            "target_price": 250.0,
            "stop_price": 180.0,
            "business_quality": "High quality, recurring revenue",
        },
    )
    assert response.status_code == 201
    assert response.json["symbol"] == "AAPL"
    assert response.json["conviction"] == 75.0
    assert response.json["target_price"] == 250.0
    assert response.json["stop_price"] == 180.0
    assert response.json["business_quality"] == "High quality, recurring revenue"


def test_create_thesis_lowercase_symbol_normalizes(client):
    """Test symbol is normalized to uppercase."""
    response = client.post(
        "/api/v1/theses",
        json={
            "symbol": "tsla",
            "title": "Tesla Growth",
            "hypothesis": "Tesla will dominate autonomous driving",
        },
    )
    assert response.status_code == 201
    assert response.json["symbol"] == "TSLA"


def test_create_thesis_missing_required_fields(client):
    """Test creating thesis without required fields."""
    response = client.post(
        "/api/v1/theses",
        json={"symbol": "NVDA"},
    )
    assert response.status_code == 400
    assert response.json["error"] == "validation_error"
    assert "title" in response.json["message"]
    assert "hypothesis" in response.json["message"]


def test_create_thesis_missing_symbol(client):
    """Test creating thesis without symbol."""
    response = client.post(
        "/api/v1/theses",
        json={
            "title": "Test Title",
            "hypothesis": "Test hypothesis",
        },
    )
    assert response.status_code == 400
    assert response.json["error"] == "validation_error"
    assert "symbol" in response.json["message"]


def test_get_thesis(client):
    """Test getting a thesis by ID."""
    # Create a thesis first
    create_response = client.post(
        "/api/v1/theses",
        json={
            "symbol": "MSFT",
            "title": "Cloud Growth",
            "hypothesis": "Azure will continue growing",
        },
    )
    thesis_id = create_response.json["id"]

    # Get the thesis
    response = client.get(f"/api/v1/theses/{thesis_id}")
    assert response.status_code == 200
    assert response.json["id"] == thesis_id
    assert response.json["symbol"] == "MSFT"
    assert response.json["title"] == "Cloud Growth"


def test_get_thesis_not_found(client):
    """Test getting non-existent thesis returns 404."""
    response = client.get("/api/v1/theses/nonexistent")
    assert response.status_code == 404
    assert response.json["error"] == "not_found"
    assert "nonexistent" in response.json["message"]


def test_update_thesis(client):
    """Test updating a thesis."""
    # Create a thesis first
    create_response = client.post(
        "/api/v1/theses",
        json={
            "symbol": "GOOGL",
            "title": "Google AI",
            "hypothesis": "Gemini will compete with GPT",
        },
    )
    thesis_id = create_response.json["id"]

    # Update the thesis
    response = client.patch(
        f"/api/v1/theses/{thesis_id}",
        json={
            "conviction": 80.0,
            "target_price": 200.0,
        },
    )
    assert response.status_code == 200
    assert response.json["conviction"] == 80.0
    assert response.json["target_price"] == 200.0
    # Original fields should be preserved
    assert response.json["symbol"] == "GOOGL"
    assert response.json["title"] == "Google AI"


def test_update_thesis_status(client):
    """Test updating thesis status."""
    # Create a thesis first
    create_response = client.post(
        "/api/v1/theses",
        json={
            "symbol": "META",
            "title": "Meta VR",
            "hypothesis": "VR will become mainstream",
        },
    )
    thesis_id = create_response.json["id"]

    # Update status to closed
    response = client.patch(
        f"/api/v1/theses/{thesis_id}",
        json={"status": "closed"},
    )
    assert response.status_code == 200
    assert response.json["status"] == "closed"


def test_update_thesis_not_found(client):
    """Test updating non-existent thesis returns 404."""
    response = client.patch(
        "/api/v1/theses/nonexistent",
        json={"conviction": 80.0},
    )
    assert response.status_code == 404
    assert response.json["error"] == "not_found"


def test_delete_thesis(client):
    """Test deleting a thesis."""
    # Create a thesis first
    create_response = client.post(
        "/api/v1/theses",
        json={
            "symbol": "AMZN",
            "title": "AWS Growth",
            "hypothesis": "AWS will maintain cloud dominance",
        },
    )
    thesis_id = create_response.json["id"]

    # Delete the thesis
    response = client.delete(f"/api/v1/theses/{thesis_id}")
    assert response.status_code == 204
    assert response.data == b""

    # Verify it's gone
    response = client.get(f"/api/v1/theses/{thesis_id}")
    assert response.status_code == 404


def test_delete_thesis_not_found(client):
    """Test deleting non-existent thesis returns 404."""
    response = client.delete("/api/v1/theses/nonexistent")
    assert response.status_code == 404
    assert response.json["error"] == "not_found"


def test_list_theses_with_items(client):
    """Test listing theses with multiple items."""
    client.post(
        "/api/v1/theses",
        json={
            "symbol": "AAPL",
            "title": "Apple Growth",
            "hypothesis": "Apple will grow",
        },
    )
    client.post(
        "/api/v1/theses",
        json={
            "symbol": "GOOGL",
            "title": "Google Growth",
            "hypothesis": "Google will grow",
        },
    )
    client.post(
        "/api/v1/theses",
        json={
            "symbol": "MSFT",
            "title": "Microsoft Growth",
            "hypothesis": "Microsoft will grow",
        },
    )

    response = client.get("/api/v1/theses")
    assert response.status_code == 200
    assert len(response.json) == 3
    symbols = [thesis["symbol"] for thesis in response.json]
    assert "AAPL" in symbols
    assert "GOOGL" in symbols
    assert "MSFT" in symbols


# --- Position API tests ---


def test_list_positions_empty(client):
    """Test listing positions when none exist."""
    response = client.get("/api/v1/positions")
    assert response.status_code == 200
    assert response.json == []


def test_create_position(client):
    """Test creating a new position."""
    response = client.post(
        "/api/v1/positions",
        json={"symbol": "AAPL", "size": 100, "entry_price": 150.0},
    )
    assert response.status_code == 201
    assert response.json["symbol"] == "AAPL"
    assert response.json["size"] == 100
    assert response.json["entry_price"] == 150.0
    assert response.json["side"] == "long"
    assert response.json["status"] == "open"
    assert "id" in response.json


def test_create_position_with_side(client):
    """Test creating a short position."""
    response = client.post(
        "/api/v1/positions",
        json={"symbol": "TSLA", "size": 50, "entry_price": 200.0, "side": "short"},
    )
    assert response.status_code == 201
    assert response.json["side"] == "short"


def test_create_position_with_thesis_id(client):
    """Test creating a position linked to a thesis."""
    # First create a thesis to link to
    thesis_response = client.post(
        "/api/v1/theses",
        json={
            "title": "Test Thesis for Position",
            "symbol": "NVDA",
            "hypothesis": "AI demand will drive growth",
        },
    )
    thesis_id = thesis_response.json["id"]

    response = client.post(
        "/api/v1/positions",
        json={
            "symbol": "NVDA",
            "size": 10,
            "entry_price": 500.0,
            "thesis_id": thesis_id,
        },
    )
    assert response.status_code == 201
    assert response.json["thesis_id"] == thesis_id


def test_create_position_lowercase_symbol(client):
    """Test symbol is normalized to uppercase."""
    response = client.post(
        "/api/v1/positions",
        json={"symbol": "goog", "size": 25, "entry_price": 140.0},
    )
    assert response.status_code == 201
    assert response.json["symbol"] == "GOOG"


def test_create_position_missing_symbol(client):
    """Test creating position without symbol returns validation error."""
    response = client.post(
        "/api/v1/positions",
        json={"size": 100, "entry_price": 150.0},
    )
    assert response.status_code == 400
    assert response.json["error"] == "validation_error"
    assert "symbol" in response.json["message"]


def test_create_position_missing_size(client):
    """Test creating position without size returns validation error."""
    response = client.post(
        "/api/v1/positions",
        json={"symbol": "AAPL", "entry_price": 150.0},
    )
    assert response.status_code == 400
    assert response.json["error"] == "validation_error"
    assert "size" in response.json["message"]


def test_create_position_missing_entry_price(client):
    """Test creating position without entry_price returns validation error."""
    response = client.post(
        "/api/v1/positions",
        json={"symbol": "AAPL", "size": 100},
    )
    assert response.status_code == 400
    assert response.json["error"] == "validation_error"
    assert "entry_price" in response.json["message"]


def test_create_position_invalid_size(client):
    """Test creating position with non-positive size returns validation error."""
    response = client.post(
        "/api/v1/positions",
        json={"symbol": "AAPL", "size": 0, "entry_price": 150.0},
    )
    assert response.status_code == 400
    assert response.json["error"] == "validation_error"


def test_create_position_invalid_side(client):
    """Test creating position with invalid side returns validation error."""
    response = client.post(
        "/api/v1/positions",
        json={"symbol": "AAPL", "size": 100, "entry_price": 150.0, "side": "invalid"},
    )
    assert response.status_code == 400
    assert response.json["error"] == "validation_error"
    assert "side" in response.json["message"].lower()


def test_get_position(client):
    """Test getting a position by ID."""
    create_response = client.post(
        "/api/v1/positions",
        json={"symbol": "MSFT", "size": 50, "entry_price": 380.0},
    )
    position_id = create_response.json["id"]

    response = client.get(f"/api/v1/positions/{position_id}")
    assert response.status_code == 200
    assert response.json["id"] == position_id
    assert response.json["symbol"] == "MSFT"


def test_get_position_not_found(client):
    """Test getting non-existent position returns 404."""
    response = client.get("/api/v1/positions/nonexistent")
    assert response.status_code == 404
    assert response.json["error"] == "not_found"
    assert "nonexistent" in response.json["message"]


def test_update_position_size(client):
    """Test updating position size."""
    create_response = client.post(
        "/api/v1/positions",
        json={"symbol": "AMZN", "size": 100, "entry_price": 180.0},
    )
    position_id = create_response.json["id"]

    response = client.patch(
        f"/api/v1/positions/{position_id}",
        json={"size": 150},
    )
    assert response.status_code == 200
    assert response.json["size"] == 150


def test_update_position_status(client):
    """Test updating position status."""
    create_response = client.post(
        "/api/v1/positions",
        json={"symbol": "META", "size": 75, "entry_price": 500.0},
    )
    position_id = create_response.json["id"]

    response = client.patch(
        f"/api/v1/positions/{position_id}",
        json={"status": "closed", "exit_price": 550.0},
    )
    assert response.status_code == 200
    assert response.json["status"] == "closed"
    assert response.json["exit_price"] == 550.0


def test_update_position_notes(client):
    """Test updating position notes."""
    create_response = client.post(
        "/api/v1/positions",
        json={"symbol": "NFLX", "size": 20, "entry_price": 600.0},
    )
    position_id = create_response.json["id"]

    response = client.patch(
        f"/api/v1/positions/{position_id}",
        json={"notes": "Earnings play"},
    )
    assert response.status_code == 200
    assert response.json["notes"] == "Earnings play"


def test_update_position_not_found(client):
    """Test updating non-existent position returns 404."""
    response = client.patch(
        "/api/v1/positions/nonexistent",
        json={"size": 100},
    )
    assert response.status_code == 404
    assert response.json["error"] == "not_found"


def test_update_position_invalid_status(client):
    """Test updating position with invalid status returns validation error."""
    create_response = client.post(
        "/api/v1/positions",
        json={"symbol": "AMD", "size": 200, "entry_price": 120.0},
    )
    position_id = create_response.json["id"]

    response = client.patch(
        f"/api/v1/positions/{position_id}",
        json={"status": "invalid"},
    )
    assert response.status_code == 400
    assert response.json["error"] == "validation_error"


def test_delete_position(client):
    """Test deleting a position."""
    create_response = client.post(
        "/api/v1/positions",
        json={"symbol": "INTC", "size": 500, "entry_price": 30.0},
    )
    position_id = create_response.json["id"]

    response = client.delete(f"/api/v1/positions/{position_id}")
    assert response.status_code == 204
    assert response.data == b""

    # Verify it's gone
    response = client.get(f"/api/v1/positions/{position_id}")
    assert response.status_code == 404


def test_delete_position_not_found(client):
    """Test deleting non-existent position returns 404."""
    response = client.delete("/api/v1/positions/nonexistent")
    assert response.status_code == 404
    assert response.json["error"] == "not_found"


def test_list_positions_with_items(client):
    """Test listing positions with multiple items."""
    client.post(
        "/api/v1/positions",
        json={"symbol": "AAPL", "size": 100, "entry_price": 150.0},
    )
    client.post(
        "/api/v1/positions",
        json={"symbol": "GOOGL", "size": 50, "entry_price": 140.0},
    )
    client.post(
        "/api/v1/positions",
        json={"symbol": "MSFT", "size": 75, "entry_price": 380.0},
    )

    response = client.get("/api/v1/positions")
    assert response.status_code == 200
    assert len(response.json) == 3
    symbols = [p["symbol"] for p in response.json]
    assert "AAPL" in symbols
    assert "GOOGL" in symbols
    assert "MSFT" in symbols
