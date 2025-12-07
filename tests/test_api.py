"""Tests for Flask API."""

import pytest

from cents.api import create_app


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
    assert "error" in response.json
