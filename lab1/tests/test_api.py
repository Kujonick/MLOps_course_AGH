from fastapi.testclient import TestClient

from app import app


def test_settings():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json().get("message", "") == "Welcome to the ML API"
