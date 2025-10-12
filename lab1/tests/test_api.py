from fastapi.testclient import TestClient

from app import app
from src.api.models.iris import PredictRequest
from src.training import load_data


def test_get():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json().get("message", "") == "Welcome to the ML API"


def test_get_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json().get("status", "") == "ok"


def test_predict():
    client = TestClient(app)
    dataset = load_data()
    data0 = dataset["data"][0]  # first value of dataset
    feature_names = PredictRequest.model_fields.keys()
    values = {name: value for name, value in zip(feature_names, data0)}

    response = client.post("/predict", json=values)
    target_names = dataset["target_names"]

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in target_names
