from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint_shape():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_rejects_empty_records():
    response = client.post("/predict", json={"records": []})
    assert response.status_code == 422
