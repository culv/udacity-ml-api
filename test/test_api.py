from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_api_locally_get_root():
    r = client.get("/")
    assert (
        r.status_code == 200
        and r.json() == "Welcome to the US Census Salary Prediction API"
    )
