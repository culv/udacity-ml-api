from fastapi.testclient import TestClient
import pytest

from main import app

client = TestClient(app)


def test_get_root():
    r = client.get("/")
    assert (
        r.status_code == 200
        and r.json() == "Welcome to the US Census Salary Prediction API"
    )


def test_post_predict_valid_datatypes():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }
    r = client.post("/predict/", json=data)
    assert r.status_code == 200


@pytest.mark.parametrize(
    "key,value",
    [
        ("age", "not int"),
        ("workclass", 1),
        ("fnlgt", "not int"),
        ("education", 1),
        ("education_num", "not int"),
        ("marital_status", 1),
        ("occupation", 1),
        ("relationship", 1),
        ("race", 1),
        ("sex", 1),
        ("capital_gain", "not int"),
        ("capital_loss", "not int"),
        ("hours_per_week", "not int"),
        ("native_country", 1),
    ],
)
def test_post_predict_invalid_datatypes(key, value):
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }
    data[key] = value
    r = client.post("/predict/", json=data)
    assert r.status_code == 422
