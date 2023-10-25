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


@pytest.mark.parametrize(
    "key,value,err",
    [
        (
            "education",
            "abc",
            "Received invalid field: education=abc. education must be one of \
['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', \
'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th', \
'Preschool', '12th']",
        ),
        (
            "marital_status",
            "abc",
            "Received invalid field: marital_status=abc. marital_status must be one \
of ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', \
'Separated', 'Married-AF-spouse', 'Widowed']",
        ),
        (
            "relationship",
            "abc",
            "Received invalid field: relationship=abc. relationship must be one of \
['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative']",
        ),
        (
            "race",
            "abc",
            "Received invalid field: race=abc. race must be one of ['White', 'Black', \
'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']",
        ),
        (
            "sex",
            "abc",
            "Received invalid field: sex=abc. sex must be one of ['Male', 'Female']",
        ),
        (
            "age",
            -1,
            "Received invalid field: age=-1. age must be in the range [0, 130]",
        ),
        (
            "age",
            131,
            "Received invalid field: age=131. age must be in the range [0, 130]",
        ),
        (
            "education_num",
            -1,
            "Received invalid field: education_num=-1. education_num must be in the \
range [0, 16]",
        ),
        (
            "education_num",
            17,
            "Received invalid field: education_num=17. education_num must be in the \
range [0, 16]",
        ),
        (
            "hours_per_week",
            -1,
            "Received invalid field: hours_per_week=-1. hours_per_week must be in the \
range [0, 120]",
        ),
        (
            "hours_per_week",
            121,
            "Received invalid field: hours_per_week=121. hours_per_week must be in the \
range [0, 120]",
        ),
    ],
)
def test_post_predict_invalid_values(key, value, err):
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
    assert r.status_code == 444 and r.json()["detail"] == err
