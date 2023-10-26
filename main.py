from pathlib import Path

from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, field_validator, ValidationInfo

from ml.data import process_data
from ml.model import inference, load_model

categorical_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

valid_categories = {
    "education": [
        "Bachelors",
        "HS-grad",
        "11th",
        "Masters",
        "9th",
        "Some-college",
        "Assoc-acdm",
        "Assoc-voc",
        "7th-8th",
        "Doctorate",
        "Prof-school",
        "5th-6th",
        "10th",
        "1st-4th",
        "Preschool",
        "12th",
    ],
    "marital_status": [
        "Never-married",
        "Married-civ-spouse",
        "Divorced",
        "Married-spouse-absent",
        "Separated",
        "Married-AF-spouse",
        "Widowed",
    ],
    "relationship": [
        "Not-in-family",
        "Husband",
        "Wife",
        "Own-child",
        "Unmarried",
        "Other-relative",
    ],
    "race": ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
    "sex": ["Male", "Female"],
}

valid_numerical_ranges = {
    "age": (0, 130),
    "education_num": (0, 16),
    "hours_per_week": (0, 120),
}


class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }

    @field_validator(*valid_categories.keys())
    @classmethod
    def validate_categorical(cls, v: str, info: ValidationInfo):
        if v not in valid_categories[info.field_name]:
            raise HTTPException(
                status_code=444,
                detail=f"Received invalid field: {info.field_name}={v}. \
{info.field_name} must be one of {valid_categories[info.field_name]}",
            )
        return v

    @field_validator(*valid_numerical_ranges.keys())
    @classmethod
    def validate_numerical_ranges(cls, v: str, info: ValidationInfo):
        min_, max_ = valid_numerical_ranges[info.field_name]
        if not (min_ <= v <= max_):
            raise HTTPException(
                status_code=444,
                detail=f"Received invalid field: {info.field_name}={v}. \
{info.field_name} must be in the range [{min_}, {max_}]",
            )
        return v


app = FastAPI()


@app.get("/")
async def hello():
    return "Welcome to the US Census Salary Prediction API"


# Load model and encoder once when the app starts up (rather than every time
# inside the predict() function)
model_path = Path(__file__).parent / "model" / "model.pkl"
encoder_path = Path(__file__).parent / "model" / "encoder.pkl"

model = load_model(model_path)
encoder = load_model(encoder_path)


@app.post("/predict")
async def predict(data: Data):
    df = pd.DataFrame(dict(data), index=[0])
    X, _, _, _ = process_data(
        df, categorical_features=categorical_features, encoder=encoder, training=False
    )
    pred = inference(model, X)
    return {"prediction": int(pred[0]), "data": data}
