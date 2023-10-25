from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator


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

    @field_validator("education")
    @classmethod
    def validate_education(cls, edu: str):
        valid_educations = ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm',
 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th',
 '1st-4th', 'Preschool', '12th']
        if edu not in valid_educations:
            raise HTTPException(status_code=444, detail=f"Received invalid field: education={edu}. education must be one of {valid_educations}")


app = FastAPI()


@app.get("/")
async def hello():
    return "Welcome to the US Census Salary Prediction API"


@app.post("/predict")
async def predict(data: Data):
    return data
