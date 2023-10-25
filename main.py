from fastapi import FastAPI
from pydantic import BaseModel


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


app = FastAPI()


@app.get("/")
async def hello():
    return "Welcome to the US Census Salary Prediction API"


@app.post("/predict")
async def predict(data: Data):
    return data
