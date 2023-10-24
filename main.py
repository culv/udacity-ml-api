from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def hello():
    return "Welcome to the US Census Salary Prediction API"


@app.post("/predict")
async def predict():
    pass
