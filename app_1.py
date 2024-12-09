from fastapi import FastAPI, File, UploadFile
from io import StringIO
import pandas as pd
from joblib import load
import uvicorn 

app = FastAPI() 

@app.get("/health")
def read_root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    classifier = load("linear_regression.joblib")
    

    df = pd.read_csv(StringIO((await file.read()).decode('utf-8')))
    model = load('model.joblib')
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}


