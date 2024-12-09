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

    features_df = pd.read_csv('selected_features.csv')
    features = features_df['0'].to_list()

    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    df = df[features]
    
    prediction = classifier.predict(df)

    return {
        "predictions": prediction.tolist()
    }


