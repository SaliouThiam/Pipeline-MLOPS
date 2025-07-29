from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.joblib")

@app.get("/")
def root():
    return {"message": "Modèle prêt à faire des prédictions."}

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}
