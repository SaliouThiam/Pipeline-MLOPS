from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
import os

# Définir un modèle pour les données d'entrée
class FeaturesInput(BaseModel):
    features: list[float]

app = FastAPI()
model = joblib.load("model.joblib")

@app.get("/")
def root():
    return {"message": "Modèle prêt à faire des prédictions."}

@app.post("/predict")
def predict(data: FeaturesInput):
    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
