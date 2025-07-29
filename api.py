from fastapi import FastAPI
import joblib
import numpy as np
import uvicorn
import os

app = FastAPI()
model = joblib.load("model.joblib")

@app.get("/")
def root():
    return {"message": "Modèle prêt à faire des prédictions."}

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render fournit le port via variable d'env
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
