from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(
    title='Iris Classifier API',
    description='Predict Iris flower species from measurements',
    version='1.0.0'
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'iris_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

CLASS_NAMES = ['setosa', 'versicolor', 'virginica']

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionOutput(BaseModel):
    species: str
    species_id: int
    confidence: float

@app.get('/health')
def health_check():
    return {'status': 'healthy', 'model': 'iris_classifier_v1'}

@app.post('/predict', response_model=PredictionOutput)
def predict(input: IrisInput):
    try:
        features = np.array([[
            input.sepal_length, input.sepal_width,
            input.petal_length, input.petal_width
        ]])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[prediction])
        return PredictionOutput(
            species=CLASS_NAMES[prediction],
            species_id=int(prediction),
            confidence=round(confidence, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
def home():
    return {"message": "Iris API is running "}