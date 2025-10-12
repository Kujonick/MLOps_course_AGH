from fastapi import FastAPI

from src.api.models.iris import PredictRequest, PredictResponse
from src.inference import load_model, predict
from src.training import load_data

app = FastAPI()
model, scaler = load_model()
data = load_data()


@app.get("/")
def welcome_root():
    return {"message": "Welcome to the ML API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_request(request: PredictRequest):
    prediction = predict(model, scaler, data, request.model_dump())
    return PredictResponse(prediction=prediction)
