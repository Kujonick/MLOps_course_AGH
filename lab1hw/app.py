from fastapi import FastAPI
from src.models import PredictRequest, PredictResponse
from src.inference import predict, load_models

app = FastAPI()

model, classifier = load_models()
responses = ["negative", "neutral", "positive"]


@app.post("/predict")
def predict_request(request: PredictRequest):
    output = predict(model, classifier, request.text)
    return PredictResponse(prediction=responses[output])
