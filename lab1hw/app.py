from fastapi import FastAPI
from src.models import PredictRequest, PredictResponse

app = FastAPI()


@app.post("/predict")
def predict_request(request: PredictRequest):
    return PredictResponse(prediction="mock answer")
