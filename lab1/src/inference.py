from typing import Tuple
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.api.models.iris import PredictRequest
from src.training import MODEL_PATH, SCALER_PATH


def load_model() -> Tuple[LogisticRegression, StandardScaler]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("model does not exist")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("scaler does not exist")
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def predict(
    model: LogisticRegression, scaler: StandardScaler, data: dict, input: dict
) -> str:
    column_order = PredictRequest.model_fields.keys()
    input = [[input[c] for c in column_order]]
    input = scaler.transform(input)
    output = model.predict(input)[0]
    return data["target_names"][output]
