from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODEL_DIR = "./resources"
MODEL_PATH = os.path.join(MODEL_DIR, "model")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler")


def load_data(return_X_y=False, as_frame=False):
    return load_iris(return_X_y=return_X_y, as_frame=as_frame)


def train_model(model: LogisticRegression, scaler: StandardScaler, X, y) -> None:
    X = scaler.fit_transform(X, y)
    model.fit(X, y)


def save_model(model, scaler) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
