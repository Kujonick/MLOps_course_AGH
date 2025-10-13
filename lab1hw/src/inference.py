from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib
from typing import Tuple


def load_models() -> Tuple[SentenceTransformer, LogisticRegression]:
    model = SentenceTransformer("resources/sentence_transformer.model")
    classifier: LogisticRegression = joblib.load("resources/classifier.joblib")
    return model, classifier


def predict(model: SentenceTransformer, classifier: LogisticRegression, text: str):
    embedding = model.encode(text).reshape(1, -1)
    output = classifier.predict(embedding)[0]
    return output
