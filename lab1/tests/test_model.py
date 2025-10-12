from src.training import load_data, train_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def test_loading_data():
    dataset = load_data()
    assert "data" in dataset
    assert "target" in dataset
    assert "target_names" in dataset


def test_training_model():
    dataset = load_data()
    X, y = dataset["data"], dataset["target"]
    model = LogisticRegression()
    scaler = StandardScaler()
    train_model(model, scaler, X, y)

    assert model.predict(X[:1]) in range(3)
