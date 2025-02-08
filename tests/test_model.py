"""
test_model.py

Unit tests for the decision tree model.
"""

import pytest
from src.model import DecisionTreeModel
from src.data import load_data


@pytest.fixture(scope="module")
def model():
    dt_model = DecisionTreeModel()
    dt_model.train()
    return dt_model


def test_model_accuracy(model):
    X, y = load_data()
    predictions = model.predict(X)
    accuracy = (predictions == y).mean()
    # Expect at least 70% accuracy on the full dataset
    assert accuracy > 0.7, "Model accuracy should be greater than 70%"


def test_model_save_load(tmp_path, model):
    model_path = tmp_path / "dt_model.joblib"
    model.save(str(model_path))
    new_model = DecisionTreeModel()
    new_model.load(str(model_path))
    X, _ = load_data()
    # Ensure that predictions are similar after loading the model
    assert (model.predict(X) == new_model.predict(X)).all(), "Loaded model predictions differ from the saved model."
