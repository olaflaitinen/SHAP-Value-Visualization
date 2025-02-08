"""
test_shap_visualizer.py

Unit tests for the SHAP visualizer functionality.
"""

import pytest
import pandas as pd
from src.data import load_data
from src.model import DecisionTreeModel
from src.shap_visualizer import SHAPVisualizer


@pytest.fixture(scope="module")
def trained_model():
    model = DecisionTreeModel()
    model.train()
    return model


@pytest.fixture(scope="module")
def background_data():
    X, _ = load_data()
    return X


@pytest.fixture(scope="module")
def shap_visualizer(trained_model, background_data):
    feature_names = list(background_data.columns)
    return SHAPVisualizer(model=trained_model.model, data=background_data, feature_names=feature_names)


def test_compute_shap_values(shap_visualizer, background_data):
    shap_values = shap_visualizer.compute_shap_values()
    # For a classification tree, shap_values is typically a list of arrays (one per class)
    # Here we check that we have at least one set of SHAP values and that dimensions match
    assert isinstance(shap_values, list) and len(shap_values) > 0, "SHAP values should be a non-empty list."
    for sv in shap_values:
        assert sv.shape[0] == background_data.shape[0], "SHAP values row count should match data row count."
