"""
run_visualization.py

Example script to train a decision tree model, compute SHAP values, and generate visualizations.
"""

import pandas as pd
from src.model import DecisionTreeModel
from src.data import load_data
from src.shap_visualizer import SHAPVisualizer

def main():
    # Load data and train the model
    X, y = load_data()
    model = DecisionTreeModel()
    metrics = model.train()
    print(f"Model training metrics: {metrics}")

    # Initialize SHAP visualizer with the trained model and background data
    feature_names = list(X.columns)
    visualizer = SHAPVisualizer(model=model.model, data=X, feature_names=feature_names)

    # Compute SHAP values
    shap_values = visualizer.compute_shap_values()

    # Generate and save a SHAP summary plot
    visualizer.plot_summary(shap_values, output_file='shap_summary.png')
    print("SHAP summary plot saved as shap_summary.png")

    # Generate and save a dependency plot for a selected feature (e.g., the first feature)
    visualizer.plot_dependency(shap_values, feature=feature_names[0], output_file='shap_dependency.png')
    print(f"SHAP dependency plot for feature '{feature_names[0]}' saved as shap_dependency.png")


if __name__ == "__main__":
    main()
