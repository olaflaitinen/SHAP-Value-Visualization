"""
shap_visualizer.py

Module for computing and visualizing SHAP values for a decision tree model.
"""

import shap
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPVisualizer:
    """
    Class for computing and visualizing SHAP values.
    """

    def __init__(self, model, data, feature_names):
        """
        Initialize the SHAPVisualizer.

        Args:
            model: Trained decision tree model.
            data (pd.DataFrame or array-like): Background data for SHAP.
            feature_names (list): List of feature names.
        """
        self.model = model
        self.data = data
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        logger.info("Initialized SHAP TreeExplainer.")

    def compute_shap_values(self):
        """
        Compute SHAP values for the provided data.

        Returns:
            shap_values: SHAP values computed for the data.
        """
        shap_values = self.explainer.shap_values(self.data)
        logger.info("Computed SHAP values.")
        return shap_values

    def plot_summary(self, shap_values, output_file="shap_summary.png"):
        """
        Generate and save a SHAP summary plot.

        Args:
            shap_values: Computed SHAP values.
            output_file (str): File path to save the plot.
        """
        plt.figure()
        # The summary_plot function creates its own figure; we suppress immediate display
        shap.summary_plot(shap_values, self.data, feature_names=self.feature_names, show=False)
        plt.savefig(output_file)
        plt.close()
        logger.info(f"SHAP summary plot saved to {output_file}")

    def plot_dependency(self, shap_values, feature, output_file="shap_dependency.png"):
        """
        Generate and save a SHAP dependency plot for a specific feature.

        Args:
            shap_values: Computed SHAP values.
            feature (str or int): Feature name or index for the dependency plot.
            output_file (str): File path to save the plot.
        """
        plt.figure()
        shap.dependence_plot(feature, shap_values, self.data, feature_names=self.feature_names, show=False)
        plt.savefig(output_file)
        plt.close()
        logger.info(f"SHAP dependency plot for feature '{feature}' saved to {output_file}")
