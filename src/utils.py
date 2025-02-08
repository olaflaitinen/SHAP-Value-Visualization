"""
utils.py

Utility functions for the SHAP Value Visualization project.
"""

import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_figure(fig, filename):
    """
    Save a matplotlib figure to a file.

    Args:
        fig: Matplotlib figure object.
        filename (str): Destination file path.
    """
    fig.savefig(filename)
    plt.close(fig)
    logger.info(f"Figure saved as {filename}")
