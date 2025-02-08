"""
model.py

Module for training and using a decision tree classifier on the Iris dataset.
"""

import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .data import load_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecisionTreeModel:
    """
    A class to train, evaluate, save, and load a decision tree classifier.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = DecisionTreeClassifier(random_state=self.random_state)

    def train(self, test_size=0.2):
        """
        Train the decision tree model on the Iris dataset.

        Args:
            test_size (float): Fraction of data to be used as the test set.

        Returns:
            dict: Dictionary with training and testing accuracy.
        """
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        test_acc = accuracy_score(y_test, self.model.predict(X_test))
        logger.info(f"Training accuracy: {train_acc:.4f}, Testing accuracy: {test_acc:.4f}")
        return {"train_accuracy": train_acc, "test_accuracy": test_acc}

    def predict(self, X):
        """
        Predict target labels for a given feature set.

        Args:
            X (pd.DataFrame or array-like): Input features.

        Returns:
            array-like: Predicted labels.
        """
        return self.model.predict(X)

    def save(self, filepath='decision_tree_model.joblib'):
        """
        Save the trained model to disk.

        Args:
            filepath (str): Destination file path.
        """
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath='decision_tree_model.joblib'):
        """
        Load a model from disk.

        Args:
            filepath (str): Path to the saved model file.
        """
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
