import joblib
import os
import pandas as pd
import numpy as np
from typing import Tuple

class ModelManager:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.payoffs = np.array([-2, -1, 0, 1, 2]) # Large Down, Medium Down, Flat, Medium Up, Large Up

    def load_model(self):
        """Load the trained model from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        self.model = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}")

    def save_model(self, model):
        """Save a trained model to disk."""
        joblib.dump(model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict probabilities and calculate EV.
        Returns: (probabilities, ev_signal)
        """
        if self.model is None:
            self.load_model()
            
        # Predict probabilities
        # Shape: (n_samples, 5)
        probs = self.model.predict_proba(X)
        
        # Calculate EV
        # EV = sum(prob * payoff)
        ev_signal = np.dot(probs, self.payoffs)
        
        return probs, ev_signal
