"""Module for linear quantile regression model."""
import numpy as np
from numpy import ndarray
from sklearn.linear_model import QuantileRegressor
from joblib import Parallel, delayed
from tqdm import tqdm


class LinearQuantileRegression:
    """Linear quantile regression model for multiple quantiles."""

    def __init__(self, quantiles) -> None:
        self.quantiles = quantiles
        self.models = {quantile: QuantileRegressor(
            quantile=quantile, alpha=0.1, solver="highs") for quantile in quantiles}

    def _fit_quantile_model(self, model, X, y, quantile) -> tuple[float, QuantileRegressor]:
        """
        Fit a single quantile regression model.
        """
        model.fit(X, y)
        return quantile, model

    def train(self, X, y, n_jobs=4) -> None:
        """
        Train the quantile regression models in parallel with progress tracking.

        Parameters:
        X (ndarray): The feature matrix (shape: n_samples, n_features)
        y (ndarray): The target values (shape: n_samples,)
        n_jobs (int): The number of parallel jobs to run (-1 means using all processors)
        """
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._fit_quantile_model)(model, X, y, quantile)
            for quantile, model in tqdm(self.models.items(), desc="Training quantile models", total=len(self.models))
        )
        self.models = dict(results)  # type: ignore

    def predict(self, X) -> ndarray:
        """
        Predict the target values for each quantile using the trained models.

        Parameters:
        X (ndarray): The feature matrix for prediction (shape: n_samples, n_features)

        Returns:
        dict: A dictionary where the keys are quantiles and the values are the predictions.
        """
        predictions = [model.predict(X) for model in self.models.values()]  # type: ignore
        return np.array(predictions).T

    def __call__(self, X) -> ndarray:
        return self.predict(X)

    def eval(self) -> None:
        """To mimic functionally of the eval method in PyTorch models."""
        pass
