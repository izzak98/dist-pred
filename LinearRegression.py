import numpy as np
from sklearn.linear_model import QuantileRegressor
from joblib import Parallel, delayed
from tqdm import tqdm

class LinearQuantileRegression:
    def __init__(self, quantiles):
        """
        Initialize the Quantile Regression class with a list of quantiles.
        
        Parameters:
        quantiles (list): List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
        """
        self.quantiles = quantiles
        self.models = {quantile: QuantileRegressor(quantile=quantile, alpha=0.1, solver="highs") for quantile in quantiles}
        
    def _fit_quantile_model(self, model, X, y, quantile):
        """
        Fit a single quantile regression model.
        """
        model.fit(X, y)
        return quantile, model
    
    def train(self, X, y, n_jobs=4):
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
        self.models = dict(results)  # Update models with trained instances
    
    def predict(self, X):
        """
        Predict the target values for each quantile using the trained models.
        
        Parameters:
        X (ndarray): The feature matrix for prediction (shape: n_samples, n_features)
        
        Returns:
        dict: A dictionary where the keys are quantiles and the values are the predictions.
        """
        predictions = [model.predict(X) for model in self.models.values()]
        return np.array(predictions).T

    def __call__(self, X):
        return self.predict(X)

    def eval(self):
        pass