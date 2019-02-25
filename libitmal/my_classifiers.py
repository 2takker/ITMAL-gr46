from sklearn.base import BaseEstimator
import numpy as np

class DummyClassifier(BaseEstimator):
    def fit(self, X, y = None):
        pass
    def predict(self, X):
        return np.zeros((len(X), ), dtype = bool)
