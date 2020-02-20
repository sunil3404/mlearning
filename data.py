from sklearn.datasets import *
from sklearn.preprocessing import StandardScaler
import numpy as np

class Data():
    
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y 

    def scale_data(self, X):
        return StandardScaler().fit_transform(X)
    
    def train_test_data(self, X, n_splits):
        size_X = len(X)
