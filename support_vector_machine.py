from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from data import Data
import pandas as pd
import numpy as np

class Support_vector_machine():
    
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def scale_data(self):
        return StandardScaler().fit_transform(self.X)

    def split_data(self, scaled_X):
        train_x, test_x, train_y, test_y = train_test_split(scaled_X, self.Y, test_size=0.2 , random_state=234)
        return train_x, test_x, train_y, test_y

    def train_model(self, train_x, train_y):
        model =  SVC()
        model.fit(train_x, train_y)
        return model

    def test_model(self, test_x, model):
        return model.predict(test_x)

    
if __name__ == "__main__":
    iris = load_breast_cancer()
    X = iris.data
    Y = iris.target
    sv = Support_vector_machine(X, Y)
    scaled_X = sv.scale_data()
    train_x, test_x, train_y, test_y = sv.split_data(scaled_X)
    model = sv.train_model(train_x, train_y)
    predicted = sv.test_model(test_x, model)
    confu_mat = confusion_matrix(predicted, test_y)
    print(confu_mat)
