from sklearn.datasets import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


class Random_Forest():
    
    def scale_data(self, X, scale_type):
        if scale_type == 'mx':
            return MinMaxScaler().fit_transform(X)
        else:
            return StandardScaler().fit_transform(X)

    def split_data(self, X, Y):
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=46)
        return train_x, test_x, train_y, test_y
        
    def train_model(self, X, Y):
        model = RandomForestClassifier()
        return model.fit(X,Y)

    def test_model(self, X, model):
        return model.predict(X)
        

if __name__ == "__main__":
    data_iris = load_iris()
    data_bc = load_breast_cancer()
    data_digits = load_digits()
    X_iris = data_iris.data
    Y_iris = data_iris.target

    X_bc = data_bc.data
    Y_bc = data_bc.target
    
    X_dig = data_bc.data
    Y_dig = data_bc.target

    rf = Random_Forest()

    scaled_iris = rf.scale_data(X_iris, "std")
    scaled_bc = rf.scale_data(X_bc, "std")
    scaled_dig = rf.scale_data(X_dig, "std")

    iris_train_x, iris_test_x, iris_train_y, iris_test_y = rf.split_data(X_iris, Y_iris)
    bc_train_x, bc_test_x, bc_train_y, bc_test_y = rf.split_data(X_bc, Y_bc)
    dig_train_x, dig_test_x, dig_train_y, dig_test_y = rf.split_data(X_dig, Y_bc)
    
    print("+++++++++++ Output on Iris data +++++++++++++++")
    model_iris = rf.train_model(iris_train_x, iris_train_y)
    predicted_iris = rf.test_model(iris_test_x, model_iris)
    print(confusion_matrix(predicted_iris, iris_test_y))
    
    print("+++++++++++ Output on breast_cancer data +++++++++++++++")
    model_bc = rf.train_model(bc_train_x, bc_train_y)
    predicted_bc = rf.test_model(bc_test_x, model_bc)
    print(confusion_matrix(predicted_bc, bc_test_y))
    
    print("+++++++++++ Output on digits data +++++++++++++++")
    model_dig = rf.train_model(dig_train_x, bc_train_y)
    predicted_dig = rf.test_model(dig_test_x, model_dig)
    print(confusion_matrix(predicted_dig, dig_test_y))




