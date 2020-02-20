import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Perceptron:

    
    
    def __init__(self, learning_rate=0.001, n_iters=1000):
        
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None
    
    def accuracy(self, predictions, y):
        return np.sum(y == predictions) / len(y)

    def data(self):
        bc_data = load_breast_cancer()
        bc_x = bc_data.data
        bc_y = bc_data.target
        
        train_x, test_x, train_y, test_y = train_test_split(bc_x, bc_y, test_size=0.2, random_state=20)
        return train_x, test_x, train_y, test_y

    def fit(self, X, y):
        n_samples, n_features = X.shape
        #initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        Y_ = np.array([1 if i> 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for id_x, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predict = self.activation_func(linear_output)


                #update
                update = self.learning_rate * (Y_[id_x] - y_predict)
                self.weights += update * x_i
                self.bias += update
                 
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)

if __name__ == "__main__":
    pc = Perceptron(learning_rate=0.001, n_iters=1000)
    train_x, test_x, train_y, test_y = pc.data()
    pc.fit(train_x, train_y)
    predictions = pc.predict(test_x)
    accur = pc.accuracy(test_y, predictions)
    print(accur)
