import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import StandardScaler 
from dimentionality_rediction import Pca_reduction, Lda_reduction
import sys
class Logistic():
    

    def data_preprocessing(self, bc_data,labels, column_names):
        df_x = pd.DataFrame(bc_data, columns = column_names)
        df_y = labels 
        return df_x, df_y

    def scale_data(self, df_x):
        scaler = StandardScaler()
        df_x  = scaler.fit_transform(df_x)
        return df_x
    
    def train_model(self, train_x, train_y):
        model = LogisticRegression()
        model.fit(train_x, train_y)
        return model

    def test_model(self, test_x, model):
        predict = model.predict(test_x)
        return predict
    
    '''
    using numpy and sigmoid function
    number of hidden layers = 1
    number of neurons in hidden_layer = 100
    number of neurons in output layer = 1
    
    parameters
    -----------
    w = np.random.rand(train_x.shape[1], 1)
    b = 1
    alpha = 0.001

    '''

class Logistic_manual():

    def __init__(self, w, b, learning_rate):
        self.w = w
        self.b = b
        self.learning_rate = learning_rate
    
    def forward_pass(self, w, b, train_x):
        z = np.dot(train_x, self.w) + self.b
        return z

    def cost_fucntion(self, train_y, y_pred):
        #return -np.mean(np.sum(train_y*np.log(y_pred)+(1-train_y)*np.log(1- y_pred)))
        return -np.mean(np.sum(train_y*np.log(y_pred)+(1-train_y)*np.log(1- y_pred)))

    def output(self, y_pred):
        return 1/(1 + np.exp(-y_pred))

    def backward_pass(self, dw, db, w, b, learning_rate):
        w = w - learning_rate * dw
        b = b - learning_rate * db

        return w, b

    def diff_weights_bias(self, w, b, train_x ,train_y, y_pred):
        error = y_pred - train_y
        dw = (1/train_y.shape[0]) * np.dot(train_x.T, error)
        db = (1/train_y.shape[0]) * np.sum(error)
        return dw, db

    def train_model(self, train_x, train_y, w, b,learning_rate):
        costs = []
        for i in range(0, 100000):
            y_pred  = self.forward_pass(self.w, self.b, train_x)
            p       = self.output(y_pred)
            cost    = self.cost_fucntion(train_y, p)
            dw, db  = self.diff_weights_bias(self.w, self.b, train_x, train_y, p)
            self.w,  self.b    = self.backward_pass(dw, db, self.w, self.b, learning_rate)
            if i % 5000 == 0:
                costs.append(cost)
        print(costs)
        return self.w, self.b, costs
    def test_model(self, w, b, test_x):
        predicts = []
        y_pred= np.dot(test_x, w) +b
        activation = self.output(y_pred)
        for i in range(0, len(activation)): 
            if activation[i] > 0.5:
                predicts.append(1)
            else:
                predicts.append(0)
        return predicts

if __name__ == "__main__":
    logic = Logistic()
    bc_data = load_breast_cancer()
    df_x, df_y = logic.data_preprocessing(bc_data.data,bc_data.target,  bc_data.feature_names)
    scaled_x = logic.scale_data(df_x)
    labels = bc_data.target
     
    train_x, test_x, train_y, test_y = train_test_split(scaled_x,df_y, test_size=0.3, random_state=42)
    print(test_y.shape)
    model = logic.train_model(train_x, train_y)
    predict = logic.test_model(test_x, model)
    conf_mat = confusion_matrix(test_y, predict)
    print("Confusion Matrix for log model through sklearn")
    print("==============================================")
    print(conf_mat)
    
    '''
    logistic regression using numpy

    '''
    np.random.seed(42)
    train_y =  np.array(train_y).reshape([-1,1])
    logic_manual = Logistic_manual(w = np.random.randn(train_x.shape[1], 1), b = 1, learning_rate=0.01)
    w, b, costs =  logic_manual.train_model(train_x, train_y, logic_manual.w, logic_manual.b, logic_manual.learning_rate)
    predict_manual = logic_manual.test_model(w, b , test_x)
    conf_mat_manual = confusion_matrix(test_y, predict_manual)
    print("costs from manual logistic regression at epochs of 5000")
    print("==============================================")
    print(costs)
    
    print("=============================================")
    print("\n")
    print("Confusion Matrix for log model through numpy")
    print("==============================================")
    print(conf_mat_manual)


    


    




    
