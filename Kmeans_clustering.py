import pandas as pd
import numpy as np
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class KMEANS_cluster():
    
    def scale_data(self, X):
        return StandardScaler().fit_transform(X)
    
    def split_data(self,X, Y):
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=34)
        return train_x, test_x, train_y, test_y
     
    def train_model(self,X):
        model =  KMeans(n_clusters=3)
        model.fit(X)
        return model

    def test_model(self, X, model):
        return model.predict(X)

if __name__ == "__main__":
    
    km = KMEANS_cluster()
    iris_d = load_iris()

    X = iris_d.data
    Y = iris_d.target

    plt.plot([1,2,3], [3,4,5])
    scaled_X = km.scale_data(X)
    train_x, test_x, train_y, test_y = km.split_data(scaled_X, Y)

    model =  km.train_model(train_x)
    predicted = km.test_model(test_x, model)
    df = pd.DataFrame(pd.Series(predicted),  columns=["predicted"])
    df['actual'] = pd.Series(test_y)
    print(df)
    #plt.scatter(df['predicted'].reshape([-1,1]), df['actual'].reshape([-1,1]))
    print(model.score(np.array(df['predicted']).reshape([-1,1]), np.array(df['actual'])).reshape([-1,1]))
