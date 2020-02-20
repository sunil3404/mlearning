import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dimentionality_rediction import Pca_reduction
from sklearn.metrics import confusion_matrix
from sklearn import tree


class Decision_Tree():
    
    def train_model(self, train_f, train_l):
        model = tree.DecisionTreeClassifier()
        return model.fit(train_f,  train_l) 
    
    def test_model(self, model, test_f):
        predicted = model.predict(test_f)
        return predicted

if __name__ == "__main__":
    ir_d = load_iris()
    features = ir_d.data
    labels = ir_d.target
    pc = Pca_reduction(features, pd.Series(labels))
    scaled_features = pc.scale_data(features)
    features_cov = pc.transform_data(scaled_features)
    eig_val,  eig_vec = pc.calc_eigen_vec_val(features_cov)
    eig_pairs = pc.pair_eig_val_vec(eig_val, eig_vec)
    eig_pairs =  sorted(eig_pairs, key=lambda k:k[0], reverse=True) 
    w = np.hstack((eig_pairs[0][1][:, np.newaxis].real,
               eig_pairs[1][1][:, np.newaxis].real))
    dc = Decision_Tree()
    
    print("+++++++ Output without PCA ++++++++++++++++")
    train_f, test_f, train_l, test_l = train_test_split(scaled_features, labels, test_size = 0.25, random_state=42)
    model = dc.train_model(train_f, train_l)
    predicted_no_pca = dc.test_model(model,test_f)
    conf_mat_nopca = confusion_matrix(predicted_no_pca, test_l)
    print(conf_mat_nopca)
    print("\n")
    print("+++++++ Output with PCA ++++++++++++++++")
    features = np.dot(scaled_features, w)
    train_f, test_f, train_l, test_l = train_test_split(features, labels, test_size = 0.25, random_state=42)
    model = dc.train_model(train_f, train_l)
    predictions  = dc.test_model(model, test_f)
    conf_mat = confusion_matrix(predictions, test_l)
    print(conf_mat)



