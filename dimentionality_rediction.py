import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



class Pca_reduction():
    
    def __init__(self, features, label):
        self.features = features
        self.label  = label
        self.unique_labels = label.unique()

    def scale_data(self, features):
        scale =  StandardScaler()
        features_scaled = scale.fit_transform(self.features)
        return features_scaled

    def transform_data(self, features):
        feature_cov = np.cov(self.features.T)
        return feature_cov

    def calc_eigen_vec_val(self, features):
        eig_val, eig_vec = np.linalg.eig(features)
        return eig_val, eig_vec

    def pair_eig_val_vec(self, eig_val, eig_vec):
        return [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]

    def display_variance(self, eig_val):
        print(eig_val)
        variance=[x/np.sum(eig_val) for x in eig_val]
        count = 0
        cum_sum = 0
        for var in variance:
            count = count +1
            cum_sum += var
            if cum_sum > .90:
                break

        return variance , count

class Lda_reduction():
    
    def mean_vecs(self):
        mean_vecs = []
        for label in unique_labels:
            mean_vecs.append(np.mean(features[label == label]))
        return mean_vecs


    #return scatter matrix within class.
    def scatter_mat_wc(self, features, mean_vecs,labels, unique_labels):
        swc = np.zeroes((features.shape[1], features.shape[1]))

        for label, mean_vecs in zip(labels, mean_vecs):
            class_matrix = np.zeroes((features.shape[1], features.shape[1]))
            for row in features[labels==label]:
                row = row.reshape(features.shape[1], 1)
                mean_vecs = mean_vecs.reshape(features.shape[1], 1)
                class_matrix = np.dot((row - mean_vecs), (row - mean_vecs).T)
            swc += class_matrix
        return swc.shape
        return swc

    def scatter_mat_btc(self, features, mean_vecs,labels, unique_labels):
        sbc = np.zeroes((features.shape[1], features.shape[1]))
        over_mean = np.mean(features)
        for mean_vec, label in zip(mean_vecs, unique_labels):
            n = features[labels == label].shape[0]
            mean_vec = mean_vec.reshape(features.shape[0], 1)
            over_mean = mean_vec.reshape(features.shape[0], 1)
            sbc = n * np.dot((mean_vec - over_mean) , (mean_vec - over_meani).T)
        print(sbc.shape)
        return sbc



            
        


    
