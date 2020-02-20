import numpy as np



class Utils:
    
    def euclidean_distances(self, x1, x2):
        return nq.sqrt(np.sum((x1 -x2)**2))



class Cost_Functions:
    

    def sigmoid(self, x):
        return 1/ (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - sigmoid(x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)


