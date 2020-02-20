import numpy as np
from decisiontree import DecisionTree
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def bootstrap_sample(X,y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size = n_samples, replace=True)
    return X[idxs], y[idxs]

def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


class RandomForest:


    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []


    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, 
                    max_depth = self.max_depth, n_feats=self.n_feats)
            x_sample, y_sample = bootstrap_sample(X,y)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)    

if __name__ == "__main__":
    bc_data = load_breast_cancer()
    bc_x = bc_data.data
    bc_y = bc_data.target


    train_x, test_x, train_y, test_y = train_test_split(bc_x, bc_y, test_size = 0.2, random_state=23)

    dc_tree = RandomForest(n_trees=3)
    dc_tree.fit(train_x, train_y)
    y_pred = dc_tree.predict(test_x)

    accur = np.sum(test_y == y_pred) / len(test_y)
    print(accur)
