import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

class RandomForestSelector:
    def __init__(self,X,Y,trees=11):
        self.X=X
        self.Y=Y
        self.trees=trees
    def fit(self):
        dtree = RandomForestClassifier(self.trees,random_state=0)
        dtree.fit(self.X, self.Y)
        self.importances = dtree.feature_importances_  # importance value of features, with higher score indicating higher importance
        self.indices = np.argsort(self.importances)[::-1]
        self.std = np.std([tree.feature_importances_ for tree in dtree.estimators_], axis=0)
        return self.indices
    def plot(self, figsize):
        plt.figure(figsize=figsize)
        plt.title("Feature importances")
        plt.bar(range(self.X.shape[1]), self.importances[self.indices],
            color="r", yerr=self.std[self.indices], align="center")
        plt.xticks(range(self.X.shape[1]), self.indices)
        plt.xlim([-1, self.X.shape[1]])
        plt.show()
