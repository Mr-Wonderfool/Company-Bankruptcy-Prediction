import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

class RandomForestSelector:
    def __init__(self,X,Y,trees=10):
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
    def plot(self, columns, figsize=(15,6), threshold=.2):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_title(f"Feature Importances with Random Forest")
        filter_ = np.count_nonzero(self.importances > threshold)
        X_indices = range(filter_)
        ax.bar(X_indices, self.importances[self.indices][:filter_], width=.5,
            color="b", yerr=self.std[self.indices][:filter_], align="center")
        plt.xticks(X_indices, self.indices[:filter_])
        ax.set_xticklabels(columns[self.indices[:filter_]], rotation=30, fontsize='small', ha='right')
        return fig, ax
