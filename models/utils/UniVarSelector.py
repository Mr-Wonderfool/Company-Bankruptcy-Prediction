from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif
import numpy as np
import matplotlib.pyplot as plt

class UniVarSelector:
    def __init__(self,X,Y,percent=20, method='f_classif'):
        assert method in ['f_classif', 'mutual_info_classif'], 'Invalid value'
        self.X=X
        self.Y=Y
        self.percent=percent
        if method == 'f_classif':
            self.method = f_classif
        elif method == 'mutual_info_classif':
            self.method = mutual_info_classif
    def fit(self):
        selector= SelectPercentile(self.method, percentile=self.percent) # choose 20% of the variables
        selector.fit_transform(self.X, self.Y)
        self.selected_feautes = selector.get_support(indices=True)
        if self.method == f_classif:
            pvalues = selector.pvalues_ # smaller values indicate the variable is more important
            pvalues = np.nan_to_num(pvalues, nan=1.)
            scores = -np.log10(pvalues) # for numerical reasons
        elif self.method == mutual_info_classif:
            scores = selector.scores_
        self.scores = scores / scores.max() # normalize
        self.indx = np.argsort(self.scores)[::-1] # descending order
        return self.selected_feautes, self.indx
    def plot(self, columns, figsize=(15,6), threshold=0.1):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_title(f"Feature importances (with score threshold = {threshold})")
        filter_ = np.count_nonzero(self.scores > threshold)
        X_indices = range(filter_)
        ax.bar(X_indices, self.scores[self.indx][:filter_], width=.75,
                label=r'mutual information score', color='b')
        ax.set_xticks(X_indices, self.indx[:filter_])
        ax.set_xticklabels(columns[self.indx[:filter_]], rotation=30, fontsize='small', ha='right')
        plt.legend()
        return ax