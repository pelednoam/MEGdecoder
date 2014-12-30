'''
Created on Nov 20, 2014

@author: noampeled
'''
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif


class ChannelsSelector(SelectKBest):

    def __init__(self, k):
        super(ChannelsSelector, self).__init__(f_classif, k=k)

    def fit(self, X, y):
        self.T = X.shape[1]
        y = yTile(y, self.T)
        X = preReshape(X)
        return SelectKBest.fit(self, X, y)

    def transform(self, X):
        X = preReshape(X)
        X = SelectKBest.transform(self, X)
        X = postReshape(X, self.T, self.k)
        return X

    def fit_transform(self, X, y):
        self.T = X.shape[1]
        y = np.tile(y, (self.T, 1)).T.reshape(-1)
        X = preReshape(X)
        SelectKBest.fit(self, X, y)
        X = SelectKBest.transform(self, X)
        X = postReshape(X, self.T, self.k)
        return X


class ChannelsSelector2():

    def __init__(self, k, kTime=20):
        # Take best kTime points in the time domain
        self.selector = SelectKBest(f_classif, k=kTime)
        self.k = k

    def fit(self, X, y, doPrint=False):
        C = X.shape[2]
        self.selectors = [None] * C
        self.scores = np.zeros(C)
        for c in range(C):
            if (doPrint):
                print('sensor {}'.format(c))
            model = self.selector.fit(X[:, :, c], y)
            scores = model.scores_[model._get_support_mask()]
            self.scores[c] = np.mean(scores)
        self.channelsIndices = np.argsort(self.scores)[::-1][:self.k]

    def transform(self, X):
        return X[:, :, self.channelsIndices]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class ChannelsPCA(PCA):

    def fit(self, X, y=None):
        self.T = X.shape[1]
        X = preReshape(X)
        # normalize the data
        self.scaler = preprocessing.StandardScaler().fit(X)
        X = self.scaler.transform(X)
        PCA.fit(self, X)
        self.printExplainedVar()
        return self

    def transform(self, X):
        X = preReshape(X)
        X = self.scaler.transform(X)
        X = PCA.transform(self, X)
        X = postReshape(X, self.T, self.n_components)
#         print(X.shape)
        return X

    def fit_transform(self, X, y=None):
        self.T = X.shape[1]
        X = preReshape(X)
        # normalize the data
        self.scaler = preprocessing.StandardScaler().fit(X)
        X = self.scaler.transform(X)
        self.printExplainedVar()
        # Fit and transform
        PCA.fit(self, X, y)
        X = PCA.transform(self, X)
        self.printExplainedVar()
        X = postReshape(X, self.T, self.n_components)
        return X

    def printExplainedVar(self):
        pass
#         print ('explained variance ratio (first %d components): %.2f'%(
#             self.n_components, sum(self.explained_variance_ratio_)))


def preReshape(X):
    N, T, C = X.shape
    return X.reshape((N * T, C))


def postReshape(X, T, k):
    N = X.shape[0] / T
    X = X.reshape((N, T, k))
    return X


def yTile(y, T):
    return np.tile(y, (T, 1)).T.reshape(-1)
