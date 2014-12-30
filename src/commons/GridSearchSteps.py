'''
Created on Jan 26, 2014

@author: noampeled
'''
from sklearn.base import BaseEstimator
from sklearn import preprocessing
from sklearn.svm import SVC
from src.commons.utils import MLUtils


class FeaturesExtractor(BaseEstimator):

    def __init__(self, extractorFunc, *args, **kwargs):
        self.extractorFunc = extractorFunc
        super(BaseEstimator, self).__init__(*args, **kwargs)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.extractorFunc(X)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def extract(self, X):
        return self.transform(X)


class TSVC(SVC):

    def __init__(self, C, kernel, gamma=0, probability=True):
        super(TSVC, self).__init__(C=C, kernel=kernel, gamma=gamma, probability=True)

    def fit(self, X, y, doShuffle=True):
        if (doShuffle):
            (X, idx) = MLUtils.shuffle(X)
            y = y[idx]
        self.scaler = preprocessing.StandardScaler().fit(X)
        X = self.scaler.transform(X)
        super(TSVC, self).fit(X, y)
        return self

    def predict(self, X, calcProbs=True):
        X = self.scaler.transform(X)
        if (calcProbs):
#             dist = self.decision_function(X)
#             probs = distanceToProb(dist)
#             probs = np.vstack((1-probs,probs)).T
            probs = super(TSVC, self).predict_proba(X)
        else:
            probs = super(TSVC, self).predict(X)
        return probs


class staticCV():

    def __init__(self, cv):
        self.cv = cv
        self.trains, self.tests = [], []
        for train, test in cv:
            self.trains.append(train)
            self.tests.append(test)

    def __iter__(self):
        for train, test in zip(self.trains, self.tests):
            yield train, test

    def __repr__(self):
        return self.cv.__repr__()

    def __len__(self):
        return self.cv.__len__()
