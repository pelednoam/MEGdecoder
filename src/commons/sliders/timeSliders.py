'''
Created on Nov 20, 2014

@author: noampeled
'''
import numpy as np
from sklearn.base import BaseEstimator


class TimeWindowSlider(BaseEstimator):

    def __init__(self, startIndex, windowSize, windowsNum, T):
        self.startIndex = startIndex
        self.windowSize = windowSize
        self.T = T
        self.windowsNum = windowsNum
        self.dt = T / windowsNum

    def fit(self, X):
        return self

    def transform(self, X, cvIndices=None, returnIndices=False):
        if (cvIndices is None):
            cvIndices = range(X.shape[0])
        if (returnIndices):
            tAxis = range(X.shape[1])
            return np.array(tAxis[self.startIndex:self.startIndex +
                self.windowSize])
        else:
            return X[cvIndices, self.startIndex:self.startIndex +
                     self.windowSize]

    def fit_transform(self, X, cvIndices=None, returnIndices=False):
        return self.transform(X, cvIndices, returnIndices)

    def extract(self, X):
        return self.transform(X)

    def windowsGenerator(self):
        return np.linspace(0, self.T - self.windowSize, self.windowsNum
                           ).astype(int)


class windowAccumulator(TimeWindowSlider):

    def transform(self, X):
        return X[:, :self.startIndex + self.windowSize]


class windowAccumulatorTwice(TimeWindowSlider):

    def transform(self, X):
        if (self.startIndex + self.dt < self.T):
            # Fix the last one
            if (self.T - self.startIndex < 2 * self.dt):
                return X[:, :self.T]
            else:
                return X[:, :self.startIndex + self.dt]
        else:
            return X[:, self.startIndex + self.dt - self.T:self.T]

    def windowsGenerator(self):
        self.dt = self.T / self.windowsNum
        t = np.arange(0, self.T, self.dt)
        t[-1] = self.T  # fix the last indice
        t2 = np.hstack((t, self.T + t[1: -2]))
        return t2
