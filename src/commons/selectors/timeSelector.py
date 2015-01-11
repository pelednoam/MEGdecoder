'''
Created on Nov 20, 2014

@author: noampeled
'''

from sklearn.feature_selection import SelectKBest, f_classif
from src.commons.utils import sectionsUtils


class TimeSelector():

    def __init__(self, alpha, sigSectionMinLength, onlyMidValue,
                 timeAxis, maxSurpriseVal=20, labels=['0', '1'],
                 doPlotSections=False):
        # Take best kTime points in the time domain
        self.selector = SelectKBest(f_classif, k='all')
        self.alpha = alpha
        self.sigSectionMinLength = sigSectionMinLength
        self.onlyMidValue = onlyMidValue
        self.xAxis = timeAxis
        self.maxSurpriseVal = maxSurpriseVal
        self.labels = labels
        self.doPlotSections = doPlotSections

    def fit(self, X, y, cvIndices=None, timeIndices=None, weights=None):
        self.sections = sectionsUtils.findSigSectionsInPValues(
            X, y, self.selector, self.alpha,
            self.sigSectionMinLength, self.xAxis,
            cvIndices, timeIndices, weights, 
            self.maxSurpriseVal, self.labels, self.doPlotSections)

    def transform(self, X, cvIndices, *args, **kwargs):
        return sectionsUtils.concatenateFeaturesFromSections(
            X, self.sections, self.onlyMidValue, cvIndices)

    def fit_transform(self, X, y, cvIndices=None, timeIndices=None, weights=None):
        self.fit(X, y, cvIndices, timeIndices)
        return self.transform(X, cvIndices)
