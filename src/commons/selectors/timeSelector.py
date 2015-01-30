'''
Created on Nov 20, 2014

@author: noampeled
'''

from sklearn.feature_selection import SelectKBest, f_classif
from src.commons.utils import sectionsUtils


class TimeSelector():

    def __init__(self, alpha, sigSectionMinLength, onlyMidValue,
                 timeAxis, maxSurpriseVal=20, labels=['0', '1'],
                 doPlotSections=False, sectionsKeys=None):
        # Take best kTime points in the time domain
        self.selector = SelectKBest(f_classif, k='all')
        self.alpha = alpha
        self.sigSectionMinLength = sigSectionMinLength
        self.onlyMidValue = onlyMidValue
        self.xAxis = timeAxis
        self.maxSurpriseVal = maxSurpriseVal
        self.labels = labels
        self.doPlotSections = doPlotSections
        self.sectionsKeys = sectionsKeys

    def fit(self, X, y, cvIndices=None, timeIndices=None, weights=None):
        self.sections, self.sectionsDic = \
            sectionsUtils.findSigSectionsInPValues(
            X, y, self.selector, self.alpha,
            self.sigSectionMinLength, self.xAxis,
            cvIndices, timeIndices, weights,
            self.maxSurpriseVal, self.labels, self.doPlotSections,
            self.sectionsKeys)

    def transform(self, X, cvIndices):
        return sectionsUtils.concatenateFeaturesFromSections(
            X, self.sections, self.onlyMidValue, cvIndices)

    def fit_transform(self, X, y, cvIndices=None, timeIndices=None,
                      weights=None):
        self.fit(X, y, cvIndices, timeIndices)
        return self.transform(X, cvIndices)
