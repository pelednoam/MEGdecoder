'''
Created on Nov 20, 2014

@author: noampeled
'''
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.datasets.base import Bunch
from src.commons.utils import sectionsUtils
from src.commons.utils import freqsUtils
from src.commons.utils import sectionsUtils as su


class FrequenciesSelector():

    def __init__(self, timeStep, alpha, sigSectionMinLength, minFreq=0,
                 maxFreq=50, onlyMidValue=True, maxSurpriseVal=20,
                 labels=['0', '1'], doPlotSections=False, sectionsKeys=None):
        self.selector = SelectKBest(f_classif, k='all')
        self.timeStep = timeStep
        self.alpha = alpha
        self.sigSectionMinLength = sigSectionMinLength
        self.minFreq = minFreq
        self.maxFreq = maxFreq
        self.onlyMidValue = onlyMidValue
        self.maxSurpriseVal = maxSurpriseVal
        self.labels = labels
        self.doPlotSections = doPlotSections
        self.sectionsKeys = sectionsKeys

    def fit(self, X, y, cvIndices=None, timeIndices=None, weights=None,
            preCalcPSS=None, preCalcFreqs=None):
        self.sections, self.pss, self.freqs, self.sectionsDic = \
            sectionsUtils.findSigSectionsPSInPValues(
            X, y, self.selector, self.timeStep, self.minFreq, self.maxFreq,
            self.alpha, self.sigSectionMinLength, cvIndices, timeIndices,
            weights, preCalcPSS, preCalcFreqs, self.maxSurpriseVal,
            self.labels, self.doPlotSections, self.sectionsKeys)

    def transform(self, X, cvIndices=None, timeIndices=None,
                  weights=None, preCalcPSS=None, preCalcFreqs=None):
        if (preCalcPSS is None):
            self.freqs, pss = freqsUtils.calcAllPS(X, self.timeStep,
                self.minFreq, self.maxFreq, cvIndices=cvIndices,
                timeIndices=timeIndices, weights=weights)
        else:
            C = su.calcSectionsNum(X, preCalcPSS, weights)
            freqs, ps = freqsUtils.cutPS(preCalcPSS[0, :, :].squeeze(),
                preCalcFreqs, self.minFreq, self.maxFreq)
            pss = np.empty((ps.shape[0], ps.shape[1], C))
            for c in range(C):
                ps = preCalcPSS[c, :, :].squeeze()
                _, ps = freqsUtils.cutPS(ps, preCalcFreqs, self.minFreq,
                    self.maxFreq)
                pss[:, :, c] = ps
            self.freqs, pss = freqs, pss

        features = sectionsUtils.concatenateFeaturesFromSections(
            pss, self.sections, self.onlyMidValue)
        return features

    def fit_transform(self, X, y, cvIndices=None, timeIndices=None,
                      weights=None, preCalcPSS=None, preCalcFreqs=None):
        self.fit(X, y, cvIndices, timeIndices, weights,
            preCalcPSS, preCalcFreqs)
        return self.transform(X, cvIndices, timeIndices, weights,
            preCalcPSS, preCalcFreqs)
