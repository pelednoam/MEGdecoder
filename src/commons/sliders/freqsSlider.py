'''
Created on Dec 11, 2014

@author: noam
'''

import numpy as np
from sklearn.base import BaseEstimator


class FreqsWindowSlider(BaseEstimator):

    def __init__(self, fromFreq, toFreq, windowWidth, windowsNum):
        '''
            fromFreq, toFreq: Frequencies range [Hz]
            windowWidth: The window width [Hz]
        '''
        self.fromFreq = fromFreq
        self.toFreq = toFreq
        self.windowWidth = windowWidth
        self.windowsNum = windowsNum

    def windowsGenerator(self):
        freqsRange = np.linspace(self.fromFreq, self.toFreq - self.windowWidth,
            self.windowsNum)
        for minFreq in freqsRange:
            yield (minFreq, minFreq + self.windowWidth)
