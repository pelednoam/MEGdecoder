'''
Created on Nov 20, 2014

@author: noampeled
'''
import os
from path3 import path


class PowerSpectrumSelector():

    def __init__(self, freqs, alpha=0.05, sigSectionMinLength=10,
                 minFreq=0, maxFreq=50, powerSpectrumDataFolder='',
                 onlyMidValue=False, doPrint=False):
        self.freqs = freqs
        self.alpha = alpha
        self.sigSectionMinLength = sigSectionMinLength
        self.minFreq = minFreq
        self.maxFreq = maxFreq
        self.powerSpectrumDataFolder = powerSpectrumDataFolder
        self.onlyMidValue = onlyMidValue
        self.doPrint = doPrint

    def fit(self, X, y):
        N, T, C = X.shape
        # Check if the power ppetrum was done elsewhere (matlab fieldtrip for example)
        if (self.powerSpectrumDataFolder != ''):
            # The data stacture is like the following:
            # label -> sensor -> file: 'spect{}_{}_{}.mat'.format(label,sensor,freqID)
            labels = path(self.powerSpectrumDataFolder).files()
            for label in labels:
                sensors = path(os.path.join(self.powerSpectrumDataFolder,label)).files()
                for sensor in sensors:
                    spectrumFiles = path(os.path.join(self.powerSpectrumDataFolder,label,sensor)).files()
                    for spectrumFile in spectrumFiles:
                        pass
