'''
Created on Nov 25, 2013

@author: noampeled
'''

from src.commons.analyzer.analyzer import Analyzer
from src.commons.analyzer.analyzerTimeSelector import AnalyzerTimeSelector
from src.commons.utils import utils
import os
import numpy as np


class AnalyzerYoni(AnalyzerTimeSelector):
    PROCS_NAMES = ['1Or2', '2or4', '1Or4']
    PROC_1_2, PROC_2_4, PROC_1_4 = range(3)
    LABELS = [['1', '2'], ['2', '4'], ['1', '4']]

    def __init__(self, *args, **kwargs):
        kwargs['indetifier'] = 'yoni'
        super(AnalyzerYoni, self).__init__(*args, **kwargs)

    def loadData(self):
        matlabFullPath = os.path.join(self.folder, self.matlabFile)
        matlabDic = utils.loadMatlab(matlabFullPath)
        return matlabDic

    def getTrialsTimeLength(self, matlabDic):
        return matlabDic['x'].shape[1]

    def dataGenerator(self, matlabDic):
        X, Y = matlabDic['x'], matlabDic['y'][0]
        nanidx = np.where(np.isnan(X[0]))[0]
        self.xAxis = np.delete(self.xAxis, nanidx, axis=0)
        for x, y in zip(X, Y):
            x = np.delete(x, nanidx, axis=0)
            x = np.reshape(x, (x.shape[0], 1))
            yield ((x, y), {})

    def getTrialShape(self, matlabDic):
        x = matlabDic['x'][0]
        nanidx = np.where(np.isnan(x))[0]
        T = x.shape[0] - len(nanidx)
        return (T, 1)  # Only one channel

    def metaDataGenerator(self, matlabDic):
        labels = matlabDic['y'][0]
        for label in labels:
            yield (label, {})

    def trialCond(self, label, trialInfo):
        flag = False
        if (self.procID == self.PROC_1_2):
            flag = (label in [1, 2])
        elif (self.procID == self.PROC_1_4):
            flag = (label in [1, 4])
        elif (self.procID == self.PROC_2_4):
            flag = (label in [2, 4])
        else:
            utils.throwException('wrong procID!')
        return flag

    def trialLabel(self, label, trialInfo):
        if (self.procID == self.PROC_1_2):
            y = 0 if label == 1 else 1
        elif (self.procID == self.PROC_1_4):
            y = 0 if label == 1 else 1
        elif (self.procID == self.PROC_2_4):
            y = 0 if label == 2 else 1
        else:
            utils.throwException('wrong procID!')
        return y

    def weightsFullFileName(self, samWeights):
        return os.path.join(self.folder, self.subject, samWeights)

    @property
    def weightsDicKey(self):
        return 'ActWgtsNoZeros'


