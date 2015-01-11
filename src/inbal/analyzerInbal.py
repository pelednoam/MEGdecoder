'''
Created on Nov 25, 2013

@author: noampeled
'''

from src.commons.analyzer.analyzer import Analyzer
from src.commons.analyzer.analyzerFreqsSelector import AnalyzerFreqsSelector
from src.commons.utils import utils
import os
import numpy as np


class AnalyzerInbalSuper(object):
    PROCS_NAMES = ['3OrRN', '3orgOr3Diff', '3Or2']
    PROC_3_RN, PROC_3ORG_3Diff, PROC_3_2 = range(3)
    LABELS = [['3', 'RN'], ['3org', '3diff'], ['3', '2']]


    def loadData(self):
        matlabFullPath = os.path.join(self.folder, self.subject,
            self.matlabFile)
        matlabDic = utils.loadMatlab(matlabFullPath)
        return matlabDic

class AnalyzerInbalClusters(AnalyzerFreqsSelector):

    def __init__(self, *args, **kwargs):
        kwargs['indetifier'] = 'inbalClusters'
        super(AnalyzerInbal, self).__init__(*args, **kwargs)


class AnalyzerInbal(AnalyzerFreqsSelector):
    PROCS_NAMES = ['3OrRN', '3orgOr3Diff', '3Or2']
    PROC_3_RN, PROC_3ORG_3Diff, PROC_3_2 = range(3)
    LABELS = [['3', 'RN'], ['3org', '3diff'], ['3', '2']]

    ONSET = 200.0  # ms
    TOTAL_TIME = 900.0  # ms

    def __init__(self, *args, **kwargs):
        kwargs['indetifier'] = 'inbal'
        super(AnalyzerInbal, self).__init__(*args, **kwargs)

    def loadData(self):
        matlabFullPath = os.path.join(self.folder, self.subject,
            self.matlabFile)
        matlabDic = utils.loadMatlab(matlabFullPath)
        return matlabDic

    def getTrialsTimeLength(self, matlabDic):
        return matlabDic['trials'][0][0][0][0].shape[1]

    def dataGenerator(self, matlabDic):
        (trials, labels, ranks) = (matlabDic['trials'][0],
            matlabDic['labels'][0], matlabDic['ranks'][0])
        T = trials[0][0][0].shape[1]
        onsetInd = self._calcTrialOnset(T)
        self.xAxis = self.xAxis[onsetInd:]
        for trial, label, rank in zip(trials, labels, ranks):
            yield ((trial[0][0][:, onsetInd:].T, label[0]),
                   {'rank': rank[0][0]})

    def getTrialShape(self, matlabDic):
        trial = matlabDic['trials'][0][0][0][0]
        T = trial.shape[1]
        onsetInd = self._calcTrialOnset(T)
        return trial[:, onsetInd:].T.shape

    def _calcTrialOnset(self, T):
        freq = T / AnalyzerInbal.TOTAL_TIME
        onsetInd = int(AnalyzerInbal.ONSET * freq)
        return onsetInd

    def metaDataGenerator(self, matlabDic):
        labels, ranks = matlabDic['labels'][0], matlabDic['ranks'][0]
        for label, rank in zip(labels, ranks):
            yield (label[0], {'rank': rank[0][0]})

    def trialCond(self, label, trialInfo):
        flag = False
        rank = trialInfo['rank']
        if (self.procID == self.PROC_3_RN):
            flag = (rank == 3 or label == 'RN')
        elif (self.procID == self.PROC_3ORG_3Diff):
            flag = (rank == 3 and label in ['T', 'TM', 'C', 'Cm'])
        elif (self.procID == self.PROC_3_2):
            flag = (rank in [2, 3])
        else:
            utils.throwException('wrong procID!')
        return flag

    def trialLabel(self, label, trialInfo):
        rank = trialInfo['rank']
        if (self.procID == self.PROC_3_RN):
            y = 0 if label == 'RN' else 1
        elif (self.procID == self.PROC_3ORG_3Diff):
            y = 0 if label in ['T', 'C'] else 1 if label in ['TM', 'Cm'] else -1
        elif (self.procID == self.PROC_3_2):
            y = 0 if rank == 2 else 1
        else:
            utils.throwException('wrong procID!')
        return y

    def weightsFullFileName(self, samWeights):
        return os.path.join(self.folder, self.subject, samWeights)

    @property
    def weightsDicKey(self):
        return 'ActWgtsNoZeros'


class AnalyzerInbalAMI(AnalyzerInbal):

    def __init__(self,folder,matlabFile,subject,procID,analID,PCACompsNum,indetifier='inbalAMI'):
        Analyzer.__init__(self, folder, matlabFile, subject, procID, analID,PCACompsNum,indetifier)

    def loadData(self):
        files = utils.filesInFolder('{}/AMI'.format(self.folder), 'Allapp_*_AI_sub{}.mat'.format(self.subject))
        matlabDic={}
        for matFile in files:
            note = matFile.name.split('_')[1]
            matlabDic[note] = utils.loadMatlab(matFile)
        return matlabDic

    def trialsAnalysis(self):
        Analyzer.trialsAnalysis(self)

    def dataGenerator(self, matlabDic):
        for note, noteDic in matlabDic.items():
            ami = noteDic['GI']
            rank = int(note) if utils.isNumber(note) else 1
            label = note if note=='RN' else ''
            for trialNum in range(ami.shape[2]):
                trial = ami[:,:,trialNum]
                for channel in xrange(trial.shape[0]):
                    trial[channel,:]=trial[channel,:]/trial[channel,0]
                yield ((trial, label), {'rank':rank})
