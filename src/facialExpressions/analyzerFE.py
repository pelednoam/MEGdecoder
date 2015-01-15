'''
Created on Nov 25, 2013

@author: noampeled
'''

from src.commons.analyzerSpacialSlidingWindow.analyzerSpacialSWFreqsSelector import AnalyzerSpacialSWFreqsSelector
from src.commons.analyzerTimeSlidingWindow.analyzerTimeSWFreqsSelector import AnalyzerTimeSWFreqsSelector
from src.commons.analyzer.analyzerFreqsSelector import AnalyzerFreqsSelector
from src.commons import scoreFunctions as sf
from src.commons.utils import MLUtils
from src.commons.misc.subjectsCV import SubjectsCV

import tablesClasses as tc
import featuresExtraction as fe

import numpy as np



class AnalyzerFESuper(object):
    PROCS_NAMES = ['StayOrLeave']
    PROC_LEAVE_STAY = 0
    LABELS = [['Stay', 'Leave']]
    ACTION_STAY, ACTION_LEAVE = range(2)
    FEATURES_NUM = 136

    def loadData(self):
        db = tc.Tables(tc.HDF5_FILE_NAME, read=True)
#         playersRounds = db.getPlayersRounds()
#         playerRounds = playersRounds[self.subject]
        self.labels = db.getActions()  #[playerRounds]

    def dataGenerator(self, matlabDic):
        db = tc.Tables(tc.HDF5_FILE_NAME, read=True)
        dataGroup = tc.GROUP_RAW_DATA
        centroidGroupType = tc.GROUP_KALMAN
        calcMainCentroids = True
        groups = self.getFacialGroups()
        for (dataTable, _, subjectID, gameID, roundID, frameRate), label in zip(db.getTables(dataGroup), self.labels):
            data = dataTable[:]
            mainCentroids = db.loadMainCentroids(centroidGroupType, subjectID, gameID, roundID)
            featuresCalculator = fe.FEPointsDistances(data, groups, calcMainCentroids=calcMainCentroids, mainCentroids=mainCentroids)
            dists = featuresCalculator.calcPointsDistanceFromNose(parameteriation=False, features=False, polar=True, doPlot=False)
            yield ((dists, label), {'timeStep': 1.0 / frameRate, 'subjectID': subjectID})

    def getFacialGroups(self):
        # from http://www.luxand.com/download/Luxand_FaceSDK_Documentation.pdf
        leftEye = [0, 23, 24, 38, 27, 37, 35, 28, 36, 29, 30]
        rightEye = [1, 25, 26, 41, 31, 42, 40, 32, 39, 33, 34]
    #    leftEyeBrow = [13, 16, 18, 19, 12]
    #    rightEyeBrow = [14, 17, 20, 21, 15]
        rightRightEyeBrow = [17, 21, 15]
        rightLeftEyeBrow = [14, 20, 17]
        leftLeftEyeBrow = [12, 18, 16]
        leftRightEyeBrow = [13, 19, 16]
        nose = [2, 49, 22, 43, 45, 47, 44, 46, 48]
        mouth = [3, 4, 54, 61, 55, 64, 56, 60, 57, 62, 58, 63, 59, 65]
        chickLeft = [50, 52]
        chickRight = [51, 53]
        # contour = [11, 9, 10, 7, 5, 6, 8
        rightChin = [6, 8, 10, 11]
        leftChin = [5, 7, 9, 11]
        groups = [leftEye, rightEye, leftLeftEyeBrow, leftRightEyeBrow, rightRightEyeBrow, rightLeftEyeBrow, nose, mouth, chickLeft, chickRight, rightChin, leftChin]
        return groups

    def featuresCV(self, y, trialsInfo, foldsNum, testSize=None):
        subjects = [trialInfo['subjectID'] for trialInfo in trialsInfo]
        return SubjectsCV(subjects)

    def calcTimeStep(self, trialsInfo):
        ''' return times bin '''
        return np.array([trialInfo['timeStep'] for trialInfo in trialsInfo])

    def getChannelsNum(self, matlabDic):
        return self.FEATURES_NUM

    def metaDataGenerator(self, matlabDic):
        for label in self.labels:
            yield (label, {})

    def getTrialsInfo(self, matlabDic):
        return {}

    def trialCond(self, label, trialInfo):
        return True

    def trialLabel(self, label, trialInfo):
        return label

    def scorer(self, ytest, probs):
        ypred = MLUtils.probsToPreds(probs)
        return sf.gmeanScore(ytest, ypred)
#         return sf.AUCScore(ytest, probs)

    def gridSearchScorer(self, ytest, probs):
        ypred = MLUtils.probsToPreds(probs)
        return sf.gmeanScore(ytest, ypred)
#         return sf.AUCScore(ytest, probs)


class AnalyzerFE(AnalyzerFESuper, AnalyzerFreqsSelector):

    def __init__(self, *args, **kwargs):
        kwargs['indetifier'] = 'FacialExpressions'
        super(AnalyzerFE, self).__init__(*args, **kwargs)


class AnalyzerCentipedeFreqsSlidingWindow(AnalyzerFESuper,
    AnalyzerTimeSWFreqsSelector):

    def __init__(self, *args, **kwargs):
        kwargs['indetifier'] = 'centipedeFSW'
        super(AnalyzerTimeSWFreqsSelector, self).__init__(*args, **kwargs)


class AnalyzerCentipedeSpacialSWFreqs(AnalyzerFESuper,
    AnalyzerSpacialSWFreqsSelector):

    def __init__(self, *args, **kwargs):
        kwargs['indetifier'] = 'centipedeSpacialSWFreqs'
        super(AnalyzerCentipedeSpacialSWFreqs, self).__init__(*args, **kwargs)
