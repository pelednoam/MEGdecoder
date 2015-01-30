'''
Created on Nov 25, 2013

@author: noampeled
'''

from src.commons.analyzerSpacialSlidingWindow.analyzerSpacialSWFreqsSelector \
    import AnalyzerSpacialSWFreqsSelector
from src.commons.analyzerTimeSlidingWindow.analyzerTimeSWFreqsSelector \
    import AnalyzerTimeSWFreqsSelector
from src.commons.analyzerFreqsSlidingWindow.analyzerFreqsSWSelector \
    import AnalyzerFreqsSWSelector
from src.commons.analyzerFreqsSlidingWindow.analyzerFreqTimeSWSelector \
    import AnalyzerFreqsTimeSWSelector

from src.commons.analyzer.analyzerFreqsSelector import AnalyzerFreqsSelector
from src.commons.utils import utils
from src.commons import scoreFunctions as sf
from src.commons.utils import MLUtils

import os
import numpy as np
from path3 import path
import operator
import itertools
from sklearn.datasets.base import Bunch
import tables


class AnalyzerCentipedeSuper(object):
    PROCS_NAMES = ['StayOrLeave', 'StayOrLeave6_10']
    PROC_LEAVE_STAY, PROC_LEAVE_STAY_6_10 = range(2)
    LABELS = [['Stay', 'Leave'], ['Stay', 'Leave']]
    ACTION_STAY, ACTION_LEAVE = range(2)

    trialsInfoDesc = {
        'round': tables.Int32Col(),
        'game': tables.Int32Col(),
        'session': tables.Int32Col(),
    }

    def loadData(self):
        matlabFullPath = os.path.join(self.folder, self.subject,
            self.matlabFile)
        matlabDic = utils.loadMatlab(matlabFullPath)
        if (self.procID == self.PROC_LEAVE_STAY_6_10):
            self.lastLeaveBeforeRound6 = self.findLastLeaveBeforeRound6(matlabDic)
        return matlabDic

    def loadDataWithArtifacts(self, matlabDic):
        if (self.matlabFileWithArtifacts == self.matlabFile):
            return matlabDic
        else:
            matlabFullPath = os.path.join(self.folder, self.subject, self.matlabFileWithArtifacts)
            if (utils.fileExists(matlabFullPath)):
                matlabDicWithArtifacts = utils.loadMatlab(matlabFullPath)
                return matlabDicWithArtifacts
            else:
                print('No matlab file with artifacts!')
                return None

    def getTrialsTimeLength(self, matlabDic):
        return matlabDic['X'].shape[2]

    def dataGenerator(self, matlabDic):
        trials, labels = np.squeeze(np.array(matlabDic['x'])), \
            np.squeeze(np.array(matlabDic['y']))
        rounds, games, sessions = self.getTrialsInfo(matlabDic)
        if (self.useSpectral):
            for n, (label, round, game, session) in enumerate(zip(labels,rounds,games,sessions)):
                # (700, 3656, 120)
                trial = trials[:,:,n]
                yield ((trial.T, label), {'round':round,'game':game, 'session':session})
        else:
            for trial, label, round, game, session in zip(trials,labels,rounds,games,sessions):
                yield ((trial.T, label), {'round':round,'game':game, 'session':session})

    def getTrialShape(self, matlabDic):
        trial = matlabDic['x'][0][0]
        return trial.T.shape

    def metaDataGenerator(self, matlabDic):
        labels = np.squeeze(np.array(matlabDic['y']))
        rounds, games, sessions = self.getTrialsInfo(matlabDic)
        for label, round, game, session in zip(labels, rounds, games, sessions):
            yield (label, {'round': round, 'game': game, 'session': session})

    def getTrialsInfo(self, matlabDic):
        rounds = np.squeeze(np.array(matlabDic['rounds']))
        games = np.squeeze(np.array(matlabDic['games']))
        sessions = np.squeeze(np.array(matlabDic['sessions']))
        return rounds, games, sessions

    def trialCond(self, label, trialInfo):
        if (self.procID == self.PROC_LEAVE_STAY_6_10):
            # take all trials before the player has realized the agent always stays in rounds 1-5, after that take only trials from round 6 and above
            before = (trialInfo['session'] <= self.lastLeaveBeforeRound6.session and
                      trialInfo['game'] <= self.lastLeaveBeforeRound6.game)
            return (before or trialInfo['round'] > self.lastLeaveBeforeRound6.round)
        else:
            return True

    def trialLabel(self, label, trialInfo):
        return label - 1

    def createEmptyTrialInfoTable(self, tab, recordsNum):
        trialInfo = tab.row
        for _ in range(recordsNum):
            trialInfo['round'] = 0
            trialInfo['game'] = 0
            trialInfo['session'] = 0
            trialInfo.append()
        tab.flush()

    def setTrialInfoRecord(self, tab, recordNum, trialInfo):
        tab.cols.round[recordNum] = trialInfo['round']
        tab.cols.game[recordNum] = trialInfo['game']
        tab.cols.session[recordNum] = trialInfo['session']

    def scorer(self, ytest, probs):
        ypred = MLUtils.probsToPreds(probs)
        return sf.gmeanScore(ytest, ypred)
#         return sf.AUCScore(ytest, probs)

    def gridSearchScorer(self,ytest,probs):
        ypred = MLUtils.probsToPreds(probs)
        return sf.gmeanScore(ytest, ypred)
#         return sf.AUCScore(ytest, probs)

    def findLastLeaveBeforeRound6(self, matlabDic=None):
        ''' The function tries to find where to player has realized that the agent always
            stays for the first 5 rounds. For that, it finds the last game where the player has 
            left in round<6 '''
        # Default
        return Bunch(round=2, game=3, session=1)

        matlabDicWithArtifacts = self.loadDataWithArtifacts(matlabDic)
        if (matlabDicWithArtifacts is not None):
            labels = np.squeeze(np.array(matlabDicWithArtifacts['y']))
            rounds, games, sessions = self.getTrialsInfo(matlabDicWithArtifacts)
            # Sort the trials info according to sessions, games and rounds
            sortedInfo = sorted(zip(rounds, games, sessions, labels),key=operator.itemgetter(2,1,0))
            sortedInfoGroupByGame = itertools.groupby(sortedInfo, key=operator.itemgetter(1))      
            leave6_10 = []
            for _, gameInfo in sortedInfoGroupByGame:
                gameInfo = list(gameInfo)
                print(gameInfo)
                if (np.any(np.array([info[3] for info in gameInfo if info[0]<6])==self.ACTION_LEAVE+1)):
                    leave6_10.append(gameInfo)
            return Bunch(game=leave6_10[-1][-1][1],session=leave6_10[-1][-1][2])
        # Default 
        return Bunch(game=2,session=1)

    def loadFreqs(self):
        freqsDic = utils.loadMatlab(os.path.join(self.folder,self.subject,'freqsArr.mat'))
        freqs = {}
        labels = self.LABELS[self.procID]
        freqs[labels[0]] = freqsDic['{}Freqs'.format(labels[0])][0]
        freqs[labels[1]] = freqsDic['{}Freqs'.format(labels[1])][0]
        return freqs

    @property
    def T(self):
        return 3.5


class AnalyzerCentipedeAllSuper(AnalyzerCentipedeSuper):

    def loadData(self):
        allFiles = {}
#         self.lengths = []
        if (self.procID == self.PROC_LEAVE_STAY_6_10):
            self.lastLeaveBeforeRound6 = self.findLastLeaveBeforeRound6()
        matFiles = list(path(self.folder).walk(self.matlabFile))
        print('{} files were found'.format(len(matFiles)))
        for matfile in matFiles:
            subject = matfile.parent.name
            if subject == 'data':
                subject = matfile.parent.parent.name
            print('Loading the file for {}'.format(subject))
            matlabDic = utils.loadMatlab(matfile)
            allFiles[subject] = matlabDic
        return allFiles

    def dataGenerator(self, matlabDics):
        for subject, matlabDic in matlabDics.iteritems():
            trials, labels = np.squeeze(np.array(matlabDic['x'])), \
                np.squeeze(np.array(matlabDic['y']))
            rounds, games, sessions = self.getTrialsInfo(matlabDic)
            for trial, label, round, game, session in zip(trials, labels, rounds, games, sessions):
                yield ((trial.T, label), {'subject': subject, 'round': round,
                    'game': game, 'session': session})

    def metaDataGenerator(self, matlabDics):
        for subject, matlabDic in matlabDics.iteritems():
            labels = np.squeeze(np.array(matlabDic['y']))
            rounds, games, sessions = self.getTrialsInfo(matlabDic)
            for label, round, game, session in zip(labels, rounds, games, sessions):
                yield (label, {'subject': subject, 'round': round,
                    'game': game, 'session': session})

    def getTrialShape(self, matlabDics):
        trial = matlabDics[0]['x'][0][0]
        return trial.T.shape

    def getChannelsNum(self, matlabDics):
        # channelsNum = self.getTrialShape(matlabDics)[2]
        # print('channels num: {}'.format(channelsNum))
        # return channelsNum
        return 248

    def setTrialInfoRecord(self, tab, recordNum, trialInfo):
        tab.cols.subject[recordNum] = trialInfo['subject']
        tab.cols.round[recordNum] = trialInfo['round']
        tab.cols.game[recordNum] = trialInfo['game']
        tab.cols.session[recordNum] = trialInfo['session']

    trialsInfoDesc = {
        'subject': tables.StringCol(16),
        'round': tables.Int32Col(),
        'game': tables.Int32Col(),
        'session': tables.Int32Col(),
    }


class AnalyzerCentipede(AnalyzerCentipedeSuper, AnalyzerFreqsSelector):

    def __init__(self, *args, **kwargs):
        kwargs['indetifier'] = 'centipede'
        super(AnalyzerCentipede, self).__init__(*args, **kwargs)


class AnalyzerCentipedeTimeSWFreqs(AnalyzerCentipedeSuper,
    AnalyzerTimeSWFreqsSelector):

    def __init__(self, *args, **kwargs):
        kwargs['indetifier'] = 'centipedeTimeSW'
        super(AnalyzerCentipedeTimeSWFreqs, self).__init__(*args, **kwargs)


class AnalyzerCentipedeSpacialSWFreqs(AnalyzerCentipedeSuper,
    AnalyzerSpacialSWFreqsSelector):

    def __init__(self, *args, **kwargs):
        kwargs['indetifier'] = 'centipedeSpacialSWFreqs'
        super(AnalyzerCentipedeSpacialSWFreqs, self).__init__(*args, **kwargs)


class AnalyzerCentipedeFreqsSW(AnalyzerCentipedeSuper,
    AnalyzerFreqsSWSelector):

    def __init__(self, *args, **kwargs):
        kwargs['indetifier'] = 'centipedeFreqsSW'
        super(AnalyzerFreqsSWSelector, self).__init__(*args, **kwargs)


class AnalyzerCentipedeFreqsTimeSW(AnalyzerCentipedeSuper,
    AnalyzerFreqsTimeSWSelector):

    def __init__(self, *args, **kwargs):
        kwargs['indetifier'] = 'centipedeFreqsTimeSW'
        super(AnalyzerFreqsTimeSWSelector, self).__init__(*args, **kwargs)


class AnalyzerCentipedeAll(AnalyzerCentipedeAllSuper, AnalyzerFreqsSelector):

    def __init__(self, *args, **kwargs):
        kwargs['indetifier'] = 'centipedeAll'
        super(AnalyzerCentipedeAll, self).__init__(*args, **kwargs)


class AnalyzerCentipedeAllFreqsSW(AnalyzerCentipedeAllSuper, AnalyzerFreqsSWSelector):

    def __init__(self, *args, **kwargs):
        kwargs['indetifier'] = 'centipedeAllFreqsSW'
        super(AnalyzerCentipedeAllFreqsSW, self).__init__(*args, **kwargs)

    # Load the save file as AnalyzerCentipedeAll
    def _dataFileName(self, stepID, folder='', noShuffle=False):
        return '{}/{}{}{}_{}_{}_{}_sub_{}.npz'.format(
            folder, 'centipedeAll',
            '_shuffled' if self.shuffleLabels and not noShuffle else '',
            '_smote' if self.useSmote and not noShuffle else '',
            self.PROCS_NAMES[self.procID],
            self.getStepName(stepID), 'FrequenciesSelector',
            self.subject)

 # class AnalyzerCentipedeSpectrum(AnalyzerPowerSpectrumSelector):
#     PROCS_NAMES = ['StayOrLeave']
#     PROC_LEAVE_STAY = 0
#     LABELS = [['Stay','Leave']]
# 
#     SPECTRAL_FOLDER = 'spectral'
# 
# 
#     def __init__(self, *args, **kwargs):
#         kwargs['indetifier']='centipedeSpectrumSelector'
#         super(AnalyzerCentipedeSpectrum, self).__init__(*args, **kwargs)
# 
#     def loadFreqs(self): 
#         freqsDic = utils.loadMatlab(os.path.join(self.spectralFolder,'freqsArr.mat'))
#         freqs = {}
#         labels = self.LABELS[self.procID]
#         freqs[labels[0]] = freqsDic['{}Freqs'.format(labels[0])][0]
#         freqs[labels[1]] = freqsDic['{}Freqs'.format(labels[1])][0]
#         return freqs
#     
#     def scorer(self,ytest,probs):
#         ypred = MLUtils.probsToPreds(probs)
#         return sf.gmeanScore(ytest, ypred)
# 
#     def gridSearchScorer(self,ytest,probs):
#         ypred = MLUtils.probsToPreds(probs)
#         return sf.gmeanScore(ytest, ypred)


