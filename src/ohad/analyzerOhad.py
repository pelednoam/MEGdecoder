'''
Created on Oct 27, 2014

@author: noampeled
'''

import os
from src.commons.utils import utils

from src.commons.analyzer.analyzerFreqsSelector import AnalyzerFreqsSelector


class AnalyzerOhad(AnalyzerFreqsSelector):
    PROCS_NAMES = ['BlackOrWhite']
    PROC_BLACK_WHITE = range(1)
    LABELS = [['Black', 'White']]
    ACTION_BLACK, ACTION_WHITE = range(2)

    def __init__(self, *args, **kwargs):
        kwargs['indetifier'] = 'ohad'
        super(AnalyzerOhad, self).__init__(*args, **kwargs)

    def loadData(self):
        matlabFullPath = os.path.join(self.folder, self.subject,
                                      self.matlabFile)
        matlabDic = utils.loadMatlab(matlabFullPath)
        return matlabDic

    def dataGenerator(self, matlabDic):
        trials,labels = np.squeeze(np.array(matlabDic['x'])),np.squeeze(np.array(matlabDic['y']))
        rounds, games, sessions = self.getTrialsInfo(matlabDic)           
        if (self.useSpectral):
            for n, (label, round, game, session) in enumerate(zip(labels,rounds,games,sessions)):
                # (700, 3656, 120)
                trial = trials[:,:,n]
                yield ((trial.T, label), {'round':round,'game':game, 'session':session})
        else:
            for trial, label, round, game, session in zip(trials,labels,rounds,games,sessions):
                yield ((trial.T, label), {'round':round,'game':game, 'session':session})

