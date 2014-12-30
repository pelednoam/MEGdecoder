# -*- coding: utf-8 -*-

'''
Created on Nov 19, 2013

@author: noampeled
'''
import numpy as np
import sys
import os
from path3 import path


from analyzerInbal import AnalyzerInbal, AnalyzerInbalAMI
from src.commons.analyzer.analyzer import Analyzer
from src.commons.utils import utils
from src.commons.utils import tablesUtils

sys.path.append('/Users/noampeled/Documents/python/scikit-learn')

FOLDER_LAB = '/Users/noampeled/Copy/Data/MEG/InbalData/Data5_35Hz'
FOLDER_HOME = '/home/noam/Documents/MEGdata/Inbal'
FOLDER_SERVER = '/home/noam/Documents/MEGdata/inbalData'
LOCAL_TEMP_FOLDER = '/home/noam/Documents/MEGdata/Inbal/svmFiles/temp'
REMOTE_FILES_FOLDER = os.path.join(FOLDER_LAB,'svmFiles')

FOLDER = FOLDER_LAB if utils.folderExists(FOLDER_LAB) else FOLDER_HOME if utils.folderExists(FOLDER_HOME) else FOLDER_SERVER
# SUBJECTS = ['2','3','4','8','11','15','17','18','19']
SUBJECTS = ['2', '3', '4', '8']  # '2',
RESULTS_FILE_NAME = path.join(path(__file__).parent.parent.parent,'output.txt')
utils.sftpInit('127.0.0.1', 'noampeled', port=9998, remoteFolder=REMOTE_FILES_FOLDER)

SAM_WEIGHTS_FILE = 'ActWgtsNoZeros.mat'

if __name__ == '__main__':
    args = sys.argv[1:]
    cpuNum = 1 if (len(args) < 1) else int(args[0])
    if (cpuNum > 1):
        print('cpuNum = %d' % cpuNum)

    t = utils.ticToc()
    Cs = np.logspace(-3, 3, 7)
    gammas = np.logspace(-9, 0, 5)
    sigSectionMinLengths = [1, 2, 3]  # [1, 2, 3]
    sigSectionAlphas = [0.05, 0.1, 0.2, 0.3]
    minFreqs = [0]  # [0, 5, 10]
    maxFreqs = [40]
    kernels = ['rbf', 'linear']
    FOLDS = 5
    TEST_SIZE = 0.2
    JOBS = cpuNum

    for SUBJECT in ['2', '3', '4', '8']:  # SUBJECTS:
        MATLAB_FILE = 'data_5_35_ForML_{}.mat'.format(SUBJECT)
        print('Subject: {}'.format(SUBJECT))
        tablesUtils.DEF_TABLES = False
        analyze = AnalyzerInbal(FOLDER, MATLAB_FILE, SUBJECT,
            procID=AnalyzerInbal.PROC_3_2, multipleByWeights=False, useSpectral=False,
            jobsNum=JOBS, samWeights=SAM_WEIGHTS_FILE)
#         analyze.preProcess(parallel=False, verbose=False)
#         analyze.splitData(heldoutSize=0)
# #         analyze.shuffleLabels()
#         analyze.process(foldsNum=FOLDS, Cs=Cs,
#             gammas=gammas, kernels=kernels, testSize=TEST_SIZE,
#             sigSectionMinLengths=sigSectionMinLengths,
#             sigSectionAlphas=sigSectionAlphas,
#             minFreqs=minFreqs, maxFreqs=maxFreqs, onlyMidValueOptions=[True])
#         analyze.calculatePredictionsScores()
#         analyze.analyzeResults()
#         analyze.calcImporatances(FOLDS, TEST_SIZE, doCalc=True,
#             doShow=False, permutationNum=2000, permutationLen=5)
        analyze.fullDataAnlysis(dataRanges=[[16, 20]],
                permutationNum=200, permutationLen=5)

    howMuchTime = utils.howMuchTimeFromTic(t)
    utils.sendResultsEmail('analyzer is done! {}'.format(howMuchTime),
        RESULTS_FILE_NAME)
    if (utils.sftp):
        utils.sftp.close()
