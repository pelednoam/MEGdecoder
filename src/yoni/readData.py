# -*- coding: utf-8 -*-

'''
Created on Nov 19, 2013

@author: noampeled
'''
import numpy as np
import sys
import os
from path3 import path


from analyzerYoni import AnalyzerYoni
from src.commons.analyzer.analyzer import Analyzer
from src.commons.utils import utils
from src.commons.utils import tablesUtils

FOLDER_LAB = '/Users/noampeled/Copy/Data/MEG/yoni'
FOLDER_HOME = '/home/noam/Documents/MEGdata/yoni'
FOLDER_SERVER = '/home/noam/Documents/MEGdata/yoni'
REMOTE_FILES_FOLDER = os.path.join(FOLDER_LAB,'svmFiles')

FOLDER = FOLDER_LAB if utils.folderExists(FOLDER_LAB) else FOLDER_HOME \
    if utils.folderExists(FOLDER_HOME) else FOLDER_SERVER
SUBJECTS = ['1']  # '2',
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
    sigSectionMinLengths = np.linspace(1, 20, 20)  # [1, 2, 3]
    sigSectionAlphas = [0.05, 0.1, 0.2, 0.3]
    onlyMidValueOptions = [True, False]
    kernels = ['rbf', 'linear']
    FOLDS = 5
    TEST_SIZE = 0.2
    JOBS = cpuNum
    SUBJECT = '1'

    for procID in [AnalyzerYoni.PROC_1_2, AnalyzerYoni.PROC_2_4,
                   AnalyzerYoni.PROC_1_4]:
        MATLAB_FILE = 'dataForML.mat'
        print('procID: {}'.format(AnalyzerYoni.PROCS_NAMES[procID]))
        tablesUtils.DEF_TABLES = False
        analyze = AnalyzerYoni(FOLDER, MATLAB_FILE, SUBJECT,
            procID=procID, multipleByWeights=False,
            jobsNum=JOBS, weights=SAM_WEIGHTS_FILE)
#         analyze.preProcess(parallel=False, verbose=True)
#         analyze.splitData(heldoutSize=0)
# #         analyze.shuffleLabels()
#         analyze.process(foldsNum=FOLDS, Cs=Cs,
#             gammas=gammas, kernels=kernels, testSize=TEST_SIZE,
#             sigSectionMinLengths=sigSectionMinLengths,
#             sigSectionAlphas=sigSectionAlphas,
#             onlyMidValueOptions=onlyMidValueOptions)
#         analyze.calculatePredictionsScores()
#         analyze.analyzeResults()
#         analyze.calcImporatances(FOLDS, TEST_SIZE, doCalc=True,
#             doShow=False, permutationNum=2000, permutationLen=5)
        analyze.fullDataAnlysis(permutationNum=0, permutationLen=5,
            doPlotSections=True, doPrintBestSections=False,
            maxSurpriseVal=12, plotDataForGivenRange=False)

    howMuchTime = utils.howMuchTimeFromTic(t)
    utils.sendResultsEmail('analyzer is done! {}'.format(howMuchTime),
        RESULTS_FILE_NAME)
    if (utils.sftp):
        utils.sftp.close()
