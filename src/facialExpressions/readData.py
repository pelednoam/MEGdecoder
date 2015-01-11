'''
Created on Dec 8, 2014

@author: noampeled
'''

import numpy as np
import sys
import os
from path3 import path
import traceback
from src.facialExpressions.analyzerFE import AnalyzerFE, AnalyzerFESuper
from src.commons.utils import tablesUtils
from src.commons.utils import utils

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

RESULTS_FILE_NAME = path.join(path(__file__).parent.parent.parent,
    'output.txt')

FOLDER_LAB = '/Users/noampeled/Copy/Data/facialExpressions'
FOLDER_HOME = '/home/noam/Documents/facialExpressions'
FOLDER_OHAD = FOLDER_HOME

FOLDER = FOLDER_LAB if utils.folderExists(FOLDER_LAB) else FOLDER_HOME if \
    utils.folderExists(FOLDER_HOME) else ''
if (FOLDER == ''):
    utils.throwException('No data folder!')
utils.DUMPER_FOLDER = utils.createDirectory(os.path.join(FOLDER, 'dumper'))

SUBJECTS = range(22)


def readData():
    Cs = np.logspace(-3, 3, 7)
    gammas = np.logspace(-9, 0, 5)
    kernels = ['rbf', 'linear']
    sigSectionMinLengths = [1, 2, 3]
    sigSectionAlphas = [0.05, 0.1, 0.2, 0.3]
    minFreqs = [0]
    maxFreqs = [40]
    onlyMidValueOptions = [True]
    FOLDS = 5
    TEST_SIZE = 0.5
    JOBS = cpuNum
    procID = AnalyzerFESuper.PROC_LEAVE_STAY
    tablesUtils.DEF_TABLES = False

    try:
        analyze = AnalyzerFE(FOLDER, '', 'all',
            doLoadOriginalTimeAxis=False, variesT=True,
            procID=procID, jobsNum=JOBS)
#         analyze.preProcess()
        analyze.process(foldsNum=FOLDS, Cs=Cs, gammas=gammas,
            kernels=kernels, testSize=TEST_SIZE,
            sigSectionMinLengths=sigSectionMinLengths,
            sigSectionAlphas=sigSectionAlphas,
            minFreqs=minFreqs, maxFreqs=maxFreqs,
            onlyMidValueOptions=onlyMidValueOptions)
    except:
        print traceback.format_exc()

if __name__ == '__main__':
    args = sys.argv[1:]
    cpuNum = 1 if (len(args) < 1) else int(args[0])
    if (cpuNum > 1):
        print('cpuNum = %d' % cpuNum)

    t = utils.ticToc()
    readData()
    howMuchTime = utils.howMuchTimeFromTic(t)
    utils.sendResultsEmail('analyzer is done! {}'.format(howMuchTime),
        RESULTS_FILE_NAME)
    if (utils.sftp):
        utils.sftp.close()
