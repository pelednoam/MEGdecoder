# -*- coding: utf-8 -*-

'''
Created on Nov 19, 2013

@author: noampeled
'''
import numpy as np
import sys
import os
from path3 import path
import traceback
from src.centipede.analyzerCentipede import (AnalyzerCentipede,
    AnalyzerCentipedeSuper, AnalyzerCentipedeTimeSWFreqs,
    AnalyzerCentipedeSpacialSWFreqs, AnalyzerCentipedeFreqsSW)
from src.commons.utils import tablesUtils
from src.commons.utils import utils
from src.commons.utils import plots

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FOLDER_LAB = '/Users/noampeled/Copy/Data/MEG/data'
FOLDER_HOME = '/home/noam/Documents/MEGdata/centipede/data'
FOLDER_OHAD = FOLDER_HOME
LOCAL_TEMP_FOLDER = '/home/noam/Documents/MEGdata/centipede/svmFiles/temp'

FOLDER = FOLDER_LAB if utils.folderExists(FOLDER_LAB) else FOLDER_HOME if \
    utils.folderExists(FOLDER_HOME) else ''
if (FOLDER == ''):
    utils.throwException('No data folder!')

# SUBJECTS = ['2','3','4','8','11','15','17','18','19']
SUBJECTS = ['darya', 'dor', 'eitan', 'idan', 'liron', 'mosheB', 'raniB',
            'shira', 'TalR2', 'yoni', 'oshrit', 'ohad', 'talia']
RESULTS_FILE_NAME = path.join(path(__file__).parent.parent.parent,
    'output.txt')

FTP_NOAM, FTP_OHAD, FTP_OHAD_FROM_LAB = range(3)
FTP_SERVER = FTP_OHAD_FROM_LAB
PORT = 22 if FTP_SERVER == FTP_OHAD_FROM_LAB else 9991
FTP_USER = 'noam' if FTP_SERVER in [FTP_OHAD, FTP_OHAD_FROM_LAB] else 'noampeled'
FTP_FOLDER = os.path.join(FOLDER_LAB, 'svmFiles') if FTP_SERVER == FTP_NOAM \
    else os.path.join(FOLDER_OHAD, 'svmFiles')
FTP_PASSWORD = '363679`0' if FTP_SERVER in [FTP_OHAD, FTP_OHAD_FROM_LAB] else '363679§0'
HOST = '132.71.85.32' if FTP_SERVER == FTP_OHAD_FROM_LAB else '127.0.0.1'
utils.sftpInit(HOST, FTP_USER, port=PORT, remoteFolder=FTP_FOLDER,
               password=FTP_PASSWORD)
utils.DUMPER_FOLDER = utils.createDirectory(os.path.join(FOLDER, 'dumper'))

DATA_FILE = 'dataForDecoding.mat'
WEIGHTS_FILE = 'pointsWeights.mat'

PLOT_FOR_PUBLICATION = True

Cs = np.logspace(-3, 3, 7)
gammas = np.logspace(-9, 0, 5)
kernels = ['rbf', 'linear']
sigSectionMinLengths = [1, 2, 3]
sigSectionAlphas = [0.05, 0.1, 0.2, 0.3]
minFreqs = [0]
maxFreqs = [80]
onlyMidValueOptions = [True]
FOLDS = 15
TEST_SIZE = 0.5
PROC_ID = AnalyzerCentipedeSuper.PROC_LEAVE_STAY_6_10
tablesUtils.DEF_TABLES = False


def readData(shuffleLabelsOptions=[True, False], useSmote=False):
    subjects = SUBJECTS

    for subject in subjects:  # SUBJECTS
        for shuffleLabels in shuffleLabelsOptions:
            try:
                analyze = AnalyzerCentipede(FOLDER, DATA_FILE, subject,
                    procID=PROC_ID, jobsNum=JOBS,
                    shuffleLabels=shuffleLabels, useSmote=useSmote)
                # analyze.preProcess()
                # analyze.process(foldsNum=FOLDS, Cs=Cs, gammas=gammas,
                #     kernels=kernels, testSize=TEST_SIZE,
                #     sigSectionMinLengths=sigSectionMinLengths,
                #     sigSectionAlphas=sigSectionAlphas,
                #     minFreqs=minFreqs, maxFreqs=maxFreqs,
                #     onlyMidValueOptions=onlyMidValueOptions)
#                 analyze.getBestEstimators(getRemoteFiles=False)
#                 analyze.analyzeResults(doPlot=True)

            except:
                print('error with subject {}'.format(subject))
                # print traceback.format_exc()

    for subject in subjects:
        try:
            analyze = AnalyzerCentipede(FOLDER, DATA_FILE, subject,
                procID=PROC_ID, jobsNum=JOBS)
            analyze.findSignificantResults(doShow=False)
        except:
            print('error with subject {}'.format(subject))
            print traceback.format_exc()


def readDataSpacialSW():
    xCubeSizes = [5]
    windowsOverlapped = [True]
    FOLDS = 3

    for subject in ['dor']:
        try:
            analyze = AnalyzerCentipedeSpacialSWFreqs(
                FOLDER, DATA_FILE, subject,
                procID=PROC_ID, multipleByWeights=False,
                weightsFileName=WEIGHTS_FILE, jobsNum=JOBS)
#             analyze.preProcess(False, parallel=True)
#             analyze.splitData(heldoutSize=0)
#             analyze.shuffleLabels()
#             analyze.process(foldsNum=FOLDS, Cs=Cs, gammas=gammas,
#                 kernels=kernels, testSize=TEST_SIZE,
#                 sigSectionMinLengths=sigSectionMinLengths,
#                 sigSectionAlphas=sigSectionAlphas,
#                 minFreqs=minFreqs, maxFreqs=maxFreqs,
#                 onlyMidValueOptions=onlyMidValueOptions,
#                 xCubeSizes=xCubeSizes, windowsOverlapped=windowsOverlapped)
#             analyze.calculatePredictionsScores()
            analyze.analyzeResults(threshold=0.5)
#                 analyze.fullDataAnlysis(permutationNum=500,
#                     permutationLen=50, doPlotSections=False,
#                     doPrintBestSections=False,
#                     maxSurpriseVal=12, plotDataForGivenRange=False)
#                 analyze.calcHeldOutPrediction()

        except:
            print('error with subject {}'.format(subject))
            print traceback.format_exc()


def readDataTimeSlidingWindow(shuffleLabelsOptions=[False, True], groupAnalyze=False):
    windowSizes = [500]
    windowsNums = [50]
    results = {}
    resultsShuffle = {}
    subjects = ['raniB', 'shira', 'TalR2', 'yoni', 'oshrit', 'ohad', 'talia']

    for subject in ['raniB']:
        for shuffleLabels in [True]: # shuffleLabelsOptions:
            try:
                analyze = AnalyzerCentipedeTimeSWFreqs(FOLDER, DATA_FILE,
                    subject, procID=PROC_ID, jobsNum=JOBS,
                    shuffleLabels=shuffleLabels)
                # analyze.preProcess()
#                 analyze.process(foldsNum=FOLDS, testSize=TEST_SIZE,
#                     n_jobs=JOBS, sigSectionMinLengths=sigSectionMinLengths,
#                     sigSectionAlphas=sigSectionAlphas,
#                     minFreqs=minFreqs, maxFreqs=maxFreqs,
#                     onlyMidValueOptions=onlyMidValueOptions,
#                     windowSizes=windowSizes, windowsNums=windowsNums,
#                     kernels=kernels, Cs=Cs, gammas=gammas,
#                     useSmote=True)
                analyze.getBestEstimators(getRemoteFiles=False)
#                 results[subject] = analyze.analyzeResults(windowSizes=[500],
#                     doPlot=False, doSmooth=False)
            except:
                print('error with subject {}'.format(subject))
                print traceback.format_exc()

    for subject in ['raniB']:
        try:
            analyze = AnalyzerCentipedeTimeSWFreqs(FOLDER, DATA_FILE,
                subject, procID=PROC_ID, jobsNum=JOBS)
            results[subject] = analyze.findSignificantResults(
                FOLDS, doPlot=True, overwrite=False, windowSizes=[500])
        except:
            print('error with subject {}'.format(subject))
            print traceback.format_exc()

    if (groupAnalyze):
        groupAnalysis(results)


def readDataFreqsTimeSlidingWindow(shuffleLabels):
    freqsWindowSizes = [5]
    freqsWindowsNums = [50]
    timeWindowSizes = [500]
    timeWindowsNums = [50]

    for subject in set(SUBJECTS):  # - set(['darya']):
        try:
            analyze = AnalyzerCentipedeFreqsSW(FOLDER, DATA_FILE,
                subject, procID=PROC_ID, jobsNum=JOBS)
#             analyze.preProcess()
#             analyze.splitData(heldoutSize=0)
            analyze.process(foldsNum=FOLDS, testSize=TEST_SIZE,
                kernels=kernels, n_jobs=JOBS, Cs=Cs, gammas=gammas,
                sigSectionMinLengths=sigSectionMinLengths,
                sigSectionAlphas=sigSectionAlphas,
                minFreqs=minFreqs, maxFreqs=maxFreqs,
                onlyMidValueOptions=onlyMidValueOptions,
                freqsWindowSizes=freqsWindowSizes,
                freqsWindowsNums=freqsWindowsNums,
                timeWindowSizes=timeWindowSizes,
                timeWindowsNums=timeWindowsNums)
#             analyze.analyzeResults(freqsSliderRange=(minFreqs[0], maxFreqs[0]),
#                                    plotPerAccFunc=False, doPlot=True)
#             analyze.fullDataAnlysis(permutationNum=500,
#                 permutationLen=50, doPlotSections=False,
#                 doPrintBestSections=False,
#                 maxSurpriseVal=12, plotDataForGivenRange=False)
#             analyze.calcHeldOutPrediction()
        except:
            print('error with subject {}'.format(subject))
            print traceback.format_exc()


def readDataFreqsSlidingWindow(shuffleLabelsOptions=[False, True], groupAnalyze=False):
    windowSizes = [5]
    windowsNums = [50]
    results = {}
    subjects = SUBJECTS

    for subject in subjects:
        for shuffleLabels in shuffleLabelsOptions:
            try:
                analyze = AnalyzerCentipedeFreqsSW(FOLDER, DATA_FILE,
                    subject, procID=PROC_ID, jobsNum=JOBS,
                    shuffleLabels=shuffleLabels)
                print ('readDataFreqsSlidingWindow {}'.format(analyze.defaultFileNameBase))
    #             analyze.preProcess()
    #             analyze.splitData(heldoutSize=0)
                analyze.process(foldsNum=FOLDS, testSize=TEST_SIZE,
                    kernels=kernels, n_jobs=JOBS, Cs=Cs, gammas=gammas,
                    sigSectionMinLengths=sigSectionMinLengths,
                    sigSectionAlphas=sigSectionAlphas,
                    minFreqs=minFreqs, maxFreqs=maxFreqs,
                    onlyMidValueOptions=onlyMidValueOptions,
                    windowSizes=windowSizes, windowsNums=windowsNums)
                analyze.getBestEstimators()
    #             results[subject] = analyze.analyzeResults(
    #                 freqsSliderRange=(minFreqs[0], maxFreqs[0]),
    #                 plotPerAccFunc=False, doSmooth=False,
    #                 doPlot=False)
            except:
                print('error with subject {}'.format(subject))
                print traceback.format_exc()

    resultsShuffle = None
    if (groupAnalyze):
        groupAnalysis(results, resultsShuffle)


def groupAnalysis(results):
    subjects = results.keys()
    xaxis = results[subjects[0]][1]

#     gmeanDiff = [abs(results[s][0]['gmean'] - resultsShuffle[s][0]['gmean']) for s in resultsShuffle.keys()]
#     aucsDiff = [abs(results[s][0]['auc'] - resultsShuffle[s][0]['auc']) for s in resultsShuffle.keys()]
#     plots.graphN(xaxis, gmeanDiff, subjects, xlabel='Time (ms)', ylabel='group accuracy')
#     plots.graphN(xaxis, aucsDiff, subjects, xlabel='Time (ms)', ylabel='group accuracy')

    # for s in subjects:
    #     plots.graph2(xaxis, results[s][0]['gmean'], resultsShuffle[s][0]['gmean'], ['gmean', 'gmean shuffle'],
    #                  xlabel='Time (ms)', ylabel='accuracy', title=s)
    #     plots.graph2(xaxis, results[s][0]['auc'], resultsShuffle[s][0]['auc'], ['auc', 'auc shuffle'],
    #                  xlabel='Time (ms)', ylabel='accuracy', title=s)

    gmeans = np.array([res[0]['gmean'] for res in results.values()])
    aucs = np.array([res[0]['auc'] for res in results.values()])

    aucsCor = np.corrcoef(aucs)
    plots.plotHierarchicalClusteringOnTopOfDistancesMatrix(aucsCor)
    gmeansCor = np.corrcoef(gmeans)
    plots.plotHierarchicalClusteringOnTopOfDistancesMatrix(gmeansCor)
#     T = spectral(gmeansCor, n_clusters=4, eigen_solver=None,
#                  assign_labels='kmeans')

    plots.graphN(xaxis, gmeans, subjects, xlabel='Time (ms)', ylabel='group accuracy')
    plots.graphN(xaxis, aucs, subjects, xlabel='Time (ms)', ylabel='group accuracy')
    plots.graph2(xaxis, np.mean(gmeans, 0), np.mean(aucs, 0), ['gmean', 'auc'],
        yerrs=(np.std(gmeans, 0), np.std(aucs, 0)), xlabel='Time (ms)', ylabel='group accuracy')


if __name__ == '__main__':
    args = sys.argv[1:]
    JOBS = 1 if (len(args) < 1) else int(args[0])
    if (JOBS > 1):
        print('cpuNum = %d' % JOBS)

    t = utils.ticToc()
#     readData()
#     readDataSpacialSW()
    readDataTimeSlidingWindow()
#     readDataFreqsSlidingWindow()
    howMuchTime = utils.howMuchTimeFromTic(t)
    utils.sendResultsEmail('analyzer is done! {}'.format(howMuchTime),
        RESULTS_FILE_NAME)
    if (utils.sftp):
        utils.sftp.close()
