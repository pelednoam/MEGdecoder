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


from src.kaggle.analyzerKaggle import AnalyzerKaggle
from src.commons.analyzer.analyzer import Analyzer
from src.commons.utils import utils

import warnings
warnings.filterwarnings("ignore")

FOLDER_LAB = '/Users/noampeled/Copy/Data/MEG/kaggle'
FOLDER_LAB2 = '/home/noam/kaggle'
FOLDER_HOME = '/home/noam/Documents/MEGdata/kaggle'

LOCAL_TEMP_FOLDER = '/Users/noampeled/Documents/MEG_SVM_TEMP_FILES/Kaggle'
REMOTE_FILES_FOLDER = os.path.join(FOLDER_LAB,'svmFiles')
REMOTE_FILES_FOLDER_OHAD = os.path.join(FOLDER_LAB2,'svmFiles')

FOLDER = [folder for folder in [FOLDER_LAB,FOLDER_LAB2,FOLDER_HOME] if utils.folderExists(folder)][0]# FOLDER_LAB if utils.folderExists(FOLDER_LAB) else FOLDER_HOME
# SUBJECTS = ['2','3','4','8','11','15','17','18','19']
SUBJECT = ''
RESULTS_FILE_NAME = path.join(path(__file__).parent.parent.parent,'output.txt')
 
# utils.sftpInit('127.0.0.1', 'noampeled', port=9998, remoteFolder=REMOTE_FILES_FOLDER)
utils.sftpInit('127.0.0.1', 'noam', '363679`0', port=9997, remoteFolder=REMOTE_FILES_FOLDER_OHAD)

if __name__ == '__main__':
    args = sys.argv[1:]
    cpuNum = 1 if (len(args) < 1) else int(args[0])
    if (cpuNum > 1): print('cpuNum = %d' % cpuNum)
    
    t = utils.ticToc()
    Cs = np.logspace(-1,1,3) # [100]
    percentiles= [20,40,60,80,100] # [60]
    gammas= [0] # np.logspace(-10, 0, 11)
    channelsNums= [50,70,90,110,130] #[10,20,30,40,50,60,70,80,90,100]
#     sigSectionMinLengths = [3,5,10,15]
#     sigSectionAlphas = [0.01,0.05,0.1,0.15]

    sigSectionMinLengths = [1,2,3]
    sigSectionAlphas = [0.05,0.1,0.2,0.3]    
    minFreqs = [0,5,10]
    maxFreqs = [40]
    onlyMidValueOptions = [True,False]

    
    windowSizes= [100] #[40,60,80,100,120] 
    windowsNum=50 #100 
    kernels = ['rbf'] #['linear', 'poly', 'rbf']
    featureExtractors = [Analyzer.FE_ALL] #, Analyzer.FE_RMS, Analyzer.FE_COR, Analyzer.FE_COV]
    FOLDS = 5
    JOBS = cpuNum

    try:    
        MATLAB_FILE = 'train_subjects.mat'
        analyze = AnalyzerKaggle(FOLDER, MATLAB_FILE, SUBJECT, procID=AnalyzerKaggle.PROC_FACES, normalizationField='subject')
    #         analyze.preProcess(False,saveTimeAxis=False)
    #         analyze.splitData(heldoutSize=0)
    #         analyze.shuffleLabels()
    
#         analyze.preparePredictionsParamters(foldsNum=FOLDS,timePercentiles=percentiles, Cs=Cs, gammas=gammas,
#                                             channelsNums=channelsNums,kernels=kernels, featureExtractors=featureExtractors,
#                                             sigSectionMinLengths=sigSectionMinLengths, sigSectionAlphas=sigSectionAlphas,
#                                             minFreqs=minFreqs, maxFreqs=maxFreqs, onlyMidValueOptions=onlyMidValueOptions, 
#                                             useFeaturesGenerator=False, n_jobs=JOBS)
        analyze.calculatePredictionsScores()
        analyze.analyzeResults()    
        analyze.calcHeldOutPrediction()
        analyze.createHeldoutPredictionReport()
    
    #     analyze.plotAveragedChannles()
    #     analyze.saveTimeAxis(AnalyzerKaggle.T)
    except: 
        traceback.print_exc()
    
    howMuchTime = utils.howMuchTimeFromTic(t)
    utils.sendResultsEmail('analyzer is done! {}'.format(howMuchTime), RESULTS_FILE_NAME)        
    if (utils.sftp): utils.sftp.close()
