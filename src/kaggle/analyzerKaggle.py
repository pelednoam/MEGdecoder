'''
Created on Nov 25, 2013

@author: noampeled
'''

from src.commons.analyzer.analyzer import Analyzer
from src.commons.analyzer.analyzerTimeSelector import AnalyzerTimeSelector
from src.commons.utils import utils
import src.commons.crossValidations as cv
from src.commons import GridSearchSteps as GSS
from sklearn.datasets.base import Bunch

import csv
import numpy as np


class AnalyzerKaggle(AnalyzerTimeSelector):
    PROCS_NAMES = ['ScramOrFace']
    PROC_FACES = 0
    LABELS = [['Scramble','Face']]
    
    TOTAL_TIME = 1500.0 #ms
    ONSET = 500.0 #ms
    BASELINE = 50 # time stamps
    T = 375 # bins
    
    def __init__(self, *args, **kwargs):
        kwargs['indetifier']='kaggle'
        super(AnalyzerKaggle, self).__init__(*args, **kwargs)

    def loadData(self):
        matlabFiles = {}
        for matlabFile in utils.filesInFolder(self.folder, 'train_subject*.mat'):
            matlabDic = utils.loadMatlab(matlabFile)
            subject = matlabFile[-6:-4]
            matlabFiles[subject] = matlabDic
        return matlabFiles

    def getTrialsTimeLength(self,matlabDic):
        return matlabDic['X'].shape[2]

    def loadTimeAxis(self, T=None):
        # Load the time axis     
        timeAxis = np.linspace(0, AnalyzerKaggle.TOTAL_TIME, T)
        np.save(self.timeAxisFileName,timeAxis)

    def dataGenerator(self, matlabFiles):
        for subject, matlabDic in matlabFiles.iteritems():
            (trials,labels) = (matlabDic['X'],matlabDic['y'])
            onsetInd = self.getOnsetInd(trials)
            for trial, label in zip(trials,labels):
#                 C,T = x.shape
                trial = Analyzer.baselineCorrection(trial,AnalyzerKaggle.BASELINE)
                yield ((trial[:,onsetInd:].T, label[0]), {'subject':subject})

    def calcTimeStep(self,T):
        ''' return time step in sec '''
        return (AnalyzerKaggle.TOTAL_TIME / T) / 1000 

    def getOnsetInd(self,x):
        T = x.shape[2]
        freq = T/AnalyzerKaggle.TOTAL_TIME
        onsetInd = int(AnalyzerKaggle.ONSET*freq)
        return onsetInd

    def trialCond(self,label,trialInfo):
        return True
    
    def trialLabel(self,label,trialInfo):
        return label

    def preparePredictionParamtersLinkFunction(self):
        return self._preparePredictionsParametersChannelsFrequenciesSelector

    def prepareCVParamsLinkFunction(self,p): 
        return self._prepareCVParamsForChannelsFrequenciesSelector(p)

    def channelsSelector(self, channelsNum):
        return GSS.ChannelsSelector2(channelsNum)

    def featuresCV(self,y,trialsInfo,foldsNum):
        subjects = [trialInfo['subject'] for trialInfo in trialsInfo]
#         return cv.TestCV()
        return cv.SubjectsKFold(subjects, foldsNum)
#         train_set = [['01','02'],['03','04'],['05','06'],['07','08'],['09','10'],['11','12'],['13','14'],['15','16']]
#         test_set = [['03','04'],['01','02'],['07','08'],['05','06'],['11','12'],['09','10'],['15','16'],['13','14']]        
#         train_set = [['01','02','03','04'],['05','06','07','08'],['09','10','11','12'],['13','14','15','16']]
#         test_set = [['05','06','07','08'],['01','02','03','04'],['13','14','15','16'],['09','10','11','12']]        
#         return SubjectsPreset(subjects, train_set, test_set)

    def loadHeldoutDataGenerator(self):
        trialsInfo_heldout = []
        for matlabFile in utils.filesInFolder(self.folder, 'test_subject*.mat'):
            matlabDic = utils.loadMatlab(matlabFile)
            x_heldout = matlabDic['X']
            onsetInd = self.getOnsetInd(x_heldout)
            x_heldout = x_heldout[:,:,onsetInd:]
            x_heldout = Analyzer.transposeTimeAndChannels(x_heldout)
            subject = matlabFile[-6:-4]
            for _ in range(matlabDic['X'].shape[0]):
                trialsInfo_heldout.append({'subject':subject})
            yield x_heldout, None, trialsInfo_heldout, subject
    
    def heldoutDataInfoGenerator(self):
        subjects = sorted([matlabFile[-6:-4] for matlabFile in utils.filesInFolder(self.folder, 'test_subject*.mat')])
        for subject in subjects:
            yield subject
    
    def createHeldoutPredictionReport(self):
        bestEstimators = utils.load(self.bestEstimatorsFileName)
        for featureExtractorName in bestEstimators.keys():
            with open(self.heldoutPredictionReportFileName(featureExtractorName), 'w') as output_file:
                file_writer = csv.writer(output_file,delimiter=',')
                file_writer.writerow(['Id','Prediction'])
                for subjectID in self.heldoutDataInfoGenerator():
                    trialnum = 0
                    heldoutPrediction = utils.load(self.heldoutPredictionFileName(featureExtractorName,subjectID))
                    for pred in heldoutPrediction.ypred:
                        file_writer.writerow(['{}{}'.format(subjectID,str(trialnum).zfill(3)), pred])
                        trialnum += 1

    def normalizeFeatures(self, x, trialsInfo, field):
        return super(AnalyzerKaggle, self).normalizeFeatures(x, trialsInfo, 'subject')

    def heldoutPredictionReportFileName(self,featureExtractorName):
        return '{}/submission_{}.xls'.format(self.dataFolder,featureExtractorName)


