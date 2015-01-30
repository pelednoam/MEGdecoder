'''
Created on Jun 1, 2014

@author: noampeled
'''

from src.commons.analyzer.analyzer import Analyzer
from src.commons.analyzer.analyzerSelector import AnalyzerSelector
from src.commons.utils import (mpHelper, MLUtils, utils, freqsUtils as fu, tablesUtils as tabu)
from collections import namedtuple
import itertools
import os
from sklearn.datasets.base import Bunch


class AnalyzerFreqsSelector(AnalyzerSelector):

    def getXY(self, stepID, p):
        if (stepID != self.STEP_PRE_PROCCESS):
            return super(AnalyzerFreqsSelector, self).getXY(stepID)
        else:
            minFreq, maxFreq = min(p['minFreqs']), max(p['maxFreqs'])
            if (tabu.DEF_TABLES and tabu.isTableInGroup(self.hdfGroup, self.xTableName)):
                pss = tabu.findTableInGroup(self.hdfGroup, self.xTableName)
                print('pss matrix exists in the hdf file ({})'.format(pss.shape))
                y = tabu.findTableInGroup(self.hdfGroup, 'y')
                trialsInfo = tabu.findTableInGroup(self.hdfGroup, 'trialsInfo')
                if (not utils.fileExists(self.freqsFileName)):
                    timeStep = self.calcTimeStep(trialsInfo)
                    x, y, trialsInfo = super(AnalyzerFreqsSelector,
                        self).getXY(stepID)
                    freqs = fu.calcMaxFreqs(x, minFreq, maxFreq, timeStep)
                    utils.save(freqs, self.freqsFileName)
                else:
                    freqs = utils.load(self.freqsFileName)
            elif (utils.fileExists(self.pssFileName)):
                print('loading {}'.format(self.pssFileName))
                pss, y, trialsInfo = utils.load(self.pssFileName)
                freqs = utils.load(self.freqsFileName)
            else:
                x, y, trialsInfo = super(AnalyzerFreqsSelector,
                    self).getXY(stepID)
                timeStep = self.calcTimeStep(trialsInfo)
                pss, freqs = fu.calcPSX(x, minFreq, maxFreq, timeStep,
                    hdfFile=self.hdfFile, hdfGroup=self.hdfGroup)
                if (self.normalizeData):
                    print('normalize pss')
                    pss = self.normalizeFeatures(pss, trialsInfo, self.normalizeDataField)
                print('save pss and freqs')
                if (not tabu.DEF_TABLES):
                    utils.save((pss, y, trialsInfo), self.pssFileName)
                else:
                    # If the trials have different lengths (self.variesT), the
                    # raw data is saved in analyzer.preProcess in a npz file,
                    # not hdf. So now, when tabu.DEF_TABLES set to True, the
                    # pss can be saved in an hdf file, so we need also to save
                    # y and trialsInfo in the same hdf file.
                    if (not tabu.isTableInGroup(self.hdfGroup, 'y')):
                        tabu.createHDF5ArrTable(self.hdfFile, self.hdfGroup,
                            'y', arr=y)
                        trialsInfoTab = tabu.createHDFTable(self.hdfFile,
                            self.hdfGroup, 'trialsInfo', self.trialsInfoDesc)
                        self.createEmptyTrialInfoTable(trialsInfoTab, len(y))
                        for k, trialInfo in enumerate(trialsInfo):
                            self.setTrialInfoRecord(trialsInfoTab, k, trialInfo)
                    # Close and open in read mode, from here we don't need 
                    # to do any changes in the hdf file
                    self.hdfFile.close()
                    self.hdfFile = tabu.openHDF5File(self.hdf5FileName, 'r')
                    pss = tabu.findTableInGroup(self.hdfGroup, self.xTableName)
                    y = tabu.findTableInGroup(self.hdfGroup, 'y')
                    trialsInfo = tabu.findTableInGroup(self.hdfGroup, 'trialsInfo')

                utils.save(freqs, self.freqsFileName)

            self.xAxis = freqs
            return pss, y, trialsInfo

    def _prepareCVParams(self, p):
        p = p.merge(p.kwargs)
        params = []
        cv = list(p.cv)
        paramsNum = len(cv)
        index = 0
        x = mpHelper.ForkedData(p.x) if not tabu.DEF_TABLES else None
        for fold, (trainIndex, testIndex) in enumerate(cv):
            if (not MLUtils.isBinaryProblem(p.y, trainIndex, testIndex)):
                print('fold {} is not binary, continue'.format(fold))
                continue
            y, shuffleIndices = self.permutateTheLabels(p.y, trainIndex,
                self.useSmote) if self.shuffleLabels else (p.y[:], None)
            subject = p.cv.usubjects[fold] if self.leaveOneSubjectOutFolds \
                else fold
            params.append(Bunch(
                x=x, y=y, subject=subject, windowIndices=range(len(self.freqs)),
                trainIndex=trainIndex, testIndex=testIndex,
                fold=fold, paramsNum=paramsNum,
                sigSectionMinLengths=p.sigSectionMinLengths,
                sigSectionAlphas=p.sigSectionAlphas, minFreqs=p.minFreqs,
                maxFreqs=p.maxFreqs, index=index,
                onlyMidValueOptions=p.onlyMidValueOptions,
                kernels=p.kernels, Cs=p.Cs, gammas=p.gammas,
                shuffleIndices=shuffleIndices,
                foldsNum=p.foldsNum, testSize=p.testSize,
                trialsInfo=p.trialsInfo[:]))
            index += 1
        print('{} records!'.format(index))
        return params

    def externalParams(self, p):
        return Bunch(fold=p.fold, subject=p.subject)

    def parametersGenerator(self, p):
        for hp in itertools.product(*(p.minFreqs, p.maxFreqs,
            p.onlyMidValueOptions, p.sigSectionMinLengths,
            p.sigSectionAlphas)):
                yield self.createParamsObj(hp)

    def createParamsObj(self, paramsTuple):
        (minFreq, maxFreq, onlyMidValue, sigSectionMinLength,
         sigSectionAlpha) = paramsTuple
        return Bunch(minFreq=minFreq, maxFreq=maxFreq,
            onlyMidValue=onlyMidValue, sigSectionMinLength=sigSectionMinLength,
            sigSectionAlpha=sigSectionAlpha)

#     def selectorFactory(self, timeStep, p, params=None, maxSurpriseVal=20,
#                         doPlotSections=False):
#         if (params is None):
#             params = p
#         return FrequenciesSelector(timeStep, p.sigSectionAlpha,
#             p.sigSectionMinLength, params.minFreq, params.maxFreq,
#             params.onlyMidValue, maxSurpriseVal,
#             self.LABELS[self.procID], doPlotSections)

    def resultItem(self, selector, p, res, params, ytest):
        return Bunch(predResults=res, fold=p.fold, ytest=ytest,
            featureExtractorName=Analyzer.FE_ALL,
            sections=selector.sections, minFreq=params.minFreq,
            maxFreq=params.maxFreq, onlyMidValue=params.onlyMidValue,
            sigSectionMinLength=p.sigSectionMinLength,
            sigSectionAlpha=p.sigSectionAlpha)

    def _predictorParamtersKeyClass(self):
        return namedtuple('predictorParamters', ['sigSectionMinLength',
            'sigSectionAlpha', 'minFreq', 'maxFreq', 'onlyMidValue', 'kernel',
            'C', 'gamma'])

    def predictorParamtersKeyItem(self, res, predRes):
        PredictorParamtersKey = self._predictorParamtersKeyClass()
        return PredictorParamtersKey(
            sigSectionMinLength=res.sigSectionMinLength,
            sigSectionAlpha=res.sigSectionAlpha,
            minFreq=res.minFreq, maxFreq=res.maxFreq,
            onlyMidValue=res.onlyMidValue,
            kernel=predRes.kernel, C=predRes.C, gamma=predRes.gamma)

    def printBestPredictorResults(self, bestEstimator, printSections=True):
        bep = bestEstimator.parameters
        print('Best results: ' + \
            'sigSectionMinLength: {}, '.format(bep.sigSectionMinLength) + \
            'sigSectionAlpha:{}, '.format(bep.sigSectionAlpha,) + \
            'minFreqs: {}, maxFreqs: {}, '.format(bep.minFreq, bep.maxFreq) + \
            'onlyMidValueOptions:{}, '.format(bep.onlyMidValue) + \
            'kernel: {}, c: {}, gamma: {}'.format(
            bep.kernel, bep.C, bep.gamma))

    def calcFeaturesSpace(self, X, channels, bep):
        T = X.shape[1]
        timeStep = self.calcTimeStep(T)
        _, pss = fu.calcAllPS(X,
            timeStep, bep.minFreq, bep.maxFreq, channels)
        return pss

    def featuresAxis(self, selector=None):
        return selector.freqs

    @property
    def featuresAxisLabel(self):
        return 'Frequency (Hz)'

    @property
    def selectorName(self):
        return 'FrequenciesSelector'

    @property
    def xTableName(self):
        return 'pss'

