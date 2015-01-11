'''
Created on Jun 1, 2014

@author: noampeled
'''

from src.commons.analyzer.analyzer import Analyzer
from src.commons.analyzer.analyzerSelector import AnalyzerSelector
from src.commons.selectors.timeSelector import TimeSelector
from src.commons.utils import mpHelper
from src.commons.utils import utils
from src.commons.utils import tablesUtils as tabu

from collections import namedtuple
import itertools
from sklearn.datasets.base import Bunch


class AnalyzerTimeSelector(AnalyzerSelector):

    def _prepareCVParams(self, p):
        p = p.merge(p.kwargs)
        params = []
        allCTSParams = list(itertools.product(*(p.sigSectionMinLengths,
            p.sigSectionAlphas)))
        paramsNum = len(list(p.cv)) * len(allCTSParams)
        index = 0
        x = None if tabu.DEF_TABLES else mpHelper.ForkedData(p.x)
        for fold, (trainIndex, testIndex) in enumerate(p.cv):
            shuffleIndices = None
            if (self.shuffleLabels):
                p.y, shuffleIndices = self.permutateTheLabels(p.y, trainIndex,
                    self.useSmote)
            params.append(Bunch(
                x=x, y=p.y, trainIndex=trainIndex, testIndex=testIndex,
                trialInfo=p.trialsInfo, fold=fold, paramsNum=paramsNum,
                sigSectionMinLengths=p.sigSectionMinLengths,
                sigSectionAlphas=p.sigSectionAlphas, index=index,
                onlyMidValueOptions=p.onlyMidValueOptions,
                kernels=p.kernels, Cs=p.Cs, gammas=p.gammas,
                shuffleIndices=shuffleIndices))
            index += 1
        return params

    def parametersGenerator(self, p):
        return p.onlyMidValueOptions

    def selectorFactory(self, timeStep, p, params=None, maxSurpriseVal=20,
                        doPlotSections=False):
        verbose = p.get('verbose', True)
        if (params is None):
            params = p
        utils.log('ss alpha: {}, ss len: {}'.format(
            p.sigSectionAlpha, p.sigSectionMinLength), verbose)
        return TimeSelector(p.sigSectionAlpha, p.sigSectionMinLength,
            params.onlyMidValue, self.xAxis, maxSurpriseVal,
            self.LABELS[self.procID], doPlotSections)

    def createParamsObj(self, paramsTuple):
        onlyMidValue = paramsTuple
        return Bunch(onlyMidValue=onlyMidValue)

    def resultItem(self, selector, p, res, params, ytest):
        return Bunch(predResults=res, fold=p.fold, ytest=ytest,
            featureExtractorName=Analyzer.FE_ALL,
            sections=selector.sections,
            onlyMidValue=params.onlyMidValue,
            sigSectionMinLength=p.sigSectionMinLength,
            sigSectionAlpha=p.sigSectionAlpha)

    def _predictorParamtersKeyClass(self):
        return namedtuple('predictorParamters',
            ['sigSectionMinLength', 'sigSectionAlpha', 'onlyMidValue',
             'kernel', 'C', 'gamma'])

    def predictorParamtersKeyItem(self, res, predRes):
        if ('onlyMidValue') not in res:
            res.onlyMidValue = True
        PredictorParamtersKey = self._predictorParamtersKeyClass()
        return PredictorParamtersKey(
            sigSectionMinLength=res.sigSectionMinLength,
            sigSectionAlpha=res.sigSectionAlpha,
            onlyMidValue=res.onlyMidValue,
            kernel=predRes.kernel, C=predRes.C, gamma=predRes.gamma)

    def printBestPredictorResults(self, bestEstimator):
        bep = bestEstimator.parameters
        print('Best results: sigSectionMinLength: {}, '.format(
            bep.sigSectionMinLength) + \
            'sigSectionAlpha:{},'.format(bep.sigSectionAlpha) + \
            'onlyMidValue:{}, kernel: {}, c: {}, gamma: {}'.format(
            bep.onlyMidValue, bep.kernel, bep.C, bep.gamma))

    def featuresAxis(self, selector=None):
        return self.xAxis

    @property
    def featuresAxisLabel(self):
        return 'Time (ms)'

#     def heldOutFeaturesExtraction(self,x,y,trialsInfo,bep):
#         print('TimeSelector')
#         channelsAndTimeSelector = GSS.TimeSelector(bep.sigSectionAlpha, bep.sigSectionMinLength, False)
#         xFeaturesTimed = channelsAndTimeSelector.fit_transform(x, y)
#         xFeaturesTimed = self.normalizeFeatures(xFeaturesTimed, trialsInfo, 'subject') 
#         return xFeaturesTimed, channelsAndTimeSelector

#     def heldOutFeaturesTransform(self, p, x_heldout, featureExtractorName):
#         xHeldoutFeaturesTimed = p.selector.transform(x_heldout)  
#         xHeldoutFeaturesTimedNormalized,_,_ = MLUtils.normalizeData(xHeldoutFeaturesTimed)    
#         return xHeldoutFeaturesTimedNormalized      
