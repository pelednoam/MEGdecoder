'''
Created on Nov 20, 2014

@author: noampeled
'''

from src.commons.analyzer.analyzer import Analyzer
from src.commons.analyzerSpacialSlidingWindow.analyzerSpatialSWSelector import \
    AnalyzerSpacialSWSelector
from src.commons.selectors.frequenciesSelector import FrequenciesSelector
from src.commons.utils import mpHelper

from sklearn.datasets.base import Bunch
import itertools
from collections import namedtuple
from src.commons.utils import tablesUtils as tabu
from src.commons.utils import utils


class AnalyzerSpacialSWFreqsSelector(AnalyzerSpacialSWSelector):

    def _prepareCVParams(self, p):
        p = p.merge(p.kwargs)
        params = []
        if ('yCubeSizes' not in p):
            p.yCubeSizes = p.xCubeSizes
        if ('zCubeSizes' not in p):
            p.zCubeSizes = p.xCubeSizes

        if (not tabu.DEF_TABLES):
            weightsDict = utils.loadMatlab(
                self.weightsFullFileName(self.weightsFileName))
            weights = weightsDict[self.weightsDicKey]
        else:
            weights = None

        allCTSParams = list(itertools.product(*(p.sigSectionMinLengths,
            p.sigSectionAlphas)))
        paramsNum = len(list(p.cv)) * len(allCTSParams)
        xlim, ylim, zlim, xstep, ystep, zstep = self.getMetaParameters(p)
        index = 0
        for fold, (trainIndex, testIndex) in enumerate(p.cv):
            x = None if tabu.DEF_TABLES else mpHelper.ForkedData(p.x)
            y = None if tabu.DEF_TABLES else p.y
            trialsInfo = None if tabu.DEF_TABLES else p.trialsInfo
            for sigSectionMinLength, sigSectionAlpha in allCTSParams:
                params.append(Bunch(
                    x=x, y=y, trainIndex=trainIndex, testIndex=testIndex,
                    trialInfo=trialsInfo, fold=fold, weights=weights,
                    sigSectionMinLength=sigSectionMinLength,
                    sigSectionAlpha=sigSectionAlpha,
                    minFreqs=p.minFreqs, maxFreqs=p.maxFreqs,
                    onlyMidValueOptions=p.onlyMidValueOptions,
                    xlim=xlim, ylim=ylim, zlim=zlim,
                    xstep=xstep, ystep=ystep, zstep=zstep,
                    xCubeSizes=p.xCubeSizes, yCubeSizes=p.yCubeSizes,
                    zCubeSizes=p.zCubeSizes,
                    windowsOverlapped=p.windowsOverlapped,
                    kernels=p.kernels, Cs=p.Cs, gammas=p.gammas,
                    index=index, paramsNum=paramsNum))
                index += 1
        return params

    def getMetaParameters(self, p={}):
        weightMetaDic = utils.loadMatlab(self.weightsFullMetaFileName(
            self.weightsFileName))
        xlim = p.xlim if 'xlim' in p else weightMetaDic['xlim'][0]
        ylim = p.ylim if 'ylim' in p else weightMetaDic['ylim'][0]
        zlim = p.zlim if 'zlim' in p else weightMetaDic['zlim'][0]
        xstep = weightMetaDic['xstep'][0][0]
        ystep = weightMetaDic['ystep'][0][0] if 'ystep' in weightMetaDic \
            else xstep
        zstep = weightMetaDic['zstep'][0][0] if 'zstep' in weightMetaDic \
            else xstep
        utils.save(Bunch(xlim=xlim, ylim=ylim, zlim=zlim,
                   xstep=xstep, ystep=ystep, zstep=zstep),
                   self.metaParametersFileName)
        return xlim, ylim, zlim, xstep, ystep, zstep

    def parametersGenerator(self, p):
        return itertools.product(*(p.minFreqs, p.maxFreqs,
            p.onlyMidValueOptions, p.xCubeSizes, p.yCubeSizes, p.zCubeSizes,
            p.windowsOverlapped))

    def createParamsObj(self, paramsTuple, p):
        (minFreq, maxFreq, onlyMidValue, xCubeSize, yCubeSize, zCubeSize,
            windowsOverlapped) = paramsTuple
        params = Bunch(minFreq=minFreq, maxFreq=maxFreq,
            onlyMidValue=onlyMidValue,
            sigSectionMinLength=p.sigSectionMinLength,
            sigSectionAlpha=p.sigSectionAlpha,
            xCubeSize=xCubeSize, yCubeSize=yCubeSize, zCubeSize=zCubeSize,
            windowsOverlapped=windowsOverlapped)
#         for k, v in meta.iteritems():
#             params[k] = v
        return params

    def selectorFactory(self, timeStep, params, maxSurpriseVal=20,
                        doPlotSections=False):
        return FrequenciesSelector(timeStep, params.sigSectionAlpha,
            params.sigSectionMinLength, params.minFreq, params.maxFreq,
            params.onlyMidValue, maxSurpriseVal,
            self.LABELS[self.procID], doPlotSections, params.voxelIndices)

    def resultItem(self, selector, p, res, hp, ytest):
        return Bunch(predResults=res, fold=p.fold, ytest=ytest,
            featureExtractorName=Analyzer.FE_ALL,
            xCubeSize=hp.xCubeSize,
            yCubeSize=hp.yCubeSize,
            zCubeSize=hp.zCubeSize,
            windowsOverlapped=hp.windowsOverlapped,
            cubeIndex=hp.cubeIndex,
            sections=selector.sections,
            sectionsDic=selector.sectionsDic,
            minFreq=hp.minFreq, maxFreq=hp.maxFreq,
            onlyMidValue=hp.onlyMidValue,
            sigSectionMinLength=p.sigSectionMinLength,
            sigSectionAlpha=p.sigSectionAlpha)

    def printBestPredictorResults(self, bestEstimator):
        bep = bestEstimator.parameters
        print('Best results: ' +
            'sigSectionMinLength: {}, '.format(bep.sigSectionMinLength) +
            'sigSectionAlpha:{}, '.format(bep.sigSectionAlpha) +
            'minFreqs: {}, maxFreqs: {}, '.format(bep.minFreq, bep.maxFreq) +
            'onlyMidValueOptions:{}, '.format(bep.onlyMidValue) +
            'xCubeSize:{}, '.format(bep.xCubeSize) +
            'yCubeSize:{}, '.format(bep.yCubeSize) +
            'zCubeSize:{}, '.format(bep.zCubeSize) +
            'windowsOverlapped:{}, '.format(bep.windowsOverlapped) +
            'kernel: {}, c: {}, gamma: {}'.format(
            bep.kernel, bep.C, bep.gamma))

    def _predictorParamtersKeyClass(self):
        return namedtuple('predictorParamters',
            ['sigSectionMinLength', 'sigSectionAlpha', 'minFreq', 'maxFreq',
             'onlyMidValue', 'xCubeSize', 'yCubeSize', 'zCubeSize',
             'windowsOverlapped', 'kernel', 'C', 'gamma'])

    def predictorParamtersKeyItem(self, res, predRes):
        if ('windowsOverlapped' not in res):
            res.windowsOverlapped = True
        PredictorParamtersKey = self._predictorParamtersKeyClass()
        return PredictorParamtersKey(
            sigSectionMinLength=res.sigSectionMinLength,
            sigSectionAlpha=res.sigSectionAlpha,
            minFreq=res.minFreq, maxFreq=res.maxFreq,
            onlyMidValue=res.onlyMidValue,
            xCubeSize=res.xCubeSize, yCubeSize=res.yCubeSize,
            zCubeSize=res.zCubeSize, windowsOverlapped=res.windowsOverlapped,
            kernel=predRes.kernel, C=predRes.C, gamma=predRes.gamma)
