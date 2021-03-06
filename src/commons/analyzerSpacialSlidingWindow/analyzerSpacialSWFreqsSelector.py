'''
Created on Nov 20, 2014

@author: noampeled
'''

from src.commons.analyzer.analyzer import Analyzer
from src.commons.analyzerSpacialSlidingWindow.analyzerSpatialSWSelector \
    import AnalyzerSpacialSWSelector
from src.commons.sliders.spacialSliders import SpatialWindowSlider
from src.commons.utils import mpHelper, MLUtils, utils, freqsUtils as fu

import numpy as np
from sklearn.datasets.base import Bunch
import itertools
from collections import namedtuple, defaultdict


class AnalyzerSpacialSWFreqsSelector(AnalyzerSpacialSWSelector):

    def getXY(self, stepID, p):
        if (stepID != self.STEP_PRE_PROCCESS):
            return super(AnalyzerSpacialSWFreqsSelector, self).getXY(stepID)
        else:
            if (utils.fileExists(self.pssFileName)):
                xCubes, y, trialsInfo = utils.load(self.pssFileName)
                self.xAxis = utils.load(self.freqsFileName)
            else:
                print('Calculate the cubes ps')
                p = utils.BunchDic(p)
                x, y, trialsInfo = super(AnalyzerSpacialSWFreqsSelector,
                    self).getXY(stepID)
                timeStep = self.calcTimeStep(trialsInfo)
                weights = self.loadWeights()
                xlim, ylim, zlim, xstep, ystep, zstep = self.getMetaParameters(p)
                if ('yCubeSizes' not in p):
                    p.yCubeSizes = p.xCubeSizes
                if ('zCubeSizes' not in p):
                    p.zCubeSizes = p.xCubeSizes
                cubesParams = list(itertools.product(*(p.xCubeSizes,
                    p.yCubeSizes, p.zCubeSizes, p.windowsOverlapped)))
                xCubes = defaultdict(dict)
                for cubeParams in cubesParams:
                    xCubeSize, yCubeSize, zCubeSize, windowsOverlapped = cubeParams
                    spacialSlider = SpatialWindowSlider(
                        weights.shape[0], xlim, ylim, zlim,
                        xstep, xCubeSize, ystep, yCubeSize,
                        zstep, zCubeSize, windowsOverlapped)
                    cubesNum = spacialSlider.calcCubesNum(weights)
                    noZerosCubeIndex = 0
                    for cubeIndex, voxelIndices in \
                            enumerate(spacialSlider.cubesGenerator()):
                        cubeWeights = weights[voxelIndices, :]
                        if (np.all(cubeWeights == 0)):
                            continue
                        print('prepare params for cube {}/{}'.format(
                            noZerosCubeIndex, cubesNum))
                        noZerosCubeIndex += 1
                        cubeWeights, zeroLinesIndices = utils.removeZerosLines(
                            cubeWeights)
                        voxelIndices = voxelIndices[zeroLinesIndices]
                        pss, freqs = fu.calcPSX(x, min(p.minFreqs),
                            max(p.maxFreqs), timeStep, weights=cubeWeights)
                        xCubes[cubeParams][cubeIndex] = Bunch(x=pss,
                            voxelIndices=voxelIndices, cubeWeights=cubeWeights)
                print('save pss and freqs')
                utils.save((xCubes, y, trialsInfo), self.pssFileName)
                utils.save(freqs, self.freqsFileName)
                self.xAxis = freqs
            return xCubes, y, trialsInfo

    def _prepareCVParams(self, p):
        p = p.merge(p.kwargs)
        params = []
        p.yCubeSizes = p.get('yCubeSizes', p.xCubeSizes)
        p.zCubeSizes = p.get('zCubeSizes', p.xCubeSizes)
        totalCubesNum = sum([len(cubes.keys()) for cubes in p.x.values()])
        cubesParams = list(itertools.product(*(p.xCubeSizes,
            p.yCubeSizes, p.zCubeSizes, p.windowsOverlapped)))
        paramsNum = len(list(p.cv)) * len(cubesParams) * totalCubesNum
        index = 0
        for cubeParams in cubesParams:
            xCubeSize, yCubeSize, zCubeSize, windowsOverlapped = cubeParams
            cubes = p.x[cubeParams]
            for cubeIndex, x in cubes.iteritems():
                for fold, (trainIndex, testIndex) in enumerate(p.cv):
                    if (not MLUtils.isBinaryProblem(p.y, trainIndex, testIndex)):
                        print('fold {} is not binary, continue'.format(fold))
                        continue
                    y, shuffleIndices = self.permutateTheLabels(p.y, trainIndex,
                        self.useSmote) if self.shuffleLabels else (p.y, None)
                    params.append(Bunch(
                        x=mpHelper.ForkedData(x), y=y, trainIndex=trainIndex,
                        testIndex=testIndex, fold=fold,
                        sigSectionMinLengths=p.sigSectionMinLengths,
                        sigSectionAlphas=p.sigSectionAlphas,
                        minFreqs=p.minFreqs, maxFreqs=p.maxFreqs,
                        onlyMidValueOptions=p.onlyMidValueOptions,
                        xCubeSize=xCubeSize, yCubeSize=yCubeSize,
                        zCubeSize=zCubeSize,
                        windowsOverlapped=windowsOverlapped,
                        cubeIndex=cubeIndex,
                        kernels=p.kernels, Cs=p.Cs, gammas=p.gammas,
                        index=index, paramsNum=paramsNum,
                        shuffleIndices=shuffleIndices))
                    index += 1

        return params

#     def calcCubesPSS(self, spacialSlider, weights, x, minFreq, maxFreq, timeStep):
#         for cubeIndex, voxelIndices in \
#                 enumerate(spacialSlider.cubesGenerator()):
#             cubeWeights = weights[voxelIndices, :]
#             if (np.all(cubeWeights == 0)):
#                 continue
#             cubeWeights, zeroLinesIndices = utils.removeZerosLines(
#                 cubeWeights)
#             voxelIndices = voxelIndices[zeroLinesIndices]
#             pss, freqs = su.calcPS(x, minFreq, maxFreq, timeStep, 
#                 weights=cubeWeights)

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
        for hp in itertools.product(*(p.minFreqs, p.maxFreqs,
            p.onlyMidValueOptions, p.sigSectionMinLengths,
            p.sigSectionAlphas)):
                params = self.createParamsObj(hp)
                params.windowsOverlapped = p.windowsOverlapped
                params.cubeIndex = p.cubeIndex
                yield params

    def createParamsObj(self, paramsTuple):
        (minFreq, maxFreq, onlyMidValue, sigSectionMinLength,
         sigSectionAlpha) = paramsTuple
        return Bunch(minFreq=minFreq, maxFreq=maxFreq,
            onlyMidValue=onlyMidValue, sigSectionMinLength=sigSectionMinLength,
            sigSectionAlpha=sigSectionAlpha)

#     def selectorFactory(self, timeStep, params, maxSurpriseVal=20,
#                         doPlotSections=False):
#         return FrequenciesSelector(timeStep, params.sigSectionAlpha,
#             params.sigSectionMinLength, params.minFreq, params.maxFreq,
#             params.onlyMidValue, maxSurpriseVal,
#             self.LABELS[self.procID], doPlotSections, params.voxelIndices)

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
