    # -*- coding: utf-8 -*- 
'''
Created on Nov 25, 2013

    @author: noampeled
    '''
from abc import abstractmethod
import numpy as np
import os
from path3 import path

from sklearn import feature_selection
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.datasets.base import Bunch

import itertools
from collections import namedtuple
import operator
import traceback

from src.commons.utils import utils
from src.commons.utils import MLUtils
from src.commons import GridSearchSteps as GSS
from src.commons import scoreFunctions as sf
from src.commons import featuresExtractors as fe
from src.commons.utils import mpHelper
from src.commons.utils import plots
from src.commons.utils import tablesUtils as tabu

KERNEL_TYPES = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']


class Analyzer(object):
    PROCS_NAMES = []
    STEPS = ['preProcess', 'splitData', 'heldOutData',
             'trialsAnalysis', 'features', '']
    STEP_PRE_PROCCESS, STEP_SPLIT_DATA, STEP_SAVE_HELDOUT_DATA, \
        STEP_TRIALS_ANALYSIS, STEP_FEATURES, STEP_NONE = range(6)

    FE_RMS, FE_ALL, FE_COR, FE_COV = 'rms', 'all', 'cor', 'cov'
    FEATURE_EXTRACTORS = {FE_RMS: fe.RMS, FE_ALL: fe.ALL,
                          FE_COR: fe.cor, FE_COV: fe.cov}
    FEATURE_EXTRACTORS_NAMES = {i: k for i, k in
                                enumerate(FEATURE_EXTRACTORS.keys())}

    def __init__(self, folder, matlabFile, subject, procID=0,
            indetifier='default', normalizationField='',
            multipleByWeights=False, weightsFileName='',
            useSpectral=False, plotForPublication=False,
            doLoadOriginalTimeAxis=True, variesT=False,
            matlabFileWithArtifacts='', shuffleLabels=None,
            useSmote=False, jobsNum=1):
        self.folder = folder
        self.matlabFile = matlabFile
        self.subject = subject
        self.procID = procID
        self.indetifier = indetifier
        if (shuffleLabels is None):
            shuffleLabels = False
            print('No value was set for shuffleLabels! Set to False')
        self.shuffleLabels = shuffleLabels
        self.useSmote = useSmote
        self.multipleByWeights = multipleByWeights
        self.useSpectral = useSpectral
        self.normalizationField = normalizationField
        self.weightsFileName = weightsFileName
        self.matlabFileWithArtifacts = matlabFileWithArtifacts
        self.jobsNum = jobsNum
        self.variesT = variesT
        if (plotForPublication):
            plots.init()
        if (doLoadOriginalTimeAxis):
            self.loadOriginalTimeAxis()
        if (tabu.DEF_TABLES):
            self.hdfFile = tabu.openHDF5File(self.hdf5FileName)
            self.hdfGroup = tabu.findOrCreateGroup(self.hdfFile,
                self.STEPS[self.STEP_PRE_PROCCESS])
        if (utils.sftp is not None):
            utils.sftp.remoteTempFolder = os.path.join(utils.sftp.remoteFolder,
                'temp', self.subject)
        print('init analyzer for {}'.format(self.subject))

    def preProcess(self, checkExistingFile=False, parallel=False,
        verbose=False):
        ''' Step 1) Load the data '''
        print('Preproceesing the data')
        if (utils.fileExists(self.dataFileName(self.STEP_PRE_PROCCESS)) and \
            checkExistingFile):
            print('{} exists, overwrite? (y/n)'.format(self.dataFileName))
            userAnswer = raw_input()
            if (userAnswer == 'n'):
                return None

        matlabDic = self.loadData()
        if (self.multipleByWeights):
            print('multiple by weights!')
            weightsDict = utils.loadMatlab(
                self.weightsFullFileName(self.weightsFileName))
            # Save the weights
            if (tabu.DEF_TABLES):
                self.hdfFile.close()
                self.hdfFile = tabu.createHDF5File(self.hdf5FileName)
                self.hdfGroup = tabu.findOrCreateGroup(self.hdfFile,
                    self.STEPS[self.STEP_PRE_PROCCESS])
                tabu.createHDF5ArrTable(self.hdfFile, self.hdfGroup,
                    'weights', arr=weightsDict[self.weightsDicKey])
                weights = weightsDict[self.weightsDicKey].T
        else:
            weights = None

        readDataFunc = self.readDataParallel if parallel else self.readData
        (trials, labels, trialsInfo) = readDataFunc(
            matlabDic, self.procID, weights, verbose)
        utils.log(utils.count(np.array(labels)), verbose)
        print('saving the data')
        self.saveXY(trials, labels, self.STEP_PRE_PROCCESS, trialsInfo)
        if (not self.variesT):
            np.save(self.timeAxisFileName, self.xAxis)

    def loadData(self):
        matlabFullPath = os.path.join(self.folder, self.subject,
            self.matlabFile)
        matlabDic = utils.loadMatlab(matlabFullPath)
        return matlabDic

    def loadOriginalTimeAxis(self):
        # Save the time axis
        timeAxisFullPath = os.path.join(self.folder, self.subject,
            'xAxis.mat')
        timeAxisDic = utils.loadMatlab(timeAxisFullPath)
        self.xAxis = np.array(timeAxisDic['xAxis'][0])

    def loadTimeAxis(self):
        timeAxis = utils.load(self.timeAxisFileName, useNumpy=True)
        return timeAxis

    def readData(self, matlabDic, procID, weights=None, verbose=False):
        recordsNum, recordsFlags = self._calcRecordsNum(matlabDic)
        if not (self.variesT):
            trialShape = self.getTrialShape(matlabDic)
            channelsNum = trialShape[1] if weights is None else weights.shape[1]
            T = trialShape[0]
            trials, labels, trialsInfo = self._createDataObjects(
                recordsNum, T, channelsNum)
        else:
            channelsNum = self.getChannelsNum(matlabDic)
            trials, labels, trialsInfo = [], [], []
        recordNum = 0
        dataGenerator = self.dataGenerator(matlabDic)
        for recordFlag, ((trial, label), trialInfo) in \
                                zip(recordsFlags, dataGenerator):
            if (recordFlag):
                if (weights is not None):
                    trial = np.dot(trial, weights)
                if not (self.variesT):
                    trials[recordNum, :, :] = trial
                    labels[recordNum] = self.trialLabel(label, trialInfo)
                    if (tabu.DEF_TABLES):
                        self.setTrialInfoRecord(trialsInfo, recordNum, trialInfo)
                    else:
                        trialsInfo[recordNum] = trialInfo
                else:
                    trials.append(trial)
                    labels.append(self.trialLabel(label, trialInfo))
                    if (tabu.DEF_TABLES):
                        self.setTrialInfoRecord(trialsInfo, recordNum, trialInfo)
                    else:
                        trialsInfo.append(trialInfo)
                recordNum += 1
                if (recordNum % 100 == 0):
                    utils.log('record {} / {}'.format(
                        recordNum, sum(recordsFlags)), verbose)
        # delete nan raws
#         nanidx = np.where(np.isnan(trials[:, 0, 0]))[0]
#         utils.log('nanidx: {}'.format(nanidx), verbose)
#         trials = np.delete(trials, nanidx, axis=0)
        return (trials, labels, trialsInfo)

    def readDataParallel(self, matlabDic, procID, weights=None, verbose=False):
        jobsParams = []
        _, recordsFlags = self._calcRecordsNum(matlabDic)
        dataGenerator = self.dataGenerator(matlabDic)
        recordIndex = 0
        for recordFlag, ((trial, label), trialInfo) in \
                        zip(recordsFlags, dataGenerator):
            if (recordFlag):
                forkedTrial = mpHelper.ForkedData(trial)
                jobsParams.append(Bunch(trial=forkedTrial, label=label,
                    trialInfo=trialInfo, weights=weights,
                    index=recordIndex, verbose=verbose))
                recordIndex += 1
        recordsNum = recordIndex
        utils.log('{} records'.format(recordsNum), verbose)
        func = self._calcTrialsParallel
        if (self.jobsNum == 1):
            mapResults = [func(p) for p in jobsParams]  # For debugging
        else:
            mapResults = utils.parmap(func, jobsParams, self.jobsNum)
        utils.log('concatenate the results', verbose)
        trialShape = self.getTrialShape(matlabDic)
        channelsNum = trialShape[1] if weights is None else weights.shape[1]
        trials, labels, trialsInfo = self._createDataObjects(
            recordsNum, trialShape[0], channelsNum)
        for mapRes in mapResults:
            trials[mapRes.index, :, :] = mapRes.trial
            labels[mapRes.index] = mapRes.label
            if (tabu.DEF_TABLES):
                self.setTrialInfoRecord(trialsInfo, mapRes.index,
                    mapRes.trialInfo)
            else:
                trialsInfo[mapRes.index] = mapRes.trialInfo
        return (trials, labels, trialsInfo)

    def _createDataObjects(self, recordsNum, T, channelsNum):
        shape = (recordsNum, T, channelsNum)
        if (tabu.DEF_TABLES):
            hdfGroup = tabu.findOrCreateGroup(self.hdfFile,
                self.STEPS[self.STEP_PRE_PROCCESS])
            trials = tabu.createHDF5ArrTable(self.hdfFile,
                hdfGroup, 'x', shape=shape)
            labels = tabu.createHDF5ArrTable(self.hdfFile,
                hdfGroup, 'y', np.dtype('int16'), (recordsNum,))
            trialsInfo = tabu.createHDFTable(self.hdfFile,
                hdfGroup, 'trialsInfo', self.trialsInfoDesc)
            self.createEmptyTrialInfoTable(trialsInfo, recordsNum)
        else:
            trials = np.empty(shape)
            trials.fill(np.nan)
            labels = np.zeros((recordsNum))
            trialsInfo = [None] * recordsNum
            self.hdfFile = None
        return trials, labels, trialsInfo

    def _calcTrialsParallel(self, p):
        trial = p.trial.value
        if (p.weights is not None):
            trial = np.dot(trial, p.weights)
        label = self.trialLabel(p.label, p.trialInfo)
#         utils.log((p.index, label), p.verbose)
        return (Bunch(trial=trial, label=label, trialInfo=p.trialInfo,
                      index=p.index))

    def _calcRecordsNum(self, matlabDic):
        metaDataGenerator = self.metaDataGenerator(matlabDic)
        recordsNum = 0
        recordsFlags = []
        for (label, trialInfo) in metaDataGenerator:
            flag = self.trialCond(label, trialInfo)
            recordsFlags.append(flag)
            if (flag):
                recordsNum += 1
        return recordsNum, recordsFlags

    def getBestEstimators(self, getRemoteFiles=False):
        print('loading all the results files')
        results = {}
        results['gmean'] = {'scores': [], 'params': []}
        results['auc'] = {'scores': [], 'params': []}

        print('calculate prediction scores')
        for fileName in self.predictorResultsGenerator(getRemoteFiles):
            _, bestScore, bestParams = utils.load(fileName)
            results['gmean']['scores'].append(bestScore.gmean)
            results['gmean']['params'].append(bestParams.gmean)
            results['auc']['scores'].append(bestScore.auc)
            results['auc']['params'].append(bestParams.auc)
        print('results for {}'.format(self.defaultFileNameBase))
        for acc in ['auc', 'gmean']:
            scores = results[acc]['scores']
            print('{}: {} results: {}'.format(acc, len(scores), np.mean(scores)))
        print('save results in {}'.format(self.bestEstimatorsFileName))
        utils.save(results, self.bestEstimatorsFileName)

    @abstractmethod
    def getTrialShape(self, matlabDic):
        ''' Gets a trial shape. Assume all the trials have the same shape '''

    @abstractmethod
    def dataGenerator(self, matlabDic):
        ''' generator for the matlab data '''

    @abstractmethod
    def metaDataGenerator(self, matlabDic):
        ''' generator for the matlab data '''

    @abstractmethod
    def trialCond(self, label, trialInfo):
        ''' According to self.procID,
        returns True when the label and trialInfo fits  '''
        return True

    @abstractmethod
    def trialLabel(self, label, trialInfo):
        ''' Retrurn the trials label (0,1) according to label and trialInfo '''
        return label

#     def shuffleLabels(self, step=None):
#         ''' In case you want to calculate the prediction base level '''
#         if (step is None):
#             step = self.STEP_PRE_PROCCESS
#         trials, labels, trialsInfo = self.getXY(step)
#         random.shuffle(labels)
#         self.indetifier += "_shuffled"
#         self.saveXY(trials, labels, step, trialsInfo)

    def splitData(self, heldoutSize=0.1):
        ''' Split the data to put aside the heldout data  '''
        if (heldoutSize > 0):
            x, y, trialsInfo = self.getXY(self.STEP_PRE_PROCCESS)
            cv = StratifiedShuffleSplit(y, 1, heldoutSize, random_state=0)
            train, heldout = next(iter(cv))
            x_heldin, x_heldout = x[train], x[heldout]
            y_heldin, y_heldout = y[train], y[heldout]
            info_heldin, info_heldout = trialsInfo[train], trialsInfo[heldout]
            self.saveXY(x_heldout, y_heldout, self.STEP_SAVE_HELDOUT_DATA,
                info_heldout)
            self.saveXY(x_heldin, y_heldin, self.STEP_SPLIT_DATA, info_heldin)
        else:
            utils.renameFile(self.dataFileName(self.STEP_PRE_PROCCESS),
                self.dataFileName(self.STEP_SPLIT_DATA))

    def calcSignalPS(self):
        pass

    def plotAveragedChannles(self):
        x, y, _ = self.getXY(self.STEP_SPLIT_DATA)
        C = x.shape[2]

        channelsSelector = GSS.TimeSelector()
        channelsSelector.fit(x, y, True)
        print(channelsSelector.channelsIndices)
        plots.barPlot(channelsSelector.scores, xlim=[0, C - 1])

        x0 = np.mean(x[y == 0, :, :], 0)
        x1 = np.mean(x[y == 1, :, :], 0)
        sensorsFolder = '{}/{}/sensors'.format(self.folder, self.subject)
        utils.createDirectory(sensorsFolder)
        for c in range(C):
            plots.graph2(self.xAxis, x0[:, c], x1[:, c],
                [self.LABELS[self.procID][0],
                 self.LABELS[self.procID][1]],
                fileName='{}/sensor{}'.format(sensorsFolder, c))

    def process(self, timePercentiles=[0],foldsNum=5, testSize=None, Cs=[1],gammas=[0],channelsNums=[30], kernels=['rbf'],
                                    windowSizes=[100], windowsNum=10, featureExtractors=None, 
                                    channelsPerWindow=True, n_jobs=-2, useExistingFiles=False):
        ''' Step 3) Processing the data ''' 
        print('Proccessing the data')
        x,y,trialsInfo = self.getXY(self.STEP_PRE_PROCCESS) 
        T = x.shape[1]

        utils.save(Bunch(windowsNum=windowsNum, featureExtractors=featureExtractors, T=T, foldsNum=foldsNum), self.predictionsParamtersFileName)
        cv = self.featuresCV(y,trialsInfo,foldsNum, testSize)
        if (featureExtractors is None): raise Exception('No feature extractors!') 
        if (useExistingFiles): files = [filename for filename in utils.filesInFolder(self.tempFolder(), '*predictionsParamters*.pkl') if 'results' not in filename]
        else: files = self.featuresGenerator(x,y,cv,channelsNums,None,n_jobs)
        params = self._prepareFeaturesGeneratorParams(utils.BunchDic(locals())) 
        
        func = self._preparePredictionsParameters
        if n_jobs==1: mapResults = [func(p) for p in params] # For debugging
        else: mapResults = utils.parmap(func, params, n_jobs)
        utils.save(mapResults, self.paramsFileName)
        utils.deleteFilesFromList(files)
        
    def _preparePredictionsParameters(self, gp):
        tic = utils.ticToc()
        p = utils.load(gp.fileName)
        results = []
        # features extraction
        for featureExtractorName in gp.featureExtractors:
            xtrainFeatures = self.featureExtractor(featureExtractorName, p.xtrainChannles)
            xtestFeatures = self.featureExtractor(featureExtractorName, p.xtestChannles)
            # Time selector
            for timePercentile in gp.timePercentiles:
                timeSelector = self.timeSelector(timePercentile)
                xtrainFeaturesTimed = timeSelector.fit_transform(xtrainFeatures, p.ytrain)
                xtestFeaturesTimed = timeSelector.transform(xtestFeatures)            
                res = self._predict(Bunch(xtrainFeatures=xtrainFeaturesTimed,ytrain=p.ytrain,
                    xtestFeatures=xtestFeaturesTimed,kernels=gp.kernels,Cs=gp.Cs,gammas=gp.gammas))
                results.append(Bunch(predResults=res,fold=p.fold,ytest=p.ytest,channelsNum=p.channelsNum,
                    featureExtractorName=featureExtractorName, timePercentile=timePercentile))
        utils.howMuchTimeFromTic(tic, '_preparePredictionsParamtersChannelsFeaturesParallel')
        resultsFileName = '{}_results.pkl'.format(gp.fileName[:-4])
        utils.save(results, resultsFileName)
        return resultsFileName

    def _predict(self, p, bs, bp, hp, verbose=False):
        for kernel, C, gamma in itertools.product(*(p.kernels, p.Cs, p.gammas)):
            try:
                svc = GSS.TSVC(C=C, kernel=kernel, gamma=gamma)
                svc.fit(p.xtrainFeatures, p.ytrain)
                probs = svc.predict(p.xtestFeatures, calcProbs=True)
                ypred = MLUtils.probsToPreds(probs)
                auc = sf.AUCScore(p.ytest, probs)
                gmean = sf.gmeanScore(p.ytest, ypred)
                # Don't let results with rates=(0,1) gets in
                if (auc > bs.auc and gmean > 0):
                    bs.auc = auc
                    bp.auc = hp
                    bp.auc.kernel, bp.auc.C, bp.auc.gamma = kernel, C, gamma
                    bp.auc.rates = sf.calcRates(p.ytest, ypred)
                    if (verbose):
                        print('best auc ', bs.auc, bp.auc)
                if (gmean > bs.gmean):
                    bs.gmean = gmean
                    bp.gmean = hp
                    bp.gmean.kernel, bp.gmean.C, bp.gmean.gamma = kernel, C, gamma
                    bp.gmean.rates = sf.calcRates(p.ytest, ypred)
                    if (verbose):
                        print('best gmean ', bs.gmean, bp.gmean)
            except:
                print('error with _predict!')
                utils.dump((p, bs, bp, hp, kernel, C, gamma), '_predict', utils.DUMPER_FOLDER)
                print traceback.format_exc()


#     def calculatePredictionsScores(self, saveResults=False):
#         results = {}
#         print('calculate prediction scores')
# #         generator = self.predictorResultsGenerator()
#         params = []
#         for res in self.predictorResultsGenerator():
#             params.append(res)
#         func = self._calculatePredictionsScoresParallel
#         if (self.jobsNum == 1):
#             mapResults = [func(p) for p in params]  # For debugging
#         else:
#             mapResults = utils.parmap(func, params, self.jobsNum)
#         # Merge results
#         print('merge the results')
#         for mapRes in mapResults:
#             for key1 in mapRes.keys():
#                 if (key1 not in results):
#                     results[key1] = {}
#                 for key2 in mapRes[key1].keys():
#                     if (key2 not in results[key1]):
#                         results[key1][key2] = []
#                     results[key1][key2].extend(mapRes[key1][key2])
#         if (saveResults):
#             utils.save(results, self.scorerFoldsResultsFileName)
#         print('find best estimator')
#         self._findBestEstimators(results)

#     def _calculatePredictionsScoresParallel(self, p):
#         newFileName, fileNum, filesNum = p
#         print('file {}/{}'.format(fileNum + 1, filesNum))
#         results = utils.load(newFileName)
#         scorerFoldsResults = {}
#         # Can't be defaultdict(lambda: defaultdict(list)) because then you might get:
#         # PicklingError: Can't pickle <type 'function'>: attribute lookup __builtin__.function failed
#         for res in results:
#             scorerFoldsResultsKey = self.scorerFoldsResultsKey(res)
#             if (scorerFoldsResultsKey not in scorerFoldsResults):
#                 scorerFoldsResults[scorerFoldsResultsKey] = defaultdict(list)
#             if (res.predResults is None):
#                 for kernel, C, gamma in self.classifierHPGenerator:
#                     predRes = Bunch(kernel=kernel, C=C, gamma=gamma, probs=None)
#                     key = self.predictorParamtersKeyItem(res, predRes)
#                     scorerFoldsResults[scorerFoldsResultsKey][key].append(
#                         self.scorerFoldsResultsItem(None, None,
#                         (0.5, 0.5), res, predRes, 0.5, 0.5))
#             else:
#                 for predRes in res.predResults:
#                     # Set the key as PredictorParamtersKey, not Bunch,
#                     # because Bunch can't be used as a dictionary key
#                     key = self.predictorParamtersKeyItem(res, predRes)
#     #                 score = self.gridSearchScorer(res.ytest, predRes.probs)
#     #                 probsScore = self.probsScorer(res.ytest, predRes.probs)
#                     rates = sf.calcRates(res.ytest, MLUtils.probsToPreds(
#                         predRes.probs))
#                     auc = sf.AUCScore(res.ytest, predRes.probs)
#                     gmean = sf.gmeanScoreFromRates(rates)
#                     scorerFoldsResults[scorerFoldsResultsKey][key].append(
#                         self.scorerFoldsResultsItem(None, None, rates, res,
#                         predRes, auc, gmean))
# #                 else:
# #                     scorerFoldsResults[scorerFoldsResultsKey][key].append(
# #                         self.scorerFoldsResultsItem(None, None, None, res, None,
# #                         None, None))
#         return scorerFoldsResults

    def scorerFoldsResultsKey(self, res):
        return res.featureExtractorName

    def _findBestEstimators(self, scorerFoldsResults):
        bestEstimators = {}
        # loop over the different features extractors
        for featureExtractorName, scorerFoldsResultsDic in \
                scorerFoldsResults.iteritems():
            bestScore = 0
            for predictorParamters, predictorResults in \
                    scorerFoldsResultsDic.iteritems():
                # sort according to fold number
                results = sorted(predictorResults,
                                 key=operator.itemgetter('fold'))
                probsScores = []
                aucs, gmeans = [], []
                for res in results:
                    aucs.append(res.auc)
                    gmeans.append(res.gmean)
                    if (not res.probsScore is None):
                        probsScores = np.concatenate((probsScores,
                            res.probsScore))
                probsScores = np.array(probsScores)
                aucs = np.array(aucs)
                gmeans = np.array(gmeans)
                if (self.bestScore(aucs) > bestScore):
                    bestEstimators[featureExtractorName] = \
                        self.bestEstimatorItem(predictorParamters, probsScores,
                        aucs, results)
                    bestScore = self.bestScore(aucs)

        if (len(bestEstimators) == 0):
            print('no scores above 0!')
        for featureExtractorName in bestEstimators.keys():
            # The namedtuple PredictorParamters isn't pickleable,
            # so convert it to Bunch
            bestEstimators[featureExtractorName].parameters = \
                utils.namedtupleToBunch(
                bestEstimators[featureExtractorName].parameters)
            utils.save(bestEstimators[featureExtractorName],
                self.bestEstimatorFileName(featureExtractorName))
        utils.save(bestEstimators, self.bestEstimatorsFileName)

    def bestEstimatorItem(self, predictorParamters, probsScores, scores, results):
        return  Bunch(parameters=predictorParamters, probsScores=probsScores, scores=scores, rates=[res.rates for res in results])

    def scorerFoldsResultsItem(self, score, probsScore, rates, res, predRes,
                               auc, gmean):
        return Bunch(score=score, probsScore=probsScore, fold=res.fold,
            rates=rates, auc=auc, gmean=gmean)

    def predictorResultsGenerator(self, getRemoteFiles=False):
        files = utils.filesInFolder(self.tempFolder,
            pattern='{}_*_results.pkl'.format(
            path(self.dataFileName(self.STEP_FEATURES)).namebase),
            getRemoteFiles=getRemoteFiles)
        if (len(files) == 0):
            utils.throwException('There are no results files!')

        for fileName in files:
#             # To allow to load the tmp files from a computer where it
#             # wasn't been written
#             newFileName = utils.getLocalFullFileName(fileName, self.folder, 3)
#             # If the file doens't exist, download it using sftp
#             if (utils.sftp):
#                 utils.sftp.getAndMove(fileName, os.path.dirname(newFileName))
#             if (not utils.fileExists(newFileName)):
#                 print('{} not exist!'.format(newFileName))
#                 continue
            if (os.path.dirname(fileName) == ''):
                fileName = os.path.join(self.tempFolder, fileName)
            yield path(str(fileName))
#             results = utils.load(newFileName)
#             print('file {}/{}'.format(fileNum, filesNum))
#             for res in results:
#                 yield res
#                 if (res.predResults is None):
#                     print('res.predResults is None!')
#                     yield (res, None)
#                 else:
#                     for predRes in res.predResults:
#                         yield (res, predRes)

    def analyzeResults(self, doPlot=True):
        bestEstimators = utils.load(self.bestEstimatorsFileName)
        predParams = utils.load(self.predictionsParamtersFileName)
#         colors = sns.color_palette(None, len(bestEstimators))
        scoresGenerator = self.scoresGenerator(bestEstimators, predParams, calcProbs=False)
        for score in scoresGenerator:
            pass        

    def findSignificantResults(self, overwrite=False, doShow=True):
        self.shuffleLabels = True
        print('load {}'.format(self.bestEstimatorsFileName))
        bestEstimatorsShuffle = utils.load(
            self.bestEstimatorsFileName, overwrite=overwrite)
        self.shuffleLabels = False
        print('load {}'.format(self.bestEstimatorsFileName))
        bestEstimators = utils.load(
            self.bestEstimatorsFileName, overwrite=overwrite)
        ps = Bunch(auc=None, gmean=None)
        scores = Bunch(auc=None, gmean=None)
        scoresShuffle = Bunch(auc=None, gmean=None)
        for acc in bestEstimators.keys():
            scores[acc] = np.array(bestEstimators[acc]['scores'])
            scoresShuffle[acc] = np.array(bestEstimatorsShuffle[acc]['scores'])
            ps[acc] = utils.ttestGreaterThan(scores[acc], scoresShuffle[acc])
        print(ps)

        plots.barGrouped2([scores.auc.mean(), scores.gmean.mean()],
            [scoresShuffle.auc.mean(), scoresShuffle.gmean.mean()],
            [scores.auc.std(), scores.gmean.std()],
            [scoresShuffle.auc.std(), scoresShuffle.gmean.std()],
            ['scores', 'shuffle'], ['auc', 'gmean'],
            title='{}: p(auc):{} p(gmean):{}'.format(self.subject,
            ps['auc'], ps['gmean']), figName='scoreVSshuffle_{}'.format(
            self.subject), doShow=doShow)

    def calcHeldOutPrediction(self):
        ''' classification report on heldout data '''
        bestEstimators = utils.load(self.bestEstimatorsFileName)
        dataLoaded = False

        for featureExtractorName, bestEstimator in bestEstimators.iteritems():
            print('Calculate a predictor for all the heldin data for the ' +
                'feature extractor {}'.format(featureExtractorName))
            if (utils.fileExists(self.allDataPredictorFileName(
                    featureExtractorName))):
                p = utils.load(self.allDataPredictorFileName(
                    featureExtractorName))
            if (True):
                if (not dataLoaded):
                    print('load all the data')
                    x, y, trialsInfo = self.getXY(self.STEP_SPLIT_DATA)
                    x, y = MLUtils.boost(x, y)
                    dataLoaded = True
                bep = bestEstimator.parameters
                bep.featureExtractorName = featureExtractorName
                xFeatures, selector = self.heldOutFeaturesExtraction(x, y,
                    trialsInfo, bep, self.normalizationField)
                utils.save((xFeatures, selector),
                    self.heldOutFeaturesExtractionResultsFileName)

                print('Predictor fitting')
                svc = self.predictor(bep.C, bep.kernel, bep.gamma)
                svc.fit(xFeatures, y)
                utils.save(Bunch(selector=selector, svc=svc),
                    self.allDataPredictorFileName(featureExtractorName))
                print('prediction for all the train data')
                self.predictOverXY(svc, xFeatures, y)

            if (utils.fileExists(self.dataFileName(
                    self.STEP_SAVE_HELDOUT_DATA))):
                for x_heldout, y_heldout, trialsInfo_heldout, info in \
                        self.loadHeldoutDataGenerator():
                    print('heldout data transformation for {}'.format(info))
                    xHeldoutFeatures = self.heldOutFeaturesTransform(self, p,
                        x_heldout, featureExtractorName)
                    print('prediction')
                    ypred = self.predictOverXY(p.svc,
                        xHeldoutFeatures, y_heldout)
    #                 probs = p.svc.predict(xHeldoutFeatures)
    #                 ypred = MLUtils.probsToPreds(probs)
    #                 if (y_heldout is not None):
    #                     heldoutScores = self.scorer(y_heldout,probs)
    #                     print('accuracy score: {}'.format(heldoutScores))
    #                     print('report for {}'.format(featureExtractorName))
    #                     MLUtils.calcConfusionMatrix(y_heldout, ypred, self.LABELS[self.procID],True) 
                    utils.save(Bunch(trialsInfo_heldout=trialsInfo_heldout,
                        ypred=ypred, featureExtractorName=featureExtractorName,
                        info=info), self.heldoutPredictionFileName(
                        featureExtractorName, info))

    def fullDataAnlysis(self):
        ''' Not yet implemented '''
        pass

    def predictOverXY(self, svc, features, y, featureExtractorName='all', reportScore=True):
        print('prediction')
        probs = svc.predict(features) 
        ypred = MLUtils.probsToPreds(probs)
        if (y is not None and reportScore):
            heldoutScores = self.scorer(y,probs)
            print('accuracy score: {}'.format(heldoutScores))
            print('report for {}'.format(featureExtractorName))
            MLUtils.calcConfusionMatrix(y, ypred, self.LABELS[self.procID],True) 
        return ypred
     
    def heldOutFeaturesExtraction(self,x,y,trialsInfo,bep):
        print('Channels selection {}'.format(bep.channelsNum))
        channlesSelector = self.channelsSelector(bep.channelsNum)
        xChannles = channlesSelector.fit_transform(x, y)
        print('Feature extractor: {}'.format(bep.featureExtractorName))
        xFeatures = self.featureExtractor(bep.featureExtractorName,xChannles)
        print('Time selection {}'.format(bep.timePercentile))
        timeSelector = self.timeSelector(bep.timePercentile)
        xFeaturesTimed = timeSelector.fit_transform(xFeatures, y)
        return xFeaturesTimed, (channlesSelector,timeSelector)

    def heldOutFeaturesTransform(self, p, x_heldout, featureExtractorName):
        channlesSelector,timeSelector = p.selector
        xHeldoutChannles = channlesSelector.transform(x_heldout)
        xHeldoutFeatures = self.featureExtractor(featureExtractorName,xHeldoutChannles)
        xHeldoutFeaturesTimed = timeSelector.transform(xHeldoutFeatures)
        xHeldoutFeaturesTimedNormalized,_,_ = MLUtils.normalizeData(xHeldoutFeaturesTimed)          
        return xHeldoutFeaturesTimedNormalized
        
    def createHeldoutPredictionReport(self):
        pass

    def scoresGenerator(self, bestEstimators, predParams, calcProbs=True,
                        printResults=True, doPlot=False):
        for featureExtractorName, bestEstimator in bestEstimators.iteritems():
            print('{}: {}'.format(featureExtractorName, np.mean(bestEstimator['scores'])))
            continue
            bep = bestEstimator.parameters
            bep.featureExtractorName = featureExtractorName
            if (calcProbs):
                yield (bestEstimator.probsScores)
            else:
                if (printResults):
#                     scoresMean = np.mean(bestEstimator.scores)
#                     scoresStd = np.std(bestEstimator.scores)
#                     print("%0.3f (+/-%0.03f)" % (scoresMean,scoresStd / 2))
#                     print('rates: ', bestEstimator.rates)
                    if ('y' in bestEstimator):
                        aucs = MLUtils.calcAUCs(bestEstimator.probs,
                            bestEstimator.y,
                            ['fold {}'.format(i) for i in
                             range(1, len(bestEstimator.probs) + 1)],
                            doPlot=doPlot)
                        print('Folds AUC:')
                        print(aucs)
                        print("%0.3f (+/-%0.03f)" % (np.mean(aucs),
                            np.std(aucs) / 2))
                    self.printBestPredictorResults(bestEstimator)
#                 yield (bestEstimator.scores)

    def printBestPredictorResults(self, bestEstimator):
        bep = bestEstimator.parameters
        print('Best results for features extractor: {}, channels: {}, timePercentile: {}, kernel: {}, c: {}, gamma: {}'.format(
            bep.featureExtractorName,bep.channelsNum,bep.timePercentile,bep.kernel,bep.C,bep.gamma))
        self.printBestPredictorScores(bestEstimator)
    
    def printBestPredictorScores(self, bestEstimator):
        scoresMean = np.mean(bestEstimator.scores)
        scoresStd = np.std(bestEstimator.scores)
        print("%0.3f (+/-%0.03f)" % (scoresMean,scoresStd / 2))
#         print('rates: ', bestEstimator.rates)
        
    def loadHeldoutDataGenerator(self):
        x_heldout, y_heldout, trialsInfo_heldout = self.getXY(self.STEP_SAVE_HELDOUT_DATA) 
        yield (x_heldout, y_heldout, trialsInfo_heldout, 'allData')
    
    def heldoutDataInfoGenerator(self):
        yield 'allData'              
 
    # *************** feature Generators ******************
      
    def featuresGeneratorFoldsChannelsFeatures(self,x,y,cv,channelsNums,featureExtractors, n_jobs=-2):
        retParams = self.featuresGeneratorParralel(x, y, cv, self._featuresGeneratorFoldsChannelsFeatures, 
                                    channelsNums,featureExtractors,n_jobs=n_jobs)
        for params in retParams: yield params

    def featuresGeneratorFoldsChannels(self,x,y,cv,channelsNums, featureExtractors=None, n_jobs=-2):
        return self.featuresGeneratorParralel(x, y, cv, self._featuresGeneratorFoldsChannels, 
                                    channelsNums, n_jobs=n_jobs)
        
    def featuresGeneratorFoldsWindows(self,x,y,cv, windowSizes, windowsNum, n_jobs=-2):
        return self.featuresGeneratorParralel(x, y, cv, self._featuresGeneratorFoldsWindows, 
                                              windowSizes=windowSizes, windowsNum=windowsNum, n_jobs=n_jobs)
    
    def featuresGeneratorParralel(self,x,y,cv,parralelFunc,channelsNums=None,featureExtractors=None,windowSizes=None,windowsNum=None, n_jobs=-2):
#         params = [Bunch(x=x, y=y, fold=fold, train=train,test=test, channelsNums=channelsNums, featureExtractors=featureExtractors, 
#                         windowSizes=windowSizes, windowsNum=windowsNum) for fold, (train,test) in enumerate(cv)]                
        params = [Bunch(xtrain=mpHelper.ForkedData(x[train_index]), ytrain=mpHelper.ForkedData(y[train_index]), 
                        xtest=mpHelper.ForkedData(x[test_index]), ytest=mpHelper.ForkedData(y[test_index]), 
                        fold=fold, channelsNums=channelsNums, featureExtractors=featureExtractors, 
                        windowSizes=windowSizes, windowsNum=windowsNum) for fold, (train_index, test_index) in enumerate(cv)]        

        if n_jobs==1: mapResults = [parralelFunc(featuresParams) for featuresParams in params] # For debugging
        else: mapResults = utils.parmap(parralelFunc, params, n_jobs)
        return [channelParams for foldParams in mapResults for channelParams in foldParams]
        
    def _featuresGeneratorFoldsChannelsFeatures(self,p):
        retParams = []
        print('fold {}'.format(p.fold))
        xtrain = p.xtrain.value
        ytrain = p.ytrain.value
        xtest = p.xtest.value
        ytest = p.ytest.value
        xtrain, ytrain = MLUtils.boost(xtrain, ytrain)
        # Channels selector
        for channelsNum in p.channelsNums:  
            channlesSelector = self.channelsSelector(channelsNum)    
            xtrainChannles = channlesSelector.fit_transform(xtrain, ytrain)
            xtestChannles = channlesSelector.transform(xtest)
            # features extraction
            for featureExtractorName in p.featureExtractors:
                xtrainFeatures = self.featureExtractor(featureExtractorName, xtrainChannles)
                xtestFeatures = self.featureExtractor(featureExtractorName, xtestChannles)
                retParams.append(Bunch(fold=p.fold,xtrain=xtrainFeatures,ytrain=ytrain,
                                       xtest=xtestFeatures,ytest=ytest,
                                       channelsNum=channelsNum,featureExtractorName=featureExtractorName))
        return retParams

        
    def _featuresGeneratorFoldsChannels(self,p):
        retFiles = []
        print('fold {}'.format(p.fold))
        xtrain = p.xtrain.value
        ytrain = p.ytrain.value
        xtest = p.xtest.value
        ytest = p.ytest.value
        xtrain, ytrain = MLUtils.boost(xtrain, ytrain)
        # Channels selector
        for channelsNum in p.channelsNums:  
            channlesSelector = self.channelsSelector(channelsNum)    
            xtrainChannles = channlesSelector.fit_transform(xtrain, ytrain)
            xtestChannles = channlesSelector.transform(xtest)
            resultsFileName = self.preparePredictionsParamtersFileName('{}_{}'.format(p.fold,channelsNum))
            utils.save(Bunch(fold=p.fold,xtrainChannles=xtrainChannles,
                                   ytrain=ytrain,xtestChannles=xtestChannles,
                                   ytest=ytest,channelsNum=channelsNum),resultsFileName)
            print('finish for channelsNum {}'.format(channelsNum))
            retFiles.append(resultsFileName)
        return retFiles

    def _featuresGeneratorFoldsWindows(self,p):
        retFiles = []
        print('fold {}'.format(p.fold))
        xtrain = p.xtrain.value
        ytrain = p.ytrain.value
        xtest = p.xtest.value
        ytest = p.ytest.value
        xtrain, ytrain = MLUtils.boost(xtrain, ytrain)
        T = xtrain.shape[1]
        for windowSize in p.windowSizes:
            timeSelector = self.timeSelector(0, windowSize,p.windowsNum,T)
            for timeSelector.startIndex in timeSelector.windowsGenerator():
                xtrainTimed = timeSelector.fit_transform(xtrain)
                xtestTimed = timeSelector.transform(xtest)
                resultsFileName = self.preparePredictionsParamtersFileName('{}_{}_{}'.format(p.fold,windowSize,timeSelector.startIndex))
                utils.save(Bunch(fold=p.fold,xtrain=xtrainTimed,
                    ytrain=ytrain,xtest=xtestTimed,ytest=ytest,windowSize=windowSize,
                    startIndex=timeSelector.startIndex),resultsFileName)
                retFiles.append(resultsFileName)
        return retFiles
    
    def featuresGeneratorChannlesFolds(self,x,y,cv,channelsNums,featureExtractors):
        # Channels selector
        for channelsNum in channelsNums:  
            print('process: channelsNum {}'.format(channelsNum))
            channlesSelector = self.channelsSelector(channelsNum)    
            xChannles = channlesSelector.fit_transform(x)
            # features extraction
            xFeatures = self.featureExtractor(xChannles)
            for fold, (train,test) in enumerate(cv):
                xtrainFeatures, ytrain, xtestFeatures, ytest = xFeatures[train], y[train], xFeatures[test], y[test] 
                xtrainFeatures, ytrain = MLUtils.boost(xtrainFeatures, ytrain)
                yield (Bunch(fold=fold,xtrain=xtrainFeatures,ytrain=ytrain,
                             xtest=xtestFeatures,ytest=ytest,channelsNum=channelsNum))    

    # *************** Params generators functions *********
    
    def _prepareFeaturesGeneratorParams(self,p):
        params = [Bunch(fileName=fileName,channelsNums=p.channelsNums,timePercentiles=p.timePercentiles,
            featureExtractors=p.featureExtractors,kernels=p.kernels,Cs=p.Cs,gammas=p.gammas) for fileName in p.files]
        return params

    def _prepareCVParams(self, p):
        paramsNum = len(list(p.cv))
        params = []
        index = 0
        for fold, (trainIndex, testIndex) in enumerate(p.cv):
            params.append(Bunch(xtrain=mpHelper.ForkedData(p.x[trainIndex]),
                ytrain=mpHelper.ForkedData(p.y[trainIndex]),
                xtest=mpHelper.ForkedData(p.x[testIndex]),
                ytest=mpHelper.ForkedData(p.y[testIndex]),
                trainTrialsInfo=p.trialsInfo[trainIndex],
                testTrialsInfo=p.trialsInfo[testIndex],
                paramsNum=paramsNum, fold=fold, kernels=p.kernels,
                Cs=p.Cs, gammas=p.gammas, index=index))
            index += 1
        return params

    def _predictorParamtersKeyClass(self):
        return namedtuple('predictorParamters',['timePercentile','channelsNum','kernel','C','gamma']) 

    def predictorParamtersKeyItem(self,  res, predRes):
        PredictorParamtersKey = self._predictorParamtersKeyClass()
        return PredictorParamtersKey(channelsNum=res.channelsNum, timePercentile=res.timePercentile,       
                            kernel=predRes.kernel,C=predRes.C,gamma=predRes.gamma)

    def passRatesThreshold(self, rates, ratesThreshold=0.5):
        return ((rates[0] > ratesThreshold and rates[1] >= ratesThreshold) or
                (rates[0] >= ratesThreshold and rates[1] > ratesThreshold))

    # *************** Linking functions *******************

    def featuresCV(self, y, trialsInfo, foldsNum, testSize=None):
        if (testSize is None):
            testSize = 1.0 / foldsNum
            print('testSize is None, set it to {}'.format(testSize))
        return StratifiedShuffleSplit(y, foldsNum, testSize, random_state=0)

    def featuresGenerator(self,x,y,cv,channelsNums,featureExtractors=None,n_jobs=-2):
        return self.featuresGeneratorFoldsChannels(x, y, cv, channelsNums, featureExtractors, n_jobs) 
    
    def preparePredictionParamtersLinkFunctions(self):
        return self._preparePredictionsParamtersChannelsFeaturesParallel
    
    def channelsSelector(self, channelsNum):
        return GSS.ChannelsSelector(channelsNum)

    def featureExtractor(self,featureExtractorFuncName,X):
        return Analyzer.FEATURE_EXTRACTORS[featureExtractorFuncName](X)

    def timeSelector(self,timePercentile):
        return feature_selection.SelectPercentile(feature_selection.f_classif, percentile=timePercentile)
    
#     def scorer(self,ytest,ypred):
#         return sf.gmeanScore(ytest,ypred)

    def scorer(self, ytest, probs):
        ypred = MLUtils.probsToPreds(probs)
        return sf.accuracyScore(ytest, ypred)

    def gridSearchScorer(self, ytest, probs):
        ypred = MLUtils.probsToPreds(probs)
        return sf.accuracyScore(ytest, ypred)

    def probsScorer(self, ytest, probs):
        return np.array([p[int(y)] for
            p, y in zip(probs, ytest)])

    def bestScore(self, scores, isProbs=False):
        if (isProbs):
            return np.mean(scores,(1,2))
        else:
            return np.mean(scores)

    def predictor(self,C,kernel,gamma):
        return GSS.TSVC(C=C,kernel=kernel,gamma=gamma)

    def dataFileName(self, stepID, folder='', noShuffle=False):
        if (folder == ''):
            folder = self.dataFolder
        if (not utils.folderExists(folder)):
            utils.createDirectory(folder)
        fileName = '{}/{}{}{}_{}_{}_{}{}_sub_{}.npz'.format(
            folder, self.indetifier,
            '_shuffled' if self.shuffleLabels and not noShuffle else '',
            'smote' if self.useSmote and not noShuffle else '',
            self.PROCS_NAMES[self.procID],
            self.getStepName(stepID), self.selectorName,
            '_Weights_{}'.format(self.weightsFileName) \
            if self.multipleByWeights else '', self.subject)
        return fileName

    @classmethod
    def staticDataFileName(cls,folder,indetifier,procID,timeSelectorName):
        return '{}/{}_{}_{}_staticData.npz'.format(folder, indetifier, cls.PROCS_NAMES[procID], timeSelectorName)        

    @property
    def timeAxisFileName(self):
        oldName = '{}/{}_{}_{}_timeAxis.npy'.format(self.dataFolder,
            self.matlabFile, self.indetifier, self.subject)
        newName = '{}/{}_{}_{}_timeAxis.npy'.format(self.dataFolder,
            self.indetifier, self.selectorName, self.subject)
        if (utils.fileExists(newName)):
            return newName
        else:
            if (utils.fileExists(oldName)):
                utils.renameFile(oldName, newName)
            return newName

    @property
    def defaultFilePrefix(self):
        return self.dataFileName(self.STEP_FEATURES)[:-4]

    @property
    def defaultFileNameBase(self):
        return path(self.defaultFilePrefix).namebase

    @property
    def hdf5FileName(self):
        dataFileName = self.dataFileName(self.STEP_NONE)
        return '{}.hdf'.format(dataFileName[:-4])

    @property
    def metaParametersFileName(self):
        return '{}_metaParameters.pkl'.format(self.defaultFilePrefix)

    @property
    def blenderFileName(self):
        return '{}_blender.csv'.format(self.defaultFilePrefix)

    @property
    def resultsFileName(self):
        return '{}_results.pkl'.format(self.defaultFilePrefix)

    @property
    def predictionsParamtersFileName(self):
        return '{}_predictionsParamters.pkl'.format(self.defaultFilePrefix)

    @property
    def predictionScoresFileName(self):
        return '{}_predictionsScores.pkl'.format(self.defaultFilePrefix)

    def preparePredictionsParamtersFileName(self,index):
        return '{}_predictionsParamters_{}.pkl'.format(self.dataFileName(self.STEP_FEATURES,self.tempFolder)[:-4],index)

    @property
    def paramsFileName(self):
        return '{}_params.pkl'.format(self.defaultFilePrefix)

    @property
    def predictResultsFileName(self):
        return '{}_predictResults.pkl'.format(self.defaultFilePrefix)

    @property
    def parametersForProcessFileName(self):
        return '{}_parametersForProcess.pkl'.format(self.defaultFilePrefix)

    @property
    def bestEstimatorsFileName(self):
        return '{}_bestEstimator.pkl'.format(self.defaultFilePrefix)

    @property
    def bestEstimatorsPerWindowFileName(self):
        return '{}_bestEstimatorsPerWindow.pkl'.format(self.defaultFilePrefix)

    @property
    def scorerFoldsResultsFileName(self):
        return '{}_scorerFoldsResults.pkl'.format(self.defaultFilePrefix)

    def weightsFullFileName(self, samWeights):
        return os.path.join(self.folder, self.subject, 'SAM', samWeights)

    def weightsFullMetaFileName(self, samWeights):
        return os.path.join(self.folder, self.subject, 'SAM',
            '{}MetaData.mat'.format(samWeights[:-4]))

    @property
    def weightsDicKey(self):
        return 'weights'

    def allDataPredictorFileName(self,featureExtractorName):
        return '{}_{}_AllDataPredictor.pkl'.format(self.defaultFilePrefix,featureExtractorName)

    def heldoutPredictionFileName(self,featureExtractorName,info):
        return '{}_{}_{}_heldoutPrediction.pkl'.format(self.defaultFilePrefix,featureExtractorName,info)
    
    def heldoutPredictionFileNames(self,featureExtractorName):
        return utils.filesInFolder(self.dataFolder, '*heldoutPrediction.pkl')
    
    def bestEstimatorFileName(self,featureExtractorName):
        return '{}_{}_bestEstimator.pkl'.format(self.defaultFilePrefix,featureExtractorName)
   
#     def figureFileName(self,stepID,figType='jpg'):
#         return '{}.{}'.format(self.dataFileName(stepID, self.figuresFolder)[:-4], figType)

    def ROCFigName(self,figType='jpg'):
        return '{}_ROC.{}'.format(self.dataFileName(self.STEP_FEATURES, self.figuresFolder)[:-4], figType)

    @property
    def subjectFiguresFolder(self):
        folder = os.path.join(self.folder, self.subject, 'figures')
        utils.createDirectory(folder)
        return folder

    def figureFileName(self, figureName):
        return os.path.join(self.subjectFiguresFolder, figureName)

    @property
    def dataFolder(self): 
        folderName = os.path.join(self.folder,'svmFiles')
        utils.createDirectory(folderName)
        return folderName  

    @property
    def dumpFolder(self):
        folderName = os.path.join(self.dataFolder,'dump')
        utils.createDirectory(folderName)
        return folderName

    @property
    def tempFolder(self):
        folderName = os.path.join(self.dataFolder, 'temp', self.subject)
        utils.createDirectory(folderName)
        return folderName

    @property
    def figuresFolder(self):
        folderName = os.path.join(self.folder, 'figures')
        utils.createDirectory(folderName)
        return folderName

    def getXY(self, stepID):
        if (tabu.DEF_TABLES):
            self.hdfFile = tabu.openHDF5File(self.hdf5FileName)
            self.hdfGroup = tabu.findOrCreateGroup(self.hdfFile,
                self.STEPS[stepID])
            return (self.hdfGroup.x, self.hdfGroup.y,
                    self.hdfGroup.trialsInfo)
        else:
            dic = utils.load(self.dataFileName(stepID, noShuffle=True),
                useNumpy=True)
            return (dic['x'], dic['y'], dic['trialsInfo'])

    def saveXY(self, x, y, stepID, trialsInfo=None):
        if (tabu.DEF_TABLES):
            self.hdfFile.close()
        else:
            np.savez(self.dataFileName(stepID), x=x, y=y,
                     trialsInfo=trialsInfo)

    def calcTimeStep(self, T=None):
        ''' return time bin '''
#         xAxis = utils.load(self.timeAxisFileName, useNumpy=True)
        return (self.xAxis[1] - self.xAxis[0])

    def getStepName(self, stepID):
        return self.STEPS[stepID]

    @staticmethod
    def transposeTimeAndChannels(x):
        N,C,T = x.shape
        newx = np.zeros((N,T,C))
        for i in range(x.shape[0]):
            newx[i,:,:] = x[i,:,:].T            
        return newx

    @staticmethod    
    def transposeChannelsAndTime(x):
        N,T,C = x.shape
        newx = np.zeros((N,C,T))
        for i in range(x.shape[0]):
            newx[i,:,:] = x[i,:,:].T            
        return newx

    @staticmethod
    def baselineCorrection(x,onsetInd):
        C,T = x.shape
        x-=np.tile(np.mean(x[:,:onsetInd-1],1),(T,1)).T
        return x 

    def normalizeFeatures(self, x, trialsInfo, field=''):
        if (field==''): utils.throwException('normalizeFeatures with empty field!')
        ids = np.array([trialInfo[field] for trialInfo in trialsInfo])
        uids = np.unique(ids)
        for id in uids:
            indices = np.where(ids==id)[0]
            x[indices],_,_ = MLUtils.normalizeData(x[indices])
        return x

    @staticmethod  
    def _parallelPrediction():
        pass
        
    @property
    def timeSelectorName(self):
        return ''
    
    @property
    def channlesFileName(self):
        return '{}_channels.pkl'.format(self.defaultFilePrefix)

    @property
    def channlesMatlabFileName(self):
        return '{}_channels'.format(self.defaultFilePrefix)

    @property
    def heldOutFeaturesExtractionResultsFileName(self):
        return '{}_heldOutFeaturesExtractionResults.pkl'.format(self.defaultFilePrefix)

    def plotMeanTimeSeriesPerSensor(self, sensors=None):
        folder = os.path.join(self.folder,self.subject,'timeSeriesPerSensor')
        utils.createDirectory(folder)
        X,y,_ = self.getXY(self.STEP_SPLIT_DATA)
        N,T,C = X.shape
        if (sensors is None): sensors = range(C)
        for c in sensors:
            x0 = np.mean(X[y==0,:,c],0)
            x1 = np.mean(X[y==1,:,c],0)
            plots.graph2(range(T), x0, x1, self.LABELS[self.procID], xlabel='Time(ms)', ylabel='Magnetic Field (T)', fileName=os.path.join(folder,'timeSeries_{}'.format(c)))

#    def plotScores(self,fs):
#         k = fs.get_params()['k']
#         xr = np.arange(1,len(fs.get_support())+1)
#         scores = -np.log10(fs.pvalues_) #fs.scores_
#         inds = np.argsort(scores)[::-1]
#         print(xr[inds][:k])
#         plt.plot(xr, scores)
#         plt.scatter(xr[fs.get_support()],scores[fs.get_support()],s=40,c='r')
#         plt.xlim([0,len(fs.pvalues_)])
#         plt.ylim([0,max(scores)*1.1])
#         plt.show()        
#     
#     def plotPValues(self):
#         estimator = utils.load(self.resultsFileName)
#         fs = estimator.best_estimator_.named_steps['anova']
# 
#         x,y,trialsInfo = self.getXY(self.STEP_TRIALS_ANALYSIS)
#         T = x.shape[1]
#         maxx = np.max(np.mean(x,0))
#         xr = np.arange(1,T+1)
#         scores = -np.log10(fs.pvalues_) 
#         scores /= (maxx/np.max(scores))
#         plt.plot(xr,scores, label='p-values')
#         plt.scatter(xr[fs.get_support()],scores[fs.get_support()],s=40,c='r', label='chosen features')
# 
#         sns.tsplot(xr,x[y==0], label=self.LABELS[self.procID][0])
#         sns.tsplot(xr,x[y==1], label=self.LABELS[self.procID][1])
#         plt.xlim([1,T+1])
#         plt.ylim([0,maxx*1.1])
#         plt.legend()    
# #         plt.xlim([0,len(fs.pvalues_)])
#         plt.show()    
#     
#     def ttest(self, stepID):
#         x,y,trialsInfo = self.getXY(stepID)
#         T = x.shape[1]
#         for i in xrange(T):
#             utils.printTtestResult(x[y==0,i], x[y==1,i], '{}_{}'.format(self.LABELS[self.procID][0],i+1),'{}_{}'.format(self.LABELS[self.procID][1],i+1),False)
#         
#     def plot(self, stepID): 
#         x,y,trialsInfo = self.getXY(stepID)
#         T = x.shape[1]
#         sns.tsplot(range(1,T+1),x[y==0], label=self.LABELS[self.procID][0])
#         sns.tsplot(range(1,T+1),x[y==1], label=self.LABELS[self.procID][1])
#         plt.xlim([1,T+1])
#         plt.legend()
#         plt.savefig(self.figureFileName(stepID))
#         plt.show()    
#     