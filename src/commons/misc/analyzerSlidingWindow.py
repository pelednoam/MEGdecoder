'''
Created on Feb 12, 2014

@author: noampeled
'''

from analyzerSemiSupervised import Analyzer

import numpy as np
import scipy
from sklearn.datasets.base import Bunch
import os

import itertools
from collections import defaultdict,namedtuple
import operator

import seaborn as sns
import matplotlib.pyplot as plt
import plots

from src.commons.utils import utils
from src.commons.utils import MLUtils
from src.commons import GridSearchSteps as GSS
from src.commons import scoreFunctions as sf

from __builtin__ import Exception


class AnalyzerSW(Analyzer):
    '''
    calculates the prediction score for a sliding window
    '''
    
    def process(self,timePercentiles=[0],foldsNum=5,Cs=[1],gammas=[0],channelsNums=[30], kernels=['rbf'],windowSizes=[100], windowsNum=10, featureExtractors=None, 
                                    channelsPerWindow=True, n_jobs=-2):
        ''' Step 3) Processing the data ''' 
        print('Proccessing the data')
        x,y,trialsInfo = self.getXY(self.STEP_SPLIT_DATA) 
        
        cv = self.featuresCV(y,trialsInfo,foldsNum)
        if (featureExtractors is None):
            raise Exception('No feature extractors!') 

        files = self.featuresGenerator(x,y,cv,channelsNums,None,windowSizes,windowsNum,channelsPerWindow,n_jobs)
        T = utils.load(files[0]).xtrain.shape[1] # Ugly patch to get T
        params = [Bunch(fileName=fileName,channelsNums=channelsNums,windowSizes=windowSizes,windowsNum=windowsNum,
            featureExtractors=featureExtractors,kernels=kernels,Cs=Cs,gammas=gammas) for fileName in files]
        
        func = self._preparePredictionsParamtersChannelsFeaturesParallel if channelsPerWindow else self._preparePredictionsParamtersWindowsFeaturesParallel
        if n_jobs==1: mapResults = [func(p) for p in params] # For debugging
        else: mapResults = utils.parmap(func, params, n_jobs)
        utils.save(mapResults, self.paramsFileName)
        for tmpfile in files: os.remove(tmpfile)
        utils.save(Bunch(windowsNum=windowsNum, featureExtractors=featureExtractors, T=T, foldsNum=foldsNum), self.predictionsParamtersFileName)

    def _preparePredictionsParamtersChannelsFeaturesParallel(self, gp):
        tic = utils.ticToc()
        p = utils.load(gp.fileName)
        results = []
        for channelsNum in gp.channelsNums:  
            channlesSelector = self.channelsSelector(channelsNum)    
            xtrainChannles = channlesSelector.fit_transform(p.xtrain, p.ytrain)
            xtestChannles = channlesSelector.transform(p.xtest)
            # features extraction
            for featureExtractorName in gp.featureExtractors:
                xtrainFeatures = self.featureExtractor(featureExtractorName, xtrainChannles)
                xtestFeatures = self.featureExtractor(featureExtractorName, xtestChannles)
                res = self._predict(Bunch(xtrainFeatures=xtrainFeatures,ytrain=p.ytrain,
                    xtestFeatures=xtestFeatures,kernels=gp.kernels,Cs=gp.Cs,gammas=gp.gammas))
                results.append(Bunch(predResults=res,fold=p.fold,ytest=p.ytest,channelsNum=channelsNum,
                    windowSize=p.windowSize,featureExtractorName=featureExtractorName,startIndex=p.startIndex))
        utils.howMuchTimeFromTic(tic, '_preparePredictionsParamtersChannelsFeaturesParallel')
        resultsFileName = '{}_results.pkl'.format(gp.fileName[:-4])
        utils.save(results, resultsFileName)
        return resultsFileName
    
    def _preparePredictionsParamtersWindowsFeaturesParallel(self, gp):
        tic = utils.ticToc()
        p = utils.load(gp.fileName)
        results = []
        # Time selector
        T = p.xtrain.shape[1]
        for windowSize in gp.windowSizes:
            timeSelector = self.timeSelector(0, windowSize,gp.windowsNum,T)
            for timeSelector.startIndex in timeSelector.windowsGenerator():
                xtrainTimed = timeSelector.fit_transform(p.xtrain)
                xtestTimed = timeSelector.transform(p.xtest)
                # features extraction
                for featureExtractorName in gp.featureExtractors:
                    xtrainFeatures = self.featureExtractor(featureExtractorName, xtrainTimed)
                    xtestFeatures = self.featureExtractor(featureExtractorName, xtestTimed)
                    res = self._predict(Bunch(xtrainFeatures=xtrainFeatures,ytrain=p.ytrain,
                        xtestFeatures=xtestFeatures,kernels=gp.kernels,Cs=gp.Cs,gammas=gp.gammas))
                    results.append(Bunch(predResults=res,fold=p.fold,ytest=p.ytest,channelsNum=p.channelsNum,
                        windowSize=windowSize,featureExtractorName=featureExtractorName,startIndex=timeSelector.startIndex))
        utils.howMuchTimeFromTic(tic, '_preparePredictionsParamtersWindowsFeaturesParallel')
        resultsFileName = '{}_results.pkl'.format(gp.fileName[:-4])
        utils.save(results, resultsFileName)
        return resultsFileName

    def process(self, n_jobs=-2):
        '''
        Nothing to do here, the prediction is done in process
        '''
        pass

    def _predict(self,p):
#         tic = utils.ticToc()
        allCSVParams = itertools.product(*(p.kernels,p.Cs,p.gammas))
        results = []
        for kernel,C,gamma in allCSVParams:
            svc = GSS.TSVC(C=C,kernel=kernel,gamma=gamma)
            svc.fit(p.xtrainFeatures,p.ytrain)
            probs = svc.predict(p.xtestFeatures)
            results.append(Bunch(probs=probs,kernel=kernel,C=C,gamma=gamma))            

#         utils.howMuchTimeFromTic(tic, 'AnalyzerSW.predict')
        return results

    def calculatePredictionsScores(self):
        predParams = utils.load(self.predictionsParamtersFileName)
        scorerFoldsResults = {}
        for featureExtractorName in predParams.featureExtractors:
            scorerFoldsResults[featureExtractorName] = defaultdict(list)
        PredictorParamtersKey = namedtuple('predictorParamters',['channelsNum','windowSize','kernel','C','gamma'])
        for res,predRes in self.predictorResultsGenerator():
            score = self.gridSearchScorer(res.ytest,predRes.probs)
            probsScore = self.probsScorer(res.ytest, predRes.probs)
            # Set the key as PredictorParamtersKey, not Bunch, because Bunch can't be used as a dictionary key
            key = PredictorParamtersKey(channelsNum=res.channelsNum,windowSize=res.windowSize,
                        kernel=predRes.kernel,C=predRes.C,gamma=predRes.gamma)
            scorerFoldsResults[res.featureExtractorName][key].append(Bunch(
                score=score, probsScore=probsScore, fold=res.fold,startIndex=res.startIndex))

        bestEstimators = {}
        bestEstimatorsPerWindow = defaultdict(dict)
        # loop over the different features extractors
        for featureExtractorName, scorerFoldsResultsDic in scorerFoldsResults.iteritems():
            bestScore = 0  
            bestScorePerWindow=defaultdict(int)
            for predictorParamters,predictorResults in scorerFoldsResultsDic.iteritems():
                # sort according to startIndex and fold number
                results = sorted(predictorResults, key=operator.itemgetter('startIndex','fold'))
                # group by startIndex
                resultsByFold = itertools.groupby(results, key=operator.itemgetter('startIndex'))
                probsScores, scores = [],[]
                for startIndex, resultsPerStartIndex in resultsByFold:
                    scoresPerWindow, probsScoresPerWindow = [],[] 
                    for res in resultsPerStartIndex:
                        scoresPerWindow.append(res.score)
                        probsScoresPerWindow = np.concatenate((probsScoresPerWindow, res.probsScore))
                    # Find the best estimator per window 
                    if (np.mean(scoresPerWindow) > bestScorePerWindow[startIndex]):
                        bestEstimatorsPerWindow[featureExtractorName][startIndex] = Bunch(
                            parameters=predictorParamters, probsScores=probsScoresPerWindow, scores=scoresPerWindow)
                        bestScorePerWindow[startIndex] = np.mean(scoresPerWindow)
                    probsScores = utils.arrAppend(probsScores,probsScoresPerWindow)
                    scores.append(scoresPerWindow)
                probsScores = np.array(probsScores)
                scores = np.array(scores)
                if (self.bestScore(scores) > bestScore):
                    bestEstimators[featureExtractorName] = Bunch(
                        parameters=predictorParamters, probsScores=probsScores, scores=scores)
                    bestScore = self.bestScore(scores)
        
        for featureExtractorName in bestEstimators.keys():
            # The namedtuple PredictorParamters isn't pickleable, so convert it to Bunch 
            bestEstimators[featureExtractorName].parameters = utils.namedtupleToBunch(bestEstimators[featureExtractorName].parameters) 
            utils.save(bestEstimators[featureExtractorName],self.bestEstimatorFileName(featureExtractorName))
            bestEstimatorsPerWindow[featureExtractorName] = utils.sortDictionaryByKey(bestEstimatorsPerWindow[featureExtractorName])
            for startIndex in bestEstimatorsPerWindow[featureExtractorName].keys():
                bestEstimatorsPerWindow[featureExtractorName][startIndex].parameters = utils.namedtupleToBunch(bestEstimatorsPerWindow[featureExtractorName][startIndex].parameters)
        utils.save(bestEstimators, self.bestEstimatorsFileName)
        utils.save(bestEstimatorsPerWindow, self.bestEstimatorsPerWindowFileName)

    def predictorResultsGenerator(self):
        files = utils.load(self.paramsFileName)
        for fileName in files:
            # To allow to load the tmp files from a computer where it wasn't been written
            newFileName = utils.getLocalFullFileName(fileName, self.folder, 3) 
            # If the file doens't exist, download it using sftp
            if (utils.sftp): utils.sftp.getAndMove(fileName,os.path.dirname(newFileName))
            results = utils.load(newFileName)
            for res in results:
                for predRes in res.predResults:
                    yield (res,predRes)

#     def calculateBestEstimatorRates(self,bestEstimator,featureExtractorName):
#         predParams = utils.load(self.predictionsParamtersFileName)
#         x,y = self.getXY(self.STEP_SPLIT_DATA) 
#         cv = self.featuresCV(y,predParams.foldsNum)
#         for train,test in cv:        

    def analyzeResults(self, calcProbs=True,probsThreshold=0.5,doPlot=True,calcOnHeldoutData=False,printResults=False):
        self.xAxis = self.loadTimeAxis()
        bestEstimators = utils.load(self.bestEstimatorsFileName)
        bestEstimatorsPerWindow = utils.load(self.bestEstimatorsPerWindowFileName)
        predParams = utils.load(self.predictionsParamtersFileName)
        colors = sns.color_palette(None, len(bestEstimators))
        featureExtractors = bestEstimators.keys()
#         scoresGenerator = self.scoresGenerator(bestEstimators, predParams, self.xAxis, calcProbs, probsThreshold)
        scoresGenerator = self.scoresGeneratorPerWindow(bestEstimatorsPerWindow, predParams, self.xAxis, calcProbs, probsThreshold,printResults)
        probScoresAggregator, scoresAggregator = {},{}
        meanScoresAggregator, meanProbScoresAggregator = [], []

        if (doPlot): fig=plt.figure()
        for (xAxis, scores, probScores),featureExtractorName,color in zip(scoresGenerator,bestEstimators.keys(),colors):
            probScoresAggregator[featureExtractorName] = probScores
            scoresAggregator[featureExtractorName] = scores
            try:
                meanProbScoresAggregator = utils.arrAppend(meanProbScoresAggregator, np.mean(probScores,1))
                meanScoresAggregator = utils.arrAppend(meanScoresAggregator, np.mean(scores,1))
            except:
                pass
            if (doPlot):
                if (calcProbs):
                    if (featureExtractorName=='all'):
                        AnalyzerSW.plotInformationTiming(probScores, xAxis, 0.8, featureExtractorName)
                else:
                    sns.tsplot(probScores.T,xAxis,label=featureExtractorName, color=color)
        
        bestProbsFeatureExtractors = np.argmax(meanProbScoresAggregator,0)
        bestFeatureExtractors = np.argmax(meanScoresAggregator,0)
        bestProbScores = np.empty(probScoresAggregator['all'].shape)
        bestScores = np.empty(scoresAggregator['all'].shape)
        for w in range(bestProbScores.shape[0]):
            bestProbsFeatureExtractorPerWindow = featureExtractors[bestProbsFeatureExtractors[w]]
            bestFeatureExtractorPerWindow = featureExtractors[bestFeatureExtractors[w]]
            bestProbScores[w,:] = probScoresAggregator[bestProbsFeatureExtractorPerWindow][w,:]
            bestScores[w,:] = scoresAggregator[bestFeatureExtractorPerWindow][w]
    
        utils.save((scoresAggregator,probScoresAggregator,bestScores,bestProbScores,xAxis),self.predictionScoresFileName)
        if (doPlot):
            plt.title('Prediction on heldin data')
            plt.xlabel('Time (ms)')
            plt.ylabel('Probabilities' if calcProbs else 'Prediction (AUC)')
            plt.xlim((xAxis[0],xAxis[-1]))
            plt.legend()
            plt.show()
            
    @staticmethod
    def plotInformationTiming(scores, randomScores, xAxis, probsThreshold, label='', subject='', color='k', ylim=None, density=1000, isProbs=False):
#         scores, xAxis = scoresDict[0]
#         randScores, _ = scoresDict[1]
#         dens,densX, peaksRatio = MLUtils.calcPeaks(scores,probsThreshold,xAxis)
#         plt.plot(densX, dens, '-', label=label)
#         randDens, _ , _ = MLUtils.calcPeaks(randScores,probsThreshold,xAxis)
#         plt.plot(densX, randDens, '--',color='r', label='random')
#         plt.plot(densX, np.ones(densX.shape)*(0.05/50.0), '--',color='g', label='chance')
        if (isProbs):
            ratio = np.sum(scores<probsThreshold)*100/float(scores.size)
            scores[scores<probsThreshold]=np.NAN
        sns.tsplot(scores.T,xAxis,label=label, color=color, estimator=scipy.nanmean)
        if (randomScores is not None): sns.tsplot(randomScores.T,xAxis,label='random', color='r')
       
        plt.xlabel('Time (ms)')
        plt.ylabel('Certainty')
        plt.xlim((xAxis[0],xAxis[-1]))
        plt.ylim((0.5 if isProbs else 0,ylim if ylim else 1))
        if (label!=''): plt.legend()
        if (isProbs and probsThreshold>0):
            plt.title('subject: {} ({:.2f}% above {:.2f})'.format(subject,ratio,probsThreshold))
        else:
            plt.title('subject: {}'.format(subject))

    @classmethod
    def plotPredictionScores(cls,subjects,folder,matlabFile,procID,calcProbs=True,probsThreshold=0.5,calcOnHeldoutData=False,doPlot=True,plotHists=False,addShuffle=True,
                             printResults=False,showProbs=False,showOnlyBestFE=True):
        scoresAggregator = defaultdict(lambda : defaultdict(dict))
        probScoresAggregator = defaultdict(lambda : defaultdict(dict))
        bestScoresAggregator,bestProbScoresAggregator = {},{}
#         ylim = {cls.PROC_3_RN:0.8, cls.PROC_3_2:0.62}
        for subject in subjects:
            for isShuffled in ([1,0] if addShuffle else [0]):
                analyze = cls(folder, matlabFile, subject, procID)
                analyze.loadTimeAxis()
                if (isShuffled): analyze.indetifier += "_shuffled"
#                     analyze.calculatePredictionsScores()
                analyze.analyzeResults(calcProbs,probsThreshold,calcOnHeldoutData=False,doPlot=False,printResults=printResults)
                subjectScoresAggregator, subjectProbScores, bestScores, bestProbScores, xAxis = utils.load(analyze.predictionScoresFileName)
                if (not isShuffled): 
                    bestProbScoresAggregator[subject]=bestProbScores
                    bestScoresAggregator[subject]=bestScores
                for ind, ((featureExtractorName, scores),probScores) in enumerate(zip(subjectScoresAggregator.iteritems(), subjectProbScores.values())):
                    probScoresAggregator[subject][featureExtractorName][isShuffled]=probScores
                    scoresAggregator[subject][featureExtractorName][isShuffled]=scores
                    if (len(probScores)==0):
                        raise Exception('empty scores! {} {}'.format(featureExtractorName,subject))
        
        aggregator = probScoresAggregator if showProbs else scoresAggregator
        bestAggregator = bestProbScoresAggregator if showProbs else bestScoresAggregator
        featureExtractors = aggregator[subjects[0]].keys()
        colors = sns.color_palette(None, len(featureExtractors))
        plt.figure(figsize=(10,10))
        for ind,subject in enumerate(subjects):
            plt.subplot(230+ind+1)
            if (showOnlyBestFE):
                    AnalyzerSW.plotInformationTiming(bestAggregator[subject],None,xAxis,probsThreshold,'',subject,'b',isProbs=showProbs)#,ylim[procID])    
            else:
                for (featureExtractorName,scoresDict),color in zip(aggregator[subject].iteritems(),colors):          
                    randomScores = scoresDict[1] if addShuffle else None  
                    AnalyzerSW.plotInformationTiming(scoresDict[0],randomScores,xAxis,probsThreshold,featureExtractorName,subject,color,isProbs=showProbs)#,ylim[procID])    
#         plt.savefig('Certainty_{}.png'.format(cls.PROCS_NAMES[procID]))
        plt.show()

        if (plotHists):
            plt.figure(figsize=(10,10))
            for ind,subject in enumerate(subjects):
                plt.subplot(220+ind+1)
                for featureExtractorName,scoresDict in scoresAggregator[subject].iteritems():
                    if (featureExtractorName=='all'):
                        
                        plots.histCalcAndPlot(scores.flatten(), binsNum=50, show=False)
                        plt.title('probs hist for subject {}'.format(subject))
            plt.show()
    
    def scoresGenerator(self,bestEstimators,predParams,timeAxis,calcProbs=True,probsThreshold=0.5,printResults=True):
        for featureExtractorName,bestEstimator in bestEstimators.iteritems():
            bep = bestEstimator.parameters
            print('Best results for features extractor: {}, channels: {}, windowSize: {}, kernel: {}, c: {}, gamma: {}'.format(
                featureExtractorName,bep.channelsNum,bep.windowSize,bep.kernel,bep.C,bep.gamma))

            timeSelector = self.timeSelector(0, bep.windowSize,predParams.windowsNum,len(timeAxis))# predParams.T)
            startIndices = np.array(timeSelector.windowsGenerator())
            xAxis = timeAxis[startIndices+bep.windowSize/2]
            if (calcProbs):
                W = bestEstimator.probsScores.shape[0]
                probsScores = np.reshape(bestEstimator.probsScores,(W,-1))
                probsScores=probsScores[:,np.max(probsScores,0)>probsThreshold]
                yield (xAxis, probsScores)
            else:
                if (printResults):
                    bestWindowIndex = np.argmax(np.mean(bestEstimator.scores,1))                    
                    scoresMean = np.mean(bestEstimator.scores[bestWindowIndex])
                    scoresStd = np.std(bestEstimator.scores[bestWindowIndex])
                    print("%0.3f (+/-%0.03f)" % (scoresMean,scoresStd / 2))
                yield (xAxis,bestEstimator.scores)

    def scoresGeneratorPerWindow(self,bestEstimatorsPerWindow,predParams,timeAxis,calcProbs=True,probsThreshold=0.5,printResults=True):
        for featureExtractorName,bestEstimatorPerWindow in bestEstimatorsPerWindow.iteritems():
            probsScores = []
            scores = []  #np.empty((len(bestEstimatorPerWindow))); scores.fill(np.NaN)
            for ind, (startIndex,bestEstimator) in enumerate(bestEstimatorPerWindow.iteritems()):
                probsScores = utils.arrAppend(probsScores, bestEstimator.probsScores)
                scores = utils.arrAppend(scores, bestEstimator.scores)
#                 scores[ind] = np.mean(bestEstimator.scores)
                bep = bestEstimator.parameters
                if (printResults):
                    print('Best results for features extractor: {}, startIndex: {}, channels: {}, windowSize: {}, kernel: {}, c: {}, gamma: {}'.format(
                        featureExtractorName,startIndex,bep.channelsNum,bep.windowSize,bep.kernel,bep.C,bep.gamma))
                
            timeSelector = self.timeSelector(0, bep.windowSize,predParams.windowsNum,len(timeAxis))# predParams.T)
            startIndices = np.array(timeSelector.windowsGenerator())
            xAxis = timeAxis[startIndices+bep.windowSize/2]
            if (printResults):
                bestWindowIndex = np.argmax(np.mean(bestEstimator.scores,1))                    
                scoresMean = np.mean(bestEstimator.scores[bestWindowIndex])
                scoresStd = np.std(bestEstimator.scores[bestWindowIndex])
                print("%0.3f (+/-%0.03f)" % (scoresMean,scoresStd / 2))
            yield (xAxis, scores, probsScores)
    
    def calcChannelsPerWindow(self, verbose=False, overwrite=False):
        ''' Calc the different channels per window over all the heldin data '''
        if (utils.fileExists(self.channlesFileName) and not overwrite):
            return utils.load(self.channlesFileName)
        x,y,trialsInfo = self.getXY(self.STEP_SPLIT_DATA)
        self.xAxis = self.loadTimeAxis()
        bestEstimatorsPerWindow = utils.load(self.bestEstimatorsPerWindowFileName)
        predParams = utils.load(self.predictionsParamtersFileName)
        T = x.shape[1]

        channels = defaultdict(dict)
        scores = defaultdict(dict)
        startIndices = {}
        for featureExtractorName,bestEstimatorPerWindow in bestEstimatorsPerWindow.iteritems():
            if (featureExtractorName!=Analyzer.FE_ALL): continue
            if (verbose): print(featureExtractorName)
            for startIndex,bestEstimator in bestEstimatorPerWindow.iteritems():
                if (verbose): print(startIndex)
                bep = bestEstimator.parameters
                timeSelector = self.timeSelector(startIndex,bep.windowSize,predParams.windowsNum,T)
                xTimed = timeSelector.fit_transform(x)
                channlesSelector = self.channelsSelector(bep.channelsNum)
                channelsModel = channlesSelector.fit(xTimed, y)
                channlesIndices = np.arange(x.shape[2])[channelsModel.get_support()]
                channels[featureExtractorName][self.xAxis[startIndex+bep.windowSize/2]]=channlesIndices
                scores[featureExtractorName][self.xAxis[startIndex+bep.windowSize/2]]=channelsModel.scores_
            startIndices[featureExtractorName] = bestEstimatorPerWindow.keys()
            channels[featureExtractorName] = utils.sortDictionaryByKey(channels[featureExtractorName])
            scores[featureExtractorName] = utils.sortDictionaryByKey(scores[featureExtractorName])
        utils.save(channels, self.channlesFileName)
        channelsAll=[x for x in channels[Analyzer.FE_ALL].values()]
        scoresAll=[x for x in scores[Analyzer.FE_ALL].values()] 
        utils.saveToMatlab({'channelsIndices':channelsAll,'scores':scoresAll, 'indices':startIndices[Analyzer.FE_ALL]}, self.channlesMatlabFileName)
        return channels
        
    def calcScoresOnHeldOutData(self):
        print('*** classification report on heldout data ***')
        bestEstimators = utils.load(self.bestEstimatorsFileName)
        predParams = utils.load(self.predictionsParamtersFileName)        
        
        x,y,trialsInfo = self.getXY(self.STEP_SPLIT_DATA)
        x,y = MLUtils.boost(x, y)
        x_heldout, y_heldout, trialsInfo_heldout = self.getXY(self.STEP_SAVE_HELDOUT_DATA)        
        
        heldoutScores = []
        for featureExtractorName,bestEstimator in bestEstimators.iteritems():
            bep = bestEstimator.parameters
            startIndices = np.array(self.timeSelector(0, bep.windowSize,predParams.windowsNum,predParams.T).windowsGenerator())
            channlesSelector = self.channelsSelector(bep.channelsNum)
            svc = self.predictor(bep.C, bep.kernel, bep.gamma)
            xChannles = channlesSelector.fit_transform(x, y)
            xFeatures = self.featureExtractor(featureExtractorName,xChannles)
            xHeldoutChannles = channlesSelector.transform(x_heldout)
            xHeldoutFeatures = self.featureExtractor(featureExtractorName,xHeldoutChannles)
            reports = []
            heldoutScores = np.zeros(len(startIndices))
            for windowNum, startIndex in enumerate(startIndices):
                timeSelector = self.timeSelector(startIndex,bep.windowSize,predParams.windowsNum,predParams.T)
                xFeaturesTimed = timeSelector.fit_transform(xFeatures)
                HeldoutFeaturesTimed = timeSelector.transform(xHeldoutFeatures)
                svc.fit(xFeaturesTimed,y)
                probs = svc.predict(HeldoutFeaturesTimed) 
                ypred = MLUtils.probsToPreds(probs)
                heldoutScores[windowNum] = self.scorer(y_heldout,probs)
#                 print('\nwindowNum: {}'.format(windowNum))
#                 print(classification_report(y_heldout, ypred))   
                _, _, _, _,report = MLUtils.calcConfusionMatrix(y_heldout, ypred, self.LABELS[self.procID],False) 
                reports.append(report)
        
            bestStartIndex = np.argmax(heldoutScores)
            print('{}: best start index: {} for windowSize: {}'.format(featureExtractorName,bestStartIndex,bep.windowSize))
            print(reports[bestStartIndex])
            xAxis = self.xAxis[startIndices+bep.windowSize/2]
            plt.plot(xAxis,heldoutScores,label=featureExtractorName)      
              
        plt.title('Prediction on held-out data for window size {}'.format(bep.windowSize))
        plt.xlabel('Time (ms)')
        plt.ylabel('Prediction (AUC)')
        plt.xlim((xAxis[0],xAxis[-1]))
        plt.legend()
        plt.show()    
        
    def timeSelector(self,startIndex,windowSize,windowsNum, T):
#         return GSS.windowAccumulatorTwice(startIndex, windowSize,windowsNum, T)
        return GSS.TimeWindowSlider(startIndex, windowSize, windowsNum, T)
    
    @property
    def timeSelectorName(self):
        return 'TimeWindowSlider'

    def featuresGenerator(self,x,y,cv,channelsNums=None,featureExtractors=None,windowSizes=None,windowsNum=None,channelsPerWindow=True,n_jobs=-2):
        if (channelsPerWindow):
            return self.featuresGeneratorFoldsWindows(x, y, cv, windowSizes, windowsNum, n_jobs) 
        else:
            return self.featuresGeneratorFoldsChannels(x, y, cv, channelsNums, n_jobs)

    def channelsSelector(self, channelsNum):
        return GSS.ChannelsSelector(channelsNum)
#         return GSS.ChannelsPCA(channelsNum)

    def bestScore(self, scores, isProbs=False):
        if (isProbs):
            return np.max(np.mean(scores,(1,2)))
        else:
            return np.max(np.mean(scores,1))
