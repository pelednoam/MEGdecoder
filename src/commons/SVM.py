'''
Created on Feb 7, 2012

@author: noam
'''

import numpy as np
from src.commons.utils import MLUtils
from src.commons.utils import utils
import math
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit
from sklearn import preprocessing
# from SubjectsKFold import SubjectsKFold, IndicesKFold

RESULTS_FILE_NAME = '../Agents/files/predictors/'
KERNEL_TYPES = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

# def classify(x, y, c=1, kernelType=0, k=10, boostFactor=5, doPlot=False, doCrossValidation=True, doCalcTrainPred=False):
# #    print('SVM %s kernel' % (KERNEL_TYPES[kernelType]))
#
#    if (doCrossValidation):
#        probs, yTestTrue, yTestPred, testIndices = crossValidation(x, y, c, kernelType, k, boostFactor, doPlot, doCalcTrainPred)
#        return (None, probs, yTestTrue, yTestPred, testIndices)
#    else:
#        (classifier, _, probas_, ytrue, ypred) = classifyAllData(x, y, c, kernelType, boostFactor)
#        return (classifier, probas_, ytrue, ypred, [])

def classifyAllData(x, y, c=1, kt=2, boostFactor=8, doScale=True, doShuffle=True):
    classifier = svm.SVC(kernel=KERNEL_TYPES[kt], probability=False, C=c)
    scaler = preprocessing.StandardScaler()
    if (doScale):
        scaler = scaler.fit(x)
        xtrain, ytrain = scaler.transform(x), y
    else:
        xtrain, ytrain = x, y
    (xtrainb, ytrainb) = MLUtils.boost(xtrain, ytrain, utils.LEAVE_CODE, boostFactor)
    if (doShuffle):
        (xtrainb, idx) = MLUtils.shuffle(xtrainb)
        ytrainb = ytrainb[idx]    
    classifier.fit(xtrainb, ytrainb)
    ypred = classifier.predict(xtrain)  
    return (classifier, scaler, ytrain, ypred)

def classifyFold(xtrain, ytrain, xtest, ytest, foldNum=1, c=1, kt=2, gamma=0.0, minorityBF=-1, doShuffle=True, doScale=True, printCM=False, getProbs=True):
    actionsCount = utils.count(ytrain)
#     print(actionsCount)
    minorityLabel = 0 if actionsCount[0]<actionsCount[1] else 1
    majorityLabel = 1-minorityLabel
    if (minorityBF == -1):
        minorityBF = (float(actionsCount[majorityLabel]) / float(actionsCount[minorityLabel])) - 1            
    classifier = svm.SVC(kernel=KERNEL_TYPES[kt], probability=True, C=c, gamma=gamma)
    if (doScale):
        scaler = preprocessing.StandardScaler().fit(xtest)
        xtrain = scaler.transform(xtrain)
        xtest = scaler.transform(xtest)
    (xtrainb, ytrainb) = MLUtils.boost(xtrain, ytrain, -1)
#    actionsCount = utils.count(ytrainb)
#    print('%d fold: stays=%d, leaves=%d, leaveBF=%.1f' % (foldNum, actionsCount[0], actionsCount[1],leaveBF))    
    if (doShuffle):
        (xtrainb, idx) = MLUtils.shuffle(xtrainb)
        ytrainb = ytrainb[idx]
    classifier.fit(xtrainb, ytrainb)
    if (getProbs):
#        probs = classifier.predict_proba(xtest)
        dists = classifier.decision_function(xtest)
        distProbs = MLUtils.distanceToProb(dists)
        yTestPred = MLUtils.probToClass(distProbs) 
        probs = np.array([[1-p, p] for p in distProbs])
    else:
        yTestPred = classifier.predict(xtest)
        probs=[]  
#    yTestPred = classifier.predict(xtest)
#    probs = classifier.predict_proba(xtest)
#    dist = classifier.decision_function(xtest)
#    pdist = math.exp(dist)/(math.exp(dist)+math.exp(-dist)) 
#    probsTrue = (probToClass(probs[0][1])==yTestPred)[0]
#    pdistTrue = (probToClass(pdist)==yTestPred)[0]
#    print(yTestPred[0],probs[0],dist[0][0],pdist,probsTrue,pdistTrue)
    if (printCM):
        (_, _, stayAccuracy, leaveAccuracy) = MLUtils.calcConfusionMatrix(ytest, yTestPred, True)
#    print (c, kt, stayAccuracy, leaveAccuracy, stayAccuracy * leaveAccuracy)
    return yTestPred, ytest, probs

def crossValidationFoldPool(params):
    xtrain, ytrain, xtest, ytest, foldNum, c, kt, boostFactor, doShuffle, doScale = params
    yTestPred, yTestTrue = classifyFold(xtrain, ytrain, xtest, ytest, foldNum, c, kt, boostFactor, doShuffle, doScale)
    return yTestPred, yTestTrue

def crossValidation(X, y, c, kt, k=10, boostFactor=5, doShuffle=True, doScale=True, samplesIDs=None, parallel=False,
                    pool=None, calcOnlyOneFold=True):
    if not samplesIDs: samplesIDs = []
    yTestTrue, yTestPred, testIndices = [], [], []
    cv = StratifiedKFold(y, k) if (len(samplesIDs) == 0) else SubjectsKFold(samplesIDs)   
    if (not parallel):
        for foldNum, (train, test) in enumerate(cv):
            yFoldTestPred, yFoldTestTrue = classifyFold(X[train], y[train], X[test], y[test], foldNum, c, kt, boostFactor, doShuffle, doScale)
            yTestPred.extend(yFoldTestPred)
            yTestTrue.extend(yFoldTestTrue)
            testIndices.extend(test)
            if (calcOnlyOneFold): break
    else:
        params = []
        for foldNum, (train, test) in enumerate(cv):
            params.append((X[train], y[train], X[test], y[test], foldNum, c, kt, boostFactor, doShuffle, doScale))
            testIndices.extend(test)            
        results = pool.map(crossValidationFoldPool, params)
        for (yTestPredFold, yTestTrueFold) in results:
            yTestPred.extend(yTestPredFold)
            yTestTrue.extend(yTestTrueFold)
    
    return (yTestTrue, yTestPred, testIndices)

def twoLevelSVM(X1, X2, y, c, kernelType, boostFactor, k=10, doShuffle=True):
    yTestTrue, yTestPred = [], []
    if (doShuffle):
        (X1, idx) = MLUtils.shuffle(X1)
        X2 = X2[idx]
        y = y[idx]
    cv = StratifiedKFold(y, k)

    classifier0 = svm.SVC(kernel=KERNEL_TYPES[kernelType[0]], probability=True, C=c[0])
    classifier1 = svm.SVC(kernel=KERNEL_TYPES[kernelType[1]], probability=True, C=c[1])
    classifier2 = svm.SVC(kernel=KERNEL_TYPES[kernelType[2]], probability=True, C=c[2])

    for (train, test) in cv:
        (x1train, ytrain, x1test, ytest, x1trainBoost, y1TrainBoost) = calcXYTrainTest(X1, y, train, test, boostFactor[1])
        (x2train, _, x2test, _, x2trainBoost, y2TrainBoost) = calcXYTrainTest(X2, y, train, test, boostFactor[2])
        
        model1 = classifier1.fit(x1trainBoost, y1TrainBoost)
        model2 = classifier2.fit(x2trainBoost, y2TrainBoost)
        probsTrain1 = model1.predict_proba(x1train)
        probsTrain2 = model2.predict_proba(x2train)
        probsTrain = np.vstack((probsTrain1[:, 0], probsTrain2[:, 0])).T
        
        probsTest1 = model1.predict_proba(x1test)
        probsTest2 = model2.predict_proba(x2test)
        probsTest = np.vstack((probsTest1[:, 0], probsTest2[:, 0])).T
        
        # It seems it better without scaling here
#        scaler = preprocessing.Scaler().fit(probsTrain)
#        probsTrain = scaler.transform(probsTrain)
#        probsTest = scaler.transform(probsTest)
        (probsTrainBoost, yProbsTrainBoost) = MLUtils.boost(probsTrain, ytrain, utils.LEAVE_CODE, boostFactor[0])

        model0 = classifier0.fit(probsTrainBoost, yProbsTrainBoost)
        probas_ = model0.predict_proba(probsTest)
        yTestPred.extend([utils.STAY_CODE if p[0] > 0.5 else utils.LEAVE_CODE  for p in probas_])
        yTestTrue.extend(ytest)     
    
    return (yTestTrue, yTestPred)

def calcXYTrainTest(X, y, trainIndices, testIndices, boostFactor):
    scaler = preprocessing.StandardScaler().fit(X[trainIndices])
    xtrain, ytrain = scaler.transform(X[trainIndices]), y[trainIndices]
    xtest, ytest = scaler.transform(X[testIndices]), y[testIndices]
    (xtrainBoost, ytrainBoost) = MLUtils.boost(xtrain, ytrain, utils.LEAVE_CODE, boostFactor)
    return (xtrain, ytrain, xtest, ytest, xtrainBoost, ytrainBoost)
    
        
def calcCrossValidationAccrodingToLabelingVec(X, y, c, kernelType, labelingVec, boostFactor=1, excludeLabels=None,
                                              doPrint=False, doCalcTrainPred=False, doShuffle=True):
    if not excludeLabels: excludeLabels = []
    yTestTrue, yTestPred, yTrainPred, yTrainTrue, probs = [], [], [], [], []

    labelingVec = np.array(labelingVec)
    if (doShuffle):
        (X, idx) = MLUtils.shuffle(X)
        y = y[idx]
        labelingVec = labelingVec[idx]
    uniquVec = [label for label in np.unique(labelingVec) if label not in excludeLabels]
    for label in uniquVec:
        test = np.where(labelingVec == label)[0]
        train = np.where(labelingVec != label)[0]            
        classifier = svm.SVC(kernel=kernelType, probability=True, C=c)
        scaler = preprocessing.StandardScaler().fit(X[train])
        xtrain, ytrain = scaler.transform(X[train]), y[train]
        xtest, ytest = scaler.transform(X[test]), y[test]
        (xtrain, ytrain) = MLUtils.boost(xtrain, ytrain, utils.LEAVE_CODE, boostFactor)
        probas_ = classifier.fit(xtrain, ytrain).predict_proba(xtest)
        yTestPredLabel = [0 if p[0] > 0.5 else 1 for p in probas_]
        yTestPred.extend(yTestPredLabel)
        yTestTrue.extend(ytest)
        probs = utils.arrAppend(probs, probas_)
        
        if (doCalcTrainPred):
            probas_ = classifier.fit(xtrain, ytrain).predict_proba(xtrain)
            yTrainPred.extend([0 if p[0] > 0.5 else 1 for p in probas_])
            yTrainTrue.extend(ytrain)
        
        (_, _, stayAccuracy, leaveAccuracy) = MLUtils.calcConfusionMatrix(ytest, yTestPredLabel, False)
        if (doPrint):
            print('%s: #n=%d, stay accuracy=%.2f, leave accuracy=%.2f' % (label, len(test), stayAccuracy, leaveAccuracy))
            
    return (probs, yTestTrue, yTestPred, yTrainTrue, yTrainPred)


def calcAccuracy(x, y, c, kernelType, k=5, doPlot=False):
    lb = []
    if (len(np.unique(y)) == 2):
        X, _, _ = MLUtils.normalizeData(x)
        accs = []
        for _ in range(1):
            acc, leaveProbs = crossValidation(X, y, c, KERNEL_TYPES[kernelType], k, doPlot)
            accs.append(acc)
            lb.extend(leaveProbs)
        acc = np.mean(accs)
    else:
        acc = calcAccuracyMultiClass(x, y, c, kernelType)
    return acc, lb
        
def calcAccuracyMultiClass(x, y, c, kernelType, k=5):
    totalAccs = []
    for _ in range(1):
        accs = []
        for uy in np.unique(y):
            yh = (y == uy)
            try:
                acc, leaveProbs = crossValidation(x, yh, c, KERNEL_TYPES[kernelType], k, False)
                if (acc != 0.5):
                    accs.append(acc)
            except: pass
        acc = np.mean(accs)
        totalAccs.append(acc)
        
    acc = np.mean(totalAccs)
    return acc

def calcBestC(x, y, kernelType):
    maxAcc = 0
    maxC = 0
    for c in np.linspace(1, 10, 10):
        acc = calcAccuracy(x, y, c, kernelType)
        if (acc > maxAcc):
            maxAcc = acc
            maxC = c
    print('c= %.2f, acc = %.2f' % (maxC , maxAcc))
    return maxAcc

def classifyLibSVM(x, y, fileName, kernelType=0, c=1):
#    lr = predictorResults()  
    X, _, _ = MLUtils.normalizeData(x, isNP=False)        
    
    prob = svm_problem(y, X)  
#    param = svm_parameter('-c %.5f -b 1' % c)
    param = svm_parameter()
    param.kernel_type = kernelType
    param.C = c
    param.probability = 1
    m = svm_train(prob, param)
    svm_save_model('%s.model' % fileName, m)
    p_label, p_acc, p_val = svm_predict(y, x, m, '-b 1')
    print(p_val)
#    utils.save(lr, fileName)
    return m

def loadSVMModel(fileName):
    return svm_load_model('%s.model' % fileName)

def modelPredict(m, x):
    nx, _, _ = MLUtils.normalizeData(x, isNP=False)
    nr_class = m.get_nr_class()
    p = svm_predict([0] * len(nx), [list(nx)], m, '-b 1')
    return p[0][0] if (nr_class == 2) else p[0]       

    
def calcAcc(y, probas_):
    fpr, tpr, _ = roc_curve(y, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    return roc_auc

def hyperParamSearch(x, y):
    best = 0
    for c in range(1, 11):
        for kt in range(3):
            for bs in [5]:
                _, _, yTestTrue, yTestPred, yTrainTrue, yTrainPred = classify(x, y, c=c, kernelType=kt, k=10, boostFactor=bs, doCalcTrainPred=True)
                (_, _, stayAccuracy, leaveAccuracy) = MLUtils.calcConfusionMatrix(yTrainTrue, yTrainPred, False)
                if (stayAccuracy * leaveAccuracy > best):
                    best = stayAccuracy * leaveAccuracy
                    result = (c, kt, bs, stayAccuracy, leaveAccuracy)
                    (_, _, stayAccuracyTest, leaveAccuracyTest) = MLUtils.calcConfusionMatrix(yTestTrue, yTestPred, False)
                    print (c, kt, bs, stayAccuracy, leaveAccuracy, stayAccuracy * leaveAccuracy, 'best!', stayAccuracyTest, leaveAccuracyTest, stayAccuracyTest * leaveAccuracyTest)
                else:
                    print (c, kt, bs, stayAccuracy, leaveAccuracy, stayAccuracy * leaveAccuracy)
    print('Best results:')
    print(result)                
    
