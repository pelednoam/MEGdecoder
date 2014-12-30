'''
Created on Jun 17, 2012

@author: noam
'''

import scipy.io
import os
import pickle
import numpy as np
import time
import csv
import scipy.stats.stats as st
from scipy import stats
import heapq
import math
from collections import Counter, OrderedDict
from itertools import product, cycle 
from path3 import path
#from lxml import etree
import subprocess 
import shlex
import smtplib
import random
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.MIMEBase import MIMEBase
from email import Encoders
import shutil
import types
from datetime import datetime
from functools import wraps


FALSE_CLAIM, TRUE_CLAIM = 'FalseClaim', 'TrueClaim'
ACTIONS = (TRUE_CLAIM, FALSE_CLAIM) 
FALSE_CLAIM_CODE, TRUE_CLAIM_CODE = range(2)

def LINUX():
    return True if (os.name == 'posix') else False

class groupby(dict):
    def __init__(self, seq, key=lambda x:x):
        for value in seq:
            k = key(value)
            self.setdefault(k, []).append(value)
    __iter__ = dict.iteritems

def getFileType(fileName):
    extension = os.path.splitext(fileName)[1][1:].lower()
    return extension.lower()

def typeNumToStr(num):
    if (num in [FALSE_CLAIM_CODE, TRUE_CLAIM_CODE]):
        return FALSE_CLAIM if num == FALSE_CLAIM_CODE else TRUE_CLAIM
    else:
        print('error in typeNumToStr(%d)' % num)
        return ''

def typeStrToNum(typeStr):
    if (typeStr in [FALSE_CLAIM, TRUE_CLAIM]):
        return FALSE_CLAIM_CODE if typeStr == FALSE_CLAIM else TRUE_CLAIM_CODE
    else:
        print('error in typeStrToNum! (%s)' % typeStr)
        return -1

def typeFromFileName(fileName):
    return fileName[0:fileName.index('_')]

def actionsFromFileName(fileName):
    s1 = fileName[fileName.find('_') + 1:]
    s2 = s1[s1.find('_') + 1:]
    return s1[:s1.find('_')], s2[:s2.find('_')]

def concatFolders(folder, subFolder):
    return folder + '//' + subFolder

def save(obj, fileName):
    pklFile = open(fileName, 'w')
    pickle.dump(obj, pklFile)   
    pklFile.close()

def load(fileName):
    pklFile = open(fileName, 'r')
    obj = pickle.load(pklFile)
    pklFile.close()
    return obj    

def printVec(vec):
    if (len(vec.shape) > 0):
        s = '['
        for i, rec in enumerate(vec):
            s += str(rec)
            if (i < len(vec) - 1): s += ', '
        s += '];'
        print(s)
    else:
        print(vec)

def getClassesProbs(ret, y, confusionMatrixProbs):
    ret[0][0].extend(confusionMatrixProbs[np.where(y == 0)[0]][:, 0])
    ret[0][1].extend(confusionMatrixProbs[np.where(y == 0)[0]][:, 1])
    ret[1][0].extend(confusionMatrixProbs[np.where(y == 1)[0]][:, 0])
    ret[1][1].extend(confusionMatrixProbs[np.where(y == 1)[0]][:, 1])

def sortConfusionMatrixProbs(confusionMatrixProbs):
    confusionMatrixProbs[0][0] = np.sort(confusionMatrixProbs[0][0])    
    confusionMatrixProbs[0][1] = np.sort(confusionMatrixProbs[0][1])
    confusionMatrixProbs[1][0] = np.sort(confusionMatrixProbs[1][0])
    confusionMatrixProbs[1][1] = np.sort(confusionMatrixProbs[1][1])

def initConfusionMatrixProbs():
    return [[[], []], [[], []]]

def arrAppend(arr, rec):
    return rec if (arr == []) else np.vstack((arr, rec))
    
def maxN(arr, n, absVals=False):
    if (absVals):
        arr = abs(arr)
    return heapq.nlargest(n, arr)

def minN(arr, n, absVals=False):
    if (absVals):
        arr = abs(arr)
    return heapq.nsmallest(n, arr)

def argmaxN(arr,n,absVals=False):
    if (absVals):
        arr = abs(arr)
    maxVals = maxN(arr,n)
    maxInds = [np.where(arr==val)[0][0] for val in maxVals]
    return maxVals,maxInds

def argminN(arr,n,absVals=False):
    if (absVals):
        arr = abs(arr)
    minVals = minN(arr,n)
    minInds = [np.where(arr==val)[0][0] for val in minVals]
    return minVals,minInds


def pointsDistance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))

def pointsAng(p1, p2):
    return math.atan2(p1[1] - p2[1], p1[0] - p2[0])

def pointsStrConverter(points, elms=2):
    ret = []
    for p in points:
        try:
            point = p.split(',')[:elms]
            for ind in range(elms):
                point[ind] = float(point[ind])
            ret.append(point)
        except:
            pass
    return ret 

def calcCentroid(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points))    
    return centroid

def writeCSVFile(data, fileName):
    with open('%s.csv'%(fileName), 'wb') as test_file:
        file_writer = csv.writer(test_file)
        for line in data:
            file_writer.writerow(line)

def readCSVFile(fileName, delimiter='\t', useNP=True, dtype='', startRowIndex=0, doPrint=False, maxErrorsNum=0, getOnlyEvenLines=False):    
    """ Read CSV File """
    if (useNP):
        if (dtype == ''):
            data = np.genfromtxt(fileName, delimiter=delimiter, dtype=None)
        else:
            data = np.genfromtxt(fileName, delimiter=delimiter)
        if (startRowIndex>0):
            data = np.delete(data, range(startRowIndex), 1)
        return data
    else:
        csvFile = open(fileName, 'r')
        errLines = 0     
        lines = 0   
        headData, timeData, featuresData = [], [], []
        getLine=True
        for line in csvFile.readlines():
            if (getOnlyEvenLines):
                if (not getLine): getLine=True; break
                else: getLine=False
            lineData = line.split('\t')
            featuresRec = np.array(pointsStrConverter(lineData[startRowIndex:-1]))
            headRec = np.array(lineData[1])
            timeRec = lineData[0]
            lines+=1
            if (len(lineData) > 3):
                featuresData.append(featuresRec)  # = arrAppend(featuresData, featuresRec.T)
                headData = arrAppend(headData, headRec)
                timeData.append(timeRec)
            else:
                errLines += 1
#                if (errLines>maxErrorsNum): break
        csvFile.close()
        if (errLines > 0 and doPrint):
            print('%s: Reading error in %d lines' % (fileName, errLines))
    return (featuresData, headData, errLines, timeData, lines)

def arrayToList(x):
    l = []
    for i in range(x.shape[0]):
        l.append(list(x[i, :]))
    return l
    
def count(x):
    cnt = Counter(x)
    return cnt

def saveToCsv(mat, fileName):
    np.savetxt('%s.csv' % fileName, mat, delimiter=";")    

def premutationWithRepetitions(x, repeat=None):
    if (repeat is None): repeat = len(x)
    return [p for p in product(x, repeat=repeat)]

def powerSet(data):
    masks = [p for p in product([0, 1], repeat=len(data))]
    ret = [[x for i, x in enumerate(data) if mask[i]] for mask in masks]
    return ret

def timeToMiliseconds(time):
    return time.hour * 60 * 60 * 1000 + time.minute * 60 * 1000 + time.second * 1000 + time.microsecond / 1000

def getGameAndRoundID(gamesFolder):
    gameID = float(gamesFolder[4:6]) + 1
    roundID = float(gamesFolder[11:13]) + 1
    return (gameID, roundID)

def parseXMLFile(xmlFileName):
    xmlFileName = path(xmlFileName)
    if (not xmlFileName.exists()):
        raise Exception("No xml file %s" % xmlFileName) 
    try:
        summary = etree.parse(xmlFileName)
    except:
        raise Exception("Can't parse the xml file %s" % xmlFileName)
    return summary

def sortDictionaryByKey(d):
    return OrderedDict(d)
    
def sortDictionaryByValue(d):
    return OrderedDict(sorted(d.items(), key=lambda t: t[1]))    

def sortIndices(lst):
    return [i[0] for i in sorted(enumerate(lst), key=lambda x:x[1])]    

def sortFolders(folders):
    names = []
    for folder in folders:
        names.append(folder.name.lower())
    idx = sortIndices(names)
    return [folders[ind] for ind in idx]

def getMaxResults(results, ind=-1):
    return (max(results, key=lambda x:x[ind]))       

def printMaxResults(results, ind=-1):
    print('Max Results:')
    print(getMaxResults(results, ind))

def gcd(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b:      
        a, b = b, a % b
    return a

def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)

def shellCall(command):
    p = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    out, err = p.communicate()
    return out

def pointsToVec(p1, p2):
    return [p1[0] - p2[0], p1[1] - p2[1]]

def pointNorm(p):
    return pow(pow(p[0], 2) + pow(p[1], 2), 0.5) 

def normPoint(p):
    norm = pointNorm(p)
    return [p[0] / norm, p[1] / norm]

def sendEmail(message='', subject='results', fromEmail='peled.noam@gmail.com', to=None):
    if not to: to = ['peled.noam@gmail.com']
    try:
        # Create the container (outer) email message.
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = fromEmail
        msg['To'] = ",".join(to)
        msg.preamble = message
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.starttls()
        server.login('peled.noam', '3636np79`0')
        server.sendmail(fromEmail, to, msg.as_string())
        server.quit()
    except:
        print('Cant send the email')    

def sendEmailWithAttchment(message='', subject='results', fromEmail='peled.noam@gmail.com', to=None, files=None):
    if not to: to = ['peled.noam@gmail.com']
    if not files: files = []
    msg = MIMEMultipart()
    msg['Subject'] = message 
    msg['From'] = fromEmail
    msg['To'] = ', '.join(to)
    msg.preamble = message
    
    for fileName in files:
        part = MIMEBase('application', "octet-stream")
        part.set_payload(open(fileName, "rb").read())
        Encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="text.txt"')
        msg.attach(part)
    
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.sendmail(fromEmail, to, msg.as_string())    
    
def emailPredictorResults(conMat, totalAccuracy, stayAccuracy, leaveAccuracy, staysStd, leavesStd, boostFactor, trainMethod, featuresMethod):
    message = conMat.__str__() + \
            '\nTotal: %.5f\nStay: %.5f\nLeave: %.5f' % (totalAccuracy, stayAccuracy, leaveAccuracy) + \
            '\nStay std = %.2f, Leave std = %.2f' % (staysStd, leavesStd) 
    sendEmail(message, 'results: boost=%.2f, trainMethod=%s, featuresMethod=%s' % (boostFactor, trainMethod, featuresMethod))
    
def saveToMatlab(x, fileName):
    scipy.io.savemat('%s.mat' % (fileName), mdict={'arr':x})

def saveDictToMatlab(mdict, fileName):
    scipy.io.savemat('%s.mat' % (fileName), mdict=mdict)

def loadMatlab(fileName, includeSuffix=False):
    includeSuffix = fileName[-4:]=='.mat'
    if (not includeSuffix): fileName = '%s.mat' % (fileName)
    x = scipy.io.loadmat(fileName)
    return x

def noNone(x):
    return np.all(~np.isnan(x))    

def replaceNone(x, val=0):
    x[np.where(np.isnan(x))] = val

def deleteFiles(folder):
    shutil.rmtree('%s/*' % (folder))
    
def timeDiff(time1, time2):
    diff = abs(time1 - time2)
    return diff.seconds + diff.microseconds / 1E6

def mean(x, axis=0):
    return st.nanmean(x, axis)

def std(x, axis=0):
    return st.nanstd(x, axis)

def isTuple(x):
    return isinstance(x, types.TupleType)

def distsToProbs(d):
    if (len(d) == 1): return np.array([1.0])
    if (d[0] == 0): d = d[1:]
    d = 1 / d
    p = d / sum(d)
    return p

def distsToAccProbs(d):
    p = distsToProbs(d)
    cumsum = np.cumsum(p)
    return cumsum

def findCutof(probs):
    d = np.cumsum(probs)
    cutof = np.random.random() * max(d)
    return sum(d < cutof)

def findCutofWithEx(probsCumSum, exList=None, inds=None):
    if not inds: inds = []
    if not exList: exList = []
    cutof = np.random.random()
    cutofInd = -1
    for valInd, val in enumerate(probsCumSum):
        if (cutof < val and inds[valInd] not in exList):
            cutofInd = valInd 
            break
    while (cutofInd == -1):
        print('error in picking valInd')
        valInd = pickIndex(len(probsCumSum))
        if (inds[valInd] not in exList): 
            cutofInd = valInd
    return cutofInd


def setKItems(k, N):
    val = k / N + 1
    ret = [val] * N
    for _ in range(sum(ret) - k):
        ind = pickIndex(len(ret))
        while(ret[ind] == 1):  # No item should be less the 1
            ind = pickIndex(len(ret))
        ret[ind] -= 1
    return ret
    
def pickIndex(N):
    return np.random.randint(0, N)

def kbin(x, k=5):
    return x / int(k) * k

def boolStrToInt(val):
    if (val is not None):
        if boolStrToBool(val):
            return 1
        else:
            return 0
    else:
        return 0

def boolStrToBool(val):
    return val.lower() in ['true', '1']

def ttest(A, B):
    """
    Returns
    -------
    t : float or array
        t-statistic
    prob : float or array
        two-tailed p-value    
    """
    A = noNanArr(A)
    B = noNanArr(B)
    if (len(A) > 0 and len(B) > 0):
        _, p = stats.ttest_ind(A, B)
    else:
        p = 1
    return p

def coupledTtest(A, B):
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html#scipy.stats.ttest_rel
    _, p = stats.ttest_rel(A, B)
    return p

def ttestCond(A, B):
    A = noNanArr(A)
    B = noNanArr(B)
    if (len(A) > 0 and len(B) > 0):
        (p, cond) = (ttest(A, B), '>') if st.nanmean(A) > st.nanmean(B) else (ttest(B, A), '<')
    else:
        (p, cond) = 1, ''
    return ('%s (!)' % cond if p <= 0.05 else cond, p) 

def printTtestResult(A, B, Aname, Bname, descriptive=True):
    cond, p = ttestCond(A, B)
    if (descriptive):
        print('%s(%.5f, std=%.5f) %s %s(%.5f, std=%.5f) (p=%.2f)' % (Aname, np.mean(A), np.std(A), cond, Bname, np.mean(B), np.std(B), p))
    else:
        print('%s %s %s (p=%.2f)' % (Aname, cond, Bname, p))

def noNanArr(arr):
    return [a for a in arr if not np.isnan(a)]

def cycler(arr):
    return cycle(arr)

def timeToDateTime(time,baseDate, timeFormat = '%d/%m/%Y %H:%M:%S.%f'):
    fullTime = '%s %s'%(baseDate, time)
    return datetime.strptime(fullTime,timeFormat)

def timeToString(time,timeFormat = '%d/%m/%Y %H:%M:%S.%f'):
    return datetime.strftime(time, timeFormat)

def timedeltaToString(timeDelta):  
    return str(timeDelta)  
      
def deleteDirectory(path):
    if (os.path.isdir(path)):
        shutil.rmtree(path) 
        
def createDirectory(directory):
    if (not os.path.exists(directory)):
        os.makedirs(directory)

def fileExists(fileName):
    return path(fileName).exists()

#def copyFile(folder, fileName, dst):
#    try:
#        shutil.copyfile(folder.files(fileName)[0], dst)
#    except:
#        print('******* no file! %s ************' %(fileName))
        
def xmlToStr(root):
    print (etree.tostring(root, pretty_print=True))        
    
def findMinorityLabel(y):
    cnt = count(y)
    return 0 if cnt[0]<cnt[1] else 1

def minorityCheck(y):
    cnt = count(y)
    return (0,1,cnt[0],cnt[1]) if cnt[0]<cnt[1] else (1,0,cnt[1],cnt[0])

def underSampling(x,y):
    minorityLabel, majorityLabel, minorityCount, majorityCount = minorityCheck(y)
    majorityInds = np.where(y == majorityLabel)[0]
    minorityIndex = np.where(y == minorityLabel)[0]
    majorityInds = random.sample(majorityInds, minorityCount)
    inds = np.concatenate((minorityIndex,majorityInds))
    return x[inds],y[inds] 
    
def removeFilesFromFolder(folder):
    deleteDirectory(folder)
    createDirectory(folder)
    
def dictArgMax(d):
    return max(d, key=d.get)    
    
def allItemAreIdentical(arr):
    return len(set(arr))==1

def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def ticToc():
    return time.clock()

def howMuchTimeFromTic(tic, functionName=''):
    toc = ticToc()
    czas = toc - tic
    czas_='%s: '%('TicToc' if functionName=='' else functionName)
    czas_=czas_ + "%0.*f" %(0,np.fix(czas/3600)) + ' hours, '
    czas_=czas_ + "%0.*f" %(0,np.fix(czas/60)-np.fix(czas/3600)*60) + ' min and '
    czas_=czas_ + "%0.*f" %(2,np.mod(czas,60)) + ' s.'
    print(czas_)    
    
def changeFolder(folder):
    os.chdir(folder)
    
def makeDir(folder):
    os.mkdir(folder)
    
def currentFolder():
    return path.cwd()
      
def copyFile(sourceFolder,destFolder,fileName):
    try:
        shutil.copy('%s/%s'%(sourceFolder,fileName), '%s/%s'%(destFolder,fileName))
    except:
        print('******* no file! %s ************' %(fileName))
        
def tictoc(f):

    @wraps(f)
    def wrapped(*args, **kwargs):
        tic=ticToc()
        r = f(*args, **kwargs)
        howMuchTimeFromTic(tic,f.func_name)
        return r

    return wrapped

def nhist(X,binsNum=10):
    mins, maxs, hists = [], [], []
    for x in X:
        mins.append(np.min(x))
        maxs.append(np.max(x))
    bins = np.linspace(min(mins), max(maxs), binsNum)
    for x in X:
        hist, _ = np.histogram(x, bins)
        hists.append(hist)
    return hists
        
def throwException(errStr='Error!'):
    raise Exception(errStr)

def filesInFolder(folder, pattern=''):
    return path(folder).files(pattern)

def calcEntropy(data):
    """ Computes entropy of label distribution. """
    data = np.array(data)
    n_labels = len(data)
    if (n_labels <= 1): return 0
    probs = data / float(sum(data))
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1: return 0
    ent = np.sum([-p * np.log2(p) for p in probs if p>0])
    return ent    

def logicOrOnVectors(vectors):
    ret = False
    for vec in vectors:
        ret = ret | vec
    return ret

def copyDics(orgDic, ext=None):
    retDic = {}
    for field in [f for f in orgDic.keys() if f not in ext]:
        retDic[field] = orgDic[field]
    return retDic