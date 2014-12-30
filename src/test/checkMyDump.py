'''
Created on Jul 20, 2014

@author: noampeled
'''

import traceback
import os
import pickle
from functools import wraps
from uuid import uuid4 as uuid
import numpy as np

from src.commons.utils import freqsUtils
from src.commons.utils import tablesUtils as tabu
from src.commons.utils import sectionsUtils as su
from src.commons.selectors.frequenciesSelector import FrequenciesSelector

DUMP_FOLDER = '/home/noam/Documents/MEGdata/centipede/data'
HDF_FILE = '/home/noam/Documents/MEGdata/centipede/data/svmFiles/dataForDecoding.mat_centipedeSpacialSWFreqs_StayOrLeave6_10___subdor.hdf'


def dump(objToDump, dumpFolder, functionName=''):
    dumpID = str(uuid())[-8:]
    print('Error{}! {}'.format(' in {}'.format(functionName) if functionName!='' else '', dumpID))
    errStr = traceback.format_exc()
    print(errStr)
    dumpFileName = os.path.join(dumpFolder, '{}_dump_{}.pkl'.format(functionName,dumpID))
    with open(dumpFileName, 'w') as pklFile:
        pickle.dump((objToDump, errStr), pklFile)

def dumper(dumpFolder, defaultReturnValue=None):
    def wrap(f):
        @wraps(f)
        def wrapped(*a, **kw):
            try:
                return f(*a,**kw)
            except:
                dump((a, kw), dumpFolder, f.__name__)
                return defaultReturnValue
        return wrapped
    return wrap


def loadDumpFile(dumpFolder, dumpFileName):
    dumpFullFileName = os.path.join(dumpFolder,
        '{}.pkl'.format(dumpFileName))
    with open(dumpFullFileName, 'r') as pklFile:
        dump = pickle.load(pklFile)
    parameters, strErr = dump
    print('Error message:')
    print(strErr)
    return parameters
#     print('parameters:')
#     print(parameters)



# The dumps folder is DUMP_FOLDER, and the default return value is 0
@dumper(DUMP_FOLDER, 0)
def errorFunction(a):
    print(1 / 0)
    return(a)

def checkCalcSlice():
    (c, cvIndices, _) = loadDumpFile(DUMP_FOLDER, '{}_dump_{}'.format('calcSlice', 'qXyk1'))
    hdfFile = tabu.openHDF5File(HDF_FILE)
    X = tabu.findTable(hdfFile, 'x', 'preProcess')
    weights = tabu.findTable(hdfFile, 'weights', 'preProcess')
    xc = np.dot(X[cvIndices, :, :], weights[c, :].T)
    print(np.sum(np.isnan(xc)))
    hdfFile.close()


def checkFfindSigSectionsPSInPValues():
    (c, y, selector, timeStep, minFreq, maxFreq, alpha, minSegSectionLen,
        cvIndices, timeIndices) = loadDumpFile(
        DUMP_FOLDER, '{}_dump_{}'.format('findSigSectionsPSInPValues', 'darpZ'))

    hdfFile = tabu.openHDF5File(HDF_FILE)
    X = tabu.findTable(hdfFile, 'x', 'preProcess')
    weights = tabu.findTable(hdfFile, 'weights', 'preProcess')

    xc = su.calcSlice(X, c, cvIndices, timeIndices, weights)
    _, ps = freqsUtils.calcPS(xc, timeStep, minFreq, maxFreq)
    model = selector.fit(ps, y, cvIndices, timeIndices, weights)
    print model

if __name__ == '__main__':
#     errorFunction(10)
#     dumpID = ''
    checkFfindSigSectionsPSInPValues()