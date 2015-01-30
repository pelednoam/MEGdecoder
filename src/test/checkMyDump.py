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
import src.commons.utils.sectionsUtils as su
from path3 import path

# DUMP_FOLDER = '/home/noam/Documents/MEGdata/centipede/data'
# DUMP_FOLDER = '/Users/noampeled/Documents/svmDumper/dumper'
DUMP_FOLDER = '/Users/noampeled/Copy/Data/MEG/data/dumper'
# DUMP_FOLDER = '/home/noam/Documents/MEGdata/centipede/data/dumper'
# HDF_FILE = '/home/noam/Documents/MEGdata/centipede/data/svmFiles/dataForDecoding.mat_centipedeSpacialSWFreqs_StayOrLeave6_10___subdor.hdf'
HDF_FILE = '/Users/noampeled/Copy/Data/MEG/data/svmFiles/centipedeAll_StayOrLeave6_10__FrequenciesSelector_sub_all.hdf'
# HDF_FILE = '/home/noam/Documents/MEGdata/centipede/data/svmFiles/centipedeAll_StayOrLeave6_10__FrequenciesSelector_sub_all.hdf'


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


def loadDumpFile(dumpFolder, dumpFileName, doPrintErr=True):
    dumpFullFileName = os.path.join(dumpFolder,
        '{}.pkl'.format(dumpFileName))
    with open(dumpFullFileName, 'r') as pklFile:
        dump = pickle.load(pklFile)
    parameters, strErr = dump
    if (doPrintErr):
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


def checkFindSigSectionsInPValues():
    hdfFile = tabu.openHDF5File(HDF_FILE)
    X = tabu.findTable(hdfFile, 'pss', 'preProcess')
#     y = tabu.findTable(hdfFile, 'y', 'preProcess')
    files = path(DUMP_FOLDER).files('.findSigSectionsInPValues*')
    for dumpFile in files:
        try:
            (c, y, selector, cvIndices, timeIndices, alpha, sigSectionMinLength) = loadDumpFile(DUMP_FOLDER, dumpFile.namebase, doPrintErr=False) # '{}_dump_{}'.format('.findSigSectionsInPValues', 'meIA5'))
            print (c)
            weights = None
            xc = su.calcSlice(X, c, cvIndices, timeIndices, weights)
            model = selector.fit(xc, y)
            sigSections = su.findSectionSmallerThan(model.pvalues_,
                alpha, sigSectionMinLength)
            print(sigSections)
        except:
            print('******** **************')
            print(dumpFile.name)
            print(traceback.format_exc())
            break
    hdfFile.close()


def checkCalcSlice():
    (c, cvIndices, _) = loadDumpFile(DUMP_FOLDER, '{}_dump_{}'.format('calcSlice', 'qXyk1'))
    hdfFile = tabu.openHDF5File(HDF_FILE)
    X = tabu.findTable(hdfFile, 'x', 'preProcess')
    weights = tabu.findTable(hdfFile, 'weights', 'preProcess')
    xc = np.dot(X[cvIndices, :, :], weights[c, :].T)
    print(np.sum(np.isnan(xc)))
    hdfFile.close()


def checkFindSigSectionsPSInPValues():
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
    checkFindSigSectionsInPValues()