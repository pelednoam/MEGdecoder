'''
Created on Jul 20, 2014

@author: noampeled
'''

from path3 import path
import os
import numpy as np
from src.commons.utils import plots
from src.commons.utils import utils

SPECTRAL_PATH = '/home/noam/Documents/MEGdata/centipede/data/dor/spectral'
DUMP_FOLDRE = '/home/noam/Documents/MEGdata/centipede/data/svmFiles/dump'

def seperateIntoFolders():
    sensorsRange = range(1,249)
    labels = ['Stay','Leave']
    for label in labels:
        print('label {}'.format(label))
        utils.createDirectory(os.path.join(SPECTRAL_PATH,label))
        for sensor in [str(s) for s in sensorsRange]:
            print('sensor {}'.format(sensor))
            utils.createDirectory(os.path.join(SPECTRAL_PATH,label,sensor))
            specFiles = path(SPECTRAL_PATH).files('spect{}_{}_*.mat'.format(label,sensor))
            print('{} files to move'.format(len(specFiles)))
            for specFile in specFiles: 
                utils.moveFile(SPECTRAL_PATH, os.path.join(SPECTRAL_PATH,label,sensor), specFile.name)
            

def plotStayLeaveFreq12():
    d = utils.loadMatlab('/Users/noampeled/Copy/Data/MEG/data/dor/varsForPS12.mat');
    plots.init()
    plots.graph2(d['time'].T, d['stay12'], d['leave12'], ['Stay','Leave'], xlabel='Time (sec)', ylabel='Power',fileName='/Users/noampeled/Dropbox/postDocMoshe/MEG/biomag2014/poster/pics/A2_12Hz.jpg')

def savePntToMat(pntFile):
    lines = utils.csvGenerator(pntFile, '\t', 1)
    mat = np.array([line for line in lines]).astype(np.double) 
    print(mat.shape)
    utils.saveToMatlab(mat, pntFile[:-4], 'points')
    

@utils.dumper
def errorFunction(a):
    print(1/0)
    return(a)

if __name__ == '__main__':
#     seperateIntoFolders()
#     plotStayLeaveFreq12()
#     savePntToMat('/home/noam/Dropbox/postDocMoshe/MEG/centipede/matlabSource/idan/pnt.txt')
#     utils.loadDumpFile(DUMP_FOLDRE, 'errorFunction_dump_0zJcl')
    print(errorFunction(10))