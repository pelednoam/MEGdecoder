'''
Created on Jul 24, 2014

@author: noam
'''

from src.commons.utils import utils
from path3 import path

FILE = '/home/noam/Documents/MEGdata/spectLeave_1_1.mat'

def loadFile():
    fileName = path(FILE).namebase
    freq = fileName.split('_')[-1]
    spectrumFile = utils.loadMatlab(FILE, True)
    print(spectrumFile)

if __name__ == '__main__':
    loadFile()