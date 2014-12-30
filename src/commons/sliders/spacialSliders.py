'''
Created on Nov 20, 2014

@author: noampeled
'''
import numpy as np
import itertools
from sklearn.base import BaseEstimator

from src.commons.utils import utils


class SpatialWindowSlider(BaseEstimator):

    def __init__(self, weightsNum, xlim, ylim, zlim, xstep, xCubeSize,
                 ystep=None, yCubeSize=None,
                 zstep=None, zCubeSize=None,
                 windowsOverlapped=True, calcLocs=False):
        ''' xlim, ylim, zlim, xstep, ystep and zstep are integers.
            All the units are in mm '''
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.xCubeSize = xCubeSize
        self.yCubeSize = xCubeSize if yCubeSize is None else yCubeSize
        self.zCubeSize = xCubeSize if zCubeSize is None else zCubeSize
        self.xstep = xstep
        self.ystep = xstep if ystep is None else ystep
        self.zstep = xstep if zstep is None else zstep
        self.xspace = range(xlim[0], xlim[1] + 1, xstep)
        self.yspace = range(ylim[0], ylim[1] + 1, ystep)
        self.zspace = range(zlim[0], zlim[1] + 1, zstep)
        print('head xspace(mm): {}'.format(self.xspace))
        print('head yspace(mm): {}'.format(self.yspace))
        print('head yspace(mm): {}'.format(self.zspace))
        self.windowsOverlapped = windowsOverlapped
        if (calcLocs):
            self.cubesIndices, self.cubesLocs = self.calcCubesIndices(calcLocs)
        else:
            self.cubesIndices = self.calcCubesIndices(calcLocs)
        print('cubes dim: {}'.format(self.cubesIndices.shape))
        print('cubes num: {}'.format(self.cubesIndices.size))
        if (weightsNum != 0 and self.cubesIndices.size != weightsNum):
            raise Exception("cubesIndices size ({})".format(
                self.cubesIndices.size) +
                " isn't equall to the weights number ({})".format(weightsNum))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.startIndex:self.startIndex + self.windowSize]

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def extract(self, X):
        return self.transform(X)

    def cubesGenerator(self):
        xhalf = int((self.xCubeSize) / 2)
        yhalf = int((self.yCubeSize) / 2)
        zhalf = int((self.zCubeSize) / 2)
        dx = xhalf if self.windowsOverlapped else self.xCubeSize
        dy = yhalf if self.windowsOverlapped else self.yCubeSize
        dz = zhalf if self.windowsOverlapped else self.zCubeSize
        xinds = np.arange(xhalf, len(self.xspace) - xhalf, dx)
        yinds = np.arange(yhalf, len(self.yspace) - yhalf, dy)
        zinds = np.arange(zhalf, len(self.zspace) - zhalf, dz)
        if (not (len(self.xspace) - xhalf) in xinds):
            xinds = np.hstack((xinds, len(self.xspace) - xhalf))
        if (not (len(self.yspace) - yhalf) in yinds):
            yinds = np.hstack((yinds, len(self.yspace) - yhalf))
        if (not (len(self.zspace) - zhalf) in zinds):
            zinds = np.hstack((zinds, len(self.zspace) - zhalf))
#         dex = 0 if self.xCubeSize % 2 == 0 else 1
#         dey = 0 if self.xCubeSize % 2 == 0 else 1
#         dez = 0 if self.xCubeSize % 2 == 0 else 1
        for xcenter in xinds:
            for ycenter in yinds:
                for zcenter in zinds:
                    xcube = np.arange(xcenter - xhalf, xcenter + xhalf)
                    ycube = np.arange(ycenter - yhalf, ycenter + yhalf)
                    zcube = np.arange(zcenter - zhalf, zcenter + zhalf)
                    cubeIndices = itertools.product(*(
                        xcube, ycube, zcube))
                    weightsIndices = []
                    for ix, iy, iz in cubeIndices:
                        weightsIndices.append(self.cubesIndices[ix, iy, iz])
                    yield (np.array(weightsIndices, dtype=np.int))

    def calcCubesNum(self, weights):
        n = 0
        for voxelIndices in self.cubesGenerator():
            cubeWeights = weights[voxelIndices, :]
            if (not np.all(cubeWeights == 0)):
                n += 1
        return n

    def calcCubesIndices(self, calcLocs=False):
        indices = np.zeros((len(self.xspace), len(self.yspace),
            len(self.zspace)))
        if (calcLocs):
            locs = {}
        index = 0
        for ix, x in enumerate(self.xspace):
            for iy, y in enumerate(self.yspace):
                for iz, z in enumerate(self.zspace):
                    indices[ix, iy, iz] = int(index)
                    if (calcLocs):
                        locs[index] = (x, y, z)
                    index += 1
        if (calcLocs):
            return (indices, locs)
        else:
            return indices
