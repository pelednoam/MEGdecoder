'''
Created on Jun 30, 2012

@author: noam
'''
import math
# import WaveletTransform as wt
# import PCA
import numpy as np
import utils
import MLUtils
# import plots

MATLAB_FOLDER = '/home/noam/Desktop/Thesis/Centipede/KalmanAll/readData/data'

# LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, NOSE, MOUTH, LEFT_CHICK, RIGHT_CHICK, RIGHT_CHIN, LEFT_CHIN = range(10)
LEFT_EYE, RIGHT_EYE, LEFT_LEFT_EYEBROW, LEFT_RIGHT_EYEBROW, RIGHT_RIGHT_EYEBROW, RIGHT_LEFT_EYEBROW, NOSE, MOUTH, LEFT_CHICK, RIGHT_CHICK, RIGHT_CHIN, LEFT_CHIN = range(12)

PARAM_FEATURES_NUM = 18  # number of features

class FeaturesExtraction(object):
    def __init__(self, data, groups, subject='', gameID= -1, roundID= -1, normalizeData=False, smoothData=False, parametrizedData=False, calcMainCentroids=False, mainCentroids=[], pointsNoiseSTD=[1, 2], convertRawData=False):
        self.subject, self.gameID, self.roundID = subject, gameID, roundID
        self.data = convertRawDataToMatrix(data) if convertRawData else data
        self.P = self.data.shape[1]
        self.T = self.data.shape[0]
        self.pointsNoiseSTD = pointsNoiseSTD
        # utils.saveToMatlab(self.data, '%s/%s_%d_%d' % (MATLAB_FOLDER, subject, gameID, roundID))
        self.groups = groups
        if (calcMainCentroids): 
            self.calcMainCentroids()
        elif (mainCentroids):
            self.loadMainCentroids(mainCentroids)
        if (normalizeData): self.normalizePoints()
        if (smoothData): self.smoothPoints()
        if (parametrizedData): self.parametrization() 

    def calcDiffsSum(self):
        sums = []
        for i in range(self.data.shape[1]):
            diffs = np.diff(self.data[:, i], 0)
#            diffsSum = np.sum(diffs, 1)
            sums = utils.arrAppend(sums, diffs)
            
#        for group in self.groups:
#            diffs = abs(np.diff(self.data[:, group], 0))
#            diffsSum = np.sum(diffs, 1)
#            sums = utils.arrAppend(sums, diffsSum)
        return sums

    def calcGroupsCentroids(self, t):
        centroids = []
        for group in self.groups:
            groupPoints = self.data[t, group]
            centroid = utils.calcCentroid(groupPoints)
            centroids.append(centroid)
        return centroids
    
    def calcNoseAngle(self, t):
        noseEyes = utils.pointsToVec(self.noseCentroids[t], self.eyesCentroids[t]) 
        return math.atan2(noseEyes[1], noseEyes[0])
    
    def calcEyesNoseRatio(self, t):
        noseLength = utils.pointsDistance(self.eyesCentroids[t], self.noseCentroids[t])
        eyesDistance = utils.pointsDistance(self.rightEyeCentroids[t], self.leftEyeCentroids[t])
        return noseLength / eyesDistance
    
    def calcNormalizeFactors(self, t):
        x = utils.pointsToVec(self.rightEyeCentroids[t], self.leftEyeCentroids[t])  # le-re
        y = utils.pointsToVec(self.eyesCentroids[t], self.noseCentroids[t])  # ec-nc
        x_hat = np.array(utils.normPoint(x))
        y_hat = np.array(utils.normPoint(y))
        norm_vec = np.array([1 / utils.pointNorm(x), 1 / utils.pointNorm(y)])
        projection_matrix = np.array([[x_hat[0], x_hat[1]], [y_hat[0], y_hat[1]]])
        return norm_vec, projection_matrix

    def calcMainCentroids(self):
        self.eyesNoseRatios = np.zeros(self.data.shape[0])
        self.rightEyeCentroids = np.zeros((self.data.shape[0], 2))
        self.leftEyeCentroids = np.zeros((self.data.shape[0], 2))
        self.noseCentroids = np.zeros((self.data.shape[0], 2))
        self.eyesCentroids = np.zeros((self.data.shape[0], 2))
        self.noseAngles = np.zeros(self.data.shape[0])
        self.noseLocations = np.zeros((self.data.shape[0], 2))        
        for t in range(self.data.shape[0]):
            self.rightEyeCentroids[t] = self.centroid(RIGHT_EYE, t)  # re
            self.leftEyeCentroids[t] = self.centroid(LEFT_EYE, t)  # le
            self.noseCentroids[t] = self.centroid(NOSE, t)  # nc
            self.eyesCentroids[t] = utils.calcCentroid([self.rightEyeCentroids[t], self.leftEyeCentroids[t]])  # ec
            self.eyesNoseRatios[t] = self.calcEyesNoseRatio(t)
            self.noseLocations[t] = self.noseCentroids[t]
            self.noseAngles[t] = self.calcNoseAngle(t)     
        self.mainCentroids = (self.eyesNoseRatios, self.rightEyeCentroids, self.leftEyeCentroids, self.noseCentroids, self.eyesCentroids, self.noseAngles, self.noseLocations)       

    def loadMainCentroids(self, mainCentroids):
        self.eyesNoseRatios, self.rightEyeCentroids, self.leftEyeCentroids, self.noseCentroids, self.eyesCentroids, self.noseAngles, self.noseLocations = mainCentroids

    def normalizePoints(self):
        for t in range(self.data.shape[0]):
            norm_vec, projection_matrix = self.calcNormalizeFactors(t)
            for f in range(self.data.shape[1]):
                self.data[t, f] -= self.noseLocations[t]  # turn the nose centroid to be (0,0)
                self.data[t, f] = np.dot(projection_matrix, self.data[t, f])  # change base
                self.data[t, f] = norm_vec * self.data[t, f]  # normalize

    def smoothPoints(self, doPlot=False):
        noiseVar = self.pointsNoiseSTD
        for p in range(self.data.shape[1]):
            newx = MLUtils.kalman1d(self.data[:, p, 0], R=noiseVar[0] ** 2)
            newy = MLUtils.kalman1d(self.data[:, p, 1], R=noiseVar[1] ** 2)
            if (doPlot):
                plots.plt.figure()
                plots.plt.plot(self.data[:, p, 0], self.data[:, p, 1], label='%d x' % p)
                plots.plt.plot(newx, newy, label='kalman %.2f' % noiseVar)
                plots.plt.legend()
#                plots.plt.savefig('%s/pics/signals/%s_%d.svg'%(fileName.getcwd(),fileName.name[:-4],p))
                plots.plt.show()        
            self.data[:, p, 0] = newx
            self.data[:, p, 1] = newy
    
    def parametrization(self):
        T = self.data.shape[0]
        self.features = np.zeros((T, PARAM_FEATURES_NUM))
        self.paramsNames = ['upper mouth y', 'lower mouth y', 'raise right eyebrow', 'raise left eyebrow']
        for t in range(T):
            self.features[t, 0] = self.data[t, 64, 1]  # self.maxGroup(MOUTH,t,1) # upper mouth y value (point 54)
            self.features[t, 1] = self.data[t, 61, 1]  # self.minGroup(MOUTH,t,1) # lower mouth y value (point 55)
            self.features[t, 2] = self.data[t, 17, 1]  # self.maxGroup(RIGHT_RIGHT_EYEBROW, t, 1) # raise right eyebrow (point 17)
            self.features[t, 3] = self.data[t, 16, 1]  # self.maxGroup(LEFT_LEFT_EYEBROW, t, 1) # raise left eyebrow (point 16)
            self.features[t, 4] = self.data[t, 11, 1]  # self.minGroup(LEFT_CHIN, t, 1) # lower point of the chin (point 11)
            self.features[t, 5] = self.data[t, 4, 0]  # self.maxGroup(MOUTH,t,0) # most right mouth x value (point 4)
            self.features[t, 6] = self.data[t, 3, 0]  # self.minGroup(MOUTH,t,0) # most left mouth x value (point 3)
            self.features[t, 7] = self.data[t, 32, 1] - self.data[t, 31, 1]  # self.maxGroup(RIGHT_EYE, t, 1)-self.minGroup(RIGHT_EYE, t, 1) # right eye open (points 32-31)
            self.features[t, 8] = self.data[t, 28, 1] - self.data[t, 27, 1]  # self.maxGroup(LEFT_EYE, t, 1)-self.minGroup(LEFT_EYE, t, 1) # left eye open (points 28-27)            
            self.features[t, 9] = self.data[t, 13, 0] - self.data[t, 14, 0]  # self.minGroup(RIGHT_LEFT_EYEBROW, t, 0) - self.maxGroup(LEFT_RIGHT_EYEBROW, t, 0) # distance between eyebrows (points 25-24)
            self.features[t, 10] = self.groupAngle(RIGHT_RIGHT_EYEBROW, t)  # right eyebrow angle
            self.features[t, 11] = self.groupAngle(RIGHT_LEFT_EYEBROW, t)  # right eyebrow angle
            self.features[t, 12] = self.groupAngle(LEFT_LEFT_EYEBROW, t)  # left eyebrow angle
            self.features[t, 13] = self.groupAngle(LEFT_RIGHT_EYEBROW, t)  # left eyebrow angle
            self.features[t, 14] = self.groupAngle(RIGHT_CHICK, t)  # right eyebrow angle
            self.features[t, 15] = self.groupAngle(LEFT_CHICK, t)  # left eyebrow angle
            self.features[t, 16] = self.groupAngle(RIGHT_CHIN, t)  # right chin angle
            self.features[t, 17] = self.groupAngle(LEFT_CHIN, t)  # left chin angle

    def plotPoints(self, interval=50):
        EXTRA = 7
        LINES = 13
        points = np.zeros((self.T, self.P + EXTRA, 2))
        lines = np.zeros((self.T, LINES, 2, 2))
        points[:, :self.P, :] = self.data
        sizes = np.zeros((points.shape[1]))
        sizes[:self.P] = 1
        sizes[-EXTRA:] = 0.8
        angs = np.zeros((8, 2))
        angsIDs = [[17, 15], [14, 17], [12, 16], [16, 13], [51, 53], [52, 50], [11, 6], [5, 11]]
        linesIDs = [[13, 14], [27, 28], [31, 32], [64, 61], [3, 4]]
        pointsIDs = [64, 61, 17, 16, 11, 4, 3]
        for t in range(self.data.shape[0]):
            for ind, groupID in enumerate([RIGHT_RIGHT_EYEBROW, RIGHT_LEFT_EYEBROW, LEFT_LEFT_EYEBROW, LEFT_RIGHT_EYEBROW, RIGHT_CHICK, LEFT_CHICK, RIGHT_CHIN, LEFT_CHIN]):
                angs[ind] = self.groupAngle(groupID, t, True)
#            points[t,-EXTRA:] = self.data[t,pointsIDs]
#            for i,angID in enumerate(angsIDs):   
#                lines[t,i] = [[self.data[t,angID[0],0],self.data[t,angID[1],0]],[self.data[t,angID[0],0]*angs[i,0]+angs[i,1],self.data[t,angID[1],0]*angs[i,0]+angs[i,1]]]
#            for i,lineID in enumerate(linesIDs):
#                lines[t,len(angsIDs)+i] = [[self.data[t,lineID[0],0],self.data[t,lineID[1],0]],[self.data[t,lineID[0],1],self.data[t,lineID[1],1]]] # distance between eyebrows (points 25-24)

        ani = plots.AnimatedScatter(points, lines, sizes, 50, [-2, 2, -3, 2], '%s game:%d round:%d' % (self.subject, self.gameID, self.roundID))  # ,movieName='onlyPoints')
        ani.run()
            
    def extract(self):
            raise NotImplementedError("abstract class")

    def centroid(self, groupID, t):
        return utils.calcCentroid(self.data[t, self.groups[groupID]])
    
    def group(self, groupID, t):
        return self.data[t, self.groups[groupID]]
    
    def maxGroup(self, groupID, t, dim):
        return max(self.group(groupID, t), key=lambda x:x[dim])[dim]
    
    def minGroup(self, groupID, t, dim):
        return min(self.group(groupID, t), key=lambda x:x[dim])[dim]
    
    def groupAngle(self, groupID, t, includeb=False):
        x = self.group(groupID, t)[:, 0]
        y = self.group(groupID, t)[:, 1]
        a, b = MLUtils.linearRegression(x, y)
        if (includeb):
            return a, b
        else:
            return a

def convertRawDataToMatrix(rawData):
    data = np.zeros((len(rawData), rawData[0].shape[0], rawData[0].shape[1]))
    for i, line in enumerate(rawData):
        for j, point in enumerate(line):
            data[i, j, 0] = point[0]
            data[i, j, 1] = point[1]
    return data
    
class FESTD(FeaturesExtraction):
    
    def extract(self):
        return np.std(self.data, 0)

class FECorr(FeaturesExtraction):
    
    def extract(self):
        groupsData = np.array(self.calcDiffsSum()).T
        corr = np.dot(groupsData.T, groupsData)
        eig = np.linalg.eigvals(corr)
        return eig

class FEMovement(FeaturesExtraction):

    def __init__(self, data, groups, windowSize):
        FeaturesExtraction.__init__(self, data, groups)
        self.windowSize = windowSize

    def extract(self, elmsNum=50):
        diffsSums = self.calcDiffsSum()
        cova = MLUtils.calcCova(diffsSums, elmsNum)
        cova = MLUtils.halfMat(cova, True)        
        return cova


#        diffsSumWT = wt.contTransform(diffsSum)
#        diffsSumWTSum = np.sum(diffsSumWT, 0)
#        ret = utils.maxN(diffsSumWTSum, 5)
#        ret.append(np.average(diffsSumWTSum))
#        ret.append(np.std(diffsSumWTSum))
#        return ret

    def wtCoefficents(self, diffsSum):
        ret = []
        diffsSumWT = wt.transform(diffsSum)
        inds = np.argsort(diffsSumWT)
        for i in range(5):
            ret.append(inds[-(i + 1)])
            ret.append(diffsSumWT[inds[-(i + 1)]])
        ret.append(np.average(diffsSum))
        ret.append(np.std(diffsSum))
        return ret

class FESlidingWindows(FeaturesExtraction):
    
    def __init__(self, data, windowSize):
        FeaturesExtraction.__init__(self, data)
        self.windowSize = windowSize
    
    def extract(self):
        return []
   
    def stds(self):
        windowsStds = []
        for rowInd in range(self.data.shape[0] - self.windowSize + 1):
            windowsData = self.data[rowInd:rowInd + self.windowSize, :]
            windowsStd = FESTD(windowsData).extract()
            windowsStds = utils.arrAppend(windowsStds, windowsStd) 
        return windowsStds
   
class FEcontWT_PCA(FeaturesExtraction):
 
    def extract(self):
        features = []
        for col in range(self.data.shape[1]):
            dotFeatures = wt.contTransform(self.data[:, col], doPlot=False)
            dotFeatures = dotFeatures.reshape([dotFeatures.size])
            features = list(dotFeatures) if (len(features) == 0) else np.vstack((features, dotFeatures))
    
        features = PCA.transform(features, 40)
        features = features.reshape([features.size])
        return features

class FEPointsDistances(FeaturesExtraction):

    def calcDistancesTimeSeries(self, headData, groupNum=0, calcCentroids=False):
        T = self.data.shape[0]
        G = len(self.groups) if calcCentroids else len(self.groups[groupNum])
        dists = np.zeros((G * (G - 1) / 2, T))
#        angles = np.zeros((G * (G - 1) / 2, T))
#        tripleDist = np.zeros((G * (G - 1) / 2, T))
        for t in range(T):
            if (calcCentroids):
                centroids = self.calcGroupsCentroids(t)
            else:
                centroids = utils.pointsStrConverter(self.data[t, self.groups[groupNum]])
#            headPosition = utils.pointsStrConverter(headData[t, :])[0]
            f = 0 
            for f1 in range(G):
                for f2 in range(G):
                    if (f2 > f1):
                        dists[f, t] = utils.pointsDistance(centroids[f1], centroids[f2])
                        # distance from head center ratio 
#                        dists[f, t] = utils.pointsDistance(centroids[f1], headPosition) / utils.pointsDistance(centroids[f2], headPosition)
#                        angles[f, t] = utils.pointsAng(centroids[f1], centroids[f2])
#                        tripleDist[f, t] = utils.pointsDistance(centroids[f1], centroids[f2]) + \
#                                                 utils.pointsDistance(centroids[f1], headPosition) + \
#                                                 utils.pointsDistance(centroids[f2], headPosition)
                        f = f + 1        
        return dists.T

    def calcPointsDistanceFromNose(self, polar=True, parameteriation=False, features=None, doPlot=False):
        param_points = [12, 16, 13, 14, 17, 15, 28, 27, 24, 25, 32, 31, 64, 61, 3, 4, 11, 6, 23, 26]
        T = self.data.shape[0]
        if (parameteriation):
            features = self.features if features is None else features
        G = self.data.shape[1] if not parameteriation else features.shape[1]  # len(param_points)
        dists = np.zeros((T, G * 2 + 4)) if not parameteriation else np.zeros((T, G + 4))
        prev_nosex, prev_nosey = self.noseCentroids[0][0], self.noseCentroids[0][1]
        for t in range(T):
            dists[t, 0] = self.noseCentroids[t][0] - prev_nosex  # nose_centroid_x diff
            dists[t, 1] = self.noseCentroids[t][1] - prev_nosey  # nose_centroid_y diff
            prev_nosex, prev_nosey = self.noseCentroids[t][0], self.noseCentroids[t][1]
            dists[t, 2] = self.eyesNoseRatios[t]
            dists[t, 3] = self.noseAngles[t]
            ind = 4
            if (parameteriation):
#                for p in param_points:
#                    point = self.data[t, p]
#                    dists[t, ind] = utils.pointNorm(point)  # r
#                    dists[t, ind + 1] = math.atan2(point[1], point[0]) + math.pi / 2  # theta from the nose
#                    ind += 2                
                dists[t, ind:] = features[t, :]
            else:
                for p in range(G):
                    point = self.data[t, p]
                    if (polar):
                        dists[t, ind] = utils.pointNorm(point)  # r
                        dists[t, ind + 1] = math.atan2(point[1], point[0]) + math.pi / 2  # theta from the nose
                    else:
                        dists[t, ind] = point[0]
                        dists[t, ind + 1] = point[1]                        
                    ind += 2
        if (doPlot):                
            for p in range(G + 4):
                fig = plots.plt.figure()
                fig.canvas.set_window_title('%d' % p) 
                plots.plt.plot(dists[:, p])
                plots.plt.show()
                

#        distsPCA,_ = PCA.transform(dists, nc, doPrint=False) if doPCA else [],[]  
        return dists

    def calcDistancesFronHeadTimeSeries(self, headData, groupNum=0, calcCentroids=False):
        T = self.data.shape[0]
        G = len(self.groups) if calcCentroids else len(self.groups[groupNum])
        dists = np.zeros((G, T))
        angles = np.zeros((G, T))
        for t in range(T):                
            if (calcCentroids):
                centroids = self.calcGroupsCentroids(t)
            else:
                centroids = utils.pointsStrConverter(self.data[t, self.groups[groupNum]])
            headPosition = utils.pointsStrConverter(headData[t, :])[0]
            for f in range(G):
                dists[f, t] = utils.pointsDistance(centroids[f], headPosition)
                angles[f, t] = utils.pointsAng(centroids[f], headPosition)         
        return (dists, angles)

    
    def extractCORR_EIG(self):
        dists = self.calcDistancesTimeSeries()
        corr = np.dot(dists, dists.T)
        features = np.linalg.eigvals(corr)
        return features
        
