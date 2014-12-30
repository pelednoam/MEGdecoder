# -*- coding: utf-8 -*-
'''
Created on Dec 12, 2013

@author: noampeled
'''

# import h5py
import sys
from multiprocessing import Pool
import numpy as np
from distancesUtils import victor_purpura_dist as vpd
import utils
from sklearn import mixture
import scipy
import pylab
from sklearn.cluster import spectral_clustering as spectral
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import scipy.spatial.distance as ssd
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from commons import plots as plt
    from commons import MLUtils
    from commons import PCA

except:
    print('cant import from commons!')

try:
    import h5py
except:
    print ('no h5py installed, try scipy.io')



ROOT_FOLDER = '/Users/noampeled/Copy/Data/MEG/abelesEvents/idan'
CATEGORIES = ['correctPress','errorPress']
FILE_NAME = 'CCDppBySNR'
DISTANCES_MATLAB_FILE = 'files/targPosition_all.mat'
CHANNELS = 700
LINKAGE_METHODS = ['single','complete','average','weighted','centroid','median','ward']
CCDPP_STRUCT_FIELDS = ['deadTime','numSegments','samplingRate','segStart']

def readEventsFiles():
    print('readEventsFile')
    events1 = loadCCDpp(0)
    events2 = loadCCDpp(1)
    emptyEvents = np.zeros((events1.shape[0],1000))
    allEvents = np.hstack((events1, emptyEvents, events2))
    print ('calcEventsTimes')
    eventsTimes = calcEventsTimes(allEvents)
    utils.save(eventsTimes, 'allEventsTimes.pkl')

def loadCCDpp(categoryID):
    return  np.array(loadCCDppStructure(categoryID)['CCDpp'])

def loadMainCCDpp():
    return  np.array(loadMainCCDppStructure()['CCDpp'])


def loadCCDppStructure(categoryID):
    return h5py.File('{}/{}/{}.mat'.format(ROOT_FOLDER,CATEGORIES[categoryID],FILE_NAME))

def loadMainCCDppStructure():
#     return h5py.File('{}/{}.mat'.format(ROOT_FOLDER,FILE_NAME))
    return utils.loadMatlab('{}/{}'.format(ROOT_FOLDER, FILE_NAME))

def saveMainCCDpp(events):
    events = np.array(events).T
    ccDppStruct = loadMainCCDppStructure()
    newccDppStruct = copyCCDppStructures(ccDppStruct,events)
    utils.saveDictToMatlab(newccDppStruct,'{}/{}_squeeze'.format(ROOT_FOLDER,FILE_NAME))

def copyCCDppStructures(orgCCdPP, events):
    newccDppStruct = {}
    if ('deadTime' in orgCCdPP.keys()): newccDppStruct['deadTime'] = orgCCdPP['deadTime'][0][0]
    if ('numSegments' in orgCCdPP.keys()): newccDppStruct['numSegments'] = orgCCdPP['numSegments'][0][0]
    if ('segStart' in orgCCdPP.keys()): newccDppStruct['segStart'] = np.array(orgCCdPP['segStart'])
    newccDppStruct['samplingRate'] = orgCCdPP['samplingRate'][0][0]
    newccDppStruct['CCDpp'] = events.T
    return newccDppStruct
    
def CCDppFileName(categoryID):
    return '{}/{}/{}.mat'.format(ROOT_FOLDER,CATEGORIES[categoryID],FILE_NAME)

def saveCCDppFiles(D,cut,method='ward'):
    newEvents = []
    events = loadMainCCDpp()
    Y = sch.linkage(D, method=method)
    T= sch.fcluster(Y, cut,'maxclust')
    for channels in channelsPerCluster(T):  
        newEvents = utils.arrAppend(newEvents, utils.logicOrOnVectors(events[channels,:]))
    saveMainCCDpp(newEvents)
        

def calcEventsDistances():
    print('calcEventsDistances')
    eventsTimes={}
    for category in CATEGORIES:
        eventsTimes = utils.load('eventsTimes_{}'.format(category))
        calcDistanceMatrix(eventsTimes,category)

def calcDistanceMatrix():
    eventsTimes = utils.load('allEventsTimes.pkl')
    print ('calcDistanceMatrix')
    params = [(i,eventsTimes) for i in range(CHANNELS)]
#     calcVPDistance((0,eventsTimes))
    
    print ('{} couples to compute'.format(len(params)))
    results = pool.map(calcVPDistance, params) 
    utils.save(results,'distancesResults.pkl')   
    
def readDistancesResults():        
    results = utils.load('distancesResults.pkl')   
    dists = np.zeros((CHANNELS,CHANNELS))

    for channelResults in results:
        for (i,j,dist) in channelResults:
            dists[i,j]=dist
    for i in range(CHANNELS):
        for j in range(i+1,CHANNELS):
            dists[j,i]=dists[i,j]
                
    utils.save(dists, 'vpd.pkl')

def calcVPDistance(params):
    i,events = params
    print('channel {}'.format(i))
    N=len(events)
    dists = []
    for j in range(i+1,N):
        distance = vpd(events[i],events[j])
        dists.append((i,j,distance))
    
    utils.save(dists,'files/dists_{}'.format(i))    
    return dists

def calcEventsTimes(events):
    eventsTimes = []
    for channelIndex, channel in enumerate(events):
        if (channelIndex % 10 ==0):
            print('channel {}'.format(channelIndex))
        eventsTimes.append(trainToTimes(channel))
    return eventsTimes

def trainToTimes(st):
    return np.where(st)[0]



def mog(X,category,pool):
    params = []
    features, pca = PCA.transform(dists, n_components=1, doPrint=True, returnModel=True)
#     print(np.cumsum(pca.explained_variance_ratio_))
#     plt.graph(range(len(features)), np.sort(features[:,0]))
    for n_components in range(1, features.shape[0]):
        params.append([features,n_components])

    results = pool.map(mog_fit, params)            
    utils.save(results, 'bics_{}.pkl'.format(category))

def mog_fit(params):
    # Fit a mixture of gaussians with EM
    X,n_components = params
    print(n_components)
    gmm = mixture.GMM(n_components=n_components, covariance_type='full')
    gmm.fit(X)
    bic = gmm.bic(X)
    return (n_components,bic)

def loadBICs(category):
    bics = utils.load('bics_{}.pkl'.format(category))
    plt.graph(xrange(len(bics)), bics, category)

def plotHierarchicalClustering(D):
#     v = squareform(D) 
    Y = sch.linkage(D, method='centroid')

    xrange = range(1,700) # np.arange(0.1,1.5,0.1)
    clusts = np.zeros(len(xrange))
    for i,cut in enumerate(xrange):
        print(cut)
        T= sch.fcluster(Y, cut,'maxclust')# 'inconsistent')
        clusts[i] = (len(np.unique(T)))

    utils.save(clusts,'maxclust.pkl')
    plt.graph(xrange, clusts,title='#clusters / cutoff index', xlabel='cutoff', ylabel='#clusters')#, fileName='clusters_vs_cutoff_index')
#     plt.plt.show()    
#     plt.plt.savefig('dendrogramAll.svg')
    
def plotHierarchicalClusteringOnTopOfDistancesMatrix(D):
    # Compute and plot dendrogram.
    fig = pylab.figure()
    axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
    Y = sch.linkage(D, method='centroid')
    Z = sch.dendrogram(Y, orientation='right')
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    
#     # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    index = Z['leaves']
    D = D[index,:]
    D = D[:,index]
    im = axmatrix.matshow(D, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
#      
#     # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
    pylab.colorbar(im, cax=axcolor)
      
#     Display and save figure.
    fig.show()    
    plt.plt.show()
    
def plotDistsHist(dists):
    distsVals = MLUtils.halfMat(dists,False)
    plt.histCalcAndPlot(distsVals,binsNum=50)

def plotHierarchicalClusteringForCategories():
    for category in CATEGORIES:
        dists = utils.load('vpd_{}.pkl'.format(category))
        np.fill_diagonal(dists,0)
        plotHierarchicalClustering(dists,category)

def plotMaxclust():
    clusts = {}
    for category in CATEGORIES:
        clusts[category] = utils.load('maxclust_{}'.format(category))
    
    plt.graph2(range(1,700), clusts[CATEGORIES[0]], clusts[CATEGORIES[1]], CATEGORIES, title='#clusters / cutoff index', xlabel='cutoff', ylabel='#clusters', fileName='clusters_vs_cutoff_index',
               legendLocation='upper left')


def analyzeMeanChannelsNumInClusters(D,load=True):
    xrange = np.arange(1,700) 
    if (not load):
        for link_method in LINKAGE_METHODS:
            if (link_method!='ward'): continue
            print(link_method)
            Y = sch.linkage(D, method=link_method)
            meanChannels = np.zeros((len(xrange)))
            medChannels = np.zeros((len(xrange)))
            for i,cut in enumerate(xrange):
                if (cut!=40):continue
                if (i % 10 == 0): print(cut)            
                T = sch.fcluster(Y, cut,'maxclust')
                channelsNum = channelsNumPerCluster(T)
                meanChannels[i] = np.mean(channelsNum)
                medChannels[i] = np.median(channelsNum)
            np.savez('files/meanChannels_{}'.format(link_method),meanChannels=meanChannels,medChannels=medChannels)
    else:
        meanChannels,medChannels={},{}
        for link_method in LINKAGE_METHODS:
            meanChannelsDic = np.load('files/meanChannels_{}.npz'.format(link_method))
            meanChannels[link_method], medChannels[link_method] = meanChannelsDic['meanChannels'], meanChannelsDic['medChannels']
            medChannels[link_method] = medChannels[link_method][:50]
        
#     plt.barPlot(meanChannels[50:],  errors=stdChannels[50:])
    plt.graph(xrange[:50], medChannels['ward'])
    plt.graphN(xrange[:50], medChannels.values(),LINKAGE_METHODS)
    
#     sns.tsplot(xrange[50:],meanChannels[50:])
#     plt.plt.show()    


def analyzeDistancesInClusters(D):
    distancesMat = utils.loadMatlab(DISTANCES_MATLAB_FILE)
    realDistances = distancesMat['D']
    realPositons = distancesMat['XYZ']
    clusts = utils.load('files/maxclust.pkl')
    xrange = range(1,CHANNELS) 
#     plt.graph(xrange,clusts)
    
    clustersMeasure = np.zeros((len(xrange)))
    Y = sch.linkage(D, method='ward')
    cuts = xrange # [30] # [60] # [40] #[300,500,600]
    for i,cut in enumerate(cuts):
        T= sch.fcluster(Y, cut,'maxclust')
        clustersNum = (len(np.unique(T)))
        print(i)
        distancesInClusters = histDistsInClusters(T,doPlot=False)
#         channelsNum = channelsNumPerCluster(T)
#         h, _ = np.histogram(channelsNum, 20)
#         ent = utils.calcEntropy(h)
        clustersMeasure[i] = np.mean(distancesInClusters[np.where(distancesInClusters)]) 
#     plt.barPlot(clustersMeasure,'Entropy',title='Entropy vs clustering cutoff',startsWithZeroZero=True)

    plt.barPlot(clustersMeasure,startsWithZeroZero=True)

def spectralClustering(D,doLoad=True):
#     eigen_solver: {None, ‘arpack’, ‘lobpcg’, or ‘amg’}
#     assign_labels : {‘kmeans’, ‘discretize’}, default: ‘kmeans’
    clustersNumRange = np.arange(1,700)
    if (not doLoad):
        medChannels = np.zeros((len(clustersNumRange)))
        for i,clustersNum in enumerate(clustersNumRange):
            print(clustersNum) 
            T = spectral(D,n_clusters=clustersNum,eigen_solver=None,assign_labels='kmeans')
            channelsNum = channelsNumPerCluster(T)
            medChannels[i] = np.median(channelsNum)
        np.save('files/spectral_med',medChannels)
    else:
        medChannels = np.load('files/spectral_med.npy')
        maxClustersIndex = np.where(medChannels==max(medChannels))[0][0]
        print(maxClustersIndex,medChannels[maxClustersIndex])
#         plt.graph(clustersNumRange,medChannels)
        
def spectralClusteringMaxMed(D):
    medChannels = np.load('files/spectral_med.npy')
    maxMedClustersNum = np.where(medChannels==max(medChannels))[0][0]
    T = spectral(D,n_clusters=maxMedClustersNum,eigen_solver=None,assign_labels='kmeans')
    histDistsInClusters(T)

def histDistsInClusters(T,doPlot=True):
    distancesMat = utils.loadMatlab(DISTANCES_MATLAB_FILE)
    realDistances = distancesMat['D']
#     realPositons = distancesMat['XYZ']

    clustersNum = calcClustersNum(T)
    distsInClustrers = np.zeros((clustersNum))
    for clust in range(clustersNum):
        channels = np.where(T==(clust+1))[0]
        if (len(channels)>1):
            distsInClustrers[clust] = np.mean([realDistances[channels[0],channels[c]] for c in range(1,len(channels))])

    if (doPlot): 
#         plt.histCalcAndPlot(distsInClustrers, binsNum=20,title='distances in clusters')
        plt.barPlot(distsInClustrers,title='Distances in clusters',startsWithZeroZero=True)

    return distsInClustrers

def channelsNumPerCluster(T,doPlot=True):
    channelsNum = np.zeros((calcClustersNum(T)))
    for clust,channels in enumerate(channelsPerCluster(T)):
        channelsNum[clust] = len(channels)
    if (doPlot):
        plt.barPlot(channelsNum,title='Channels per cluster',startsWithZeroZero=True)
    return channelsNum

def channelsPerCluster(T):    
    for clust in range(calcClustersNum(T)):
        yield np.where(T==(clust+1))[0]

def calcClustersNum(T):
    return len(np.unique(T))

if __name__ == '__main__':
    args = sys.argv[1:]
    cpuNum = 1 if (len(args) < 1) else int(args[0])
    if (cpuNum > 1): print('cpuNum = %d' % cpuNum)
    pool = Pool(processes=cpuNum)
    
    tic = utils.ticToc()
    
#     readEventsFiles()
#     calcDistanceMatrix()
#     readDistancesResults()

    dists = utils.load('files/vpd.pkl')
    plotHierarchicalClusteringOnTopOfDistancesMatrix(dists)
    plotHierarchicalClustering(dists)
#     analyzeMeanChannelsNumInClusters(dists)
#     analyzeDistancesInClusters(dists)
#     saveCCDppFiles(dists,30)
#     spectralClustering(dists)
#     spectralClusteringMaxMed(dists)

 
    utils.howMuchTimeFromTic(tic)

