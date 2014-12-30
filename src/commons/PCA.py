'''
Created on Dec 17, 2011

@author: noam
'''
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
import sklearn.decomposition as deco
from sklearn.pls import PLSRegression
from sklearn.lda import LDA
from src.commons.utils import MLUtils
from numpy.random import RandomState
from sklearn.cluster import MiniBatchKMeans
from src.commons.utils.plots import plt
import numpy as np

def transform(x, n_components=3, y=None, doPlot=False, doPrint=False, returnModel=False):
    if not y: y = []
    x, _, _ = MLUtils.normalizeData(x)
    pca = deco.PCA(n_components)
    x_r = pca.fit(x).transform(x)
    if (doPrint):
        print ('explained variance ratio (first %d components): %.2f'%(n_components, sum(pca.explained_variance_ratio_)))
    if (doPlot):
        show(x_r, y, n_components)
    if (returnModel):
        return x_r, pca 
    else:
        return x_r


def ICA(x,n_components):
    ica = deco.FastICA(n_components=n_components, whiten=True, max_iter=10)
    return ica.fit_transform(x)
#    ica.fit(x.T)
#    return ica.components_.T

def PLS(x,y,n_components):
    pls = PLSRegression(n_components=n_components)
    pls.fit (x, y)
#     ssy = np.dot(pls.y_weights_.T, pls.y_weights_)
#     ssx = np.dot(pls.x_weights_.T, pls.x_weights_)
#     explained_variance = sum(ssx)/sum(ssy)
#     x_r = pls.transform(x, copy=True)
    return pls.x_scores_, pls



def randomizedPCA(x,n_components):
    rpca = deco.FastICA(n_components=n_components, whiten=True)
#    rpca.fit(x.T)
    return rpca.fit_transform(x)
#    return rpca.components_.T

def miniBatchSparsePCA(x,n_components):
    rng = RandomState(0)
    mbspca = deco.MiniBatchSparsePCA(n_components=n_components, alpha=0.8,
                                      n_iter=100, chunk_size=3,
                                      random_state=rng)
    x_r = mbspca.fit_transform(x)
    return x_r

def NMF(x,n_components):
    nmf = deco.NMF(n_components=n_components, init='nndsvda', beta=5.0,
                       tol=5e-3, sparseness='components')
    return nmf.fit_transform(x)

def calcLDA(x, y, n_components=3, doPlot=False):
    lda = LDA(n_components)
    x_r = lda.fit(x, y).transform(x)
    if (doPlot):
        show(x_r, y, n_components)
    return x_r
    
def show(x_r, y, n_components):
    fig = plt.figure()
    if (n_components == 2):  ax = fig.add_subplot(111)
    elif (n_components == 3): ax = fig.add_subplot(111, projection='3d')
    for c, i, target_name in zip("rg", [0, 1], ['accept', 'reject']):
        if (n_components == 2): ax.scatter(x_r[y == i, 0], x_r[y == i, 1], c=c, label=target_name)
        elif (n_components == 3): ax.scatter(x_r[y == i, 0], x_r[y == i, 1], x_r[y == i, 2], c=c, label=target_name)
    plt.legend()
    plt.title('PCA')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':  
    B = np.array([[1,2,8],[2,6,4],[3,3,3],[5,7,7],[6,3,5]])
    print(B)
    x_r, pca = transform(B,3,returnModel=True)
    print(x_r)
    print(np.cov(x_r))
