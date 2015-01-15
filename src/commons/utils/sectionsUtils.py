'''
Created on Nov 9, 2014

@author: noam
'''

import plots
import utils
import freqsUtils
import numpy as np


def findSigSectionsInPValues(X, y, selector, alpha, sigSectionMinLength,
            xAxis, cvIndices=None, timeIndices=None, weights=None,
            maxSurpriseVal=20, labels=['0', '1'],
            doPlotSections=True, sectionsKeys=None):
    C = calcSectionsNum(X, None, weights)
    sections, sectionsDic = {}, {}
    if (sectionsKeys is None):
        sectionsKeys = [None] * C
    for c, sectionKey in zip(range(C), sectionsKeys):
        xc = calcSlice(X, c, cvIndices, timeIndices, weights)
        model = selector.fit(xc, y)
        sigSections = findSectionSmallerThan(model.pvalues_,
            alpha, sigSectionMinLength)
        if (len(sigSections) > 0):
            sections[c] = sigSections
            sectionsDic[c] = sectionKey
        if (doPlotSections):
            plot2PSAndSctions(xAxis, model, sigSections, X[:, :, c], y,
                alpha, maxSurpriseVal=maxSurpriseVal,
                xlabel='Time (ms)', labels=labels)
    return sections, sectionsDic


def findSigSectionsPSInPValues(X, y, selector, timeStep, minFreq, maxFreq,
            alpha, minSegSectionLen, cvIndices=None, timeIndices=None,
            weights=None, preCalcPSS=None, preCalcFreqs=None,
            maxSurpriseVal=20, labels=['0', '1'],
            doPlotSections=True, sectionsKeys=None):
    C = calcSectionsNum(X, preCalcPSS, weights)
    sections, sectionsDic = {}, {}
    if (sectionsKeys is None):
        sectionsKeys = [None] * C
    for c, sectionKey in zip(range(C), sectionsKeys):
        if (preCalcPSS is None):
            if (X.ndim > 1):
                xc = calcSlice(X, c, cvIndices, timeIndices, weights)
                freqs, ps = freqsUtils.calcPS(xc, timeStep, minFreq, maxFreq)
            else:
                # Todo: Finish the code for varied T
                for x in X[cvIndices]:
                    freqs, ps = freqsUtils.calcPS(x[:, c], timeStep, minFreq, maxFreq)
        else:
            freqs = preCalcFreqs
            ps = preCalcPSS[c, :, :].squeeze()
            freqs, ps = freqsUtils.cutPS(ps, freqs, minFreq, maxFreq)

        model = selector.fit(ps, y)
        sigSections = findSectionSmallerThan(model.pvalues_, alpha,
            minSegSectionLen)
        if (len(sigSections) > 0):
            sections[c] = sigSections
            sectionsDic[c] = sectionKey
            if (doPlotSections):
                plot2PSAndSctions(freqs, model, sigSections, ps, y, alpha,
                    minFreq, maxFreq, maxSurpriseVal, 'Frequency (Hz)', labels)
        if (preCalcPSS is None):
            if (c == 0):
                pss = np.empty((ps.shape[0], ps.shape[1], C))
                pss.fill(np.NaN)
            pss[:, :, c] = ps

    if (preCalcPSS is not None):
        pss = preCalcPSS

    if (sectionsKeys is None):
        return sections, pss, freqs
    else:
        return sections, pss, freqs, sectionsDic


def calcSlice(X, c, cvIndices, timeIndices, weights):
    if (X.ndim > 1):
        if (cvIndices is None and timeIndices is None):
            xc = X[:, :, c] if weights is None else \
                 np.dot(X[:, :, :], weights[c, :].T)
        elif (cvIndices is not None and timeIndices is None):
            xc = X[cvIndices, :, c] if weights is None else \
                 np.dot(X[cvIndices, :, :], weights[c, :].T)
        elif (cvIndices is None and timeIndices is not None):
            xc = X[:, timeIndices, c] if weights is None else \
                 np.dot(X[:, timeIndices, :], weights[c, :].T)
        else:
            xc = X[:, timeIndices, c][cvIndices, :] if weights is None else \
                 np.dot(X[cvIndices, timeIndices, :], weights[c, :].T)
            # X[cvIndices, timeIndices, c] gives the following error:
            # ValueError: shape mismatch: objects cannot be broadcast
            # to a single shape
    else:
        # Each time series in X has different length
        if (cvIndices is None and timeIndices is None):
            xc = [x[:, c] for x in X] if weights is None else \
                 [np.dot(x[:, c], weights[c, :].T) for x in X]
        elif (cvIndices is not None and timeIndices is None):
            xc = [x[:, c] for x in X[cvIndices]] if weights is None else \
                 [np.dot(x[:, c], weights[c, :].T) for x in X[cvIndices]]
        elif (cvIndices is None and timeIndices is not None):
            xc = [x[timeIndices, c] for x in X] if weights is None else \
                 [np.dot(x[timeIndices, c], weights[c, :].T) for x in X]
        else:
            xc = [x[timeIndices, c] for x in X[cvIndices]] if weights is None else \
                 [np.dot(x[timeIndices, c], weights[c, :].T) for x in X[cvIndices]]
        xc = np.array(xc)
# 
#     print('c is first dim!')
#     xc = np.array(X[c])
#     if (cvIndices is not None and timeIndices is None):
#         xc = xc[cvIndices] if weights is None else \
#              np.dot(xc[cvIndices], weights[c, :].T)
#     elif (cvIndices is None and timeIndices is not None):
#         xc = [x[timeIndices] for x in xc] if weights is None else \
#              [np.dot(x[timeIndices], weights[c, :].T) for x in xc]
#     else:
#         xc = [x[timeIndices] for x in xc[cvIndices]] if weights is None else \
#              [np.dot(x[timeIndices], weights[c, :].T) for x in xc[cvIndices]]
    return xc


def findSectionSmallerThan(x, threshold, sigSectionMinLength,
        smallerOrEquall=True, removeSmallGaps=True):
    if (smallerOrEquall):
        sigIndices = np.where(x <= threshold)[0]
    else:
        sigIndices = np.where(x < threshold)[0]
    sigDiff = np.diff(sigIndices)
    if (removeSmallGaps):
        smallGaps = np.where(sigDiff == 2)[0]
        for gap in smallGaps:
            sigIndices = np.hstack((sigIndices[:gap + 1], [sigIndices[gap] + 1],
                sigIndices[gap + 1:]))
        sigDiff = np.diff(sigIndices)
    sigSections = []
    ind = 0
    while(ind < len(sigDiff)):
        if (sigDiff[ind] == 1 and  # ind + sigSectionMinLength < len(sigDiff) and
                sum(sigDiff[ind:ind + sigSectionMinLength]) ==
                sigSectionMinLength):
            endIndices = np.where(sigDiff[ind:] > 1)[0]
            endIndex = endIndices[0] + ind if (len(endIndices) > 0) \
                else len(sigDiff) #- 1
            minIndex = np.argmin(x[sigIndices[ind]:sigIndices[endIndex] + 1]) \
                + sigIndices[ind]
            sigSections.append((sigIndices[ind], sigIndices[endIndex],
                                minIndex))
#             if (sigIndices[endIndex] - sigIndices[ind] < sigSectionMinLength):
#                 utils.throwException('Error in findSectionSmallerThan')
            ind = endIndex + 1
        else:
            ind += 1
    return sigSections


def concatenateFeaturesFromSections(X, sections, onlyMidValue=True, cvIndices=None):
    features = []
    for c, sensorSections in sections.iteritems():
        indices = []
        for sec in sensorSections:
            if (onlyMidValue):
                indices.append(sec[2])
            else:
                indices.extend(range(sec[0], sec[1] + 1))
        if len(indices) > 0:
            features = utils.matHAppend(features, X[:, indices, c][cvIndices, :]) \
                if (cvIndices is not None) else \
                utils.matHAppend(features, X[:, indices, c])
    return np.array(features)


def plot2PSAndSctions(fs, model, sigSections, X, y, alpha,
                      minVal=None, maxVal=None, maxSurpriseVal=20,
                      xlabel='', labels=['0', '1'], fileName=''):
    if (minVal is None):
        minVal = min(fs)
    if (maxVal is None):
        maxVal = max(fs)
    plots.plt.figure(figsize=(12, 6))
    plots.plt.subplot(121)
    plotSection(fs, model, sigSections, alpha, minVal, maxVal, xlabel,
                maxSurpriseVal, False)
    plots.plt.subplot(122)
    plot2Averages(X, y, fs, minVal, maxVal, xlabel, labels, fileName)


def plot2Averages(X, y, fs, minVal, maxVal, xlabel, labels=['0', '1'], fileName=''):
    x0 = X[y == 0, :]
    x1 = X[y == 1, :]
    x0std = np.std(x0, 0)
    x1std = np.std(x1, 0)
    plots.graph2(fs, np.mean(x0, 0), np.mean(x1, 0),
        xlim=[minVal, maxVal], xlabel=xlabel, doPlot=(fileName == ''),
        yerrs=[x0std, x1std], fileName=fileName, labels=labels)


def plotSection(fs, model, sigSections, alpha, minVal, maxVal, xlabel,
                maxSurpriseVal=20, doShow=True):
    plots.plt.plot(fs, model.scores_, 'b-')
    plots.plt.plot(fs, np.ones(len(model.pvalues_)) *
                   -np.log(alpha), 'r--')
    for sec in sigSections:
        plots.plt.scatter(fs[sec[0]], -np.log(alpha), 20, 'g', marker='o')
        plots.plt.scatter(fs[sec[1]], -np.log(alpha), 20, 'g', marker='o')
        plots.plt.vlines([fs[sec[0]], fs[sec[1]]], 0, maxSurpriseVal,
                         colors='k', linestyles='dashed', label='')
        plots.plt.scatter(fs[sec[2]], model.scores_[sec[2]], 100, 'r',
                          marker='o')
    plots.plt.xlim([minVal, maxVal])
    plots.plt.ylim([0, maxSurpriseVal])
    plots.plt.xlabel(xlabel)
    plots.plt.ylabel('Surprise')
    if (doShow):
        plots.plt.show()


def calcSectionsNum(X, preCalcPSS, weights):
    if (weights is None):
        if (preCalcPSS is None):
            if (X.ndim > 1):
                C = X.shape[2]
            else:
                C = X[0].shape[1]
        else:
            C = preCalcPSS.shape[0]
    else:
        C = weights.shape[0]
    return C
