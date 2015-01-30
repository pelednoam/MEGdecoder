'''
Created on Apr 11, 2011

@author: noam
'''

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import cnames
from matplotlib import rcParams
from matplotlib import animation
from matplotlib.figure import Figure
import matplotlib.cm as cmx

from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D

import pylab
import scipy.cluster.hierarchy as sch

from src.commons.utils.utils import tryCall
from src.commons.utils import MLUtils

try:
    from PyQt4 import QtGui
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
except:
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    print('No QtGui!')

try:
    import seaborn as sns
except:
    print('no seaborn')


def init():
    # set plot attributes
    fig_width = 12  # width in inches
    fig_height = 9  # height in inches
    fig_size = [fig_width, fig_height]
    params = {'backend': 'Agg',
              'axes.labelsize': 22,
              'axes.titlesize': 20,
              'text.fontsize': 20,
              'legend.fontsize': 22,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'lines.linewidth': 5,
              'figure.figsize': fig_size,
              'savefig.dpi': 600,
              'font.family': 'sans-serif'}
    rcParams.update(params)


def autolabel(rects, fontSize=24):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%.2f' % height,
                 ha='center', va='bottom', fontsize=fontSize)


# @tryCall
def barGrouped2(x1, x2, x1std=None, x2std=None, labels=None, xtick=None, ylim=None,
                ylabel='', title='', xrotate=0, figName='', doShow=True):
    if not labels:
        labels = ['', '']
    init()
    locs = np.arange(1, len(x1) + 1)
    width = 0.33
    if (x1std is None):
        x1std = np.zeros((len(x1)))
    if (x2std is None):
        x2std = np.zeros((len(x2)))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.bar(locs + width / 2, x1, width=width, yerr=x1std, ecolor='black', facecolor=cnames['darkcyan'],
            label=labels[0])
    plt.bar(locs + width * 1.5, x2, width=width, yerr=x2std, ecolor='black', facecolor=cnames['darkkhaki'],
            label=labels[1])
    plt.xticks(locs + width * 1.5, locs)
    if (xtick is not None):
        ax.set_xticklabels(xtick)
    plt.legend(loc="upper right")
    plt.ylabel(ylabel)
    if (ylim):
        plt.ylim([ylim[0], ylim[1]])
    plt.grid(True)
    if (xrotate):
        plt.xticks(rotation=xrotate)
    if (title != ''):
        plt.title(title)
    # fig.autofmt_xdate()
    if (figName != ''):
        plt.savefig('{}.jpg'.format(figName))
    if (doShow):
        plt.show()


@tryCall
def barPlot(x, yLabel='', xTickLabels='', ylim=None, xlim=None, labels=None,
            xLabelsFontSize=26, yFontSize=26, title='', titleFontSize=26,
            doPlotValues=False, valuesFontSize=26, errors=None, xrotate=0,
            startsWithZeroZero=False, fileName='',
            doShow=True, fileType='jpg'):
    x = np.array(x)
    if not ylim:
        ylim = []
    if not xlim:
        xlim = []
    if not labels:
        labels = ['', '', '']
    init()
    ind = range(len(x))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if (errors is None):
        errors = np.zeros((len(x)))
    rects = ax.bar(ind, x, facecolor='#777777', align='center',
                   ecolor='black', yerr=errors)
    if (doPlotValues):
        autolabel(rects, valuesFontSize)
    if (startsWithZeroZero):
        if (ylim):
            ylim[0] = 0
        else:
            ylim = [0, max(x * 1.1)]
        if (xlim):
            xlim[0] = -0.5
        else:
            xlim = [-0.5, len(x)]
    if (ylim):
        plt.ylim(ylim)
    if (xlim):
        plt.xlim(xlim)
    if (xTickLabels != ''):
        plt.xticks(ind)
        ax.set_xticklabels(xTickLabels, fontsize=xLabelsFontSize)
    # plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=xLabelsFontSize)
    plt.ylabel(yLabel, fontsize=yFontSize)
    if (xrotate):
        plt.xticks(rotation=xrotate)
    if (title != ''):
        ax.set_title(title, fontsize=titleFontSize)
    # fig.autofmt_xdate()
    # fig.autofmt_xdate()
    if (fileName != ''):
        plt.savefig('{}.{}'.format(fileName, fileType))
    if (doShow):
        plt.show()
    plt.close()


@tryCall
def bens3GErrBars(x1Avg, x1Std, x2Avg, x2Std, x3Avg, x3Std, yLabel='Benefit', yLim=130, labels=None, xTickLabels=None,
                  errBars=True):
    if not labels: labels = ['SIGAL', 'EQ', 'People']
    if not xTickLabels: xTickLabels = ['Proposer in round 1', 'Proposer in round 2', 'Global']
    init()
    locs = np.arange(1, 4)
    eps = 0.01
    width = 0.25
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if (errBars):
        plt.bar(locs + width, x1Avg, width=width, yerr=x1Std, ecolor='black', facecolor=cnames['darkblue'],
                label=labels[0]);
        plt.bar(locs + width * 2 + eps, x2Avg, width=width, yerr=x2Std, ecolor='black', facecolor=cnames['green'],
                label=labels[1]);
        plt.bar(locs + width * 3 + eps * 2, x3Avg, width=width, yerr=x3Std, ecolor='black', facecolor=cnames['brown'],
                label=labels[2]);
    else:
        plt.bar(locs + width, x1Avg, width=width, facecolor=cnames['darkblue'], label=labels[0]);
        plt.bar(locs + width * 2 + eps, x2Avg, width=width, facecolor=cnames['green'], label=labels[1]);
        plt.bar(locs + width * 3 + eps * 2, x3Avg, width=width, facecolor=cnames['brown'], label=labels[2]);
    plt.xticks(locs + width * 2.5, locs);
    ax.set_xticklabels(xTickLabels)
    plt.legend(loc="upper right")
    plt.ylabel(yLabel)
    plt.ylim([0, yLim])
    plt.grid(True)
    # fig.autofmt_xdate()
    plt.show()
    # plt.savefig('%s.svg'%figame)


@tryCall
def twoBarsPlot(x1, x2, xlabel, ylabel, xtick1, xtick2):
    init()
    ind = np.arange(2)
    width = 0.33
    plt.bar(ind, [np.mean(x1), np.mean(x2)], width=width, yerr=[np.std(x1), np.std(x2)], ecolor='black',
            facecolor=cnames['darkcyan']);
    plt.xticks(ind + width / 2, [xtick1, xtick2]);
    plt.ylim([0, 1.5])
    plt.grid(True)
    # fig.autofmt_xdate()
    plt.show()
    # plt.savefig('%s.svg'%figName)


# @tryCall
def graph(x, y, title='', xlabel='', ylabel='', xlim=None, ylim=None,
          fileName='', yerr=None, doShow=True, fileType='jpg'):
    if (xlim is None): xlim = []
    if (ylim is None): ylim = []
    if (yerr is None): yerr = []
    if (x is None): x = range(len(y))
    if (doShow): plt.figure()
    plt.plot(x, y, '-')
    if (len(yerr) > 0):
        plt.errorbar(x, y, yerr)  # , color='%s' % colors[i])  # , linestyle='-', label=labels[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if (xlim): plt.xlim(xlim)
    if (ylim): plt.ylim(ylim)
    if (fileName != ''):
        plt.savefig('{}.{}'.format(fileName, fileType))
    if (doShow):
        plt.show()
    plt.close()


# @tryCall
def graph2(x, y1, y2, labels=['0', '1'], xlim=None, ylim=None, title='',
           yerrs=None, xlabel='', ylabel='', fileName='', x2=None,
           markers=('b-', 'g-'), legendLocation='upper right', doPlot=True):
    #     plt.figure()
    y1 = np.array(y1)
    y2 = np.array(y2)
    x2 = x if x2 is None else x2
    plt.plot(x, y1, markers[0], label=labels[0])
    plt.plot(x2, y2, markers[1], label=labels[1])
    if (yerrs is not None and len(yerrs) == 2):
        yerrs0 = np.array(yerrs[0])
        yerrs1 = np.array(yerrs[1])
        plt.fill_between(x, y1 - yerrs0, y1 + yerrs0,
                         alpha=0.2, edgecolor='#1B2ACC', facecolor='b')
        #             linewidth=4, linestyle='dashdot', antialiased=True)
        plt.fill_between(x2, y2 - yerrs1, y2 + yerrs1,
                         alpha=0.2, edgecolor='#1B2ACC', facecolor='g')
    #             linewidth=4, linestyle='dashdot', antialiased=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if (xlim):
        plt.xlim(xlim)
    if (ylim):
        plt.ylim(ylim)
    plt.legend(loc=legendLocation)
    if (fileName != ''):
        plt.savefig('%s.png' % fileName)
    if (doPlot):
        plt.show()
    plt.close()


# @tryCall
def graphN(x, ys, labels=(), yerrs=None, xlabel='', ylabel='', title='',
           xlim=None, ylim=None, legendLoc='upper right',
           doSmooth=False, smoothWindowSize=21, smoothOrder=3,
           fileName='', doShow=True, poster=True):
    if (yerrs is None):
        yerrs = [[]] * len(ys)
    pp = []
    lines = cycle(['-'] * len(ys)) if poster else linesCycler()
    #    colors = ['r', 'b', 'g']
    for y, yerr in zip(ys, yerrs):
        if (doSmooth):
            y = MLUtils.savitzkyGolaySmooth(y, smoothWindowSize, smoothOrder)
        p = plt.plot(x, y, next(lines), lw=4 if poster else 3)
        pp.append(p[0])
        if (len(yerr) > 0):
            yerr = np.array(yerr)
            plt.fill_between(x, y - yerr, y + yerr,
                             alpha=0.2, edgecolor='#1B2ACC', facecolor=p[0]._color)

    if (len(labels) > 0):
        plt.legend(pp, labels, loc=legendLoc, numpoints=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if (xlim):
        plt.xlim(xlim)
    if (ylim):
        plt.ylim(ylim)
    if (fileName != ''):
        plt.savefig('%s.png' % fileName)
    if (doShow):
        plt.show()


@tryCall
def confInt(x, y, stds, label=''):
    y, stds = np.array(y), np.array(stds)
    plt.plot(x, y, '-', label=label)
    plt.fill(np.concatenate([x, x[::-1]]), \
             np.concatenate([y - stds, (y + stds)[::-1]]), \
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.legend()
    plt.show()


@tryCall
def scatterPlot(x, y, size=10, title='', xlabel='', ylabel='', doShow=True):
    fig = plt.figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='-', color='0.75')
    ax.scatter(x, y, size, color='tomato');
    plt.show()
    # canvas.print_figure('revelationVSscore.png',dpi=500)


@tryCall
def histCalcAndPlot(x, minx=-1, maxx=-1, binsNum=10, xlabel='', title='', show=True, fileName=''):
    if (minx == -1): minx = np.min(x)
    if (maxx == -1): maxx = np.max(x)
    bins = np.linspace(minx, maxx, binsNum)
    #    histPlot(x, bins)
    plt.hist(x, bins, alpha=0.5)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.xlim([minx, maxx])
    if (fileName != ''):
        plt.savefig('%s.png' % fileName)
    if (show): plt.show()


@tryCall
def histCalcAndPlotN(X, binsNum=10, labels=None, xlabel='', title=''):
    if not labels: labels = []
    mins, maxs = [], []
    for x in X:
        mins.append(np.min(x))
        maxs.append(np.max(x))
    bins = np.linspace(min(mins), max(maxs), binsNum)
    for i, x in enumerate(X):
        label = str(i) if labels == [] else labels[i]
        plt.hist(x, bins, alpha=0.5, label=label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()


@tryCall
def histBarPlot(hist, bins):
    plt.bar(bins[:-1], hist)
    plt.show()


@tryCall
def setHistBarPlot(counter):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.bar(range(len(counter)), counter.values())
    labels = []
    for key in counter.keys(): labels.append(str(key))
    ax.set_xticks(range(len(counter)))
    ax.set_xticklabels(counter.keys())
    plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=12)
    plt.show()


@tryCall
def histPlot(x, min, max, binsNum):
    bins = np.linspace(min, max, binsNum)
    plt.hist(x, bins, alpha=0.5)
    plt.show()


@tryCall
def twoHistsPlot(x1, x2, binsNum=10, label1='', label2='', xmin=None, xmax=None):
    if not xmin: xmin = []
    if not xmax: xmax = []
    if (not xmin): xmin = min([min(x1), min(x2)])
    if (not xmax): xmax = max([min(x1), max(x2)])
    bins = np.linspace(xmin, xmax, binsNum)
    #    fig = plt.figure()
    #    ax = fig.add_subplot(1, 1, 1)
    plt.hist(x1, bins, alpha=0.5, edgecolor='black', label=label1)
    plt.hist(x2, bins, alpha=0.5, edgecolor='black', label=label2)
    plt.legend()
    #    ax.set_xticks(np.linspace(xmin,xmax,30))
    #    plt.xticks(rotation=70)
    plt.show()


@tryCall
def threeHistsPlots(x1, x2, x3, min, max, binsNum):
    bins = np.linspace(min, max, binsNum)
    plt.hist(x1, bins, alpha=0.9, facecolor='yellow', edgecolor='black')
    plt.hist(x2, bins, alpha=0.9, facecolor='blue', edgecolor='black')
    plt.hist(x3, bins, alpha=0.9, facecolor='red', edgecolor='black')
    plt.show()


@tryCall
def twoHistsPlot2(x1, x2, binsNum=10, xmin=None, xmax=None, label1='', label2='', title=''):
    if not xmin: xmin = []
    if not xmax: xmax = []
    if (not xmin): xmin = min([min(x1), min(x2)])
    if (not xmax): xmax = max([min(x1), max(x2)])
    bins = np.linspace(xmin, xmax, binsNum)
    plt.subplot(111)
    hist1, bins1 = np.histogram(x1, bins)
    hist1 /= float(np.sum(hist1))
    center1 = (bins1[:-1] + bins1[1:]) / 2
    hist2, bins2 = np.histogram(x2, bins)
    hist2 /= float(np.sum(hist2))
    center2 = (bins2[:-1] + bins2[1:]) / 2
    width = 1.5
    plt.bar(center1, hist1, width, color='k', label=label1)
    plt.bar(center2 + width, hist2, width, color='w', label=label2)
    plt.legend()
    plt.title(title)
    plt.show()


@tryCall
def histMeans(x, y, binsNum=20):
    bins = np.linspace(np.min(x), np.max(x), binsNum)
    hist, bins = np.histogram(x, bins)
    centers = (bins[:-1] + bins[1:]) / 2
    ind = 0
    ymeans = []
    for itemsNum in hist:
        ymeans.append(np.mean(y[ind:ind + itemsNum]))
        ind += itemsNum + 1
    scatterPlot(centers, ymeans)


@tryCall
def regressionLine(x, y, yh):
    plt.scatter(x, y, 10, color='k');
    plt.scatter(x, yh, 10, color='tomato');
    # plt.plot(x, y, 'k', x, yh, 'r-')
    # plt.plot([x1, x2], [y1, y2], 'k')
    plt.show()


@tryCall
def scoresScatter(xProposer, yResponder, xLabel, yLabel, labels, markers=None):
    if not markers: markers = ['o', (5, 0), '>']
    init()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(xLabel, fontsize=14)
    ax.set_ylabel(yLabel, fontsize=14)
    ax.grid(True, linestyle='-', color='0.75')

    for ind, lab in enumerate(labels):
        plt.scatter(xProposer[ind], yResponder[ind], marker=markers[ind], s=100, label=lab)

    #    for label, _x, _y in zip(['round 1', 'round 2', 'round 1', 'round 2'], xProposer, yResponder):
    #        plt.annotate(label,
    #                xy=(_x, _y), xytext=(-20, 20),
    #                textcoords='offset points', ha='right', va='bottom',
    #                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    #                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.legend(loc="upper left")
    plt.show()


@tryCall
def matShow(z, title='', xlabel='', ylabel='', xticks=None, yticks=None):
    if (xticks is None): xticks = []
    if (yticks is None): yticks = []

    fig = plt.figure()

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.8])
    im = axmatrix.matshow(z, aspect='auto', origin='lower')
    axmatrix.set_xticks(xticks)
    axmatrix.set_yticks(yticks)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Plot colorbar.
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.8])
    plt.colorbar(im, cax=axcolor)

    if (title != ''): plt.title(title)
    # Display and save figure.
    #     fig.show()
    plt.show()


def linesCycler():
    return cycle(["--", "-.", ":", "-", "--"])


class Animate(object):
    def __init__(self, dataFunc):
        self.fig = plt.figure()
        self.ax = plt.axes()  # xlim=(0, 2), ylim=(-2, 2))
        self.line, = self.ax.plot([], [], lw=2)
        self.dataFunc = dataFunc

    def init(self):
        self.line.set_data([], [])
        return self.line,

    def animate(self, t):
        x, y = self.dataFunc(t)
        self.line.set_data(x, y)
        return self.line,

    def run(self):
        self.anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init, frames=200, interval=20,
                                            blit=True)
        #    anim.save('basic_animation.mp4', fps=30)#, extra_args=['-vcodec', 'libx264'])
        plt.show()


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, data, lines, sizes, interval=10, lims=None, figureName='', movieName=''):
        if not lims: lims = [-10, 10, -10, 10]
        self.numpoints = data.shape[1]
        self.stream = self.data_stream()
        self.fig, self.ax = plt.subplots()
        self.linesNum = lines.shape[1]
        self.line = [None] * self.linesNum
        for i in range(self.linesNum):
            self.line[i], = self.ax.plot([], [], lw=2)
        self.data = data
        self.lines = lines
        self.sizes = sizes
        self.interval = interval
        self.lims = lims
        self.figureName = figureName
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=interval,
                                           init_func=self.setup_plot, blit=True)
        if (not movieName == ''):
            self.ani.save('%s.mp4' % movieName)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        points, _ = next(self.stream)
        plt.hold(True)
        self.scat = self.ax.scatter(points[:, 0], points[:, 1], c=self.sizes, s=self.sizes * 30, animated=True)
        self.ax.axis(self.lims)
        for i in range(self.linesNum):
            self.line[i].set_data([], [])
            self.line[i].set_color('b')
        ret = list([self.scat])
        for i in range(self.linesNum): ret.append(self.line[i])
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.fig.canvas.set_window_title(self.figureName)
        return ret

    def data_stream(self):
        for points, lines in zip(self.data, self.lines):
            yield points, lines

    def update(self, i):
        """Update the scatter plot."""
        points, lines = next(self.stream)
        self.scat.set_offsets(points)
        for i, line in enumerate(lines):
            self.line[i].set_data([line[0]], [line[1]])
        ret = list([self.scat])
        for i in range(self.linesNum): ret.append(self.line[i])
        return ret

    def run(self):
        plt.show()


class AnimatedBar(FigureCanvas):
    def __init__(self, dataGenerator, dtTimer=1000, ylim=None, xlim=None, startsWithZeroZero=False, title=''):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.title = title
        self.dataGen = dataGenerator
        x, t = self.dataGen.next()
        self.setTitle(t)
        self.counter = 1
        self.width = 0.8
        self.locs = np.arange(len(x))
        self.bars = self.ax.bar(self.locs, x, self.width, color='#6a7ea6')
        if not ylim: ylim = []
        if not xlim: xlim = []
        if (startsWithZeroZero):
            if (ylim):
                ylim[0] = 0
            else:
                ylim = [0, max(x * 1.1)]
            if (xlim):
                xlim[0] = -0.5
            else:
                xlim = [-0.5, len(x)]
        if (ylim): self.ax.set_ylim(ylim[0], ylim[1])
        if (xlim): self.ax.set_xlim(xlim[0], xlim[1])
        self.fig.canvas.draw()
        self.show()
        self.timer = self.startTimer(dtTimer)

    def timerEvent(self, evt):
        # update the height of the bars, one liner is easier
        x, t = self.dataGen.next()
        if (x is None):
            self.killTimer(self.timer)
        else:
            [bar.set_height(x) for bar, x in zip(self.bars, x)]
            self.setTitle(t)
            self.fig.canvas.draw()

    def setTitle(self, t):
        self.setWindowTitle('{0} t={1:.2f} sec'.format(self.title, t))


def runAnimatedBar(dataGenerator, dtTimer=1000, title='', ylim=None, xlim=None, startsWithZeroZero=False):
    app = QtGui.QApplication(sys.argv)
    w = AnimatedBar(dataGenerator, dtTimer, ylim, xlim, startsWithZeroZero, title)
    sys.exit(app.exec_())


def colors(k):
    return sns.color_palette(None, k)


def plot3d(X, color='b', xlabel='', ylabel='', mark='o'):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(X[:, 0], X[:, 1], X[:, 2], mark, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


#     LinearSegmentedColormap.from_list(name, colors, N, gamma)


def scatter3d(X, cs=None, colorsMap='jet'):
    if (cs is not None):
        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    if (cs is None):
        ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=scalarMap.to_rgba(cs))
        scalarMap.set_array(cs)
        fig.colorbar(scalarMap)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def arrToColors(x, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(x), vmax=max(x))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    return scalarMap.to_rgba(x)


def plotHierarchicalClusteringOnTopOfDistancesMatrix(D):
    # Compute and plot dendrogram.
    fig = pylab.figure()
    axdendro = fig.add_axes([0.09, 0.1, 0.2, 0.8])
    Y = sch.linkage(D, method='centroid')
    Z = sch.dendrogram(Y, orientation='right')
    axdendro.set_xticks([])
    axdendro.set_yticks([])

    #     # Plot distance matrix.
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.8])
    index = Z['leaves']
    D = D[index, :]
    D = D[:, index]
    im = axmatrix.matshow(D, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    #     # Plot colorbar.
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.8])
    pylab.colorbar(im, cax=axcolor)
    #     Display and save figure.
    fig.show()
    plt.show()
