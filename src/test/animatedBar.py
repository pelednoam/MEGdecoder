import sys
from PyQt4 import QtGui

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

import numpy as np


class Monitor(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)

        self.dataGen = self.dataGenerator()
        data = self.dataGen.next()

        self.counter = 1
        self.width = 0.8
        self.locs = np.arange(len(data))
        self.bars = self.ax.bar(self.locs, data, self.width, color='#6a7ea6')
        self.ax.set_xticks(self.locs+0.5)
        self.ax.set_xticklabels(['Red', 'Green', 'Black', 'Orange', 'Yellow'])
        self.ax.set_xlim(0., len(data))
        self.ax.set_ylim(0., 1.,)
        self.fig.canvas.draw()
        self.timer = self.startTimer(1000)

    def timerEvent(self, evt):
        # update the height of the bars, one liner is easier
        data = self.dataGen.next()
        if (len(data)==0):
            self.killTimer(self.timer)
        else:
            [bar.set_height(x) for bar,x in zip(self.bars,data)]
            self.fig.canvas.draw()

    def dataGenerator(self):
        data = [np.array(map(float, line.strip().split())) for line in open('data.txt').readlines()]
        for l in data:
            yield l

def runMonitor():
    app = QtGui.QApplication(sys.argv)
    w = Monitor()
    w.setWindowTitle("Convergence")
    w.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    runMonitor()