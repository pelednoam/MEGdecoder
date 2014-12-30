from PyQt4 import QtGui, QtCore


class MyWidget(QtGui.QWidget):
    mysignal = QtCore.pyqtSignal(list)
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self,parent)
        self.button = QtGui.QPushButton("OK", self)
        self.text=QtGui.QLineEdit()
        self.spin=QtGui.QSpinBox()
        self.grid=QtGui.QGridLayout()
        self.grid.addWidget(self.button,0,1)
        self.grid.addWidget(self.spin,0,0)
        self.grid.addWidget(self.text,0,2)
        self.setLayout(self.grid)
        self.button.clicked.connect(self.OnClicked)
        self.mysignal.connect(self.OnPrint)
        
    def OnClicked(self):
        val=self.spin.value()
        #self.emit(QtCore.SIGNAL('mysignal'),range(val))
        self.mysignal.emit(range(val))

    def OnPrint(self,val):
        s= ' '
        for el in val:
            s+=str(el)+' '
        self.text.setText(s)
if __name__ == '__main__':
 import sys
 app = QtGui.QApplication(sys.argv)
 w = MyWidget()
 w.show()
 app.exec_()