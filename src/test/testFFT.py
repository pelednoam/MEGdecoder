'''
Created on May 22, 2014

@author: noam
'''
from src.commons import utils

import scipy
import numpy as np
import scipy.fftpack
import pylab
from scipy import pi
t = scipy.linspace(0,120,4000)
acc = lambda t: 10*scipy.sin(2*pi*2.0*t) + 5*scipy.sin(2*pi*8.0*t) + 2*scipy.random.random(len(t))
signal = acc(t)

signals=[]
for i in range(100):
    signals = utils.arrAppend(signals, acc(t))
    
FFT = abs(scipy.fft(signal))
ps = np.abs(np.fft.fft(signal))**2

FFTs = abs(scipy.fft(signals))
pss = np.abs(np.fft.fft(signals))**2
freqs = scipy.fftpack.fftfreq(signals[0[].size, t[1]-t[0])

# freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
idx = np.argsort(freqs)

pylab.subplot(211)
pylab.plot(freqs[idx], ps[idx])
pylab.subplot(212)
for ps in pss:
    pylab.plot(freqs[idx], ps[idx])
pylab.show()

# pylab.subplot(211)
# pylab.plot(freqs[idx],20*scipy.log10(FFT),'x')
# pylab.subplot(212)
# pylab.plot(freqs[idx],20*scipy.log10(FFTs[0,:]),'x')
# pylab.show()
