__author__ = 'noam'

from src.commons.utils import utils
from src.commons.utils import plots
import numpy as np

# pss, freqs = utils.load('/home/noam/Documents/facialExpressions/pssFreqs.pkl')
# utils.save((pss[0], freqs),'/home/noam/Documents/facialExpressions/pssFreqs0.pkl')
pss0, freqs = utils.load('/home/noam/Documents/facialExpressions/pssFreqs0.pkl')
lengths = [len(f) for f in freqs]
maxLenInd = np.argmax(lengths)
maxFreqs = freqs[maxLenInd]

C = 136
for k in range(len(pss0)):
    ret = np.interp(maxFreqs, freqs[k], pss0[k])
    plots.graph2(maxFreqs, ret, pss0[k], ['interp', 'org'], x2=freqs[k])

