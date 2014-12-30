'''
Created on May 18, 2014

@author: noampeled
'''

from src.commons.utils import plots
import matplotlib.pyplot as plt

import numpy as np

def girAndEl():
    gHeight = np.random.randn(50, 1)*40 + 250
    gWeight = np.random.randn(50, 1)*30 + 100 

    gWeight = np.random.rand(50,1)*100 + 200
    gHeight = 0.8*gWeight + np.random.randn(50, 1)*20

    eHeight = np.random.randn(50, 1)*30 + 150
    eWeight = np.random.randn(50, 1)*20 + 250 

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Weight')
    ax.set_ylabel('Height')
    ax.grid(True, linestyle='-', color='0.75')
    ax.scatter(gWeight, gHeight, 30, color='tomato', label='Giraffes');
#     ax.scatter(eWeight, eHeight, 30, color='tomato', label='Elephants');
#     plt.title('Elephants and Giraffes')
    plt.title('Giraffes')
    plt.xlim([200,300])
#     plt.xlim([min(min(gWeight),min(eWeight))-10,max(max(gWeight),max(eWeight))+10])
#     plt.ylim([min(min(eHeight),min(eHeight))-10,max(max(gHeight),max(eHeight))+10])
#     plt.legend()
    
    plt.show()


if __name__ == '__main__':
    girAndEl()