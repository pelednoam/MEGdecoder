'''
Created on Dec 12, 2013

@author: noampeled
'''

import numpy as np

def victor_purpura_dist(tli,tlj,cost=1):
    """
    d=spkd(tli,tlj,cost) calculates the "spike time" distance
    as defined [DA2003]_ for a single free parameter, 
    the cost per unit of time to move a spike.
    
    :param tli: vector of spike times for first spike train
    :param tlj: vector of spike times for second spike train
    :keyword cost: cost per unit time to move a spike
    :returns: spike distance metric 
    
    Translated to Python by Nicolas Jimenez from Matlab code by Daniel Reich.
    
    .. [DA2003] Aronov, Dmitriy. "Fast algorithm for the metric-space analysis 
                of simultaneous responses of multiple single neurons." Journal 
                of Neuroscience Methods 124.2 (2003): 175-179.
    
    Here, the distance is 1 because there is one extra spike to be deleted at 
    the end of the the first spike train:
    
    >>> spike_time([1,2,3,4],[1,2,3],cost=1)
    1 
    
    Here the distance is 1 because we shift the first spike by 0.2, 
    leave the second alone, and shift the third one by 0.2, 
    adding up to 0.4:
    
    >>> spike_time([1.2,2,3.2],[1,2,3],cost=1)
    0.4

    Here the third spike is adjusted by 0.5, but since the cost 
    per unit time is 0.5, the distances comes out to 0.25:  
    
    >>> spike_time([1,2,3,4],[1,2,3,3.5],cost=0.5)
    0.25
    """
    
    nspi=len(tli)
    nspj=len(tlj)

    if cost==0:
        d=abs(nspi-nspj)
        return d
    elif cost==np.Inf:
        d=nspi+nspj;
        return d

    scr = np.zeros( (nspi+1,nspj+1) )

    # INITIALIZE MARGINS WITH COST OF ADDING A SPIKE

    scr[:,0] = np.arange(0,nspi+1)
    scr[0,:] = np.arange(0,nspj+1)
           
    if nspi and nspj:
        for i in range(1,nspi+1):
            for j in range(1,nspj+1):
                scr[i,j] = min([scr[i-1,j]+1, scr[i,j-1]+1, scr[i-1,j-1]+cost*abs(tli[i-1]-tlj[j-1])])
        
    d=scr[nspi,nspj]
    return d
