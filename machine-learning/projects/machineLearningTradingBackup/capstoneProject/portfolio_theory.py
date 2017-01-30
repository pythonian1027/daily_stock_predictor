# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 18:22:18 2017

@author: rcortez
"""

import numpy as np
import matplotlib.pyplot as plt



def get_weights(data_frame):
    rets = np.log(data_frame / data_frame.shift(1))   
    rets.mean() * 252       
    rets.cov() * 252    
    
#==============================================================================
#     The Basic Theory
#==============================================================================
    num_symbols = data_frame.shape[1]
    weights = np.random.random(num_symbols)
    weights /= np.sum(weights)    
    prets = list()
    pvols = list()    
    #list containing a tuple of sharpe, weights, returns and std
    srwrs = list()
    
    for p in range (1000):
        weights = np.random.random(num_symbols)
        weights /= np.sum(weights)
        exp_return = np.sum(rets.mean() * weights) * 252
        exp_volat = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
        prets.append(exp_return)
        pvols.append(exp_volat)
        if exp_return > 0.08 and exp_volat < 0.40: #arbitrary values 
            srwrs.append((exp_return/exp_volat, weights, exp_return, exp_volat))
#            print W
                
    prets = np.array(prets)       
    pvols = np.array(pvols)    

    plt.figure(figsize=(8, 4))
    plt.scatter(pvols, prets, c=prets / pvols, marker='o')
    try:        
        print 'Sharpe Ratio: {}\nExp. Return: {}\nExp. Risk: {}'.format(max(srwrs)[0], max(srwrs)[2], max(srwrs)[3])        
        plt.plot(max(srwrs)[3], max(srwrs)[2], 'r*', markersize=15.0)
        plt.annotate('Sharpe Ratio={}'.format(round(max(srwrs)[0],2)), (max(srwrs)[3]*(-1.1), max(srwrs)[2]*1.3))
    except Exception:
        pass                
    plt.grid(True)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
#    prets_ticks = np.arange(0.12, 0.21, 0.01)
#    pvols_ticks = np.arange(0.12, 0.16, 0.005)
#    bnds = np.array([prets_ticks[0], prets_ticks[-1]])/(np.array([pvols_ticks[0], pvols_ticks[-1]]))
#    plt.xticks(pvols_ticks)
#    plt.yticks(prets_ticks)
#    plt.colorbar(label='Sharpe ratio')    
#    plt.colorbar(label='Sharpe ratio', boundaries = bnds)    
    print plt.gca()
    return max(srwrs) # returns tuple (sharpe ratio, weights, return, std)
