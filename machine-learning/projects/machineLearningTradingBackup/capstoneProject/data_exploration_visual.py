# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:56:48 2017

@author: rcortez
"""
"""scatterplots"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#plt.switch_backend('cairo')
import sys

sys.path.insert(0, '/home/rcortez/Projects/umlNanoDegee/machine-learning/projects/machineLearningTradingBackup/examples/')

from utility_func import get_data, plot_data
from daily_returns import compute_daily_returns

    
def test_run():
#    plt.close("all")
    dates = pd.date_range('2011-01-01', '2017-01-01')
    symbols = ['AMZN', 'NFLX', 'AAPL', 'GOOG']
    df = get_data(symbols, dates)
    plot_data(df)
    
#    #Compute daily returns
    daily_returns = compute_daily_returns(df)
#    
#    
#   Scatter SPY vs AMZN
    for s in symbols:
        daily_returns.plot(kind='scatter', x='SPY', y=s)
        beta_, alpha_ = np.polyfit(daily_returns['SPY'], daily_returns[s], 1)
        print "beta_{} = {}".format(s, beta_.round(3))
#        print "alpha_{} = {}".format(s, alpha_.round(3))
        
        plt.plot(daily_returns['SPY'], beta_*daily_returns['SPY'] + alpha_, '-', color = 'r')
        plt.annotate('Beta={}'.format(beta_.round(3)), (-0.07, 0.15))        
        plt.grid()
        
    
##   Scatter SPY vs NFLX
#    daily_returns.plot(kind='scatter', x='SPY', y='NFLX')
#    beta_XOM, alpha_XOM = np.polyfit(daily_returns['SPY'], daily_returns['NFLX'], 1)
#    print "beta_NFLX = ", beta_XOM
#    print "alpha_NFLX = ", alpha_XOM
#    plt.plot(daily_returns['SPY'], beta_XOM*daily_returns['SPY'] + alpha_XOM, '-', color = 'r')
#    plt.grid()    
    
##    Scatter SPY vs GLD
#    daily_returns.plot(kind='scatter', x='SPY', y='GLD')
#    beta_GLD, alpha_GLD = np.polyfit(daily_returns['SPY'], daily_returns['GLD'], 1)
#    print "beta_GLD = ", beta_GLD
#    print "alpha_GLD = ", alpha_GLD
#    plt.plot(daily_returns['SPY'], beta_GLD*daily_returns['SPY'] + alpha_GLD, '-', color = 'r')    
#    plt.grid()    
    
    #Calculate correlation coefficient
    print daily_returns.corr(method = 'pearson')
    
if __name__ == "__main__":
    test_run()