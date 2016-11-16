# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 21:19:16 2016

@author: rcortez
"""

"""scatterplots"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#plt.switch_backend('cairo')


from utility_func import get_data, plot_data
from daily_returns import compute_daily_returns

    
def test_run():
#    plt.close("all")
    dates = pd.date_range('2009-01-01', '2012-12-31')
    symbols = ['SPY', 'XOM', 'GLD']
    df = get_data(symbols, dates)
    plot_data(df)
    
#    #Compute daily returns
    daily_returns = compute_daily_returns(df)
#    
#    
#   Scatter SPY vs XOM
    daily_returns.plot(kind='scatter', x='SPY', y='XOM')
    beta_XOM, alpha_XOM = np.polyfit(daily_returns['SPY'], daily_returns['XOM'], 1)
    print "beta_XOM = ", beta_XOM
    print "alpha_XOM = ", alpha_XOM
    plt.plot(daily_returns['SPY'], beta_XOM*daily_returns['SPY'] + alpha_XOM, '-', color = 'r')
    plt.grid()
    
#    Scatter SPY vs GLD
    daily_returns.plot(kind='scatter', x='SPY', y='GLD')
    beta_GLD, alpha_GLD = np.polyfit(daily_returns['SPY'], daily_returns['GLD'], 1)
    print "beta_GLD = ", beta_GLD
    print "alpha_GLD = ", alpha_GLD
    plt.plot(daily_returns['SPY'], beta_GLD*daily_returns['SPY'] + alpha_GLD, '-', color = 'r')    
    plt.grid()    
    
    #Calculate correlation coefficient
    print daily_returns.corr(method = 'pearson')
    
if __name__ == "__main__":
    test_run()