

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 20:35:50 2016

@author: rcortez
"""

import sys
sys.path.insert(0, '/home/rcortez/Projects/umlNanoDegee/machine-learning/projects/machineLearningTradingBackup/examples/')

import pandas as pd
import matplotlib.pyplot as plt

from utility_func import get_data, plot_data

def compute_daily_returns(df):
    """Compute and return the daily return values"""
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    daily_returns.ix[0, :] = 0 # set row 0 to o
    return daily_returns

def test_run():
    #Read data 
#    plt.close("all")
    dates = pd.date_range('2011-01-01', '2017-01-01')
    symbols = ['AMZN', 'NFLX', 'AAPL', 'GOOG']
    df = get_data(symbols, dates)
    plot_data(df)
          
    #Plot histogram
    for s in symbols:        
       #Compute daily returns
        daily_returns = compute_daily_returns(df)
        plot_data(daily_returns[s], title = 'Daily returns {}'.format(s), ylabel= 'Daily returns')        
        
        daily_returns['SPY'].hist(bins = 100, label = 'SPY')
        daily_returns[s].hist(bins = 100, label = s)
        plt.legend(loc='upper right')        
        
        #Get stats
        mn = daily_returns['SPY'].mean()
        print "mean = {}".format(mn)
        std = daily_returns['SPY'].std()
        print "std = {}".format(std)
        
        plt.axvline(mn, color = 'g', linestyle = 'dashed', linewidth = 2)
        plt.axvline(std, color = 'r', linestyle = 'dashed', linewidth = 2)
        plt.axvline(-std, color = 'r', linestyle = 'dashed', linewidth = 2)
        plt.show()
        
        #Compute kurtosis
        print "Kurtosis {}: {}".format(s, daily_returns[s].kurtosis())
    
if __name__ == "__main__":
    test_run()