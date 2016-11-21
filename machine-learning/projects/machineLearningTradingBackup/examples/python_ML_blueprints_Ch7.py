# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 23:15:49 2016

@author: rcortez
"""

import pandas as pd
import numpy as np
#from pandas_datareader import data, wb
import matplotlib.pyplot as plt
import os
import pandas as pd
#import pandas_datareader as pdr



cwd = os.getcwd()
print cwd
os.chdir('../')
cwd = os.getcwd()
#path = '/home/rcortez/projects/python/projects/umlNanoDegee/machine-learning/projects/machineLearningTradingBackup'
#path = os.path.abspath(os.path.join(cwd, os.pardir))
#print path
def symbol_to_path(symbol, base_dir=cwd + '/data/'):
    """Return CSV file path given ticker symbol."""    
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
    return df


def test_run():
    
    
    dates = pd.date_range('2010-01-01','2016-03-01')
    symbols = ['SPY']
    
    
    spy = get_data(symbols, dates)
    
#    spy_c = spy['Close']
#    pd.set_option('display.max_colwidth', 200)
    
    fig, ax = plt.subplots(figsize=(15,10))
    spy.plot(color='k')
    plt.title("SPY", fontsize=20)
    
    long_on_rtn = ((spy['Open'] - sp['Close'].shift(1))/sp['Close'].shift(1))*100
#    spy['Daily Change'] = pd.Series(spy['Close'] - spy['Open'])
#    spy['Daily Change'].sum()
#    np.std(spy['Daily Change'])
#    spy['Overnight Change'] = pd.Series(spy['Open'] - spy['Close'].shift(1))
#    np.std(spy['Overnight Change'])
#    
#    # daily returns
#    daily_rtn = ((spy['Close'] - spy['Close'].shift(1))/spy['Close'].shift(1))*100
#    daily_rtn

if __name__ == "__main__":
    test_run()