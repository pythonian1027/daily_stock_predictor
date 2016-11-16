# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 10:38:09 2016

@author: rcortez
"""

"""Utility functions"""

import os
import pandas as pd
import matplotlib.pyplot as plt

cwd = os.getcwd()
path = os.path.abspath(os.path.join(cwd, os.pardir))

def symbol_to_path(symbol, base_dir=path + '/data/'):
    """Return CSV file path given ticker symbol."""    
    print os.path.join(base_dir, "{}.csv".format(str(symbol)))
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        # TODO: Read and join data for each symbol
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date', 
            parse_dates = True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns = {'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY': #drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df
    
def plot_selected(df, columns, start_index, end_index):
    """Plot the desired columns over index values in the given range."""
    plot_data(df.ix[start_index:end_index, columns], title = "Selected data")
    
def plot_data(df, title = 'Stock prices', ylabel = 'Date', xlabel= 'Price'):
    ax = df.plot(title = title, fontsize = 10) # ax is a plot handler
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.grid()
    plt.show()

#def plot_data(df):
#    df.plot()
#    plt.grid()
#    plt.show()
#    
    
def normalize_data(df):
    """Normalize stock prices using the first row of the dataframe"""
    return df/df.ix[0,:]

def print_stats(df):
    
    mn = df.mean()
    std = df.std()
    
    print '\nMean:\n {}\n\nStd:\n{}'.format(mn, std)

def test_run():
    # Define a date range
    dates = pd.date_range('2010-01-22', '2011-01-26')

    # Choose stock symbols to read
    symbols = ['GOOG', 'IBM']
    
    # Get stock data
    df = get_data(symbols, dates)

#   Statistics
    print_stats(df)
    
    #slicing
#    print df.ix['2010-03-10':'2010-03-15', ['SPY', 'IBM']]
    
    # Slice and plot
#    plot_selected(df, ['SPY', 'IBM'], '2010-03-01', '2010-04-01')
    plot_data(normalize_data(df))
#    print df


if __name__ == "__main__":
    test_run()
