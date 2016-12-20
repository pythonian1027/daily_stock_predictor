# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:28:39 2016
based on stock_market.py from Chapter 04 Machine Learning Cookbook
@author: rcortez
"""

import json
import datetime
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance, cluster
#from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_yahoo
import os
import sys


cwd = os.getcwd()
print cwd
os.chdir('../')
cwd = os.getcwd()

#sys.path.append("/home/rcortez/Projects/umlNanoDegee/machine-learning/projects/machineLearningTradingBackup/examples/Python-Machine-Learning-Cookbook/Chapter04/")

#
## Choose a time period
#start_date = datetime.datetime(2004, 4, 5)
#end_date = datetime.datetime(2007, 6, 2)
#
## Load the symbol map
#with open(symbol_file, 'r') as f:
#    symbol_dict = json.loads(f.read())
#
#symbols, names = np.array(list(symbol_dict.items())).T
#
#quotes = [quotes_yahoo(symbol, start_date, end_date, asobject=True) 
#                for symbol in symbols]
def symbol_to_path(symbol, base_dir=cwd + '/data/'):
    """Return CSV file path given ticker symbol."""    
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data_all(symbol, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    df = pd.read_csv(symbol_to_path(symbol), index_col='Date',
            parse_dates=True, na_values=['nan'])   
#    if symbol == 'SPY':  # drop dates SPY did not trade
#    df = df.dropna(how='any')            
    return df

def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'High', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp['Open'] - df_temp['Adj Close']
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=['SPY'])
    return df

if __name__ == "__main__":
#    run()
    import pickle
    #data contains a list of the S&P500 symbols
    with open('sector_symbol_list.pkl', 'rb') as f:
        data = pickle.load(f)
    
    symbols = list()
    for k, v in data.iteritems():        
        for s in v:
            symbols.append(s)
    
    
        

    start_date = datetime.datetime(2004, 4, 5)
    end_date = datetime.datetime(2007, 6, 2)   
    dates = pd.date_range(start_date, end_date)
    base_dir  = cwd + '/data/'
    quotes = list()
#    for item in os.listdir(base_dir):
#        if os.path.isfile(os.path.join(base_dir, item)):
#            if item.endswith('.csv'):
##                print item
#                quotes.append(get_data_all(item[:-4], dates))                
        
    # Input symbol file
#    symbol_file = cwd+'/data/' + 'symbol_map.json'    


    # Load the symbol map
#    with open(symbol_file, 'r') as f:
#        symbol_dict = json.loads(f.read())
#
#    symbols, names = np.array(list(symbol_dict.items())).T        
                
#    for symbol in symbols:
#        try:                
#            quotes.append(get_data_all(symbol, dates))
#        except:
#            print "symbol: {} does not exists".format(symbol)                    
            
#    quotes = [get_data_all(symbol, dates) for symbol in symbols]    
#  len  print quotes
#        
                
    
    
    
    # Extract opening and closing quotes
    opening_quotes = np.array([quote['Open'] for quote in quotes]).astype(np.float)
    closing_quotes = np.array([quote['Close'] for quote in quotes]).astype(np.float)
    
    # The daily fluctuations of the quotes 
    delta_quotes = closing_quotes - opening_quotes
    
    # Build a graph model from the correlations
    edge_model = covariance.GraphLassoCV()
    
    # Standardize the data 
    X = delta_quotes.copy().T
    X /= X.std(axis=0)
    
    # Train the model
    with np.errstate(invalid='ignore'):
        edge_model.fit(X)
    
    # Build clustering model using affinity propagation
    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    num_labels = labels.max()
    
    # Print the results of clustering
    for i in range(num_labels + 1):
        print "Cluster", i+1, "-->", ', '.join(names[labels == i])

