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
from sklearn.utils import shuffle
#from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_yahoo
import os
import sys
import pickle


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

def get_data(symbols, dates, base_dir):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    #read first from SPY to determine the size of the np array
    df_temp = pd.read_csv(symbol_to_path('SPY'), index_col = 'Date', parse_dates = True, 
        usecols = ['Date', 'Open', 'Adj Close'], na_values = ['nan'])
    spy_len = df_temp.shape[0]
    
    np_delta = np.zeros(shape= (spy_len, len(symbols)) ) 
    k_iter = enumerate(symbols)
    num_cols = 0
    for sym in k_iter:      
        filename = sym[1] + '.csv'
        if filename in os.listdir(base_dir):
            if os.path.isfile(os.path.join(base_dir, filename)):
#                print sym[1]
                df_temp = pd.read_csv(symbol_to_path(sym[1]), index_col='Date',
                    parse_dates=True, usecols=['Date', 'Open','Adj Close'], na_values=['nan'])
                # if number of trading days differs from SPY then pass                     
                if df_temp.shape[0] == spy_len :                     
                    np_delta[:, num_cols ] = np.array(df_temp['Open'] - df_temp['Adj Close']).astype(np.float)
                    num_cols += 1    
                    
                else: 
                    pass                    
                
#                if df_temp.isnull().values.any():
#                    pass
#                else:
#                    df = df.join(df_temp)
#                    if symbol == 'SPY':  # drop dates SPY did not trade
#                        df = df.dropna(subset=['SPY'])
    return np_delta[:, 0:num_cols - 1]

if __name__ == "__main__":
    base_dir  = cwd + '/data/'
    start_date = datetime.datetime(2014, 4, 5)
    end_date = datetime.datetime(2015, 6, 2)   
    dates = pd.date_range(start_date, end_date)

#==============================================================================
############## LOAD FROM symbol_map.json ######################################

##     Input symbol file
#    symbol_file = cwd+'/data/' + 'symbol_map.json'    
#
#
##     Load the symbol map
#    with open(symbol_file, 'r') as f:
#        symbol_dict = json.loads(f.read())
#
#
#    syms, names = np.array(list(symbol_dict.items())).T        
#    symbols = list()           
#    for s in syms:
#        filename = s + '.csv'
#        if os.path.isfile(os.path.join(base_dir, filename)):
#            symbols.append(s)           
#    quotes = [get_data_all(symbol, dates) for symbol in symbols]    
#    
##     Extract opening and closing quotes
#    opening_quotes = np.array([quote['Open'] for quote in quotes]).astype(np.float)
#    closing_quotes = np.array([quote['Close'] for quote in quotes]).astype(np.float)
#    print type(opening_quotes)
#    print (opening_quotes.shape)    
#    
#    # The daily fluctuations of the quotes 
#    delta_quotes = closing_quotes - opening_quotes    
#==============================================================================    
#   data contains a list of the S&P500 symbols
    with open('sector_symbol_list.pkl', 'rb') as f:
        data = pickle.load(f)
    
    symbols = list()
    for k, v in data.iteritems():        
        for s in v:
            symbols.append(s)     


    quotes = list()
    delta_quotes = get_data(symbols, dates, base_dir)
    print delta_quotes.shape
###    for item in os.listdir(base_dir):
###        if os.path.isfile(os.path.join(base_dir, item)):
###           if item.endswith('.csv'):
###                quotes.append(get_data_all(item[:-4], dates))                
       

                
    num_symbols = 200
    delta_quotes = delta_quotes[:, :num_symbols]
    symbols = symbols[:num_symbols]
    delta_quotes = delta_quotes.T
    
    names = np.array(symbols)
#==============================================================================      

    
    # Build a graph model from the correlations
    #GraphLassoCV produces a sparse inverse convariance matrix (fit model takes 
    # an nd-array w/ shape (n_samples, n_feautes))
    edge_model = covariance.GraphLassoCV()
    
    # Standardize the data 
    X = delta_quotes.copy().T
    X /= X.std(axis=0)
    
    # Train the model
    with np.errstate(invalid='ignore'):
        edge_model.fit(X)
    
    print 'here'
    # Build clustering model using affinity propagation
    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    num_labels = labels.max()
    
    # Print the results of clustering
    for i in range(num_labels + 1):
        print "Cluster", i+1, "-->", ', '.join(names[labels == i])

