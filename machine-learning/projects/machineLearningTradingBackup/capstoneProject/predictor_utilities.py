# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:20:12 2017

@author: rcortez
"""

import pandas.io.data as web
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_yahoo


import os
import sys
import pickle

proj_path = os.getcwd()


def load_symbols(filename, path=proj_path):        
    try:
        handle =  open(path + '/' + filename, 'rb')
    except: 
        handle = open(filename)
                
    
    if filename.endswith('.pickle'):       
        symbols = pickle.load(handle)
        return symbols        
    elif filename.endswith('.txt'):
        s = handle.readline()
        s = s.strip()        
        s = s.split(',')   
        symbols = [k.strip() for k in s]    
        if symbols[-1].endswith('\n'):
            temp = symbols[-1][:-1]
            symbols.remove(symbols[-1])
            symbols.append(temp)
        return symbols
    else:
        print 'File not found'
        sys.exit()
    

def download_symbol(s, start_date, end_date):
    try:
        dframe = web.DataReader(name= s, data_source = 'yahoo', start = start_date, end = end_date)
        dframe.to_csv(proj_path + '/data/{}.csv'.format(s), index_label="Date")
    except Exception:
        print 'Unable to downloaddata for symbol: {}'
         
         
#symbols can be either a list of symbols or a portfolio name     
def download_hist_data(symbols, start_date, end_date):
    if os.path.isdir(proj_path) is False:
        print 'Project path {} does not exist'.format(proj_path)
        sys.exit(0)
    
    symbols_loaded = list()
    if 'SPY' not in symbols:
        symbols.insert(0,'SPY')           
    for s in symbols: 
        print 'searching file {}.csv'.format(s)                              
        if os.path.isfile(proj_path + '/data/{}.csv'.format(s)) is False:
#            print 'File Not Found: ' + proj_path + '/data/{}.csv'.format(s)
            try:                    
                print 'Downloading data for symbol {}'.format(s)                     
                dframe = web.DataReader(name=s, data_source='yahoo', start=start_date, end=end_date)                    
#                print 'writing to file %s symbol' %s
    
                dframe.to_csv(proj_path + '/data/{}.csv'.format(s), index_label="Date")
                symbols_loaded.append(s)
            except Exception:
                print 'unable to download data for symbol: {}'.format(s)                  
        else:
            symbols_loaded.append(s)                            
    print 'Finished downloading data\n'      
    return symbols_loaded   
    
def symbol_to_path(symbol, base_dir= proj_path + '/data/'):
    """Return CSV file path given ticker symbol."""    
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))
    
#train size is the fraction of the dataset dedicated to training
def get_train_test_sets(dframe, train_size):
    '''Train size is the fraction of the dataset dedicated to training,
    returns training and testing sets'''    
    train_data_sz = int(dframe.shape[0]*train_size)
    train_set = dframe.ix[:train_data_sz]
    test_set = dframe.ix[train_data_sz:]
    return train_set, test_set     

def get_train_test_idxs(_dates, _train_size):
    _train_data_sz = int(_dates.shape[0]*(_train_size))    
    _train_idxs = _dates[:_train_data_sz]
    _test_idxs = _dates[_train_data_sz:]
    return _train_idxs, _test_idxs        
    
def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        # TODO: Read and join data for each symbol
        if os.path.isfile(symbol_to_path(symbol)):            
            df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date', 
                parse_dates = True, usecols=['Date', 'Adj Close'], na_values=['nan'])
            df_temp = df_temp.rename(columns = {'Adj Close': symbol})
            df = df.join(df_temp)
            if symbol == 'SPY': #drop dates SPY did not trade
                df = df.dropna(subset=["SPY"])
#        else:
#            download_symbol(symbol)                        
    return df
    
def performance_metric_r2(y_true, y_predict):    
    score = r2_score(y_true, y_predict)
    return score

def get_partition(n_elem_train, n_elem_test):
    '''Generates indexes for CV sets (similar to KFold sets) used in GridSearchCV'''       
    idxs_train = np.arange(n_elem_train)        
    idxs_test = np.arange(idxs_train[-1] + 1, n_elem_train + n_elem_test, 1)        
    while True:
        yield idxs_train, idxs_test
        idxs_train += n_elem_test
        idxs_test += n_elem_test          

def generate_cv_sets(numel_train, numel_test, numfolds):
    '''Generates CV sets'''
    cvset = list()
    s = get_partition(numel_train,numel_test)
    for _ in range(numfolds):                
        idxs = next(s)
        u = idxs[0]
        v = idxs[1]
        cvset.append((u.copy(), v.copy()))    
    return cvset    