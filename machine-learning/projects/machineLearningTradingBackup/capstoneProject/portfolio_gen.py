# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 21:51:04 2017

@author: rcortez
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:28:39 2016
based on stock_market.py from Chapter 04 Machine Learning Cookbook
@author: rcortez
"""

import pandas_datareader.data as web
import datetime
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance, cluster
from sklearn.utils import shuffle
from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_yahoo
from sklearn.metrics import mean_squared_error, explained_variance_score

import os
import sys
import pickle


proj_path ='/home/rcortez/projects/python/projects/umlNanoDegee/machine-learning/projects/machineLearningTradingBackup' 
ptf_fnames = dict()
ptf_fnames['buffett'] = 'buffett_port_syms.pickle'        

def load_symbols(filename):
    with open(filename, 'rb') as handle:
        symbols = pickle.load(handle)
    return symbols        
    
#symbols can be either a list of symbols or a portfolio name     
def download_hist_data(portfolio, start_date, end_date):

    if type(portfolio) == str: # true then it is a portfolio name        
        symbs = load_symbols(ptf_fnames[portfolio])
        symbols = list()
        
        if 'SPY' not in symbs:
            symbs.insert(0,'SPY')        
        for s in symbs:                               
            if os.path.isfile(proj_path + '/{}_data/{}.csv'.format(portfolio, s)) is False:
                try:                    
                    print 'Downloading data from Yahoo for symbos {}'.format(s)                     
                    dframe = web.DataReader(name=s, data_source='yahoo', start=start_date, end=end_date)                    
                    print 'writing to file %s symbol' %s
                    if os.path.isdir(proj_path + '/{}_data'.format(portfolio)) is False:
                        os.mkdir(proj_path + '/{}_data'.format(portfolio))
        
                    dframe.to_csv(proj_path + '/{}_data/{}.csv'.format(portfolio, s), index_label="Date")
                    symbols.append(s)
                except Exception:
                    print 'unable to download data for symbol: {}'.format(s)                  
            else:
                symbols.append(s)                            
        #Add SPY for test base
    
    print 'Finished downloading data\n'  
    
    return symbols   
    
def symbol_to_path(symbol, base_dir= proj_path + '/buffett_data/'):
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
    
#train size is the fraction of the dataset dedicated to training
def get_train_test_sets(dframe, train_size):
    train_data_sz = int(dframe.shape[0]*train_size)
    train_set = dframe.ix[:train_data_sz]
    test_set = dframe.ix[train_data_sz:]
    return train_set, test_set        

def print_results(mod, predictions, target):
    print '{} : MSE = {}'.format(mod, mean_squared_error(predictions, target))
    result = 100*np.mean((target.values - predictions)/target.values)
    print 'avg. percentage error {}'.format(result)
    evs = explained_variance_score(target, predictions)
    print 'Explained Variance Score = {}'.format(evs)
    
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
#def get_data(symbols, dates, base_dir):
#    """Read stock data (adjusted close) for given symbols from CSV files."""
#    df = pd.DataFrame(index=dates)
#    if 'SPY' not in symbols:  # add SPY for reference, if absent
#        symbols.insert(0, 'SPY')
#
#    #read first from SPY to determine the size of the np array
#    df_temp = pd.read_csv(symbol_to_path('SPY'), index_col = 'Date', parse_dates = True, 
#        usecols = ['Date', 'Open', 'Adj Close'], na_values = ['nan'])
#    spy_len = df_temp.shape[0]
#    
#    np_delta = np.zeros(shape= (spy_len, len(symbols)) ) 
#    k_iter = enumerate(symbols)
#    num_cols = 0
#    for sym in k_iter:      
#        filename = sym[1] + '.csv'
#        if filename in os.listdir(base_dir):
#            if os.path.isfile(os.path.join(base_dir, filename)):
##                print sym[1]
#                df_temp = pd.read_csv(symbol_to_path(sym[1]), index_col='Date',
#                    parse_dates=True, usecols=['Date', 'Open','Adj Close'], na_values=['nan'])
#                # if number of trading days differs from SPY then pass                     
#                if df_temp.shape[0] == spy_len :                     
#                    np_delta[:, num_cols ] = np.array(df_temp['Open'] - df_temp['Adj Close']).astype(np.float)
#                    num_cols += 1    
#                    
#                else: 
#                    pass                    
                
#                if df_temp.isnull().values.any():
#                    pass
#                else:
#                    df = df.join(df_temp)
#                    if symbol == 'SPY':  # drop dates SPY did not trade
#                        df = df.dropna(subset=['SPY'])
    return np_delta[:, 0:num_cols - 1]

if __name__ == "__main__":
    
#    symbols = load_symbols(ptf_fnames['buffett'])
    
    start_date = datetime.datetime(2014, 4, 5)
    end_date = datetime.datetime(2015, 6, 2)   
    dates = pd.date_range(start_date, end_date)  
    symbols  = download_hist_data('buffett', start_date, end_date )
    
    
    quotes = list()    
    miss_data = list()
    for s in symbols:
        data = get_data_all(s, dates)
        if s == 'SPY':
            len_data = data.shape[0]
        if  data.shape[0] < len_data:
            print 'missing dates for symbol {} ...omitted from clustering'.format(s)
            miss_data.append(s)                
        else:
            quotes.append(data)
    
#   remove symbols with missing data from symbols list    
    for k in miss_data:
        symbols.remove(k)
    
#     Extract opening and closing quotes
    opening_quotes = np.array([quote['Open'] for quote in quotes]).astype(np.float)
    closing_quotes = np.array([quote['Close'] for quote in quotes]).astype(np.float)
 
    
    # The daily fluctuations of the quotes 
    delta_quotes = closing_quotes - opening_quotes    

#    # Build a graph model from the correlations
    #GraphLassoCV produces a sparse inverse convariance matrix (fit model takes 
    # an nd-array w/ shape (n_samples, n_feautes))
    edge_model = covariance.GraphLassoCV()
    
#    # Standardize the data 
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
        print "\nCluster", i+1, "-->", ', '.join(np.array(symbols)[labels == i])


    #analyze risk by std
    var = [np.var(d) for d in delta_quotes]
#    std = [np.std(q['Adj Close']) for q in quotes]
#    d = {'std' : pd.Series(std), 'var': pd.Series(var)}
    d  = {'var':pd.Series(var)}
    df = pd.DataFrame(d)    
    
    df.index = symbols
#    df_sorted = df.sort_index(by='var')
    
    
    #model and prediction 
    lookup_days = 1
    tot_days_back = 7
    mse = list()
    for q in quotes:             
        for i in range(lookup_days, tot_days_back, 1):
            q.loc[:,'Adj Close Minus ' + str(i)] = q['Adj Close'].shift(i)
        
        sp20 = q[[x for x in q.columns if 'Adj Close Minus' in x or x == 'Adj Close']].iloc[tot_days_back - 1:,]    
        sp20 = sp20.iloc[:,::-1]    
        
        from sklearn.svm import SVR
        clf = SVR(kernel='linear')
    
        sp20_train, sp20_test = get_train_test_sets(sp20, 0.9)
        X_train = sp20_train.ix [ : , : -1] # use previous closing prices as features
        y_train = sp20_train.ix[ :,  -1] # use last column, adjcls, as a target
        reg = clf.fit(X_train, y_train)                  
        
        y_predict = reg.predict(sp20_test.ix[:, : -1]) #n_lookup is to shift data so that dates matches w/ previous algo
        y_true = sp20_test.ix[:, -1] #n_lookup is to shift data so that dates matches w/ previous algo        

        print_results('SVR', y_predict, y_true)
        mse.append(mean_squared_error(y_predict, y_true))
    
    
    df['mse_{}_days'.format(lookup_days)] = mse
    
    
    