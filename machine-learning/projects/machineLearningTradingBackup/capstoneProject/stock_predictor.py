# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 19:12:20 2016

@author: rcortez
"""


#back testing
from sklearn.metrics import r2_score, make_scorer, fbeta_score
from sklearn.grid_search import GridSearchCV
from sklearn.tree import  DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.svm import SVR
import numpy as np
import math

import os
import pandas as pd
import matplotlib.pyplot as plt
import sys, traceback


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

#def get_data_all(symbol, dates):
#    """Read stock data (adjusted close) for given symbols from CSV files."""
#    df = pd.DataFrame(index=dates)
#    df = pd.read_csv(symbol_to_path(symbol), index_col='Date',
#            parse_dates=True, na_values=['nan'])   
##    if symbol == 'SPY':  # drop dates SPY did not trade
##    df = df.dropna(how='any')            
#    return df

def get_adj_close_data(symbols, dates):
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

def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'High', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol + '_adcls'})
#        df_temp = df_temp.rename(columns={'Volume': symbol + '_vol'})        
        df_temp = df_temp.rename(columns={'High': symbol + '_hi'})        
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY_adcls"])
    return df

#def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
#    """Plot stock prices with a custom title and meaningful axis labels."""
#    ax = df.plot(title=title, fontsize=12)
#    ax.set_xlabel(xlabel)
#    ax.set_ylabel(ylabel)
#    plt.show()

def rsquare_performance(real_data, pred_data):        
    return r2_score(real_data, pred_data)
    

def get_partition(n_elem_train, n_elem_test):
    idxs_train = np.arange(n_elem_train)        
    idxs_test = np.arange(idxs_train[-1] + 1, n_elem_train + n_elem_test, 1)        
#    idxs_test = np.arange(n_elem_test)   
    while True:
        yield idxs_train, idxs_test
        idxs_train += n_elem_test
        idxs_test += n_elem_test                        
#    return idxs_train, idxs_test
    
def performance_metric_r2(y_true, y_predict):    
    score = r2_score(y_true, y_predict)
    return score
    
def performance_metric_mse(y_true, y_predict):
    score = mean_squared_error(y_true, y_predict)
    return score    
    
def fit_model(X,y, model, cv_sets):

#    cv_sets = generate_cv_sets(num_elem_train, num_elem_test, n_folds)    
#    print cv_sets
    if model == 'DTR':
        regressor = DecisionTreeRegressor(random_state = None)
        params = {'max_depth':[2,3, 4, 5, 7, 8, 20], 'min_samples_split' : [2, 8, 16, 32]}
#        scoring_fnc = make_scorer(performance_metric_r2)
#        scoring_fnc = make_scorer(performance_metric_mse)
        #    grid = GridSearchCV(regressor, params, scoring_fnc, cv = get_partition(100, 0.3))
        
    elif model == 'SVR':
        regressor = SVR() 
#        params = { 'C':[ 10, 3, 0.01, 100, 1e3, 1e4, 1e5],            
#               'gamma': [0.0001, 0.001, 0.01, 0.1]}  
        params = { 'C':[1e3, 5e3, 1e-3, 1e-6],'gamma': [0.0001, 0.1, 1, 1e3], 'kernel': ['rbf']}                 
#        scoring_fnc = make_scorer(performance_metric_mse)               
        
    scoring_fnc = make_scorer(performance_metric_mse, greater_is_better = False)                              
    grid = GridSearchCV(regressor, params, scoring_fnc, cv = cv_sets)    
    grid = grid.fit(X,y)        
    return grid.best_estimator_    
    

def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.grid()
    plt.show()

def data_nan_check( sbl, X, y, csets):
    nfolds = len(csets)
    num_miss_blks = 0
    num_tuple = len(csets[0])
    for n in range(nfolds):
        for m in range(num_tuple):
            if X.ix[csets[n][m]].isnull().values.any() :
                num_miss_blks += 1
            if y.ix[csets[n][m]].isnull().values.any() :
                num_miss_blks += 1            
    #if more than 30% of data is missing, through a message
    if num_miss_blks > int(0.3*nfolds):
        print 'Symbol {} is missing more than 30% of data'.format(sbl)
        try:
            pass
        except KeyboardInterrupt:
            print "shutdown requested ... exiting"
        except Exception:
            traceback.print_exc(file=sys.stdout)            
        sys.exit(0)

#train size is the fraction of the dataset dedicated to training
def get_training_sets(dframe, train_size):
    train_data_sz = int(dframe.shape[0]*train_size)
    train_set = dframe.ix[:train_data_sz]
    test_set = dframe.ix[train_data_sz:]
    return train_set, test_set
                    
#def calculate_performance(y_true, y_predict):
#    score = r2_score(y_true, y_predict)
#    return score                

def generate_cv_sets(numel_train, numel_test, numfolds):
    cvset = list()
    s = get_partition(numel_train,numel_test)
    for _ in range(numfolds):                
        idxs = next(s)
        u = idxs[0]
        v = idxs[1]
        cvset.append((u.copy(), v.copy()))    
    return cvset    

def fit_model_inputs(train_size, n_lookup, n_folds, test_sz, train_sz):
    numel_train = int(math.floor(train_size - n_lookup)/(1. + n_folds*(float(test_sz)/train_sz)))
    numel_test = int(math.floor( numel_train * (float(test_sz)/train_sz) ))
    cvsets = generate_cv_sets(numel_train, numel_test, n_folds)                        
    return numel_train, numel_test, cvsets                        
                        
if __name__ == "__main__":
#    run()

#'2010-07-01', '2016-09-21'  
#'2008-07-01', '2016-09-21'
    dates = pd.date_range('2010-01-01', '2015-09-21')  # one month only
#    feats = ['_hi', '_vol', '_adcls']
    feats = ['_hi', '_adcls']
#    symbols = ['SPY', 'XOM', 'WYNN']    
    symbols = []
    test_sz = 0.2
    train_sz = (1 - test_sz)
    n_folds = 5
    n_lookup = 3

    
    df = get_data(symbols, dates)   
    print 'df.shape[0]: {}, dates: {}'.format(df.shape[0], dates)    
    #   length of training data set to 90%
    #   remaining 10% is for final testing    
    df_train, df_test = get_training_sets(df, 0.9)

    
    #preprocess data/ append adj column for next day (shifteed by 1) as additional 
#    for symbol in symbols:
#        df_temp = df.ix[:,[symbol + feats[2]]]
#        df_temp = df_temp.rename(columns={symbol + feats[2]: symbol + 'adl_cls_target'})               
#        df = df.join(df_temp[1:])            
    
#   number of days ahead to be predicted

    
#==============================================================================
# #    num_elem_train and num_elem_test calculated within the train data set
#==============================================================================
    #test_sz is the training test size within the training set to be used in GridSearchCV

            
    #num_elem_train, num_elem_test are affected but the number of days for the 
    #prediction n_lookup, ie. a dataframe with a 100 days and 10 days lookup 
    #only have 90 days to be divided between training and testing sets            
#    num_elem_train = int(math.floor(df_train.shape[0] - n_lookup)/(1 + n_folds*(test_sz/train_sz)))
#    num_elem_test = int(math.floor( num_elem_train * (test_sz/train_sz) ))
#    cv_sets = generate_cv_sets(num_elem_train, num_elem_test, n_folds)
            
#==============================================================================
    num_elem_train, num_elem_test, cv_sets = fit_model_inputs(df_train.shape[0], n_lookup, n_folds, test_sz, train_sz)
#     
    print 'train ratio {}'.format(float(num_elem_train)/(num_elem_train + num_elem_test))    
    print 'df.shape[0]: {}, df_train: {},  n days lookup {}:'.format(df.shape[0], df_train.shape[0] ,n_lookup)
    print 'num_elem_train', num_elem_train
    print 'num_elem_test', num_elem_test    
     
    print 'df: {}, df_train: {},\ndf_test:  {},\ndf_test.shape[0]: {}\n\n'\
    .format(df.index[0], df_train.index[0], df_test.index[0], df_test.shape[0])     
    #calculate feature and target set with n_lookup days in advance
    for s in symbols:        
         #X_train and y_train are Dataframes of the same size, shifted by n_lookup
         #training features are trained with targets n_lookup days in the future         
        X_train = df_train.ix [ : -n_lookup, [s+feats[0], s+feats[1]]]
        y_train = df_train.ix[ n_lookup :, s+feats[1]]    
        #if data contains more than 30% of cv_sets with "nan's" then through a message
        data_nan_check(s, X_train, y_train, cv_sets )
 
 #      try:
        models = [ 'DTR']
        for m in models:
            print type(X_train), type(y_train)
            reg = fit_model(X_train, y_train, m, cv_sets) #takes in training data
             # Produce the value for 'max_depth'
            print "Params for SYMBOL {}, model {} are {}".format(s, m, reg.get_params())           
            y_predict = reg.predict(df_test.ix[: -n_lookup, [s+feats[0], s+feats[1]] ])
            y_true = df_test.ix[ n_lookup :, s+feats[1] ]
            print 'MSE 1{} : '.format(performance_metric_mse(y_predict, y_true))
#            t = np.arange(0, y_true.shape[0])
 #            plt.plot(t, y_predict, 'r', t, y_true, 'b')
             
            #take values to convert dataframe to np arrays to be consistent with y_predict
            result = 100*np.mean((y_true.values - y_predict)/y_true.values)
            print 'avg % within actual value : {}'.format(result)
            t = np.arange(0, y_true.shape[0])
            plt.plot(t, y_predict, 'r', t, y_true, 'b')
            plt.show()

    print '\n'            
#==============================================================================


#==============================================================================
# #Python ML Blueprints
# Need to test for Kernel Linear as in the tutorial
    symbol = 'SPY'
    sp = get_adj_close_data([symbol], dates)
    sp_len = sp.shape[0]
    nprior = 3
    print 'sp.shape[0]: {}, dates: {}'.format(sp.shape[0], dates)
#   number of prior closing days to predict next closing day: nprior
    for i in range(1, nprior + 1, 1):
        sp.loc[:,'Close Minus ' + str(i)] = sp[symbol].shift(i)
    sp20 = sp[[x for x in sp.columns if 'Close Minus' in x or x == symbol]].iloc[nprior:,]
    sp20 = sp20.iloc[:,::-1] 
    
#    this train test sets are for fit model (ing) only, the final test is done with 
    df_train, df_test = get_training_sets(sp20, 0.9)    
    num_elem_train, num_elem_test, cv_sets = fit_model_inputs(df_train.shape[0], n_lookup, n_folds, test_sz, train_sz)    
    
#    train_sz = int(0.8*sp_len)
#    test_sz = sp_len - train_sz
    test_sz = 0.2
    train_sz = (1 - test_sz)    

#    this train test sets are for fit model (ing) only, the final test is done with 
#    df_train, df_test = get_training_sets(sp20, 0.9)    
#    num_elem_train, num_elem_test, cv_sets = fit_model_inputs(df_train.shape[0], n_lookup, n_folds, test_sz, train_sz)    
   
    X_train = df_train.ix[:-n_lookup]
    y_train = df_train.ix[n_lookup:, symbol ]       
    reg  = fit_model(X_train, y_train, 'SVR', cv_sets)
    print "Params for modelare {}".format(reg.get_params())           
    y_predict = reg.predict(df_test.ix[: -n_lookup])
    y_true = df_test.ix[ n_lookup :, [symbol] ]
    print 'MSE 2 {} : '.format(performance_metric_mse(y_predict, y_true))
    result = 100*np.mean((y_true.values - y_predict)/y_true.values)
    print 'avg % within actual value : {}'.format(result)   
    t = np.arange(0, y_true.shape[0])
    plt.plot(t, y_predict, 'r', t, y_true, 'b')
    plt.show()
    
    print 'train ratio {}'.format(float(num_elem_train)/(num_elem_train + num_elem_test))    
    print 'sp20.shape[0]: {}, df_train: {},  n days lookup {}:'.format(sp20.shape[0], df_train.shape[0] ,n_lookup)
    print 'num_elem_train', num_elem_train
    print 'num_elem_test', num_elem_test        
    print 'sp: {}, df_train: {},\ndf_test:  {},\ndf_test.shape[0]: {}\n\n'\
    .format(sp.index[0], df_train.index[0], df_test.index[0], df_test.shape[0])

            
#==============================================================================
#        except:
#            print 'error'
            
           
#1 DONT FORGET TO NORMALIZE DATA
#2 boston housing uses fit_morel with inputs x_training, y_training but then cv is generated within with
#3 testing and training sets | also train_test_split generates x_train and x_test but only 
# x_train seems to be used
#boston housing never utilized the test set from the first test_train_split 
#4 Why results are not consisntent from run to run ????
#5 Learning curves is used in Boston housing with Decision Tree Regressor, can curves be used with others?
#6 Does data need to be balances? Read ipynb student_intervention DTrees weaknesses
#7 add kernel to gridsearch for SVR
#Hos is accuracy measure, why %within differs so much from r_2score, what does it means?           
#References to visit           
#www.svms.org/finance           




























            
        
    
#    X_train = np.random.rand(100,2)    
#    y_train = np.random.rand(100,1)
#    X_train = np.arange(100)
#    y_train = np.arange(100)
#    reg = fit_model(features, target, cv_sets)
    # Produce the value for 'max_depth'
#    print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])    
    
#    rs = ShuffleSplit(X.shape[0], n_iter = 3, random_state = 0, test_size = 0.5)
#    for train_idx, test_idx in rs:    
#        print 'train, tes: {} \t {} '.format(train_idx, test_idx)
        

    
    
#    print get_partition(100, 0.2)
#    grid = grid.fit(X,y)
    
    
    

#    
#    for _ in range(num_folds):
#        cv = next(s)
#        print df.shape
#        X_train =  df.ix[cv[0]]
#        X_test = df.ix[cv[1]]
#        print ([cv[0]], [cv[1]])
#    


#dates = pd.date_range('2010-01-01', '2015-09-21')  # one month only
#feats = ['_hi', '_adcls']
#symbols = ['SPY', 'XOM', 'WYNN']    
#test_sz = 0.2
#train_sz = (1 - test_sz)
#n_folds = 5
#n_lookup = 1
#
#<class 'pandas.core.frame.DataFrame'> <class 'pandas.core.series.Series'>
#Params for SYMBOL SPY, model DTR are {'presort': False, 'splitter': 'best', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 8, 'min_weight_fraction_leaf': 0.0, 'criterion': 'mse', 'random_state': None, 'max_features': None, 'max_depth': 20}
#score 0.805726509474 : 
#avg % within actual value : 0.271628894725
#<class 'pandas.core.frame.DataFrame'> <class 'pandas.core.series.Series'>
#Params for SYMBOL SPY, model SVR are {'kernel': 'linear', 'C': 1000.0, 'verbose': False, 'degree': 3, 'epsilon': 0.1, 'shrinking': True, 'max_iter': -1, 'tol': 0.001, 'cache_size': 200, 'coef0': 0.0, 'gamma': 0.0001}
#score -1.30860213337 : 
#avg % within actual value : -4.0299941938
#<class 'pandas.core.frame.DataFrame'> <class 'pandas.core.series.Series'>
#Params for SYMBOL XOM, model DTR are {'presort': False, 'splitter': 'best', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'mse', 'random_state': None, 'max_features': None, 'max_depth': 5}
#score 0.930081105088 : 
#avg % within actual value : -0.0314422417914
#<class 'pandas.core.frame.DataFrame'> <class 'pandas.core.series.Series'>
#Params for SYMBOL XOM, model SVR are {'kernel': 'linear', 'C': 1000.0, 'verbose': False, 'degree': 3, 'epsilon': 0.1, 'shrinking': True, 'max_iter': -1, 'tol': 0.001, 'cache_size': 200, 'coef0': 0.0, 'gamma': 0.0001}
#score 0.832282233694 : 
#avg % within actual value : -1.78060880421
#<class 'pandas.core.frame.DataFrame'> <class 'pandas.core.series.Series'>
#Params for SYMBOL WYNN, model DTR are {'presort': False, 'splitter': 'best', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'criterion': 'mse', 'random_state': None, 'max_features': None, 'max_depth': 8}
#score 0.947043354266 : 
#avg % within actual value : -0.438871343241
#<class 'pandas.core.frame.DataFrame'> <class 'pandas.core.series.Series'>
#Params for SYMBOL WYNN, model SVR are {'kernel': 'linear', 'C': 1000.0, 'verbose': False, 'degree': 3, 'epsilon': 0.1, 'shrinking': True, 'max_iter': -1, 'tol': 0.001, 'cache_size': 200, 'coef0': 0.0, 'gamma': 0.0001}
#score 0.918404314274 : 
#avg % within actual value : 3.98401839843
#
#
#<class 'pandas.core.frame.DataFrame'> <class 'pandas.core.series.Series'>
#Params for modelare {'kernel': 'linear', 'C': 1000.0, 'verbose': False, 'degree': 3, 'epsilon': 0.1, 'shrinking': True, 'max_iter': -1, 'tol': 0.001, 'cache_size': 200, 'coef0': 0.0, 'gamma': 0.0001}
#score 0.0787410382611 : 
#avg % within actual value : -0.229713402256