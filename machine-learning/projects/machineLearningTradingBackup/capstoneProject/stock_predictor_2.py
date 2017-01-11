# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 19:12:20 2016

@author: rcortez
"""


#back testing
from sklearn.metrics import r2_score, make_scorer, fbeta_score
from sklearn.grid_search import GridSearchCV
from sklearn.tree import  DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from portfolio_gen import get_data

from sklearn.svm import SVR
import numpy as np
import math

import os
import pandas as pd
import matplotlib.pyplot as plt
import sys, traceback

#project_path ='/home/rcortez/projects/python/projects/umlNanoDegee/machine-learning/projects/machineLearningTradingBackup' 
cwd = os.getcwd()
os.chdir('../')
project_path = os.getcwd()

def symbol_to_path(symbol, base_dir=project_path + '/data/'):
    """Return CSV file path given ticker symbol."""    
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

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

#def get_data(symbols, dates):
#    """Read stock data (adjusted close) for given symbols from CSV files."""
#    df = pd.DataFrame(index=dates)
#    if 'SPY' not in symbols:  # add SPY for reference, if absent
#        symbols.insert(0, 'SPY')
#
#    for symbol in symbols:
#        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
#                parse_dates=True, usecols=['Date', 'High', 'Adj Close'], na_values=['nan'])
#        df_temp = df_temp.rename(columns={'Adj Close': symbol + '_adcls'})
#        df_temp = df_temp.rename(columns={'High': symbol + '_hi'})        
#        df = df.join(df_temp)
#        if symbol == 'SPY':  # drop dates SPY did not trade
#            df = df.dropna(subset=["SPY_adcls"])
#    return df

def plot_feature_importances(feature_importances, title, feature_names):
    # Normalize the importance values 
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    # Sort the values and flip them
    
    index_sorted = np.flipud(np.argsort(feature_importances))

    # Arrange the X ticks
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # Plot the bar graph
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()

def rsquare_performance(real_data, pred_data):        
    return r2_score(real_data, pred_data)
    

def get_partition(n_elem_train, n_elem_test):
    idxs_train = np.arange(n_elem_train)        
    idxs_test = np.arange(idxs_train[-1] + 1, n_elem_train + n_elem_test, 1)        
    while True:
        yield idxs_train, idxs_test
        idxs_train += n_elem_test
        idxs_test += n_elem_test                        
    
def performance_metric_r2(y_true, y_predict):    
    score = r2_score(y_true, y_predict)
    return score
    
def performance_metric_mse(y_true, y_predict):
    score = mean_squared_error(y_true, y_predict)
    return score    
    
def fit_model(X,y, model, cv_sets):
    if model == 'DTR':
        regressor = DecisionTreeRegressor(random_state = None)
        params = {'max_depth':[2,3, 4, 5, 7, 8, 20], 'min_samples_split' : [2, 8, 16, 32]}
    elif model == 'SVR':
        regressor = SVR() 
        params = { 'C':[1e-3],'gamma': [0.0001], 'kernel': ['linear']}                 
    elif model == 'AdaBoost':
        regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 4), n_estimators = 50, random_state = None) 
        params = {}
        
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
def get_train_test_sets(dframe, train_size):
    train_data_sz = int(dframe.shape[0]*train_size)
    train_set = dframe.ix[:train_data_sz]
    test_set = dframe.ix[train_data_sz:]
    return train_set, test_set    

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

def print_results(mod, predictions, target):
    print '{} : MSE = {}'.format(mod, mean_squared_error(predictions, target))
    result = 100*np.mean((target.values - predictions)/target.values)
    print 'Avg. Percentage Error {}'.format(result)
    evs = explained_variance_score(target, predictions)
    print 'Explained Variance Score = {}'.format(evs)
                     
def plot_results(predictions, target):
    t = np.arange(0, target.shape[0])
    plt.plot(t, predictions, 'r', t, target, 'b')
    plt.show()    
                        
def back_test(model, predictions, target):            
    print_results(model, predictions, target)
    plot_results(predictions, target)                        
                        
if __name__ == "__main__":
    days_back = 14
    dates = pd.date_range('2007-01-01', '2014-01-01')  

#    feats = ['_adcls']
#    symbols = ['SPY', 'XOM', 'WYNN']    
    symbols = ['SPY']    
    test_sz = 0.2
    train_sz = (1 - test_sz)
    n_folds = 5
    n_lookup = 7

    
    df1 = get_data(symbols, dates) 
    df = df1.ix[days_back - 1:,: ]
    
    print 'df.shape[0]: {}, dates: {}'.format(df.shape[0], dates)    
    #   length of training data set to 90%
    #   remaining 10% is for final testing    
    df_train, df_test = get_train_test_sets(df, 0.9)    

    
    
   
    for s in symbols:       
        sp = df1.ix[:,[s]]    
        for i in range(n_lookup, days_back + n_lookup, 1):
            sp.loc[:,'Close Minus ' + str(i)] = sp[s].shift(i)
        
        sp20 = sp[[x for x in sp.columns if 'Close Minus' in x or x == s]].iloc[days_back + n_lookup - 1:,]
        sp20 = sp20.iloc[:,::-1]
        
        sp20_train, sp20_test = get_train_test_sets(sp20, 0.9)
        X_train = sp20_train.ix [ : , : -1] # use previous closing prices as features
        y_train = sp20_train.ix[ :,  -1] # use last column, adjcls, as a target
        y_true = sp20_test.ix[:, -1] 
        
#        models = [ 'DTR', 'AdaBoost', 'SVR']
        models = [ 'SVR']                
        for m in models:
            print type(X_train), type(y_train)
            num_elem_train, num_elem_test, cv_sets = fit_model_inputs(sp20_train.shape[0], 
                                                                      n_lookup, 
                                                                      n_folds, test_sz, train_sz)
        
            reg = fit_model(X_train, y_train, m, cv_sets) #takes in training data
             # Produce the value for 'max_depth'
            print "Params for model {} are {}".format(m, reg.get_params())           
            y_predict = reg.predict(sp20_test.ix[:, : -1]) 
            back_test(m, y_predict, y_true)            
