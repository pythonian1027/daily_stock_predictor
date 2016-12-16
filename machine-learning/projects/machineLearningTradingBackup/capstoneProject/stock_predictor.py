# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 19:12:20 2016

@author: rcortez
"""


#back testing
from sklearn.metrics import r2_score, make_scorer, fbeta_score
from sklearn.grid_search import GridSearchCV
from sklearn.tree import  DecisionTreeRegressor


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
    
def performance_metric(y_true, y_predict):
    from sklearn.metrics import r2_score
    score = r2_score(y_true, y_predict)
    return score
    
def fit_model(X,y, cv_sets, model):
    
    if model == 'DTR':
        regressor = DecisionTreeRegressor(random_state = None)
        params = {'max_depth':[2,3, 4, 7, 9, 11, 15,  20, 5, 57, 93]}
        scoring_fnc = make_scorer(performance_metric)
        #    grid = GridSearchCV(regressor, params, scoring_fnc, cv = get_partition(100, 0.3))
        
    elif model == 'SVR':
        regressor = SVR() 
#        params = { 'C':[ 10, 3, 0.01, 100, 1e3, 1e4, 1e5],            
#               'gamma': [0.0001, 0.001, 0.01, 0.1]}  
        params = { 'C':[ 10, 3, 1e3],'gamma': [0.0001, 0.1], 
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}                 
        scoring_fnc = make_scorer(performance_metric)               
               
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
    nfolds = len(cv_sets)
    num_miss_blks = 0
    num_tuple = len(cv_sets[0])
    for n in range(nfolds):
        for m in range(num_tuple):
            if X.ix[cv_sets[n][m]].isnull().values.any() :
                num_miss_blks += 1
            if y.ix[cv_sets[n][m]].isnull().values.any() :
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
            
def calculate_performance(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score                
                        
if __name__ == "__main__":
#    run()

#'2010-07-01', '2016-09-21'  
    dates = pd.date_range('2008-07-01', '2016-09-21')  # one month only
#    feats = ['_hi', '_vol', '_adcls']
    feats = ['_hi', '_adcls']
    symbols = ['SPY', 'XOM', 'WYNN']
    df = get_data(symbols, dates)   
    
#   length of training data set to 90%
#   remaining 10% is for final testing    
    train_data_sz = int(df.shape[0]*0.9)
    df_train = df.ix[:train_data_sz]
    df_test = df.ix[train_data_sz:]
    
    #preprocess data/ append adj column for next day (shifteed by 1) as additional 
#    for symbol in symbols:
#        df_temp = df.ix[:,[symbol + feats[2]]]
#        df_temp = df_temp.rename(columns={symbol + feats[2]: symbol + 'adl_cls_target'})               
#        df = df.join(df_temp[1:])            
    
#   number of days ahead to be predicted
    n_lookup = 1
    
#==============================================================================
# #    num_elem_train and num_elem_test calculated within the train data set
#==============================================================================
    n_folds = 4    
    #test_sz is the training test size within the training set to be used in GridSearchCV
    test_sz = 0.2
    train_sz = (1 - test_sz)
    #num_elem_train, num_elem_test are affected but the number of days for the 
    #prediction n_lookup, ie. a dataframe with a 100 days and 10 days lookup 
    #only have 90 days to be divided between training and testing sets
    num_elem_train = int(math.floor(df_train.shape[0] - n_lookup)/(1 + n_folds*(test_sz/train_sz)))
    num_elem_test = int(math.floor( num_elem_train * (test_sz/train_sz) ))
    print 'train ratio {}'.format(float(num_elem_train)/(num_elem_train + num_elem_test))    
    print 'df.shape train - n days lookup:' , df_train.shape[0] - n_lookup
    print 'num_elem_train', num_elem_train
    print 'num_elem_test', num_elem_test
    
    cv_sets = list()    
    s = get_partition(num_elem_train,num_elem_test)
    for _ in range(n_folds):                
        idxs = next(s)
        u = idxs[0]
        v = idxs[1]
        cv_sets.append((u.copy(), v.copy()))    
    
    #calculate feature and target set with n_lookup days in advance
    for s in symbols:        
        #X_train and y_train are Dataframes of the same size, shifted by n_lookup
        #training features are trained with targets n_lookup days in the future         
        X_train = df.ix [ : -n_lookup, [s+feats[0], s+feats[1]]]
        y_train = df.ix[ n_lookup :, s+feats[1]]
        
        #if data contains more than 30% of cv_sets with "nan's" then through a message
        data_nan_check(s, X_train, y_train, cv_sets )

#        try:
        models = ['SVR']
        for m in models:
            reg = fit_model(X_train.values, y_train.values, cv_sets, m) #takes in training data
            # Produce the value for 'max_depth'
            print "Params for model {} are {}".format(m, reg.get_params())           
            y_predict = reg.predict(df_test.ix[: -n_lookup, [s+feats[0], s+feats[1]] ])
            y_true = df_test.ix[ n_lookup :, s+feats[1] ]
            print 'score {} : '.format(calculate_performance(y_predict, y_true))
#                plt.plot(y_predict)
            
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
