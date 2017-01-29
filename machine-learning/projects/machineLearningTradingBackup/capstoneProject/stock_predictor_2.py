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
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from portfolio_theory import get_weights
from portfolio_gen import  get_data, load_symbols, download_hist_data


from sklearn.svm import SVR
import numpy as np
import math

import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import sys, traceback

#project_path ='/home/rcortez/projects/python/projects/umlNanoDegee/machine-learning/projects/machineLearningTradingBackup' 
cwd = os.getcwd()
os.chdir('../')
project_path = os.getcwd()


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
        params = { 'C':[1e-3],'gamma': [1e-4], 'kernel': ['linear']}                 
    elif model == 'AdaBoost':
        regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 4), n_estimators = 50, random_state = None) 
        params = {}
        
    scoring_fnc = make_scorer(performance_metric_r2, greater_is_better = True)                              
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
    print '{} : R2 = {}'.format(mod, r2_score(predictions, target))
    result = 100*np.mean((target.values - predictions)/target.values)
    print 'Avg. Percentage Error {}'.format(result)
    evs = explained_variance_score(target, predictions)
    print 'Explained Variance Score = {}'.format(evs)
                     
def plot_results(predictions, target):
    t = np.arange(0, target.shape[0])
    plt.plot(t, predictions, 'r', t, target, 'b')
    plt.show()    
                        
def back_test(model, predictions, target, df_test):            
    print_results(model, predictions, target)
#    plot_results(predictions, target)        
    
    #creating a dataframe of previous closing price, actual price, predicted and signal
    outputs = df_test.ix[:, -2:]
    dframe_pred = pd.DataFrame({'Predicted':y_predict}, index= outputs.index)
    outputs = outputs.join(dframe_pred)            
    outputs = outputs.assign(signal = outputs.apply(get_signal, axis = 1))
    outputs = outputs.assign(rets = outputs.apply(get_return, axis = 1))
#    print outputs
    
    profit =  (outputs[outputs['signal']==1][s] - outputs[outputs['signal'] == 1]['Close Minus {}'.format(n_lookup)]).sum()
    profit = profit/outputs.shape[0]    
    r2score = performance_metric_r2(target, predictions)
    stock_rets =  outputs['rets'].sum()             
    return profit, r2score, stock_rets
                        
def get_signal(k):
    if k['Predicted'] > k['Close Minus {}'.format(n_lookup)]:
        return 1
    elif k['Predicted'] < k['Close Minus {}'.format(n_lookup)]:
        return -1        
    else:
        return 0
def get_return(k):
    if k['signal'] == 1:
        return (k[s] - k['Close Minus {}'.format(n_lookup)])/k['Close Minus {}'.format(n_lookup)]           
    else: 
        return 0                                      
                                                
if __name__ == "__main__":
    
    user_select = raw_input('Select 1 for portfolio analysis:\nSelect 2 for individual stock price prediction:\n')    
    
    if int(user_select) == 1:
        fname = raw_input('Input path to portfolio file location:\n')
        while os.path.exists(fname) == False:
            print 'File not found at: {}'.format(fname) 
            fname = raw_input('Input path to portfolio file location:\n')
        stocks = load_symbols(fname)                
            
                    
#        fname = str(fname)
        
    else:
        s = raw_input('Input ticker symbol: ')        
        stocks = [k.strip() for k in s]

    start_date = input('Input starting date (YYYY,MM,DD):\n')
    start_date = datetime.datetime(start_date[0], start_date[1], start_date[2])
    end_date = input('Input ending date (YYYY,MM,DD):\n')
    end_date = datetime.datetime(end_date[0], end_date[1], end_date[2])        
            
    days_back = 100
#    start_date = datetime.datetime(2011,01,01)
#    end_date = datetime.datetime(2017,01,01)
    dates = pd.date_range(start_date, end_date)  

    test_sz = 0.2
    train_sz = (1 - test_sz)
    n_folds = 10
    n_lookup = 1
    
#    stocks = load_symbols('buffett_port_syms.pickle' )    
    symbols  = download_hist_data(stocks, start_date, end_date )
#    symbols = ['WYNN']    
    
    df = get_data(symbols, dates)  
    data = df.dropna(axis = 1) #get rid of symbols with missing data
#    data = data.ix[:, :-35]  #for speedy tests, remove it
    symbols = data.columns #update columns after dropna    
    weights = get_weights(data)
    weights = weights[1]
    print weights
    
    portfolio_profits = list()    
    portfolio_r2 = list()
    predictor_model = dict()
    for s in symbols:       
        frame = df.ix[:,[s]]    
        #create columns for previous closing days
        for i in range(n_lookup, days_back + n_lookup, 1):
            frame.loc[:,'Close Minus ' + str(i)] = frame[s].shift(i)
        
        #removed the rows affected by shifting (nan's)
        frame_db = frame[[x for x in frame.columns if 'Close Minus' in x or x == s]].iloc[days_back + n_lookup - 1:,]
        #reverse the order of the colunms from oldest to latest prices
        frame_db = frame_db.iloc[:,::-1]
        
        frame_db_train, frame_db_test = get_train_test_sets(frame_db, 0.9)
        X_train = frame_db_train.ix [ : , : -1] # use previous closing prices as features
        y_train = frame_db_train.ix[ :,  [s]] # use last column, adjcls, as a target
        X_test  = frame_db_test.ix[:, : -1]
        y_target = frame_db_test.ix[:, [s]] 
        
#        models = [ 'DTR', 'AdaBoost', 'SVR']
        models = [ 'SVR']                     
        for m in models:
            print type(X_train), type(y_train)
#           Generate Cross-Validation sets for GridSearchCV
            num_elem_train, num_elem_test, cv_sets = fit_model_inputs(frame_db_train.shape[0], 
                                                                      n_lookup, 
                                                                      n_folds, test_sz, train_sz)
#           Fit Model                                                                     
            reg = fit_model(X_train, np.ravel(y_train), m, cv_sets) #takes in training data            
#            print "Params for model {} are {}".format(m, reg.get_params())           
            predictor_model[s] = reg
            
#           Predict prices            
            y_predict = reg.predict(X_test) 
            
#           Backtest results            
            stk_profit, r2score, stk_rets = back_test(m, y_predict, y_target, frame_db_test)            
            portfolio_profits.append(stk_profit)
            portfolio_r2.append(r2score)
                    
    dataframe = pd.DataFrame([portfolio_profits, portfolio_r2, list(weights)], columns = list(data.columns))    
    dataframe = dataframe.T
    dataframe = dataframe.rename(index =str , columns={0:'profits', 1:'R2', 2:'weights'})
    performance_metric_r2(dataframe.loc[:,['profits']], dataframe.loc[:,'R2'])
    (dataframe.loc[:,['profits']]).sum()
#            
    print dataframe.sort_values(by='profits', ascending = False)                    
    
    
    
            
            
            
            
            
            
            
            
            
            
#WYNN
#Params for model SVR are {'kernel': 'linear', 'C': 1, 'verbose': False, 'degree': 3, 'epsilon': 0.1, 'shrinking': True, 'max_iter': -1, 'tol': 0.001, 'cache_size': 200, 'coef0': 0.0, 'gamma': 0.0001}
#SVR : MSE = 7.70260175941
#Avg. Percentage Error -0.29066499597
#Explained Variance Score = 0.737236365318
#-14.848718
#
#
#
#SPY
#<class 'pandas.core.frame.DataFrame'> <class 'pandas.core.series.Series'>
#Params for model SVR are {'kernel': 'linear', 'C': 0.001, 'verbose': False, 'degree': 3, 'epsilon': 0.1, 'shrinking': True, 'max_iter': -1, 'tol': 0.001, 'cache_size': 200, 'coef0': 0.0, 'gamma': 0.0001}
#SVR : MSE = 2.31500270263
#Avg. Percentage Error -0.0283733557465
#Explained Variance Score = 0.925574538209
#19.712812