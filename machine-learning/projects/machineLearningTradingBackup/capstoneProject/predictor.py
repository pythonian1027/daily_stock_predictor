# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:43:20 2017

@author: rcortez
"""
from sklearn.metrics import r2_score, make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
import numpy as np

from predictor_utilities import  get_data, load_symbols, download_hist_data, \
                                get_train_test_sets, get_train_test_idxs, \
                                performance_metric_r2, generate_cv_sets
                                

import math
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import sys, traceback
import scipy.optimize as sco

project_path = os.getcwd()           
            
def fit_model(X,y, model, cv_sets):
    '''Model fitting using GridSearchCV and CV sets from get_partition func    '''
    if model == 'DTR':
        regressor = DecisionTreeRegressor(random_state = None)
        params = {'max_depth':[2,3, 4, 5, 7, 8, 20], 'min_samples_split' : [2, 8, 16, 32]}
    elif model == 'SVR':
        regressor = SVR() 
        params = { 'C':[1e-3],'gamma': [1e-4], 'kernel': ['linear']}                 
    elif model == 'AdaBoost':
        regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 4), n_estimators = 500, random_state = None) 
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
    '''Throws a message if missing data in dataframes'''
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

def fit_model_inputs(train_size, n_lookup, n_folds, test_sz, train_sz):
    '''Calculates the number of elements in training and testing points and uses 
    them to generate cv sets'''
    numel_train = int(math.floor(train_size - n_lookup)/(1. + n_folds*(float(test_sz)/train_sz)))
    numel_test = int(math.floor( numel_train * (float(test_sz)/train_sz) ))
    cvsets = generate_cv_sets(numel_train, numel_test, n_folds)                        
    return numel_train, numel_test, cvsets   

def calculate_error_prediction(mod, predictions, target):
    '''Prints out model results obtained during backtesint'''
    accuracy = 100*np.mean((target.values - predictions)/target.values)    
    return accuracy                     
                        
def back_test(model, predictions, target, df_test):   
    '''Generates/returns stats during the testing period'''
    '''Outputs:
                profit_buy:    profits generated with 'Buy' signal
                profit_sell:   profits generated with 'Sell' signal
                r2score:       r-square coefficient
                stock_rets:    stock return over test period
                accuracy:      average percentage error
                positive_hits: ratio of correct predictions
                num_days:      total number of days during testins
                outputs:       dataframe generated (for debugging purposes)'''
                
    accuracy = calculate_error_prediction(model, predictions, target)    #prints R2 and Pred Error
    
    outputs = df_test.ix[:, -2:] #creates a dataframe of prior closing price, actual price, predicted and signal
    dframe_pred = pd.DataFrame({'Predicted':y_predict}, index= outputs.index)
    outputs = outputs.join(dframe_pred)            
    
    #adding signal and gains to output/ gains are daily profits
    outputs = outputs.assign(signal = outputs.apply(get_signal, axis = 1))
    outputs = outputs.assign(gains = outputs.apply(get_daily_gains, axis = 1))    
    outputs['cum_gains'] = outputs['gains'].cumsum()
    outputs['{}_preds'.format(s)] = outputs['cum_gains'] + outputs[s][0] #add initial asset value to cum_gains

    #hits is 1 one when prediction is correct (predicted selling when prior selling 
    #price was lower than next day), -1 otherwise    
    outputs['hits']  = outputs.apply(lambda row : (1 if (row['signal'] == 1 and (row[s] > row['Close Minus {}'.format(n_lookup)])) or 
    (row['signal'] == -1 and (row[s] < row['Close Minus {}'.format(n_lookup)])) else 0), axis = 1)
    
    #for counting transactions, necessary when implementing 'Hold' signal
    outputs['transactions'] = outputs.apply(lambda row : (abs(row['signal'])), axis = 1)
    
    #calculates total profits (or losses) for testing period
    profit_buy =  (outputs[outputs['signal']==1][s] - outputs[outputs['signal'] == 1]['Close Minus {}'.format(n_lookup)]).sum()
    profit_sell = (outputs[outputs['signal'] == -1]['Close Minus {}'.format(n_lookup)] - outputs[outputs['signal']== -1][s] ).sum()    
      
    r2score = performance_metric_r2(target, predictions)
    profits = profit_buy + profit_sell
    stock_rets = (outputs.ix[0, s] + profits)/(outputs.ix[0, s]) - 1 #stock rate of return per period
    
    #positive_hits is the ratio of correct recommendations over the total number of recommendations per stock
    positive_hits = float((outputs.ix[:, ['hits']]).sum())/float((outputs.ix[:, ['transactions']]).sum() + 1e-6) #use 1e-6 tp avoid division by zero
    num_days = outputs.shape[0]

    return profit_buy, profit_sell, r2score, stock_rets, accuracy, positive_hits, num_days, outputs
                        
def get_signal(k):    
    '''Generates signals in DataFrame, 1 for 'Buy', -1 for 'Sell' '''
    try:
        if k['Predicted'] > k['Close Minus {}'.format(n_lookup)]:
            return 1
        elif k['Predicted'] < k['Close Minus {}'.format(n_lookup)]:
            return -1        
        else:
            return 0
    except KeyError:
        if k['Tomorrow'] > k['Today']:
            return 'Buy'
        elif k['Tomorrow'] < k['Today']:
            return 'Sell'
        else:
            return 'Hold'                        

def get_accuracy(k):
    try:
        return k['R2']
    except KeyError:
        print 'unhandled exception (get_accuracy)'
                    
def get_asset_weight(k):        
    try:
        return k['weight']
    except KeyError:        
        print 'unhandled exception (get_asset_weight)'
            
def get_daily_gains(k):
    '''returns the profit (or loss) from the signal'''
    if k['signal'] == 1:
        return k[s] - k['Close Minus {}'.format(n_lookup)]
    else:
        return -(k[s] - k['Close Minus {}'.format(n_lookup)])
        
def norm_return_to_benchmark(k):
    '''uses the return of the predictions to generate a signal
    comparable to the benchmark. Used for plotting benchmark vs predicted
    outputs: 
    ret:  float benchmark return 
    '''
    ret = k['SPY']*(1+k['rets'])
    return ret         

def print_benchmark_return(bm):
    '''Prints benchmark return over test period'''
    ret = (bm.ix[-1,'SPY'] - bm.ix[0, 'SPY'])/bm.ix[0, 'SPY']      
    print '\nSPY (benchmark) returns are :{}%. Test period from \'{} to {}\''.format((100*ret).round(3), bm.index[0].date(), bm.index[-1].date() )

def statistics(weights):
    '''returns portfolio statistics
    inputs:
    weights : array-like weights for 
    differrent securities in portfolio
    rets : natural log returns
    
    outputs:
    pret: float expected portfolio return
    pvol: float expected portfolio volatility
    pret/pvol : float Sharpe Ration for risk free = 0
    '''
    weights  = np.array(weights)
    pret = np.sum(rets.mean()*weights)*252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov()*252, weights)))
    return np.array([pret, pvol, pret/pvol])    

#maximization of Sharpe ratio
def minimize_sharpe_ratio(weights):
    return -statistics(weights)[2]
        
def max_sharpe_ratio(num_syms):
    #set portfolio constrainsts, weights add up to 1
    cons = ({'type': 'eq', 'fun':lambda x: np.sum(x) - 1})        
    #bounds, weights must be within 0 and 1
    bnds = tuple((0,1) for x in range(num_syms))
    #initial weight guess 
    wo = num_syms*[1./num_syms]
    maxSR = sco.minimize(minimize_sharpe_ratio, wo, method='SLSQP',bounds=bnds, constraints = cons)
    return maxSR
    
if __name__ == "__main__":    
    
#==============================================================================
#  User Interface    
    user_select = raw_input('Select 1 for portfolio analysis by file input:\nSelect 2 to input stock symbols:\n')        
    
    
    if int(user_select) == 1:
        fname = raw_input('Input path to portfolio file location:')
        while os.path.exists(fname) == False:
            print 'File not found at: {}'.format(fname) 
            fname = raw_input('Input path to portfolio file location:')
        stocks = load_symbols(fname)                                                        
    else:
        s = raw_input('Input ticker symbol(s) (comma separated): ')    
        s = s.split(',')
        
        stocks = [k.strip().upper() for k in s] #remove spaces and capitalized letters

    

    year = (datetime.datetime.now()).year
    month = (datetime.datetime.now()).month
    day = (datetime.datetime.now()).day
    
    start_date = datetime.datetime(year - 6,month, day) #analysis takes 6 years back of historical data
    end_date = datetime.datetime(year, month, day)

    dates = pd.date_range(start_date, end_date)
    days_back = 100
    test_sz = 0.1 # for cross validation
    train_sz = (1 - test_sz)
    n_folds = 10
    n_lookup = 1    
    
#==============================================================================

    
    symbols  = download_hist_data(stocks, start_date, end_date )
    
    df = get_data(symbols, dates)  
    data = df.dropna(axis = 1) #remove symbols with missing data
    if len(data.columns) < len(df.columns):
        print 'The following symbols have been removed \
    because of data missing \'NaN values: {}'.format(list(set(df) - set(data) ))
    
    SPY = data.ix[:, ['SPY']]
    data = data.drop('SPY', axis = 1) # drop benchmark from analysis
    symbols = data.columns #update columns after dropna    
    train_idxs, test_idxs = get_train_test_idxs(data.index, 0.8)
          
    portfolio_profits_buy = list()    
    portfolio_profits_sell = list()    
    portfolio_r2 = list()
    portfolio_accuracy = list()
    portfolio_rets = list()
    portfolio_hits = list()
    long_position_return = list()
    predictor_model = dict()    
    pred_dframe = pd.DataFrame(index = test_idxs)
    
    for s in symbols:       
        frame = df.ix[:,[s]]    
        #create columns for previous closing days
        for i in range(n_lookup, days_back + n_lookup, 1):
            frame.loc[:,'Close Minus ' + str(i)] = frame[s].shift(i)
        
        #removed the nan rows (from shift)
        frame_db = frame[[x for x in frame.columns if 'Close Minus' in x or x == s]].iloc[days_back + n_lookup - 1:,]
        #reverse the order of the colunms from oldest to latest prices
        frame_db = frame_db.iloc[:,::-1]
        
        frame_db_train, frame_db_test = get_train_test_sets(frame_db, 0.8)
        X_train = frame_db_train.ix [ : , : -1] # use prior days closing prices as features
        y_train = frame_db_train.ix[ :,  [s]] # use last column, adjcls, as a target
        X_test  = frame_db_test.ix[:, : -1]
        y_target = frame_db_test.ix[:, [s]] 
        
        #calculate the performance assuming a long position over the testing period
        long_position_return.append(np.float64(y_target.ix[-1,].values/y_target.ix[0,].values) - 1)
        models = [ 'SVR']
        
        for m in models:
            #Generate Cross-Validation sets for GridSearchCV
            num_elem_train, num_elem_test, cv_sets = fit_model_inputs(frame_db_train.shape[0], 
                                                                      n_lookup, 
                                                                      n_folds, test_sz, train_sz)
            #Fit Model and save model to predictor model dictionary                                                                    
            reg = fit_model(X_train, np.ravel(y_train), m, cv_sets) #takes in training data            
            predictor_model[s] = reg
            
            #Predict prices            
            y_predict = reg.predict(X_test) 
            
            #Backtest results            
            stk_profit_buy, stk_profit_sell, r2score, stk_rets, accur, hits, ndays, outputs = back_test(m, y_predict, y_target, frame_db_test)                 
                        
            portfolio_profits_buy.append(stk_profit_buy)
            portfolio_profits_sell.append(stk_profit_sell)
            portfolio_r2.append(r2score)
            portfolio_accuracy.append(accur)
            portfolio_rets.append(stk_rets)
            portfolio_hits.append(hits)
            pred_dframe[s] = outputs.ix[:,[s]]
            pred_dframe['{}_rets'.format(s)] = outputs.ix[:, ['cum_gains']]/outputs.ix[0,s]# normalized by the corresponding
            #initial symbol since cum_gains first time step is already a prediction of outputs.ix[0,s] 
            pred_dframe['{}_preds'.format(s)] = outputs.ix[:,['{}_preds'.format(s)]]
                    
                    
    idx_train = frame_db_train.index  #use training set for returns and portfolio weights calculation
    weight_period = 128 #about 6 months 
    weight_period_dates = idx_train[-weight_period:] #calculate weight from last 6 months of the training period
    
    rets = np.log(data.ix[weight_period_dates, ] / data.ix[weight_period_dates, ].shift(1))       
    max_sr = max_sharpe_ratio(data.shape[1])
    weights = list(max_sr.x)#mean-variance portfolio weights
    

    #Create a dataframe 'results' containing portfolio information
    results = pd.DataFrame([portfolio_profits_buy, portfolio_profits_sell, portfolio_rets, portfolio_r2, weights, portfolio_hits], columns = list(data.columns))    
    results= results.T
    results = results.rename(index =str , columns={0:'profits buy', 1:'profits sell', 2:'returns', 3:'R2', 4:'weight', 5:'hits'})
    results['Asset return'] = results.apply(lambda row: (row['returns']*row['weight']), axis=1)    


    print '#'*30 + '  TEST RESULTS  ' + '#'*30
    print results.round(3).sort_values(by='returns', ascending = False)

#==============================================================================
#    Calculate Benchmark Results (SPY results for the same test period)
    idx_test = frame_db_test.index    
    bframe = SPY.ix[idx_test, ['SPY']]
    print_benchmark_return(bframe)
        

    print '\nTotal Number of Traded days: {}'.format(ndays)
    print 'Algorithm Total Profit: {}'.format(results.ix[:, 'profits buy'].sum() +
                                results.ix[:, 'profits sell'].sum())
                                
    asset_ret = np.float64(results.ix[:, 'Asset return'].sum())                                
    print 'Algorithm Portfolio Return : {}%'.format((100*asset_ret).round(3))         
    
    print 'Mean-Variance analysis results: '
    mva = statistics(max_sr['x']).round(3)
    print '\t\t\t\tExpected Portfolio Return : {}'.format(mva[0])
    print '\t\t\t\tExpected Portfolio Volatility : {}'.format(mva[1])
    print '\t\t\t\tPortfolio Sharpe Ratio : {}'.format(mva[2])
    
    #calculate actual weighted returns assuming long positions during testing period
    r = np.array(long_position_return)
    w = np.array(results.ix[:, 'weight'])
    
    print 'Long Position Portfolio Return : {}%'.format(100*(np.dot(r.T, w)).round(3))
    print 'Period over which portfolio weights were calculated from {} to {}'.format(weight_period_dates[0].date(), weight_period_dates[-1].date())
                                                      
#==============================================================================
#   Generate recommendations based on tomorrow's predictions     
                                                      
    today = idx_test[-1].date()
    print '\n'
    print '#'*30 + '  FORECAST  ' + '#'*30   
    print '\nRecommendations for today\'s date: {}\n'.format(today)
    
    predictions = list()    
    for s in symbols:
        X = data.ix[-days_back:, [s]]
        predictions.append(np.asscalar(predictor_model[s].predict(X.T)))
        
    df_pred = pd.DataFrame(((data.ix[-1,:]).ravel()).T, index = symbols, columns = ['Today'])
    df_pred['Tomorrow'] = np.array(predictions).T
    df_pred = df_pred.assign(Recomm = df_pred.apply(get_signal, axis = 1))
    df_pred = df_pred.assign(r_square = results.apply(get_accuracy, axis = 1))
    #recalculate weights based on most recent 6 months returns
    weight_period_dates = idx_test[-weight_period:] #calculate weight from last 6 months of the training period    
    rets = np.log(data.ix[weight_period_dates, ] / data.ix[weight_period_dates, ].shift(1))       
    max_sr = max_sharpe_ratio(data.shape[1])    

    df_pred['Weights'] = list(max_sr.x)#mean-variance portfolio weights
    print df_pred.round(3)    
        
    print '\n' + '#'*80     
#==============================================================================