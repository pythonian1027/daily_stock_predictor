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

from portfolio_gen import  get_data, load_symbols, download_hist_data


import math
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import sys, traceback
import scipy.optimize as sco

cwd = os.getcwd()
os.chdir('../')
project_path = os.getcwd()
              
    
def performance_metric_r2(y_true, y_predict):    
    score = r2_score(y_true, y_predict)
    return score
        
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

def fit_model_inputs(train_size, n_lookup, n_folds, test_sz, train_sz):
    '''Calculates the number of elements in training and testing points and uses 
    them to generate cv sets'''
    numel_train = int(math.floor(train_size - n_lookup)/(1. + n_folds*(float(test_sz)/train_sz)))
    numel_test = int(math.floor( numel_train * (float(test_sz)/train_sz) ))
    cvsets = generate_cv_sets(numel_train, numel_test, n_folds)                        
    return numel_train, numel_test, cvsets   

def print_results(mod, predictions, target):
    '''Prints out model results obtained during backtesint'''
    print '{} : R2 = {}'.format(mod, r2_score(target, predictions))
    accuracy = 100*np.mean((target.values - predictions)/target.values)
    print 'Avg. Percentage Error {}'.format(accuracy)
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
                
    accuracy = print_results(model, predictions, target)    #prints R2 and Pred Error
    
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
        if k['1-day pred'] > k['Today']:
            return 'Buy'
        elif k['1-day pred'] < k['Today']:
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

def print_benchmarks(bm):
    '''Prints benchmark return over test period'''
    ret = (bm.ix[-1,'SPY'] - bm.ix[0, 'SPY'])/bm.ix[0, 'SPY']  
    print type(ret)
    print 'SPY return during test period from \'{} to {}\' is: {}%'.format(bm.index[0].date(), bm.index[-1].date(), (100*ret).round(3) )

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
    
    user_select = raw_input('Select 1 for portfolio analysis:\nSelect 2 for individual stock price prediction:\n')        
    if int(user_select) == 1:
        fname = raw_input('Input path to portfolio file location:\n')
        while os.path.exists(fname) == False:
            print 'File not found at: {}'.format(fname) 
            fname = raw_input('Input path to portfolio file location:\n')
        stocks = load_symbols(fname)                                                
        
    else:
        s = raw_input('Input ticker symbol(s) (comma separated): ')    
        s = s.split(',')
        
        stocks = [k.strip().upper() for k in s]

    start_date = datetime.datetime(2011,01,01)
    end_date = datetime.datetime(2017,01,01)
    dates = pd.date_range(start_date, end_date)
    is_date_instanciated = False
#    while is_date_instanciated is False:        
#        start_date = input('Input starting date (YYYY,MM,DD):\n')
#        start_date = datetime.datetime(start_date[0], start_date[1], start_date[2])
#        end_date = input('Input ending date (YYYY,MM,DD):\n')
#        end_date = datetime.datetime(end_date[0], end_date[1], end_date[2])             
#        dates = pd.date_range(start_date, end_date)   
#        if len(dates) <= 1:
#            print '\nStart date and end date must be different' 
#            print 'Start date must be older than end date'
#        else:
#            is_date_instanciated = True

    days_back = 100
    test_sz = 0.1 # for cross validation
    train_sz = (1 - test_sz)
    n_folds = 10
    n_lookup = 1
    
    symbols  = download_hist_data(stocks, start_date, end_date )
    
    df = get_data(symbols, dates)  
    data = df.dropna(axis = 1) #get rid of symbols with missing data
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
        
        #removed the rows affected by shifting (nan's)
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
#           Generate Cross-Validation sets for GridSearchCV
            num_elem_train, num_elem_test, cv_sets = fit_model_inputs(frame_db_train.shape[0], 
                                                                      n_lookup, 
                                                                      n_folds, test_sz, train_sz)
#           Fit Model and save model to predictor model dictionary                                                                    
            reg = fit_model(X_train, np.ravel(y_train), m, cv_sets) #takes in training data            
            predictor_model[s] = reg
            
#           Predict prices            
            y_predict = reg.predict(X_test) 
            
#           Backtest results            
            stk_profit_buy, stk_profit_sell, r2score, stk_rets, accur, hits, ndays, outputs = back_test(m, y_predict, y_target, frame_db_test)                 
            
            #plots predicted and actual stock prices 
#            plot_preds = outputs.ix[:, ['{}_preds'.format(s), s]]
#            plot_preds.plot(grid=True)
            
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
                    
#    stats = get_weights(data) #get weights for the most recent 1 yr period 
#    weights = stats[1]
    idx_train = frame_db_train.index     
    #use training set for returns and portfolio weights calculation
    weight_period = 128
    weight_period_dates = idx_train[-weight_period:] #use only the last year    
    
    rets = np.log(data.ix[weight_period_dates, ] / data.ix[weight_period_dates, ].shift(1))       
    mSR = max_sharpe_ratio(data.shape[1])
    weights = list(mSR.x)
    


    dataframe = pd.DataFrame([portfolio_profits_buy, portfolio_profits_sell, portfolio_rets, portfolio_r2, weights, portfolio_hits], columns = list(data.columns))    
    dataframe = dataframe.T
    dataframe = dataframe.rename(index =str , columns={0:'profits buy', 1:'profits sell', 2:'returns', 3:'R2', 4:'weight', 5:'hits'})
    dataframe['Asset return'] = dataframe.apply(lambda row: (row['returns']*row['weight']), axis=1)    

#==============================================================================
#     calculate weighted returns
    print pred_dframe
    pred_dframe = pred_dframe.ix[frame_db_test.index, ] #removed nans from pre-processing 
    for s in symbols:
        pred_dframe['{}_wr'.format(s)] = pred_dframe['{}_rets'.format(s)]*dataframe['weight'][s]
#    pred_dframe['pred_totals'] = pred_dframe.T.sum()   
    
#    pred_dframe = pred_dframe.drop(pred_dframe.columns[:-2], axis = 1)
    #normalize
#    pred_dframe1 = ( pred_dframe / pred_dframe.ix[0]) 
#    pred_dframe1.ix[0, :] = 0 #pandas leaves the 0th row with NaNs
#    print pred_dframe1

    weighted_returns = pred_dframe.ix[:, -len(symbols):]
#    s = pred_dframe.ix[:, ['GOOG_wr', 'AMZN_wr', 'AAPL_wr']]
    
#    benchmark = SPY.ix[frame_db_test.index,:]   
    total_ret_series = weighted_returns.sum(axis = 1)
    total_ret = SPY.ix[frame_db_test.index,:]   
    total_ret['rets'] = total_ret_series
    total_ret = total_ret.assign( SVR= total_ret.apply(norm_return_to_benchmark, axis = 1))
    bench = total_ret.drop('rets', axis = 1)
    ax = bench.plot(grid = True, title='SVR vs. SPY', fontsize = 12)
#==============================================================================
#     Plot All Companies prices
    symbols_spy = symbols.insert(0, 'SPY') #add benchmark for plot results
    plot_data(df.ix[X_test.index, symbols_spy], title='Stock Prices')
#==============================================================================
    
    
#    pred_dframe1 = ( pred_dframe - pred_dframe.ix[0]  )
#    pred_dframe1.plot(grid = True)
#==============================================================================
    
#    SPY = dataframe.ix[['SPY'], :]
#    dataframe = dataframe.drop(['SPY'], axis = 0)
         
    print dataframe.sort_values(by='profits buy', ascending = False)   
    
    
#==============================================================================
#     Calculate Benchmark Results (SPY results for the same test period)
    idx_test = frame_db_test.index    
    bframe = SPY.ix[idx_test, ['SPY']]
    print_benchmarks(bframe)
        
#==============================================================================
#   Calculate Performance long position performance over testing period
        

#     
#==============================================================================
    print '\nTotal Number of Traded days: {}'.format(ndays)
    print 'Total Profit : {}'.format(dataframe.ix[:, 'profits buy'].sum() +
                                dataframe.ix[:, 'profits sell'].sum())
                                
    asset_ret = np.float64(dataframe.ix[:, 'Asset return'].sum())                                
    print 'Portfolio Return : {}%'.format((100*asset_ret).round(3))         
    print 'Statistics Exp. return, Exp. volatility, SR: {}'.format(statistics(mSR['x']).round(3))
    
    #calculate actual weighted returns assuming long positions during testing period
    r = np.array(long_position_return)
    w = np.array(dataframe.ix[:, 'weight'])
    
    print 'Long Position Portfolio Return : {}%'.format(100*(np.dot(r.T, w)).round(3))
    print 'Period over which weights were calculated from {} to {}'.format(weight_period_dates[0].date(), weight_period_dates[-1].date())
                                                      
#==============================================================================
#     
    todays_date = datetime.date.today()
    month = (datetime.datetime.now()).month
    day = (datetime.datetime.now()).day
    
    y = 2017
    month = 01
    day  = 01
    #retrieve data starting 4 months ago to use 100 days for training
    if month - 4 <= 0: 
        y = (datetime.datetime.now()).year - 1
        m = month + 8 # -4 months + 12                 
    else:
        y = (datetime.datetime.now()).year
        m = month
    
    start_date = datetime.datetime(y, m, day)        
    dates_pred = pd.date_range(start_date, todays_date)
    predictions = list()    
    for s in symbols:
        X = data.ix[-days_back:, [s]]
        predictions.append(np.asscalar(predictor_model[s].predict(X.T)))
        
    df_pred = pd.DataFrame(((data.ix[-1,:]).ravel()).T, index = symbols, columns = ['Today'])
    df_pred['1-day pred'] = np.array(predictions).T
    df_pred = df_pred.assign(Signal = df_pred.apply(get_signal, axis = 1))
    df_pred = df_pred.assign(r_square = dataframe.apply(get_accuracy, axis = 1))
    df_pred = df_pred.assign(Weights = dataframe.apply(get_asset_weight, axis = 1))
        
