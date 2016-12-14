# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 19:12:20 2016

@author: rcortez
"""


#back testing
from sklearn.metrics import r2_score, make_scorer, fbeta_score
from sklearn.grid_search import GridSearchCV
from sklearn.tree import  DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import numpy as np

import os
import pandas as pd
import matplotlib.pyplot as plt


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
                parse_dates=True, usecols=['Date', 'High', 'Volume', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol + '_adcls'})
        df_temp = df_temp.rename(columns={'Volume': symbol + '_vol'})        
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
    
#datafram features : Volume     
#def split_data(df):
#    features = df['Volume']
#    adj_close = df['Adj Close']
#    X_train, X_test, y_train, y_test = train_test_split( features, df, test_size = 0.2, random_state = 23 )
    
#def test_run():
#    kf = KFold(n = 200, n_folds = 5, shuffle = False, random_state = None)
#    for train, test in kf:
#        print("%s %s" % (train, test))

def get_partition(n_elems, test_sz):
    n_elems_train = int(n_elems - n_elems*test_sz)
    n_elems_test = n_elems - n_elems_train
    idxs_train = np.arange(n_elems_train)        
    idxs_test = np.arange(idxs_train[-1] + 1, n_elems, 1)        
    while True:
        yield idxs_train, idxs_test
        idxs_train += n_elems_test
        idxs_test += n_elems_test                        
#    return idxs_train, idxs_test
    
def performance_metric(y_true, y_predict):
    from sklearn.metrics import r2_score
    score = r2_score(y_true, y_predict)
    return score
    
def fit_model(X,y, cv_sets):
    cv_sets = list()
    s = next(get_partition(100, 0.3))
    cv_sets.append((s[0], s[1]))

#    for train_idx, test_idx in cv_sets:    
#        print 'train, tes: {} \t {} '.format(train_idx.shape, test_idx.shape)
    regressor = DecisionTreeRegressor(random_state = None)
    params = {'max_depth':[2,3, 4, 7, 9, 11, 15,  20, 57]}
    scoring_fnc = make_scorer(performance_metric)
#    grid = GridSearchCV(regressor, params, scoring_fnc, cv = get_partition(100, 0.3))
    grid = GridSearchCV(regressor, params, scoring_fnc, cv = cv_sets)    
    grid = grid.fit(X,y)        
    return grid.best_estimator_    
    

def run():    
    dates = pd.date_range('2010-07-01', '2016-07-31')  # one month only
    feats = ['_hi', '_vol', '_adcls']
    symbols = ['AAPL']
    df = get_data(symbols, dates)   
    
    #preprocess data/ append adj column for next day (shifteed by 1) as additional 
    for symbol in symbols:
#        print symbol
        df_temp = df.ix[:,['AAPL' + feats[2]]]
        df_temp = df_temp.rename(columns={'AAPL' + feats[2]: symbol + 'adl_cls_target'})       
        
#        print df_temp.shift(1)

        df = df.join(df_temp[1:])    
        print df
#    print df
    
    n_folds = 3    
    test_sz = 0.2
    train_sz = (1 - test_sz)
    train_to_test_ratio = train_sz/test_sz
    num_elem_test = int(df.shape[0]/(train_to_test_ratio + n_folds))
    num_elem_train = int(num_elem_test*train_to_test_ratio)
    print 'train ratio {}'.format(float(num_elem_train)/(num_elem_train + num_elem_test))    
    cv_sets = list()    
    s = get_partition(num_elem_train+num_elem_test, test_sz)
#    s = get_partition(10, 0.2)
    for k in range(n_folds):                
        idxs = next(s)
        u = idxs[0]
        v = idxs[1]
        cv_sets.append((u.copy(), v.copy()))    
#        print cv_sets
        
    feats = ['_hi', '_vol', '_adcls']
    
    #calculation for 1 day in advance
    for s in symbols:        
        print s
        features = df.ix[:-1, [s+feats[0], s+feats[1], s+feats[2]]]
        target = df.ix[1:, s+feats[2]]
        reg = fit_model(features, target, cv_sets)
        # Produce the value for 'max_depth'
        print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])   
        
        #predict          
#        print reg.predict()
 

        
if __name__ == "__main__":
    run()
































            
        
    
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
