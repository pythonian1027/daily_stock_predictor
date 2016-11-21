# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:40:08 2016

@author: rcortez
"""

#back testing
from sklearn.metrics import r2_score, make_scorer, fbeta_score
from sklearn.grid_search import GridSearchCV
from sklearn.tree import  DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
import numpy as np

import os
import pandas as pd
import matplotlib.pyplot as plt


cwd = os.getcwd()
print cwd
os.chdir('../../')
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
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
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
    
if __name__ == "__main__":
#    test_run()
    dates = pd.date_range('2010-07-01', '2016-07-31')  # one month only
    symbols = ['SPY','XOM']
    df = get_data(symbols, dates)
#    plot_data(df)
    
    s = get_partition(100, 0.3)        
    for _ in range(5):
        cv = next(s)
        print df.shape
        X_train =  df.ix[cv[0]]
        print ([cv[1]])
    