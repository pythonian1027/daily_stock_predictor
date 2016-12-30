# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 23:15:49 2016

@author: rcortez
"""

import pandas as pd
import numpy as np
#from pandas_datareader import data, wb
import matplotlib.pyplot as plt
import os
import pandas as pd
#import pandas_datareader as pdr



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
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
    return df
    
start_date = pd.to_datetime('2000-01-01')
stop_date = pd.to_datetime('2016-03-01')




def test_run():
    
    
    dates = pd.date_range('2010-01-01','2016-03-01')
    symbols = ['SPY']
    
    
    spy = get_data(symbols, dates)
    
#    spy_c = spy['Close']
#    pd.set_option('display.max_colwidth', 200)
    
    fig, ax = plt.subplots(figsize=(15,10))
    spy.plot(color='k')
    plt.title("SPY", fontsize=20)
    
    long_on_rtn = ((spy['Open'] - sp['Close'].shift(1))/sp['Close'].shift(1))*100
#    spy['Daily Change'] = pd.Series(spy['Close'] - spy['Open'])
#    spy['Daily Change'].sum()
#    np.std(spy['Daily Change'])
#    spy['Overnight Change'] = pd.Series(spy['Open'] - spy['Close'].shift(1))
#    np.std(spy['Overnight Change'])
#    
#    # daily returns
#    daily_rtn = ((spy['Close'] - spy['Close'].shift(1))/spy['Close'].shift(1))*100
#    daily_rtn

if __name__ == "__main__":
    test_run()

#spy_c = spy['Close']
fig, ax = plt.subplots(figsize=(15,10))
spy['Close'].plot(color = 'k')
plt.title('SPY', fontsize = 20)


long_day_rtn = ((spy['Close'] - spy['Close'].shift(1))/spy['Close'].shift(1))*100
long_id_rtn = ((spy['Close'] - spy['Open'])/spy['Open'])*100

long_on_rtn = ((spy['Open'] - 
spy['Close'].shift(1))/spy['Close'].shift(1))*100

#This code gives us each day's closing price along with the prior 20 all on the same line.
#This will form the basis of the X array that we will feed our model. However, before we're ready for this, there are a few additional steps.

for i in range(1, 21, 1):
    spy.loc[:,'Close Minus ' + str(i)] = spy['Close'].shift(i)
sp20 = spy[[x for x in spy.columns if 'Close Minus' in x or x == 'Close']].iloc[20:,]
print sp20

#First, we'll reverse our columns so that time runs left to right, as follows:
sp20 = sp20.iloc[:, ::-1]

#Use SVM regressor from sklearn.svm import SVR, total number of points os over 4000
#so over 3000 for training and the remainder for testing
clf = SVR(kernel='linear')
X_train = sp20[:-1000]
y_train = sp20['Close'].shift(-1)[:-1000]
X_test = sp20[-1000:]
y_test = sp20['Close'].shift(-1)[-1000:]

model = clf.fit(X_train, y_train)
preds = model.predict(X_test)


#spy_c.plot(color='k')
#plt.title("SPY", fontsize=20)
#spy['Daily Change'] = pd.Series(spy['Close'] - spy['Open'])
#spy['Daily Change'].sum()
#np.std(spy['Daily Change'])
#spy['Overnight Change'] = pd.Series(spy['Open'] - spy['Close'].shift(1))
#np.std(spy['Overnight Change'])
#
## daily returns
#daily_rtn = ((spy['Close'] - spy['Close'].shift(1))/spy['Close'].shift(1))*100
#print daily_rtn
#daily_rtn.hist(bins=50, color='lightblue', figsize=(12,8))
#
## intra day returns
#id_rtn = ((spy['Close'] - spy['Open'])/spy['Open'])*100
#id_rtn
#
#id_rtn.hist(bins=50, color='lightblue', figsize=(12,8))
#
## overnight returns
#on_rtn = ((spy['Open'] - spy['Close'].shift(1))/spy['Close'].shift(1))*100
#on_rtn
#
#on_rtn.hist(bins=50, color='lightblue', figsize=(12,8))
#
#def get_stats(s, n=252):
#    s = s.dropna()
#    wins = len(s[s>0])
#    losses = len(s[s<0])
#    evens = len(s[s==0])
#    mean_w = round(s[s>0].mean(), 3)
#    mean_l = round(s[s<0].mean(), 3)
#    win_r = round(wins/losses, 3)
#    mean_trd = round(s.mean(), 3)
#    sd = round(np.std(s), 3)
#    max_l = round(s.min(), 3)
#    max_w = round(s.max(), 3)
#    sharpe_r = round((s.mean()/np.std(s))*np.sqrt(n), 4)
#    cnt = len(s)
#    print('Trades:', cnt,\
#          '\nWins:', wins,\
#          '\nLosses:', losses,\
#          '\nBreakeven:', evens,\
#          '\nWin/Loss Ratio', win_r,\
#          '\nMean Win:', mean_w,\
#          '\nMean Loss:', mean_l,\
#          '\nMean', mean_trd,\
#          '\nStd Dev:', sd,\
#          '\nMax Loss:', max_l,\
#          '\nMax Win:', max_w,\
#          '\nSharpe Ratio:', sharpe_r)
#          
#get_stats(daily_rtn)          
>>>>>>> Stashed changes
