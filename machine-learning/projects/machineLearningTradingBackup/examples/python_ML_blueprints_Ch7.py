# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 23:15:49 2016

@author: rcortez
"""

import pandas as pd
import numpy as np
from pandas_datareader import data, wb
import matplotlib.pyplot as plt

#%matplotlib inline
pd.set_option('display.max_colwidth', 200)

import pandas_datareader as pdr

start_date = pd.to_datetime('2010-01-01')
stop_date = pd.to_datetime('2016-03-01')

spy = pdr.data.get_data_yahoo('SPY', start_date, stop_date)

spy_c = spy['Close']
fig, ax = plt.subplots(figsize=(15,10))
spy_c.plot(color='k')
plt.title("SPY", fontsize=20)
spy['Daily Change'] = pd.Series(spy['Close'] - spy['Open'])
spy['Daily Change'].sum()
np.std(spy['Daily Change'])
spy['Overnight Change'] = pd.Series(spy['Open'] - spy['Close'].shift(1))
np.std(spy['Overnight Change'])

# daily returns
daily_rtn = ((spy['Close'] - spy['Close'].shift(1))/spy['Close'].shift(1))*100
daily_rtn