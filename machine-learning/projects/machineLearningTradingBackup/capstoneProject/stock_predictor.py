# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 19:12:20 2016

@author: rcortez
"""

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from pandas.io.data import DataReader


class stock_predictor():
    def __init__ (self, symbol, bars):
        self.symbol = symbol
        self.bars = bars
        self.create_periods()
        
    def fit_model(self)     
    
    def predict_prices(self):
        #abstract?
    
    def backtest_model(self):
        
        
    
        
        
if __name__ == "__name__":        
    start_test = datetime.datetime(2010, 01, 01)
    end_test = datetime.datetime(
    
    )