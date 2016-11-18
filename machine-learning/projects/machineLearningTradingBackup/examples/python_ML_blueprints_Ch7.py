# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 23:15:49 2016

@author: rcortez
"""
start_date = pd.to_datetime('2000-01-01')
stop_date = pd.to_datetime('2016-03-01')
sp = pdr.data.get_data_yahoo('SPY', start_date, stop_date)


