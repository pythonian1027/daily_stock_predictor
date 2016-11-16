# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 22:06:34 2016

@author: rcortez
"""

import pandas_datareader.data as web
import pickle
import os

start_date = '2010-01-01'
end_date = '2016-01-01'


symbol = 'GOOG'
dframe = web.DataReader(name=symbol, data_source='yahoo', start=start_date, end=end_date)
dframe.to_csv("./data/{}.csv".format(symbol), index_label="Date")
path = '/home/rcortez/projects/machineLearningTrading/data'


import urllib2
#import pytz
import pandas as pd

from bs4 import BeautifulSoup
#from datetime import datetime
#from pandas.io.data import DataReader


SITE = "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"



def scrape_list(site):
    hdr = {'User-Agent': 'Mozilla/5.0'}
    req = urllib2.Request(site, headers=hdr)
    page = urllib2.urlopen(req)
    soup = BeautifulSoup(page)

    table = soup.find('table', {'class': 'wikitable sortable'})
    sector_tickers = dict()
    for row in table.findAll('tr'):
        col = row.findAll('td')
        if len(col) > 0:
            sector = str(col[3].string.strip()).lower().replace(' ', '_')
            ticker = str(col[0].string.strip())
            if sector not in sector_tickers:
                sector_tickers[sector] = list()
            sector_tickers[sector].append(ticker)
    return sector_tickers


def download_hist_data(sector_tickers, start_date, end_date):
#    sector_hist_data = {}
    for sector, symbol in sector_tickers.iteritems():
        print 'Downloading data from Yahoo for %s sector' % sector
        for root, dirs, files in os.walk("."):
            print dirs        
#        for s in symbol:            
#            try:
#                dframe = web.DataReader(name=s, data_source='yahoo', start=start_date, end=end_date)
#                print 'writing to file %s symbol' %s
#                if os.path.isdir(path + '/{}'.format(sector)):
#                    dframe.to_csv(path + '/{}.csv'.format(s), index_label="Date")                
#                else:
#                    os.mkdir(path + '/{}'.format(sector))
#                    dframe.to_csv(path + '/{}.csv'.format(s), index_label="Date")
#            except Exception:
#                pass
            #Add SPY for test base
    dframe = web.DataReader(name='GLD', data_source='yahoo', start=start_date, end=end_date)
    dframe.to_csv(path + '/GLD.csv', index_label = "Date")
                
 

#        data = DataReader(symbol, 'yahoo', start_date, end_date)
#        dframe = web.DataReader(name=symbol, data_source='yahoo', start=start_date, end=end_date)
#        dframe.to_csv("./data/{}.csv".format(symbol), index_label="Date")
#        sector_hist_data[sector] = dframe
    print 'Finished downloading data'
#    return sector_hist_data
#
#
#def store_HDF5(sector_hist_data, path):
#    with pd.get_store(path) as store:
#        for sector, hist_data in sector_hist_data.iteritems():
#            store[sector] = hist_data


def get_snp500():
    sector_tickers = scrape_list(SITE)
    save_obj(sector_tickers, 'sector_symbol_list')
    download_hist_data(sector_tickers, start_date, end_date)
#    store_HDF5(sector_hist_data, './data/snp500.h5')
#    return sector_tickers
    


    
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)    


if __name__ == '__main__':
    get_snp500()
#    sectors_list = load_obj('sector_symbol_list')
#    for sector in sectors_list.keys():
#        newPath = path + '/' + sector
#        print newPath
#        os.mkdir(newPath, 755)
