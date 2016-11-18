# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:18:19 2016

@author: rcortez
"""

#back testing
from sklearn.metrics import r2_score, make_scorer, fbeta_score
from sklearn.grid_search import GridSearchCV
from sklearn.tree import  DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
import numpy as np


def rsquare_performance(real_data, pred_data):        
    return r2_score(real_data, pred_data)
    
#datafram features : Volume     
def split_data(df):
    features = df['Volume']
    adj_close = df['Adj Close']
    X_train, X_test, y_train, y_test = train_test_split( features, df, test_size = 0.2, random_state = 23 )
    
def set_split(n_elems, test_size, n_folds):
    num_train_elems = int(n_elems - n_elems*test_size)
    idxs_train = np.arange(num_train_elems)
    idxs_test = np.arange(num_train_elems, n_elems)            
    return idxs_train, idxs_test
    
    
def test_run():
<<<<<<< HEAD
    kf = KFold(200, n_folds=5, random_state = None, shuffle = False)
    for train, test in kf:        
=======
    kf = KFold(n = 200, n_folds = 5, shuffle = False, random_state = None)

    for train, test in kf:
>>>>>>> 20e27db286b0aaec1bd694ace2a07c07d8ca16a7
        print("%s %s" % (train, test))

def set_partition():
    print np.arange(50)    
    
    
if __name__ == "__main__":
<<<<<<< HEAD
#    train, test = set_split(101, 0.3)
#    print train, test
    test_run()
=======
#    test_run()
    set_partition()
>>>>>>> 20e27db286b0aaec1bd694ace2a07c07d8ca16a7
