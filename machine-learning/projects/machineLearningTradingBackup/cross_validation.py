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


def rsquare_performance(real_data, pred_data):        
    return r2_score(real_data, pred_data)
    
#datafram features : Volume     
def split_data(df):
    features = df['Volume']
    adj_close = df['Adj Close']
    X_train, X_test, y_train, y_test = train_test_split( features, df, test_size = 0.2, random_state = 23 )
    
def test_run():
    kf = KFold(200, n_folds=5)
    for train, test in kf:
        print("%s %s" % (train, test))
    
    
    
if __name__ == "__main__":
    test_run()
