# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 21:25:28 2017

@author: rcortez
"""
import pickle
import numpy as np
import pandas as pd
#m = list()
#for k in range(1, 12):
#    try:
#        filename = 'weight_vs_unweight_{}_4500.pkl'.format(k)  
#        with open(filename, 'rb') as f:
#            b = pickle.load(f)
#            m.append(b)
#    except:
#        continue      
#      
#for k in range(0, 10):    
#    print k
#    if k == 0:
#        we = np.array(m[0][0])
#        unwe = np.array(m[0][1])
#    else:
#        
#        we = np.append(we, np.array(m[k][0])) 
#        unwe = np.append(unwe, np.array(m[k][1]))


#==============================================================================
# RETURNS

m = list()
for k in range(1, 12):
    try:
        filename = 'returns_{}_4500.pkl'.format(k)    
        with open(filename, 'rb') as f:
            b = pickle.load(f)
            m.append(b)
    except:
        continue      
      
for k in range(0, 10):    
    print k
    if k == 0:
        l1, l2 = zip(*m[0])
        sym = np.array(l1)
        ret = np.array(l2)
    else:
        l1, l2 = zip(*m[k])
        sym = np.append(sym, l1) 
        ret = np.append(ret, l2)

s = {'sym' : sym}
df = pd.DataFrame(s)
df['ret'] = ret
df = df.drop_duplicates()
print df.sort_values(by='ret', ascending = False)

#==============================================================================

# HITS
m = list()
for k in range(1, 12):
    try:
        filename = 'hits_{}_4500.pkl'.format(k)    
        with open(filename, 'rb') as f:
            b = pickle.load(f)
            m.append(b)
    except:
        continue      
      
for k in range(0, 10):    
    print k
    if k == 0:
        l1, l2 = zip(*m[0])
        sym = np.array(l1)
        hit = np.array(l2)
    else:
        l1, l2 = zip(*m[k])
        sym = np.append(sym, l1) 
        hit = np.append(hit, l2)

s = {'sym' : sym}
df_hits = pd.DataFrame(s)
df_hits['hits'] = hit
df_hits = df_hits.drop_duplicates()
df_hits =  df_hits.sort_values(by='hits', ascending = False)
idx = np.where(df_hits['sym']=='RRC')
print df_hits.ix[df_hits['sym']=='RRC']

#==============================================================================

#weight_vs_unweight = np.array([svr_weigthed, svr_unweighted])
#shape  = 2, 4521

#weights_montecarlo = list()
#weights_montecarlo.append((we, unwe))                   
#with open('weight_vs_unweight_montecarlo.pickle', 'wb') as handle:
#    pickle.dump(weights_montecarlo, handle, protocol=pickle.HIGHEST_PROTOCOL)                   