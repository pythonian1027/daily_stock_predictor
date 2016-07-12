# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 05:30:11 2016

@author: rcortez
Udacity Machine Learning Nanodegree
Count Words in Python 
"""


string = "betty bought a bit of butter but the butter was bitter"
aList = string.split()
d = {}

for word in aList:
    if d.has_key(word):
        d[word] += 1
    else:
        d[word] = 1

print d

l = list()
for w in sorted(d, key=d.get, reverse = True) :
    l.append((w, d[w]))


#print [(key, value) for (key, value) in sorted(d, key=d.get)]