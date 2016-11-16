# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 00:27:29 2016

@author: rcortez
"""

#import h5py
#
#
def printname(name):
    print name
#
#with h5py.File('./data/snp500.h5','r') as hf:
#    hf.visit(printname)
    
import h5py    # HDF5 support

fileName = "./data/snp500.h5"
f = h5py.File(fileName,  "r")

for item in f.attrs.keys():
    print item + ":", f.attrs[item]
    
for k,v in f.iteritems():
    print v
#mr = f['/real_state/axis0']       
#i00 = f['/entry/mr_scan/I00']
#print "%s\t%s\t%s" % ("#", "mr", "I00")
#for i in range(len(mr)):
#    print "%d\t%g\t%d" % (i, mr[i], i00