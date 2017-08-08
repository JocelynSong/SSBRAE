# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:13:49 2017

@author: q
"""
import sys
import random

def savefile(filename,lines):
    f=open(filename,'w')
    for line in lines:
        f.write(line)
    f.close()

def extract_train_dev_test(filename,file1):
    f=open(filename)
    lines=f.readlines()
    length=len(lines)
    single=int(length/250)
    train_dev_test=random.sample(lines, single*5)
    #train=train_dev_test[:single*25]
    #dev=train_dev_test[single*25:int(single*25.5)]
    #test=train_dev_test[int(single*25.5):]
    savefile(file1,train_dev_test)
    #savefile(file2,dev)
    #savefile(file3,test)
    f.close()

if __name__=='__main__':
    extract_train_dev_test(sys.argv[1],sys.argv[2])
    
    
    