# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 20:13:23 2017

@author: song
"""
import sys

def merge_file(filename,file1,file2,file3):
    f=open(filename,'w')
    f1=open(file1)
    for line in f1.readlines():
        f.write(line)
    f1.close()
    f2=open(file2)
    for line in f2.readlines():
        f.write(line)
    f2.close()
    f3=open(file3)
    for line in f3.readlines():
        f.write(line)
    f3.close()
    f.close()

if __name__=='__main__':
    merge_file(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
    
    

