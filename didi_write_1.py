# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 11:03:25 2016

@author: zhang
"""
import pandas as pd
import numpy as np
import matplotlib as plt
a=np.zeros((330,9),dtype=float)
#for i in range(9):
#    a[:,i]=np.load('6_7_'+str(i+1)+'.npy')



##############################################
all_result=np.load('6_8_ALL.npy')
for i in range(9):
    a[:,i]=all_result[i*330:(i+1)*330]
    
 ####注释之间是我改的########
    
start=[0,1,0,1,0]
out=[]
for i in range(5):
    t=66*i
    for j in range(start[i],9):
        for k in range(t,t+66):
            out.append(a[k,j])



textfile='V1.csv'
resultfile=pd.read_csv(textfile,header=None,names=['region','time','value'],sep=',')
num=resultfile['region'].values.size

x=resultfile.iloc[i][2]
x=out
resultfile['value']=pd.Series(x)
resultfile.to_csv('predict_6_9.csv', sep=',',index=None,header=None)
