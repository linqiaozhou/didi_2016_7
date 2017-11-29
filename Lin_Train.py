# -*- coding: utf-8 -*-
"""
Created on Mon Jun 06 11:02:32 2016

@author: jiayou
"""
import copy
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from diditrain import*

def Get3Gap(p):
    time=[42,54,66,78,90,102,114,126,138]
    T=time[p-1]    
    gap=np.load('test_gap.npy')
    feature=np.zeros((5*66,3),dtype=float)
    t=0
    for k in [1,2,3,4,5]:
        for i in range(66):
            feature[t,:]=gap[T:T+3,i,k-1]
            t+=1
    for j in range(len(feature)):
        if feature[j,0]==-1 and feature[j,1]==-1 and feature[j,2]==-1:
            feature[j,:]=[0,0,0]
        else:
            s=feature[j,:][feature[j,:]!=-1].mean()
            for i in [0,1,2]:
                if feature[j,i]==-1:
                    feature[j,i]=s
    return feature
def Linear(p,Gap):
    predict=np.zeros(len(Gap),dtype=float)
    if p==5:
         predict[:]=1
#    elif p==1:
#        for i in range(len(Gap)):
#            predict[i]=max(1,(Gap[i,2]*0.85+Gap[i,1]*0.25)/2)
#    elif p==3:
#        for i in range(len(Gap)):
#            predict[i]=max(1,(Gap[i,2]))    
    else:
        for i in range(len(Gap)):
            predict[i]=max(1,(Gap[i,2]*0.65+Gap[i,1]*0.25+Gap[i,0]*0.15)/2)
    return predict
def test(p,days):
  train_feature=np.zeros((len(days)*66,3),dtype=float)
  feature,out=gettrain_timesplit(days,p)
  train_feature=feature[:,4:]
  out=np.array(out)
 
  train_result=Linear(p,train_feature)
  erro1=predicterro(out,train_result)
  full_one=np.ones(out.shape)
  erro2=predicterro(out,full_one)
  return erro1,erro2
def GetErrorMap():
    ErrorMap=np.zeros(144,66,dtype=float)
    gap_truth=np.load('train_gap.npy')
    Gap_avg=np.sum(gap_truth,axi=3)
    
    
    
    
days=[17,18,19,20,21]
days_all=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
cnt=0
cnt1=0
for i in range(9):
    result_Lin=test(i+1,[6])
    cnt+=result_Lin[0]
    cnt1+=result_Lin[1]
    print result_Lin[0],result_Lin[1]
print'******''Linear''**************'  
print cnt/9,cnt1/9