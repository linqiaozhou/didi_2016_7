# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:25:07 2016

@author: Lin
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from math import  floor
import xgboost as xgb

def predicterro(testo,predictres,model=0):
    out=0
    if model==0:
        for i in range(len(predictres)):
            if testo[i]!=0:
                out+=abs(predictres[i]-testo[i])/testo[i]
    else:
        for i in range(len(predictres)):
            if testo[i]!=0:
                out+=abs(predictres[i]-testo[i])
    out=out/len(predictres)
    return out

def erro_analysis(testo,predictres):
    out=np.zeros((5,2),dtype=float)
    for i in range(len(testo)):
        if testo[i]>0:
            if testo[i]<=1:
                out[0,0]+=abs(predictres[i]-testo[i])/testo[i]
                out[0,1]+=1
            elif testo[i]<=5:
                out[1,0]+=abs(predictres[i]-testo[i])/testo[i]
                out[1,1]+=1
            elif testo[i]<=20:
                out[2,0]+=abs(predictres[i]-testo[i])/testo[i]
                out[2,1]+=1
            elif testo<=100:
                out[3,0]+=abs(predictres[i]-testo[i])/testo[i]
                out[3,1]+=1
            else:
                out[4,0]+=abs(predictres[i]-testo[i])/testo[i]
                out[4,1]+=1
    for i in range(5):
       out[i,0]=out[i,0]/(out[i,1]+1)
    return out
def refineresult1(result):
    Result=copy.deepcopy(result)
    for i in range(len(result)):
        if result[i]<=2:
            Result[i]=1
        elif result[i]<=100:
            Result[i]=Result[i]/2   
        else:
            Result[i]=floor(Result[i])    
    return Result
def refineresult2(result):
    Result=copy.deepcopy(result)
    for i in range(len(result)):
        if result[i]<=2:
            Result[i]=1
        else:
            Result[i]=floor(Result[i])    
    
    return Result    

def showresult(testo,result):
    plt.subplot(121)
    plt.plot(result, color="blue", linewidth=2.5, linestyle="-", label="predict")
    plt.ylim(0,2000)
    plt.show()
    plt.subplot(122)
    plt.plot(testo, color="red", linewidth=2.5, linestyle="-", label="true")
    plt.ylim(0,2000)
    plt.show()
    #plt.savefig('/home/zhang/pycharm-community-5.0.4/python_pro/didi')
def gettrain_timesplit(days,p,is_test=False):
    time_long=5
    mutlpy=10/time_long
    if is_test:
        time=[mutlpy*45,mutlpy*57,mutlpy*69,mutlpy*81,mutlpy*93,mutlpy*105,mutlpy*117,mutlpy*129,mutlpy*141]                
    else:
        time=np.zeros(mutlpy*(144-40),dtype=int)
        for i in range(mutlpy*(144-40)):
            time[i]=i+mutlpy*39              #为了方便处理，把前38个时间片和最后两个时间片丢弃
        
    gap=np.load('train_gap_5.npy')
    poi=np.load('poi.npy')
    traffic=np.load('train_traffic.npy')
    feature=np.zeros((len(days)*len(time)*66,25+3*mutlpy+4*3),dtype=float)
    #T=time[p-1]
#    Gap_5=[]
#    Gap_10=[]
    out=[]
    t=0
    for T in time:
        for i in range(66):
            for k in days:             
                feature[t,:25]=poi[i,:]                
                feature[t,25:25+3*mutlpy]=gap[T-3*mutlpy:T,i,k-1]
                temp=traffic[T/2-3:T/2,i,k-1,:].reshape(12)
                #feature[t,25+3*mutlpy:25+3*mutlpy+4*3*mutlpy]=traffic[T-3*mutlpy:T,i,k,:].reshape(24)
                feature[t,25+3*mutlpy:25+3*mutlpy+4*3]=temp
#                Gap_5.append(float(gap[T,i,k-1]))
#                Gap_10.append(float(gap[T,i,k-1]+gap[T+1,i,k-1]))  #用于测试误差的的真实值
                out.append(float(gap[T,i,k-1]+gap[T+1,i,k-1]))
                t+=1
    return feature,out
def gettest_timesplit():
    time_long=5
    mutlpy=10/time_long
    time=[mutlpy*45,mutlpy*57,mutlpy*69,mutlpy*81,mutlpy*93,mutlpy*105,mutlpy*117,mutlpy*129,mutlpy*141]                
    #T=time[p-1]     
    gap=np.load('test_gap_5.npy')
    cluser=np.load('poi.npy')
    feature=np.zeros((5*66*9,25+3*mutlpy),dtype=float)
    t=0
    for T in time:
        for k in [1,2,3,4,5]:
            for i in range(66):
                regionclass=cluser[i,:]                
                feature[t,:25]=regionclass
                feature[t,25:]=gap[T-3*mutlpy:T,i,k-1]
                t+=1
    for j in range(len(feature)):
        if feature[j,25]==-1 and feature[j,26]==-1 and feature[j,27]==-1 and feature[j,28]==-1 and feature[j,29]==-1 and feature[j,30]==-1:
            feature[j,25:]=[0,0,0,0,0,0]
        else:
            s=feature[j,25:][feature[j,25:]!=-1].mean()
            for i in [25,26,27,28,29,30]:
                if feature[j,i]==-1:
                    feature[j,i]=s
    return feature
    
def GBRegressor_predict(feature,out,testf):
    
    clf = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1,max_depth=3, random_state=0, loss='lad')
    
    clf = clf.fit(feature,out)
    result=clf.predict(testf)
    #print clf.feature_importances_
    return result
    
def XGBoost_predict(feature,out,testf):
    dtrain = xgb.DMatrix(feature,label = out)
    param = {'max_depth':3, 'eta':0.5, 'silent':1, 'objective':'reg:linear' }
    num_tree=200
    model = xgb.train(param, dtrain, num_tree)
    xgb.plot_importance(model)
    test_data = xgb.DMatrix(testf)
    predict = model.predict(test_data)    
    return predict

def didi_train(model_type,p,compare=1):
    days=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,20]
    #days=[7,8,9,10,11,12,13,14,15,16]
    #test_days=[17,18,19,20,21]
    test_days=[15,16,17,19,21]
    feature,out=gettrain_timesplit(days,p,False)
    if compare==1:
        testf,testo=gettrain_timesplit(test_days,p,True)
    else:
        testf=gettest_timesplit()
    if model_type=='GBDT':
        result=XGBoost_predict(feature,out,testf)
        #result=2*result #预测的是5分钟的Gap，10分钟的Gap简单的处理，即5分钟的Gap乘以2
        if compare==1:
            #showresult(testo,result)
            result1=refineresult1(result)
            result2=refineresult2(result)
            R=predicterro(testo,result,model=0)
            R1=predicterro(testo,result1,model=0)
            R2=predicterro(testo,result2,model=0)
            out=erro_analysis(testo,result)
            out_1=erro_analysis(testo,result1)
            out_2=erro_analysis(testo,result2)
            r={'R':R,'R1':R1,'R2':R2,'out':out,'out1':out_1,'out2':out_2}
            print '5_2:',R,R1,R2
            return result,r
        else:
            Result=refineresult2(result)
            np.save('./6_8_ALL',Result)

            return result

if(__name__=='__main__'):
    result=didi_train('GBDT',1,compare=1)
#    for p in range(9):
#        result=didi_train('GBDT',p+1,1)
#    result=np.zeros(9,dtype=float)
#    cnt=0
#    for p in range(9):
#        temp=didi_train('GBDT',p+1,compare=1)
#        temp1=temp[1]
#        result[p]=temp1['Rre']
#        cnt+=result[p]
#        print result[p]
#print'********************'  
#
#print cnt/9
#
#print '********************'

    