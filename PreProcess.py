# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:21:58 2016

@author: Administrator
"""
import datetime
import pandas as pd
#import calendar   

# 将字符串转换成datetime类型  
def strToDateTime(datestr,format):      
    return datetime.datetime.strptime(datestr,format).date() 

# 时间转换成字符串,格式为2008-08-02  
def dateToStr(date):    
    return   str(date)[0:10] 
    
#时间转换成时间片
def TimeConvert(str_time):                 
   return (int(str_time[11:13])*60+int(str_time[14:16]))+1    
   
   
   
'''区域对应的hash值'''
def region2num():
    area_map=pd.read_csv('Trainning_data/cluster_map',header=None,names=['mark','num'],sep='\t')
    area_mark=area_map['mark'].values
    area_num=area_map['num'].values
    dict_mark2num=dict(zip(area_mark,area_num))
    return dict_mark2num
def num2region():
    area_map=pd.read_csv('Trainning_data/cluster_map',header=None,names=['mark','num'],sep='\t')
    area_mark=area_map['mark'].values
    area_num=area_map['num'].values
    dict_num2mark=dict(zip(area_num,area_mark))
    return dict_num2mark

if __name__ == "__main__":
    dateNow = strToDateTime('2016-05-26 17:30:15', "%Y-%m-%d %H:%M:%S")
    print(dateNow.weekday())