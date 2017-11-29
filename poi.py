# -*- coding: utf-8 -*-
"""
Created on Mon May 23 22:00:28 2016

@author: jiayou
"""
import numpy as np
from PreProcess import*
import matplotlib.pyplot as plt
poi_data=np.zeros((66,25),dtype='int32')
file = open('Trainning_data/poi_data')
for line in file:
    list_line=line.split('\t')
    pos=list_line[0]
    buid_type=list_line[1:]
    for item in buid_type:
        lable=''
        num=''
        flag1=0
        flag2=0
        for i in item:
            if (i!='#')and(i!=':')and(flag1==0):
                lable+=i
            elif ((i=='#')or(i==':'))and(flag1==0):
                flag1=1
                if(i==':'):
                    flag2=1
            elif(i==':')and(flag2==0):
                flag2=1
            elif(flag2==1)and(i.isdigit()):
                num+=i
        dict_mark2num=region2num()
        pos_int=dict_mark2num[pos]      
        poi_data[pos_int-1][int(lable)-1]+=int(num)   #python 对应的是0~65和0~24
order_data=pd.read_csv('Trainning_data/order_data_2016-01-02',header=None,names=['order_id','driver_id','passenger_id','start_district_hash','dest_district_hash','Price','Time'],sep='\t')
order_num=order_data['order_id'].values.size
'''区域对应的hash值'''
dict_mark2num=region2num()
'''不同区域在不同时期在不同时间对应的Gap'''
#dict_area_gap=[0 for regions in range(64)]
gap_matrix = [[0 for regions in range(66)] for times in range(144)]  
for i in range(order_num):
#    if math.isnan(order_data.iloc[i][1]):
    if isinstance(order_data.iloc[i][1], str):
        continue
    else:
        start_district=order_data.iloc[i][3]
        start_num=dict_mark2num[start_district]
        int_time=TimeConvert(order_data.iloc[i][6])
        gap_matrix[int_time-1][start_num-1]+=1
np.save("poi.npy",poi_data)
for i in range(66):
    plt.plot(poi_data[i,:])
    plt.savefig('.\\img\\'+str(i)+'.jpg')
    plt.close()
    