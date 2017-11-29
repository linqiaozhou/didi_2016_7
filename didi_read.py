import pandas as pd
import os
import numpy as np
from PreProcess import*

def region2num():
    path='C:/Users/jiayou/Desktop/DIDI/data/citydata/season_1/test_set_1'
    area_map=pd.read_csv(path+'/cluster_map/cluster_map',header=None,names=['mark','num'],sep='\t')
    area_mark=area_map['mark'].values
    area_num=area_map['num'].values
    dict_mark2num=dict(zip(area_mark,area_num))
    dict_num2mark=dict(zip(area_num,area_mark))
    return dict_mark2num,dict_num2mark

def read_order(path,dict_mark2num):   
    files_train=('01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21')
    files_test=('22','24','26','28','30')
    gap_matrix=np.zeros((1440,66,21),dtype=int)
    total=np.zeros((1440,66,21),dtype=int)
    num=0
    for file in files_train:
        
        #order_file=path+'/order_data/order_data_2016-01-'+file+'_test'
        order_file=path+'/order_data/order_data_2016-01-'+file
        print order_file
        order_data=pd.read_csv(order_file,header=None,
                               names=['order_id','driver_id','passenger_id','start_district_hash','dest_district_hash','Price','Time'],sep='\t')
        order_num=order_data['order_id'].values.size
        
        
        for i in range(order_num):
            start_district=order_data.iloc[i][3]
            start_num=dict_mark2num[start_district]
            int_time=TimeConvert(order_data.iloc[i][6])
            if isinstance(order_data.iloc[i][1], str):
                total[int_time-1,start_num-1,num]+=1
            else:
                gap_matrix[int_time-1,start_num-1,num]+=1
        print file+' is done'
        num+=1
    np.save('train_gap_1',gap_matrix)
    np.save('train_total_1',total)
def read_order_new(path,dict_mark2num):   
    files_train=('01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21')
    files_test=('22','24','26','28','30')
    gap_matrix=np.zeros((288,66,21),dtype=int)
    gap_matrix[:,:,:]=-1
    total=np.zeros((288,66,21),dtype=int)
    num=0
    for file in files_train:
        
        #order_file=path+'/order_data/order_data_2016-01-'+file+'_test'
        order_file=path+'/order_data/order_data_2016-01-'+file
        print order_file
        order_data=pd.read_csv(order_file,header=None,
                               names=['order_id','driver_id','passenger_id','start_district_hash','dest_district_hash','Price','Time'],sep='\t')
        order_num=order_data['order_id'].values.size
        
        
        for i in range(order_num):
            start_district=order_data.iloc[i][3]
            start_num=dict_mark2num[start_district]
            int_time=TimeConvert(order_data.iloc[i][6])
            if isinstance(order_data.iloc[i][1], str):
                total[int_time-1,start_num-1,num]+=1
            else:
                if gap_matrix[int_time-1,start_num-1,num]==-1:
                    gap_matrix[int_time-1,start_num-1,num]=0
                else:
                    gap_matrix[int_time-1,start_num-1,num]+=1
        print file+' is done'
        num+=1
    np.save('train_gap_5',gap_matrix)
    np.save('train_total_5',total)
if __name__=='__main__':
    #path='C:/Users/jiayou/Desktop/DIDI/data/citydata/season_1/test_set_1'
    path='C:/Users/jiayou/Desktop/DIDI/data/citydata/season_1/training_data'
    dict_mark2num,dict_num2mark=region2num()
    read_order(path,dict_mark2num)
    

