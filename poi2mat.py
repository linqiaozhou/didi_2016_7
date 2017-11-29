# -*- coding: utf-8 -*-


import scipy.io as sio
import numpy as np
'''
matfn='/home/zhang/matlab_program/didi/auto_region.mat'
data=sio.loadmat(matfn)
poi=data['y']
np.save('poi_autocoder',poi)
'''
sio.savemat('flow_train_29.mat',{'x':flow_feature_29,'y':flow_out_29})