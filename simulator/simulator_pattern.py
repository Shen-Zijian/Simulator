# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 19:20:13 2018

@author: kejintao

input information:
1. demand patterns (on minutes)
2. demand databases
3. drivers' working schedule (online/offline time)

** All the inputs are obtained from env, thus we do not need to alter parameters here
"""

import numpy as np
import pandas as pd
from copy import deepcopy
import random
from config import *
from path import *
import pickle
import sys
class SimulatorPattern(object):
    def __init__(self, **kwargs):
        # read parameters
        self.simulator_mode = kwargs.pop('simulator_mode', 'simulator_mode')
        self.request_file_name = kwargs['request_file_name']
        self.driver_file_name = kwargs['driver_file_name']

        if self.simulator_mode == 'toy_mode':
            self.request_all = pickle.load(open(data_path + self.request_file_name + '.pickle', 'rb'))
            # print(self.request_all)
            self.driver_info = pickle.load(open(load_path + self.driver_file_name + '.pickle', 'rb'))#.head(env_params['driver_num'])
            # print(np.min(self.driver_info['lng'].values), np.max(self.driver_info['lng'].values),np.average(self.driver_info['lng'].values))
            # print(np.average(self.driver_info['lat'].values), np.max(self.driver_info['lat'].values),np.average(self.driver_info['lat'].values))
            # print(all_requests)
            # print(type(env_params['west_lng']),env_params['west_lng'])
            # print(type(self.driver_info))
            # driver_info_1 = self.driver_info.loc[(self.driver_info['lng'] > env_params['west_lng'])]
            # print(driver_info_1['lng'].values)
            # print(self.driver_info['lng'] > float(env_params['west_lng']))
            # driver_info_2 = self.driver_info.loc[(self.driver_info['lng'] > 114.13)]
            # print(driver_info_2['lng'].values)
            # print(self.driver_info['lng'] > 114.13)
            # west_lng_value = env_params['west_lng']
            # driver_info_3 = self.driver_info.loc[(self.driver_info['lng'] >west_lng_value)]
            # print(driver_info_3['lng'].values)
            # self.driver_info = self.driver_info.loc[(self.driver_info['lng'] > 114.13)&(self.driver_info['lng'] < 114.235)&(self.driver_info['lat'] < 22.285)&(self.driver_info['lat'] > 22.23)]
            # print(self.driver_info)
            # print(self.driver_info['lng'] > west_lng_value)
            # self.driver_info = self.driver_info.loc[(self.driver_info['lat'] < env_params['north_lat']) & (self.driver_info['lat'] > env_params['south_lat'])]
            # self.driver_info = self.driver_info.sample(n=env_params['driver_num'])
            # print(self.driver_info)









