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
            self.driver_info = pickle.load(open(load_path + self.driver_file_name + '.pickle', 'rb'))
            # self.driver_info = pd.read_csv(load_path + self.driver_file_name + '.csv')
            self.driver_info = self.driver_info.sample(n=env_params['driver_num'])
            # self.driver_info['driver_id'] = range(len(self.driver_info))

            print(np.sum(self.driver_info['end_time'].values - self.driver_info['start_time'].values))
            gtime = 0
            temp_request = []
            num_request = 0
            while gtime <= 86400:
                if 0 <= gtime <= 86400:
                    for time in range(gtime-2, gtime):
                        if time in self.request_all.keys():
                            temp_request.extend(self.request_all[time])
                            database_size = len(temp_request)
                            num_request += int(np.rint(1 * database_size))
                            temp_request = []

                # if 25200 <= gtime <= 32400:
                #     for time in range(gtime-2, gtime):
                #         if time in self.request_all.keys():
                #             temp_request.extend(self.request_all[time])
                #             database_size = len(temp_request)
                #             num_request += int(np.rint(0.2 * database_size))
                #             temp_request = []
                #
                # if 61200 <= gtime <= 68400:
                #     for time in range(gtime-2, gtime):
                #         if time in self.request_all.keys():
                #             temp_request.extend(self.request_all[time])
                #             database_size = len(temp_request)
                #             num_request += int(np.rint(0.2 * database_size))
                #             temp_request = []
                #
                # if 0 <= gtime <= 18000:
                #     for time in range(gtime-2, gtime):
                #         if time in self.request_all.keys():
                #             temp_request.extend(self.request_all[time])
                #             database_size = len(temp_request)
                #             num_request += int(np.rint(0.2 * database_size))
                #             temp_request = []
                gtime+=2
            print(num_request)
            # sys.exit()









