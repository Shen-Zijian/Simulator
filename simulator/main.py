import gc

import numpy
import pandas as pd
import torch.cuda

from Driver_behavour import train_model
# from MLP import MLP_nn
import config
# from simulator_env import Simulator
import simulator_env
import pickle
import numpy as np
from config import *
from path import *
import time
from tqdm import tqdm
import warnings
import json
warnings.filterwarnings("ignore")
import os
import generate_evaluation_data

if __name__ == "__main__":
    driver_num = [750]
    # max_distance_num = [5]
    train_set = pd.DataFrame()
    cruise_flag = [CRUISE]
    pickup_flag = ['rg']
    delivery_flag = ['rg']
    radius_list = [10]
    lr_model = train_model()
    # mlp_model = MLP_nn()
    mlp_model = None
    # radius_model = MLP_nn()
    # pred_list = [2.601150, 5, 36005, 40.7679674, -73.9682154, 27.7, 8]
    # print('预测结果：', radius_model.predict([pred_list]))
    # time_list = ['midnight','morning', 'evening', 'other']
    time_list = ['whole_day']
    time_dict = {'morning': [25200, 32400], 'evening': [61200, 68400], 'midnight': [0, 18000], 'other': [68400, 86400],'whole_day':[0,86400]}
    label_list = ['lstm_fix']
    driver_route_data = {
        'drivers': []
    }
    action_data = {}
    with open('driver_route.json', 'w') as f:
        json.dump(driver_route_data, f, indent=2)
    with open('actions.json', 'w') as f:
        json.dump(action_data, f, indent=2)
    # print(torch.cuda.is_available())
    # for cur_radius in radius_list:
    for item_label in label_list:
        for time_period in time_list:
            for single_driver_num in driver_num:
                # env_params['broadcasting_scale'] = cur_radius
                env_params['label_name'] = item_label
                env_params['t_initial'] = time_dict[time_period][0]
                env_params['t_end'] = time_dict[time_period][1]
                env_params['time_period'] = time_period
                train_set_result = pd.DataFrame(columns=['trip_distance', 'wait_time', 'reward'])
                # for single_max_distance_num in max_distance_num:
                env_params['driver_num'] = single_driver_num
                # env_params['maximal_pickup_distance'] = single_max_distance_num
                # 每次循环更新环境参数
                simulator = simulator_env.Simulator(**env_params)  # 环境参数输入Simulator
                # print('simulator_cruise_flag =',simulator.cruise_flag)
                simulator.reset()
                track_record = []
                t = time.time()  #
                for step in tqdm(range(simulator.finish_run_step)):  # 进度条函数，输出一个可视化进度条
                    new_tracks = simulator.step(lr_model, mlp_model)
                    track_record.append(new_tracks)
                # print('loop_end')
                data_radius_5 = simulator.matched_requests
                # all_requests = simulator.all_requests
                # all_requests.to_csv("./experiment/train_grid" + f"{env_params['time_period']}" + "_model.csv",
                #                     mode='a', index=False, sep=',')

                print("=====saving======")
                match_and_cancel_track_list = simulator.match_and_cancel_track



