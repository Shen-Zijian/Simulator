import gc

import numpy
import pandas as pd
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

warnings.filterwarnings("ignore")
import os
import generate_evaluation_data

# python D:\Feng\drl_subway_comp\main.py

if __name__ == "__main__":
    driver_num = [500]
    # max_distance_num = [5]
    train_set = pd.DataFrame()
    cruise_flag = [CRUISE]
    pickup_flag = ['rg']
    delivery_flag = ['rg']
    # track的格式为[{'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]],
    # 'driver_2' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]},
    # {'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]}]
    # order_info, matched_num, order_pick_ratio, order_info_mean, order_info_max, driver_state_info_mean = generate_evaluation_data.calculate_metrics(
    #     './output3/rg_rg_cruise=False/records_driver_num_500.pickle', 5, 1)
    #
    # print("order_info", order_info)
    # print("matched_num", matched_num)
    # print("order_pick_ratio", order_pick_ratio)
    # print("order_info_mean", order_info_mean)
    # print("order_info_max", order_info_max)
    # print("driver_state_info_mean", driver_state_info_mean)
    # config.env_params['broadcasting_scale'] = 5
    # radius_list = env_params['radius_list']
    radius_list = [1]
    lr_model = train_model()
    # mlp_model = MLP_nn()
    mlp_model = None
    # radius_model = MLP_nn()
    # pred_list = [2.601150, 5, 36005, 40.7679674, -73.9682154, 27.7, 8]
    # print('预测结果：', radius_model.predict([pred_list]))
    time_list = ['morning','evening','midnight', 'other']
    # time_list = ['other']
    time_dict = {'morning': [25200, 32400], 'evening': [61200, 68400], 'midnight': [0, 18000], 'other': [84390, 86400]}
    # label_list = ['fixed_2s']
    # for item_label in label_list:
    for time_period in time_list:
        for cur_radius in radius_list:
            env_params['broadcasting_scale'] = env_params['model_name'] + env_params['label_name']
            # env_params['label_name'] = item_label
            env_params['t_initial'] = time_dict[time_period][0]
            env_params['t_end'] = time_dict[time_period][1]
            env_params['time_period'] = time_period
            train_set_result = pd.DataFrame(columns=['trip_distance', 'wait_time', 'reward'])
            # for single_max_distance_num in max_distance_num:
            # env_params['driver_num'] = single_driver_num
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
            # file_path = './output3/' + pc_flag + "_" + dl_flag + "_" + "cruise=" + str(cr_flag)
            # if not os.path.exists(file_path):
            #     os.makedirs(file_path)
            # pickle.dump(track_record,
            #             open(file_path + '/records_driver_num_' + str(single_driver_num) + '.pickle',
            #                  'wb'))
            # pickle.dump(simulator.requests, open(
            #     file_path + '/passenger_records_driver_num_' + str(single_driver_num) + '.pickle',
            #     'wb'))
            #
            # pickle.dump(match_and_cancel_track_list,
            #             open(file_path + '/match_and_cacel_' + str(single_driver_num) + '.pickle',
            #                  'wb'))
            temp_set = simulator.cal_reward(simulator.matched_requests)
            # train_set['order_lat'] = simulator.matched_requests['origin_lat']
            # train_set['order_lng'] = simulator.matched_requests['origin_lng']
            train_set['order_grid_id'] = simulator.matched_requests['origin_grid_id']
            train_set['wait_time'] = simulator.matched_requests['wait_time']
            train_set['trip_time'] = simulator.matched_requests['trip_time']
            train_set['trip_distance'] = simulator.matched_requests['trip_distance']
            train_set['price'] = simulator.matched_requests['weight']
            train_set['pickup_distance'] = simulator.matched_requests['pickup_distance']
            train_set['busy_ratio'] = len(simulator.matched_requests) / driver_num[0]
            # train_set['radius'] = '%s' % config.env_params['broadcasting_scale']
            train_set['reward'] = temp_set
            train_set['time_period'] = time_period
            train_set_result = pd.concat([train_set_result, train_set], axis=0, ignore_index=True)
            # train_set_result.append(train_set,ignore_index=True)
            print(train_set_result)

        # train_set_result.to_csv("./experiment/matched_data_grid_based_model_matching_rate.csv", mode='a', index=True,
        #                         sep=',')


