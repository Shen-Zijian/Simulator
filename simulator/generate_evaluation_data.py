#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: zhangyuhao
@file: generate_evaluation_data.py
@time: 2022/7/8 下午10:42
@email: yuhaozhang76@gmail.com
@desc: 
"""
import os
import pickle
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(record_path, time_interval, sample_frac):
    records = pickle.load(open(record_path, 'rb'))
    order = pickle.load(open('./input/order.pickle', 'rb'))
    order_num_time = {}
    matched_rate_time = {}
    current_num = 0
    matched_num = 0
    time_to_order = {}
    driver_no_cruising_time = {}
    for i in range(36000, 79200, time_interval):
        temp_count = 0
        for j in range(0, time_interval):
            if (i+j) in order.keys():
                current_num += len(order[i+j])
                temp_count += len(order[i+j])
        order_num_time[i] = np.rint(current_num*sample_frac)
        time_to_order[i] = np.rint(temp_count*sample_frac)
    for i, time in enumerate(tqdm(records, desc="generate driver info")):
        for driver in time:
            if driver not in driver_no_cruising_time.keys():
                driver_no_cruising_time[driver] = [set(), set()]
            if isinstance(time[driver][0], list):
                record = time[driver]
                matched_num += 1
                for single_record in record:
                    if single_record[-2] == 1.0:
                        for second in range(i*time_interval+36000, int(single_record[-1])):
                            driver_no_cruising_time[driver][0].add(second)
                        for second in range(int(single_record[-1]), int(record[-1][-1])):
                            driver_no_cruising_time[driver][1].add(second)
                        break
        matched_rate_time[i*time_interval+36000] = matched_num/order_num_time[i*time_interval+36000]
    driver_state_info = np.array([])
    for key in matched_rate_time.keys():
        cruising_count = 0
        for driver in driver_no_cruising_time.keys():
            if key not in driver_no_cruising_time[driver][1]:
                cruising_count += 1
        driver_state_info = np.append(driver_state_info, cruising_count)
    order_info = np.array(list(time_to_order.values()))
    return order_info.sum(), matched_num, matched_num/order_info.sum(), order_info.mean(), order_info.max(), driver_state_info.mean()


def calculate_metrics_passenger_by_time(record_path):
    records = pickle.load(open(record_path, 'rb'))
    order = pickle.load(open('./input/order.pickle', 'rb'))
    order_to_time = {}
    prematching_time = {}
    for i in range(36000, 79200, 5):
        for j in range(0, 5):
            if (i + j) in order.keys():
                for single_order in order[i+j]:
                    order_to_time[single_order[0]] = i

    prematching_time_temp_list = []
    postmatching_time_temp_list = []
    tripmatching_time_temp_list = []
    for i, time in enumerate(tqdm(records, desc="generate passenger info")):
        temp_pre = []
        temp_post = []
        temp_trip = []
        if prematching_time_temp_list == []:
            prematching_time[i*5+36000] = [0, 0, 0]
        else:
            if len(prematching_time_temp_list) < 360:
                if (sum(len(i) for i in prematching_time_temp_list)) == 0:
                    prematching_time[i * 5 + 36000] = [0]
                else:
                    prematching_time[i*5+36000] = [sum(sum(i) for i in prematching_time_temp_list)/(sum(len(i) for i in prematching_time_temp_list))]
                if (sum(len(i) for i in postmatching_time_temp_list)) == 0:
                    prematching_time[i * 5 + 36000].append(0)
                else:
                    prematching_time[i*5+36000].append(sum(sum(i) for i in postmatching_time_temp_list)/(sum(len(i) for i in postmatching_time_temp_list)))
                if (sum(len(i) for i in tripmatching_time_temp_list)) == 0:
                    prematching_time[i * 5 + 36000].append(0)
                else:
                    prematching_time[i*5+36000].append(sum(sum(i) for i in tripmatching_time_temp_list)/(sum(len(i) for i in tripmatching_time_temp_list)))
            else:
                prematching_time[i * 5 + 36000] = [
                    sum(sum(i) for i in prematching_time_temp_list[1:]) / (sum(len(i) for i in prematching_time_temp_list[1:]))]
                prematching_time[i * 5 + 36000].append(sum(sum(i) for i in postmatching_time_temp_list[1:]) / (
                    sum(len(i) for i in postmatching_time_temp_list[1:])))
                prematching_time[i * 5 + 36000].append(sum(sum(i) for i in tripmatching_time_temp_list[1:]) / (
                    sum(len(i) for i in tripmatching_time_temp_list[1:])))
                prematching_time_temp_list = prematching_time_temp_list[1:]
                postmatching_time_temp_list = postmatching_time_temp_list[1:]
                tripmatching_time_temp_list = tripmatching_time_temp_list[1:]
        for driver in time:
            if isinstance(time[driver][0], list):
                record = time[driver]
                matching_time = record[0][-1]
                pickup_end_time = 79200
                for single_record in record:
                    if single_record[-2] == 1:
                        pickup_end_time = min(single_record[-1], pickup_end_time)
                        break
                temp_pre.append(matching_time - order_to_time[record[0][2]])
                temp_post.append(pickup_end_time - matching_time)
                temp_trip.append(record[-1][-1] - pickup_end_time)
        prematching_time_temp_list.append(temp_pre)
        postmatching_time_temp_list.append(temp_post)
        tripmatching_time_temp_list.append(temp_trip)

    data = np.array(list(prematching_time.values()))
    return data[:, 0].mean(), data[:, 1].mean(), data[:, 2].mean()


def calculate_metrics_passenger(record_path):
    records = pickle.load(open(record_path, 'rb'))
    order = pickle.load(open('./input/order.pickle', 'rb'))
    order_to_time = {}
    for i in range(36000, 79200, 5):
        for j in range(0, 5):
            if (i + j) in order.keys():
                for single_order in order[i+j]:
                    order_to_time[single_order[0]] = i

    data = []
    for i, time in enumerate(tqdm(records, desc="generate passenger info")):
        for driver in time:
            temp_data = []
            if isinstance(time[driver][0], list):
                record = time[driver]
                matching_time = record[0][-1]
                pickup_end_time = 79200
                for single_record in record:
                    if single_record[-2] == 1:
                        pickup_end_time = min(single_record[-1], pickup_end_time)
                        print(pickup_end_time)
                        break
                temp_data.append(matching_time - order_to_time[record[0][2]])
                temp_data.append(pickup_end_time - matching_time)
                print(pickup_end_time - matching_time)
                temp_data.append(record[-1][-1] - pickup_end_time)
                data.append(temp_data)
    data = np.array(data)
    return data[:, 0].mean(), data[:, 1].mean(), data[:, 2].mean()


def generate_simulator_evaluation_data(save_dir):

    # print("订单总数目：", count)
    result_path = './final_experiment/ma_rg_cruise=True/'
    driver_dir_list = os.listdir(result_path)
    # driver_dir_list = [item for item in driver_dir_list if item.startswith('driver')]
    for driver_dir in driver_dir_list:
        driver_num = driver_dir.split('_')[2]
        sample_dir_list = os.listdir(result_path+driver_dir)
        # sample_dir_list = [item for item in sample_dir_list if item.startswith('sample')]
        for sample_dir in sample_dir_list:
            sample_frac = float(sample_dir.split('_')[2])
            record_path = result_path+driver_dir+'/'+sample_dir+'/records'
            record_file_list = os.listdir(record_path)
            for record_file in record_file_list:
                record_file_path = record_path+'/'+record_file
                time_interval = record_file.split('.')[-2].split('_')[-1]
                matching_time, pickup_time, trip_time = calculate_metrics_passenger(record_file_path)
                total_requests, matched_requests, matching_rate, mean_waiting_orders, max_waiting_orders, vacant_vehicles = calculate_metrics(record_file_path, int(time_interval), sample_frac)
                result = {'fleet_size': int(driver_num), 'total_time': 43200, 'total_requests': total_requests, 'speed': 6.33,
                          'matched_requests': matched_requests, 'matching_rate': matched_requests/43200, 'matching_time': matching_time,
                          'pickup_time': pickup_time, 'trip_time': trip_time, 'effective_orders_total_waiting_time': matching_time+pickup_time,
                          'mean_waiting_orders': mean_waiting_orders, 'max_waiting_orders': max_waiting_orders, 'vacant_vehicles': vacant_vehicles
                          }
                file_name = '/orders_' + str(sample_frac) + '_drivers_' + str(driver_num) + '_record.pickle'
                pickle.dump(result, open(save_dir+file_name, 'wb'))


if __name__ == '__main__':
    # path = '../evaluation/simulator_evaluation/Results/orders_0.2_drivers_100_record.pickle'
    # path = './new_experiment/rg_rg_cruise=True/driver_num_200/sample_frac_0.1/records/records_max_' \
    #        'distance_2.0_time_interval_5.pickle'
    # file = pickle.load(open(path, 'rb'))
    # print(file)
    # calculate_metrics_passenger(path)
    save_dir = './Result/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    generate_simulator_evaluation_data(save_dir)
    # for order in range(1, 11):
    #     info = pickle.load(open('./Result/orders_'+str(order/10)+'_drivers_1200_record.pickle', 'rb'))
    #     print(info)
    #     print(info['matching_rate'])
    #     info = pickle.load(open('../evaluation/simulator_evaluation/Results/orders_' + str(round(order / 10*6.2, 2)) + '_drivers_1200_record.pickle', 'rb'))
    #     print(info)
    #     print(info['matching_rate'])