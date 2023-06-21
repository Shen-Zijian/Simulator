import gc

import numpy as np
import pandas as pd
from numpy import *
import pickle
from copy import deepcopy
from operator import itemgetter
import main
import utilities
from config import env_params
import random
from datetime import datetime
import matplotlib.pylab as plt
import pylab
from datetime import datetime

RATIO_NOISE = 0.02
# from numba import jit, vectorize, int64


def sigmoid(x):
    """ softmax function """

    x = np.exp(x) / (1 + np.exp(x))
    return x


def z_score(data):
    """ z_score function """
    mean_data = np.mean(data)
    std_data = np.std(data)
    data = (data - mean_data) / std_data
    return data


def normalization(data):
    """

    :param data: list, the data need to be normalization
    :return: list,normalized data
    """

    scale = data.max() - data.min()
    result = (data - data.min()) / scale
    return result

driver_behaviour_scalar = pickle.load(open('./input/driver_behaviour_scaler.pkl', 'rb'))
# @jit(nopython=True)
def driver_decision(distance, reward, lr_model):
    """

    :param reward: numpyarray, price of order
    :param distance: numpyarray, distance between current order to all drivers
    :param numpyarray: n, price of order
    :return: pandas.DataFrame, the probability of drivers accept the order.
    """
    r_dis, c_dis = distance.shape
    temp_ = np.dstack((distance, reward)).reshape(-1, 2)
    temp_ = driver_behaviour_scalar.transform(temp_)
    result = lr_model.predict_proba(temp_).reshape(r_dis, c_dis, 2)
    result = np.delete(result, 0, axis=2)
    result = np.squeeze(result, axis=2)
    # result = np.zeros([r_dis, c_dis],dtype='float32')
    # # prob_1 = z_score(distance + RATIO_NOISE*np.random.normal(loc=distance.mean(), scale=1.0, size=(r_dis, c_dis)))
    # # prob_2 = z_score(reward + RATIO_NOISE*np.random.normal(loc=reward.mean(), scale=1.0, size=(r_reward, c_reward)))
    # prob_1 = distance
    # prob_2 = reward
    # r = 0.5  # 乘法+调参
    # for i in range(r_dis):
    #     for j in range(c_dis):
    #         # result[i, j] = r * (1 - prob_1[i][j]) + (1 - r) * prob_2[j][0]
    #         # result[i, j] = sigmoid(result[i, j])
    #         result[i, j] = lr_model.predict_proba([[prob_1[i][j], prob_2[i][j]]])[0, 1]
    #     # calculate probability of drivers' decision

    return result


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def Plotting(x_list, y_list, name):
    z1 = np.polyfit(x_list, y_list, 5)  # 曲线拟合，返回值为多项式的各项系数
    p1 = np.poly1d(z1)  # 返回值为多项式的表达式，也就是函数式子
    y_pred = p1(x_list)  # 根据函数的多项式表达式，求解 y
    plot1 = pylab.plot(x_list, y_list, 'o', label='original values', color=randomcolor())
    # plot2 = pylab.plot(x_list, y_pred, 'r', label=name, color=randomcolor())
    pylab.title('Distribution')
    pylab.ylabel('Probability')
    pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))


def Distribution_graph(distance_array, price_array, lr_model):
    PRICE_RATIO = 3.0
    Price = 2.0
    DIS_RATIO = 3.0
    Dis = 2.0
    prob_driver_array = driver_decision(distance_array, price_array, lr_model)
    # prob_driver_array_1 = driver_decision(distance_array, PRICE_RATIO * price_array,lr_model)
    # prob_driver_array_2 = driver_decision(distance_array, 2 * PRICE_RATIO * price_array,lr_model)
    # prob_driver_array = driver_decision(distance_array, price_array,lr_model)
    # prob_driver_array_1 = driver_decision(DIS_RATIO * distance_array, price_array,lr_model)
    # prob_driver_array_2 = driver_decision(2 * DIS_RATIO * distance_array, price_array,lr_model)
    price_distrubute_x_axis = []
    price_distrubute_y_axis = []
    price_distrubute_y_axis_1 = []
    price_distrubute_y_axis_2 = []
    list_dict = {}
    for i in range(len(price_array)):
        # line_index = np.flatnonzero(order_driver_info['reward_units'] == float(price_distrubute_x_axis[i]))[0]
        # list_sort.append(order_driver_info.at[line_index, 'order_id'])
        list_dict[i] = float(price_array[i])
    price_sort = sorted(list_dict.items(), key=lambda kv: (kv[1], kv[0]))
    for i in range(len(price_array)):
        price_distrubute_x_axis.append(price_sort[i][1])
        price_distrubute_y_axis.append(np.mean(prob_driver_array, axis=0)[price_sort[i][0]])
        # price_distrubute_y_axis_1.append(np.mean(prob_driver_array_1, axis=0)[price_sort[i][0]])
        # price_distrubute_y_axis_2.append(np.mean(prob_driver_array_2, axis=0)[price_sort[i][0]])
    Plotting(price_distrubute_x_axis, price_distrubute_y_axis, 'dis=%.2f' % Dis)
    pylab.xlabel('Price')
    plt.show()
    # Plotting(price_distrubute_x_axis, price_distrubute_y_axis_1, 'dis=%.2f' % (DIS_RATIO * Dis))
    # Plotting(price_distrubute_x_axis, price_distrubute_y_axis_2, 'dis=%.2f' % (DIS_RATIO * Dis * 2))

    distance_distribute_x_axis = []
    distance_distribute_y_axis = []
    distance_distribute_y_1_axis = []
    distance_distribute_y_2_axis = []
    list_dict_prob = {}
    list_dict_prob_1 = {}
    list_dict_prob_2 = {}
    num_raw, num_col = prob_driver_array.shape
    for i in range(num_raw):
        for j in range(num_col):
            # print("the dis =",distance_array[i, j])
            # print("the prob =",prob_driver_array[i, j])
            list_dict_prob[distance_array[i, j]] = prob_driver_array[i, j]
            # list_dict_prob_1[distance_array[i, j]] = prob_driver_array_1[i, j]
            # list_dict_prob_2[distance_array[i, j]] = prob_driver_array_2[i, j]
    dis_sort = sorted(list_dict_prob.items(), key=lambda kv: (kv[1], kv[0]))
    dis_sort_1 = sorted(list_dict_prob_1.items(), key=lambda kv: (kv[1], kv[0]))
    dis_sort_2 = sorted(list_dict_prob_2.items(), key=lambda kv: (kv[1], kv[0]))
    for i in range(len(list_dict_prob)):
        distance_distribute_x_axis.append(dis_sort[i][0])
        distance_distribute_y_axis.append(dis_sort[i][1])
        # distance_distribute_y_1_axis.append(dis_sort_1[i][1])
        # distance_distribute_y_2_axis.append(dis_sort_2[i][1])
    Plotting(distance_distribute_x_axis, distance_distribute_y_axis, 'price=%.2f' % Price)
    pylab.xlabel('Distance')
    # Plotting(distance_distribute_x_axis, distance_distribute_y_1_axis, 'price=%.2f' % (PRICE_RATIO * Price))
    # Plotting(distance_distribute_xs_axis, distance_distribute_y_2_axis, 'price=%.2f' % (PRICE_RATIO * Price * 2))
    pylab.show()
    return


def generate_random_num(length):
    if length < 1:
        res = 0
    else:
        res = random.randint(0, length)
    return res


# @jit(nopython=True)
def dispatch_broadcasting(order_driver_info, dis_array, lr_model, mlp_model, cur_time,driver_table):
    """

    :param order_driver_info: the information of drivers and orders
    :param broadcasting_scale: the radius of order broadcasting
    :return: matched driver order pair
    """

    start_time = datetime.now()
    columns_name = ['origin_lng', 'origin_lat', 'order_id', 'reward_units', 'origin_grid_id', 'driver_id',
                    'pick_up_distance']
    # list_driver = driver_table['driver_id'].tolist()
    # list_driver_grid_id = driver_table['grid_id'].tolist()
    # driver_grid_dict = dict(zip(list_driver,list_driver_grid_id))
    # print(driver_grid_dict)
    order_driver_info = pd.DataFrame(order_driver_info, columns=columns_name)

    # id of orders and drivers
    id_order = order_driver_info['order_id'].unique()
    id_driver = order_driver_info['driver_id'].unique()

    # num of orders and drivers
    num_order = order_driver_info['order_id'].nunique()
    num_driver = order_driver_info['driver_id'].nunique()
    new_all_requests = order_driver_info
    new_all_requests['time'] = cur_time
    new_all_requests['time_period'] = env_params['time_period']
    new_all_requests['num_wait_requests'] = num_order
    new_all_requests['num_available_drivers'] = num_driver
    if env_params['model_name'] != 'fixed':
        new_all_requests['radius'] = itemgetter(*order_driver_info['origin_grid_id'].values)(env_params['grid_radius_dict'])
    else:
        new_all_requests['radius'] = env_params['broadcasting_scale']
    new_all_requests['match_state'] = 4
    new_all_requests['origin_grid_id'] = order_driver_info['origin_grid_id']
    new_all_requests['pickup_distance'] = dis_array
    # new_all_requests.drop(columns=['origin_lng','origin_lat','order_id','driver_id'],inplace=True)
    dis_array = np.array(dis_array, dtype='float32')
    distance_driver_order = dis_array.reshape(num_order, num_driver)
    price_array = np.array(order_driver_info['reward_units'], dtype='float32').reshape(num_order, num_driver)
    order_grid_id_array = np.array(order_driver_info['origin_grid_id']).reshape(num_order, num_driver)
    radius_array = np.array(new_all_requests['radius']).reshape(num_order, num_driver)

    match_state_array = np.array(order_driver_info['match_state']).reshape(num_order, num_driver)
    # for i in range(num_driver):
    #     for j in range(num_order):
    #         distance_driver_order[i, j] = order_driver_info.loc[(order_driver_info['order_id'] == id_order[j]) & (
    #                     order_driver_info['driver_id'] == id_driver[i]), 'pick_up_distance'].values[0]
    # print("half")
    time_dict = {'evening': [1, 0, 0, 0], 'midnight': [0, 1, 0, 0], 'morning': [0, 0, 1, 0], 'other': [0, 0, 0, 1]}

    driver_decision_info = driver_decision(distance_driver_order, price_array, lr_model)
    driver_decision_time = datetime.now()
    # print(f"driver decision time:{(driver_decision_time - start_time).seconds}")
    # Distribution_graph(distance_driver_order, price_order, lr_model)     #distuibution graphre
    # randomly decide whether the drivers pick the order or not
    # radius = env_params['broadcasting_scale']
    # radius_list = [1, 2, 3, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10]

    # temp_1 = np.dstack((order_grid_id_array, price_array,distance_driver_order,time_dict)).reshape(-1, 2)
    for i in range(num_order):
        for j in range(num_driver):
            # temp_radius_list = []
            # for temp_rad in radius_list:
            #     temp_radius_list.append(mlp_model.predict([[temp_rad, order_lat_array[i,j], order_lng_array[i,j], price_array[i,j], distance_driver_order[i,j]]]))
            # radius = radius_list[temp_radius_list.index(max(temp_radius_list))]

            # print([order_grid_id_array[i,j], price_array[i,j], distance_driver_order[i,j]]+time_dict[env_params['time_period']])
            # radius = mlp_model.predict([[order_grid_id_array[i,j], price_array[i,j], distance_driver_order[i,j]]+time_dict[env_params['time_period']]])
            # radius = 1.5
            if distance_driver_order[i, j] > radius_array[i, j]:
                driver_decision_info[i, j] = 0  # delete drivers further than broadcasting_scale
                match_state_array[i, j] = 2
    driver_decision_info_time = datetime.now()
    # print(f"generate driver decision info time:{(driver_decision_info_time - driver_decision_time).seconds}")

    random.seed(10)
    temp_random = np.random.random((num_order, num_driver))
    # temp_random = 0
    # driver_pick_flag = (driver_decision_info > temp_random) + 0
    # index_1 = np.argwhere(driver_decision_info == 0).tolist()
    # index_2 = np.argwhere(driver_pick_flag == 0).tolist()
    # index_1 = set(tuple(item) for item in index_1)
    # index_2 = set(tuple(item) for item in index_2)
    #
    # set_res = index_2 - index_1
    # index_diff = np.array([list(item) for item in set_res])
    # # print(index_diff)
    # if len(index_diff) != 0:
    #     index_diff = np.array(index_diff)
    #     diff_row = index_diff[:, 0]
    #     diff_col = index_diff[:, 1]
    #     match_state_array[diff_row, diff_col] = 3
    # temp_random = 0.1
    driver_pick_flag = (driver_decision_info > temp_random) + 0
    mask = driver_decision_info < 0.1

    # 使用布尔数组将 driver_decision_info 中小于0.3的值置为0
    driver_decision_info[mask] = 0
    # print(match_state_array)
    # 使用相同的布尔数组将 match_state_array 中对应位置的值修改为3
    match_state_array[mask] = 3

    # for i in range(num_order):
    #     for j in range(num_driver):
    #         temp_random = np.random.random()
    #         if temp_random <= driver_decision_info[i,j]:
    #             driver_pick_flag[i, j] = 1
    #         else:
    #             driver_pick_flag[i, j] = 0

    #         if (i == max_index_dec[j]) & (max_dec[j] != 0):
    #             driver_pick_flag[i, j] = 1
    #         else:
    #             driver_pick_flag[i, j] = 0

    driver_id_list = []
    order_id_list = []
    reward_list = []
    pick_up_distance_list = []
    index = 0
    for row in driver_pick_flag:
        temp_line = np.argwhere(row == 1)
        if len(temp_line) >= 1:
            temp_num = generate_random_num(len(temp_line) - 1)
            # temp_num = 1
            # driver_pick_flag[:, temp_line[temp_num, 0]] = 0
            row[:] = 0
            row[temp_line[temp_num, 0]] = 1
            driver_pick_flag[index, :] = row
            driver_pick_flag[index + 1:, temp_line[temp_num, 0]] = 0

        index += 1

    driver_pick_flag_time = datetime.now()
    # print(f"generate driver pick up flag time:{(driver_pick_flag_time - driver_decision_info_time).seconds}")

    matched_pair = np.argwhere(driver_pick_flag == 1)
    match_state_array[np.where(driver_pick_flag == 1)] = 1
    match_state_array = match_state_array.reshape(-1).tolist()
    new_all_requests['match_state'] = match_state_array
    new_all_requests['num_driver'] = num_driver

    matched_dict = {}
    for item in matched_pair:
        matched_dict[id_order[item[0]]] = [id_driver[item[1]], price_array[item[0], item[1]],
                                           distance_driver_order[item[0], item[1]]]
        driver_id_list.append(id_driver[item[1]])
        order_id_list.append(id_order[item[0]])
        reward_list.append(price_array[item[0], item[1]])
        pick_up_distance_list.append(distance_driver_order[item[0], item[1]])
    # print("order_id_list",order_id_list)
    # print("id_order",id_order.tolist())
    # print("reward_list",reward_list)
    # print("distance_list",pick_up_distance_list)
    result = []
    for item in id_order.tolist():
        if item in matched_dict:
            result.append([item, matched_dict[item][0], matched_dict[item][1], matched_dict[item][2]])
    # print("the result is",result)
    # new_all_requests.to_csv("./experiment/train_grid" + f"_{env_params['time_period']}" + "_model.csv",
    #                     mode='a', index=False, sep=',')

    return result, new_all_requests

