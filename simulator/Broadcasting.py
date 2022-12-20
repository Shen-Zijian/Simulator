import numpy as np
import pandas as pd
from numpy import *
import pickle
from copy import deepcopy

import main
import utilities
from config import env_params
import random
from datetime import datetime
import matplotlib as plt
import pylab

RATIO_NOISE = 0.02


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


def driver_decision(distance, reward):
    """

    :param reward: numpyarray, price of order
    :param distance: numpyarray, distance between current order to all drivers
    :param numpyarray: n, price of order
    :return: pandas.DataFrame, the probability of drivers accept the order.
    """
    r_dis, c_dis = distance.shape
    r_reward, c_reward = reward.shape  # get size of info
    result = np.zeros([r_dis, c_dis])
    # prob_1 = z_score(distance + RATIO_NOISE*np.random.normal(loc=distance.mean(), scale=1.0, size=(r_dis, c_dis)))
    # prob_2 = z_score(reward + RATIO_NOISE*np.random.normal(loc=reward.mean(), scale=1.0, size=(r_reward, c_reward)))
    prob_1 = distance
    prob_2 = reward
    r = 0.5  # 乘法+调参
    for i in range(r_dis):
        for j in range(c_dis):
            result[i, j] = r * (1 - prob_1[i][j]) + (1 - r) * prob_2[j][0]
            result[i, j] = sigmoid(result[i, j])
        # calculate probability of drivers' decision
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
    plot2 = pylab.plot(x_list, y_pred, 'r', label=name, color=randomcolor())
    pylab.title('Distribution')
    pylab.xlabel('Price/Distance')
    pylab.ylabel('Probability')
    pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))


def Distribution_graph(distance_array, price_array):
    PRICE_RATIO = 3.0
    Price = 2.0
    DIS_RATIO = 3.0
    Dis = 2.0
    prob_driver_array = driver_decision(distance_array, price_array)
    prob_driver_array_1 = driver_decision(distance_array, PRICE_RATIO * price_array)
    prob_driver_array_2 = driver_decision(distance_array, 2 * PRICE_RATIO * price_array)
    # prob_driver_array = driver_decision(distance_array, price_array)
    # prob_driver_array_1 = driver_decision(DIS_RATIO * distance_array, price_array)
    # prob_driver_array_2 = driver_decision(2 * DIS_RATIO * distance_array,  price_array)
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
        price_distrubute_y_axis_1.append(np.mean(prob_driver_array_1, axis=0)[price_sort[i][0]])
        price_distrubute_y_axis_2.append(np.mean(prob_driver_array_2, axis=0)[price_sort[i][0]])
    # Plotting(price_distrubute_x_axis, price_distrubute_y_axis,'dis=%.2f' % Dis)
    # Plotting(price_distrubute_x_axis, price_distrubute_y_axis_1,'dis=%.2f'% (DIS_RATIO * Dis))
    # Plotting(price_distrubute_x_axis, price_distrubute_y_axis_2,'dis=%.2f'% (DIS_RATIO * Dis * 2))

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
            list_dict_prob_1[distance_array[i, j]] = prob_driver_array_1[i, j]
            list_dict_prob_2[distance_array[i, j]] = prob_driver_array_2[i, j]
    dis_sort = sorted(list_dict_prob.items(), key=lambda kv: (kv[1], kv[0]))
    dis_sort_1 = sorted(list_dict_prob_1.items(), key=lambda kv: (kv[1], kv[0]))
    dis_sort_2 = sorted(list_dict_prob_2.items(), key=lambda kv: (kv[1], kv[0]))
    for i in range(len(list_dict_prob)):
        distance_distribute_x_axis.append(dis_sort[i][0])
        distance_distribute_y_axis.append(dis_sort[i][1])
        distance_distribute_y_1_axis.append(dis_sort_1[i][1])
        distance_distribute_y_2_axis.append(dis_sort_2[i][1])
    Plotting(distance_distribute_x_axis, distance_distribute_y_axis, 'price=%.2f' % Price)
    Plotting(distance_distribute_x_axis, distance_distribute_y_1_axis, 'price=%.2f' % (PRICE_RATIO * Price))
    Plotting(distance_distribute_x_axis, distance_distribute_y_2_axis, 'price=%.2f' % (PRICE_RATIO * Price * 2))
    pylab.show()
    return


def generate_random_num(length):
    if length < 1:
        res = 0
    else:
        res = random.randint(0, length)
    return res


def dispatch_broadcasting(order_driver_info):
    """

    :param order_driver_info: the information of drivers and orders
    :param broadcasting_scale: the radius of order broadcasting
    :return: matched driver order pair
    """

    columns_name = ['order_id', 'driver_id', 'reward_units', 'order_driver_flag']

    order_driver_info = pd.DataFrame(order_driver_info, columns=columns_name)
    # print(order_driver_info)
    # id of orders and drivers
    id_order = order_driver_info['order_id'].unique()
    id_driver = order_driver_info['driver_id'].unique()

    # num of orders and drivers
    num_order = order_driver_info['order_id'].nunique()
    num_driver = order_driver_info['driver_id'].nunique()

    # price of orders
    price_order = []
    for i in range(num_order):
        line_index = np.flatnonzero(order_driver_info['order_id'] == id_order[i])[0]
        # price_order.append(2.0)
        price_order.append(order_driver_info.at[line_index, 'reward_units'])
    price_order = np.array(price_order)
    price_order.shape = (num_order, 1)

    # distance of driver and order
    distance_driver_order = np.empty([num_driver, num_order])

    # get coordinate of drivers and orders
    coord_driver = utilities.get_coordinate_from_nodeId(id_driver)
    coord_order = utilities.get_coordinate_from_nodeId(id_order)
    coord_driver = np.array(coord_driver)
    coord_order = np.array(coord_order)

    # get the distance between driver and order
    for i in range(num_driver):
        for j in range(num_order):
            # distance_driver_order[i, j] = 2.0
            distance_driver_order[i, j] = utilities.distance(coord_driver[i], coord_order[j])

    driver_decision_info = driver_decision(distance_driver_order, price_order)
    #Distribution_graph(distance_driver_order, price_order)
    # randomly decide whether the drivers pick the order or not
    for i in range(num_driver):
        for j in range(num_order):
            if distance_driver_order[i][j] > env_params['broadcasting_scale']:
                driver_decision_info[i, j] = 0  # delete drivers further than broadcasting_scale
    driver_pick_flag = np.zeros((num_driver, num_order), dtype=int)
    for i in range(num_driver):
        for j in range(num_order):
            rand_num = random.random()
            if rand_num <= driver_decision_info[i, j]:
                driver_pick_flag[i][j] = 1
            else:
                driver_pick_flag[i][j] = 0
    driver_id_list = []
    for i in range(num_order):
        temp_line = list(np.where(driver_pick_flag[:, i] == 1)[0])
        temp_num = generate_random_num(len(temp_line) - 1)
        if len(temp_line) != 0:
            driver_id_list.append(id_driver[temp_line[temp_num]])
            driver_pick_flag[temp_line[temp_num],1:] = 0
        # print("i =", i)
        # print(pd.DataFrame(driver_pick_flag))
        # print(driver_id_list)

    # save result
    result = []
    order_id_list = id_order.tolist()
    driver_id_list = list(pd.Series(driver_id_list).unique())
    reward_list = list(order_driver_info['reward_units'].unique())
    order_driver_flag_list = list(order_driver_info['order_driver_flag'].unique())
    if min(len(driver_id_list), len(order_id_list),len(reward_list),len(order_driver_flag_list)) != 0:
        # print("\nloop_num =", min(len(driver_id_list), len(order_id_list)))
        # print("num_1", len(order_id_list))
        # print("num_2", len(driver_id_list))
        # print("num_3", len(reward_list))
        # print("num_4", len(order_driver_flag_list))
        for i in range(min(len(driver_id_list), len(order_id_list),len(reward_list),len(order_driver_flag_list))):
            result.append([order_id_list[i], driver_id_list[i], reward_list[i], order_driver_flag_list[i]])
    else:
        result = []
    return result
