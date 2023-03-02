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
import matplotlib.pylab as plt
import pylab
from datetime import datetime
RATIO_NOISE = 0.02
from numba import jit, vectorize, int64

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


# @jit(nopython=True)
def driver_decision(distance, reward, lr_model):
    """

    :param reward: numpyarray, price of order
    :param distance: numpyarray, distance between current order to all drivers
    :param numpyarray: n, price of order
    :return: pandas.DataFrame, the probability of drivers accept the order.
    """
    print(distance.shape)
    r_dis, c_dis = distance.shape
    temp_ = np.dstack((distance,reward)).reshape(-1,2)
    result = lr_model.predict_proba(temp_).reshape(r_dis,c_dis,2)
    result = np.delete(result,0,axis=2)
    result = np.squeeze(result)
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
def dispatch_broadcasting(order_driver_info, dis_array, lr_model, mlp_model):
    """

    :param order_driver_info: the information of drivers and orders
    :param broadcasting_scale: the radius of order broadcasting
    :return: matched driver order pair
    """
    # print("=======matching=====")
    start_time = datetime.now()
    columns_name = ['origin_lng', 'origin_lat', 'order_id', 'reward_units', 'origin_grid_id', 'driver_id',
                    'pick_up_distance']

    order_driver_info = pd.DataFrame(order_driver_info, columns=columns_name)

    # print(order_driver_info)
    # id of orders and drivers
    id_order = order_driver_info['order_id'].unique()
    id_driver = order_driver_info['driver_id'].unique()

    # num of orders and drivers
    num_order = order_driver_info['order_id'].nunique()
    num_driver = order_driver_info['driver_id'].nunique()
    dis_array = np.array(dis_array,dtype='float32')
    distance_driver_order = dis_array.reshape(num_order, num_driver)
    price_array = np.array(order_driver_info['reward_units'],dtype='float32').reshape(num_order, num_driver)
    order_lng_array = np.array(order_driver_info['origin_lng']).reshape(num_order, num_driver)
    order_lat_array = np.array(order_driver_info['origin_lat']).reshape(num_order, num_driver)
    order_grid_id_array = np.array(order_driver_info['origin_grid_id']).reshape(num_order, num_driver)
    # for i in range(num_driver):
    #     for j in range(num_order):
    #         distance_driver_order[i, j] = order_driver_info.loc[(order_driver_info['order_id'] == id_order[j]) & (
    #                     order_driver_info['driver_id'] == id_driver[i]), 'pick_up_distance'].values[0]
    # print("half")
    time_dict = {'evening':[1,0,0,0],'midnight':[0,1,0,0],'morning':[0,0,1,0],'other':[0,0,0,1]}
    driver_decision_info = driver_decision(distance_driver_order, price_array, lr_model)
    driver_decision_time = datetime.now()
    print(f"driver decision time:{(driver_decision_time - start_time).seconds}")
    # Distribution_graph(distance_driver_order, price_order, lr_model)     #distuibution graphre
    # randomly decide whether the drivers pick the order or not
    # radius = env_params['broadcasting_scale']
    # radius_list = [1, 2, 3, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10]
    for i in range(num_order):
        for j in range(num_driver):
            # temp_radius_list = []
            # for temp_rad in radius_list:
            #     temp_radius_list.append(mlp_model.predict([[temp_rad, order_lat_array[i,j], order_lng_array[i,j], price_array[i,j], distance_driver_order[i,j]]]))
            # radius = radius_list[temp_radius_list.index(max(temp_radius_list))]

            # print([order_grid_id_array[i,j], price_array[i,j], distance_driver_order[i,j]]+time_dict[env_params['time_period']])
            # radius = mlp_model.predict([[order_grid_id_array[i,j], price_array[i,j], distance_driver_order[i,j]]+time_dict[env_params['time_period']]])
            radius = 1.5
            if distance_driver_order[i, j] > radius:
                driver_decision_info[i, j] = 0  # delete drivers further than broadcasting_scale
    driver_decision_info_time = datetime.now()
    print(f"generate driver decision info time:{(driver_decision_info_time - driver_decision_time).seconds}")
    # driver_pick_flag = np.zeros([num_order, num_driver], dtype=int)
    # max_index_dec = np.argmax(driver_decision_info, axis=0)
    # max_dec = np.max(driver_decision_info, axis=0)
    random.seed(10)
    temp_random = np.random.random((num_order,num_driver))
    driver_pick_flag = (driver_decision_info>temp_random)+0
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
            temp_num = generate_random_num(len(temp_line)-1)
            # temp_num = 1
            driver_pick_flag[:, temp_line[temp_num, 0]] = 0
            row[:] = 0
            row[temp_line[temp_num,0]] = 1
            driver_pick_flag[index,:] = row
            driver_pick_flag[index+1:,temp_line[temp_num,0]] = 0

        index += 1

    driver_pick_flag_time = datetime.now()
    print(f"generate driver pick up flag time:{(driver_pick_flag_time - driver_decision_info_time).seconds}")

    matched_pair = np.argwhere(driver_pick_flag == 1)
    matched_dict = {}
    for item in matched_pair:
        matched_dict[id_order[item[0]]] = [id_driver[item[1]],price_array[item[0],item[1]],distance_driver_order[item[0],item[1]]]
        driver_id_list.append(id_driver[item[1]])
        order_id_list.append(id_order[item[0]])
        reward_list.append(price_array[item[0],item[1]])
        pick_up_distance_list.append(distance_driver_order[item[0],item[1]])
    # print("order_id_list",order_id_list)
    # print("id_order",id_order.tolist())
    # print("reward_list",reward_list)
    # print("distance_list",pick_up_distance_list)
    result = []
    for item in id_order.tolist():
        if item in matched_dict:

            result.append([item, matched_dict[item][0], matched_dict[item][1], matched_dict[item][2]])
    # print("the result is",result)
    # return result
    # save result
    # result = [[27787, '70', 20.143628131236394, 2.6983672848835987], [27786, '111', 20.143628131236394, 2.635975811541658], [27783, '151', 56.71625107671233, 0.8263673363455498], [27782, '317', 12.378608416318077, 2.0680143652281653], [27788, '408', 7.917412361884604, 3.3488798447393324], [27785, '412', 7.917412361884604, 1.753299364346717], [27784, '426', 7.917412361884604, 2.3291334114665077]]

    # order_id_list = id_order.tolist()
    # if len(driver_id_list) != 0:
    #     for i in range(len(driver_id_list)):
    #         result.append([order_id_list[i], driver_id_list[i], reward_list[i], pick_up_distance_list[i]])
    # else:
    #     result = []

    print(result)
    return result
    # result = []
    # # order_id_list = id_order.tolist()
    # driver_id_list = list(pd.Series(driver_id_list).unique())
    # # reward_list = list(order_driver_info['reward_units'].unique())
    # # pick_up_distance_list = list(order_driver_info['pick_up_distance'].unique())
    #
    # if min(len(driver_id_list), len(order_id_list), len(reward_list), len(pick_up_distance_list)) != 0:
    #     # print("\nloop_num =", min(len(driver_id_list), len(order_id_list)))
    #     # print("num_1", len(order_id_list))
    #     # print("num_2", len(driver_id_list))
    #     # print("num_3", len(reward_list))
    #     # print("num_4", len(pick_up_distance_list))
    #
    #     for i in range(min(len(driver_id_list), len(order_id_list), len(reward_list), len(pick_up_distance_list))):
    #         result.append([order_id_list[i], driver_id_list[i], reward_list[i], pick_up_distance_list[i]])
    # else:
    #     result = []
    # print("the temp result is ")
    # print(result)
    # return result