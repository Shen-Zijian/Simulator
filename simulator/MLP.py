from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import pandas as pd
import pickle
import numpy as np
import json
import pandas as pd
import time
import seaborn as sns
pd.options.mode.chained_assignment = None
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
gps_1 = pd.read_csv('/home/shenzijian/etaxi/extra.csv')
gps_2 = pd.read_csv('/home/shenzijian/etaxi/extra_1_360059998_final.csv')
gps_3 = pd.read_csv('/home/shenzijian/etaxi/extra_2_377_059_998_final.csv')
gps_4 = pd.read_csv('/home/shenzijian/etaxi/extra_3_380000000_final.csv')
gps_dist_list = []
order_list = []
def load_cancel(file_name):
    info_pd = pd.read_csv(file_name,'r')
    column = ''
    for i in list(info_pd.columns):
        column+= i
    column = column.split(',')
    column = list(filter(None, column))
    info_pd.columns = column
    line_num = 0
    for line in info_pd['id']:
        line = line.split(',')
        line = list(filter(None, line))
        for i in range(len(column)):
            info_pd[column[i]][line_num] = line[i]
        line_num += 1
    return info_pd

def distance_array(coord_1, coord_2):
    """
    :param coord_1: array of coordinate
    :type coord_1: numpy.array
    :param coord_2: array of coordinate
    :type coord_2: numpy.array
    :return: the array of manhattan distance of these two-point pair
    :rtype: numpy.array
    """
    coord_1 = coord_1.astype(float)
    coord_2 = coord_2.astype(float)
    coord_1_array = np.radians(coord_1)
    coord_2_array = np.radians(coord_2)
    dlon = np.abs(coord_2_array[:, 0] - coord_1_array[:, 0])
    dlat = np.abs(coord_2_array[:, 1] - coord_1_array[:, 1])
    r = 6371

    alat = np.sin(dlat / 2) ** 2
    clat = 2 * np.arctan2(alat ** 0.5, (1 - alat) ** 0.5)
    lat_dis = clat * r

    alon = np.sin(dlon / 2) ** 2
    clon = 2 * np.arctan2(alon ** 0.5, (1 - alon) ** 0.5)
    lon_dis = clon * r

    manhattan_dis = np.abs(lat_dis) + np.abs(lon_dis)

    return manhattan_dis

def distance(coord_1, coord_2):
    """
    :param coord_1: the coordinate of one point
    :type coord_1: tuple -- (latitude,longitude)
    :param coord_2: the coordinate of another point
    :type coord_2: tuple -- (latitude,longitude)
    :return: the manhattan distance between these two points
    :rtype: float
    """
    lat1, lon1 = coord_1
    lat2, lon2 = coord_2
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = abs(lon2 - lon1)
    dlat = abs(lat2 - lat1)
    r = 6371
    alat = np.sin(dlat / 2) ** 2
    clat = 2 * np.arctan2(alat ** 0.5, (1 - alat) ** 0.5)
    lat_dis = clat * r
    alon = np.sin(dlon / 2) ** 2
    clon = 2 * np.arctan2(alon ** 0.5, (1 - alon) ** 0.5)
    lon_dis = clon * r
    manhattan_dis = abs(lat_dis) + abs(lon_dis)

    return manhattan_dis

def save_obj(obj, name ):
    with open('/home/shenzijian/下载/Transpotation_Simulator-1/simulator/output3/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    obj = pickle.load(open('/home/shenzijian/下载/Transpotation_Simulator-1/simulator/output3/' + name + '.pkl', 'rb'))
    return obj

def cal_gps_to_driver(driver_id_list,order_list):
    # driver_gps_dict = {}
    # for item in driver_id_list:
    #     driver_gps_dict[item] = []
    # print("======Processing======")
    # for index, row in gps_1.iterrows():
    #     if row['driver_id'] in driver_id_list:
    #         driver_gps_dict[row['driver_id']].append([row['latitude'],row['longitude'],row['created_at']])
    # save_obj(driver_gps_dict,'driver_gps_dict')
    manhatten_dis_list = []
    driver_gps_dict = load_obj('driver_gps_dict')
    for index, row in order_list.iterrows():
        temp_gps_list = []
        print(index)
        for item in driver_gps_dict[row['driver_id']]:
            if (time.mktime(time.strptime(item[2], '%Y-%m-%d %H:%M:%S')) > time.mktime(time.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S')))&(time.mktime(time.strptime(item[2], '%Y-%m-%d %H:%M:%S')) < time.mktime(time.strptime(row['pick_up_time'], '%Y-%m-%d %H:%M:%S'))):
                temp_gps_list.append([item[0],item[1]])
        gps_dist = 0.0
        if len(temp_gps_list) > 1:
            for i in range(len(temp_gps_list)):
                if i-1 < 0:
                    vec1 = np.array([temp_gps_list[1],temp_gps_list[1]])
                    vec2 = np.array([temp_gps_list[0],temp_gps_list[0]])
                else:
                    vec1 = np.array([temp_gps_list[i],temp_gps_list[i]])
                    vec2 = np.array([temp_gps_list[i-1],temp_gps_list[i-1]])
                gps_dist += distance(vec1,vec2)
            manhatten_dis_list.append(gps_dist)
    save_obj(manhatten_dis_list, 'manhatten_dis_list')
    sns.set()  # 切换到sns的默认运行配置
    sns.distplot(manhatten_dis_list)
    plt.show()
    return driver_gps_dict

def cal_dist_error(driver_id, date, pickup_distance,pick_up_at):
    create_time = time.mktime(time.strptime(date, '%Y-%m-%d %H:%M:%S'))
    pick_up_time = time.mktime(time.strptime(pick_up_at, '%Y-%m-%d %H:%M:%S'))
    max_gps_1 = time.mktime(time.strptime(gps_1['created_at'][gps_1.shape[0] - 1], '%Y-%m-%d %H:%M:%S'))
    min_gps_1 = time.mktime(time.strptime(gps_1['created_at'][0], '%Y-%m-%d %H:%M:%S'))
    max_gps_2 = time.mktime(time.strptime(gps_2['created_at'][gps_2.shape[0] - 1], '%Y-%m-%d %H:%M:%S'))
    min_gps_2 = time.mktime(time.strptime(gps_2['created_at'][0], '%Y-%m-%d %H:%M:%S'))
    max_gps_3 = time.mktime(time.strptime(gps_3['created_at'][gps_3.shape[0] - 1], '%Y-%m-%d %H:%M:%S'))
    min_gps_3 = time.mktime(time.strptime(gps_3['created_at'][0], '%Y-%m-%d %H:%M:%S'))
    max_gps_4 = time.mktime(time.strptime(gps_4['created_at'][gps_4.shape[0] - 1], '%Y-%m-%d %H:%M:%S'))
    min_gps_4 = time.mktime(time.strptime(gps_4['created_at'][0], '%Y-%m-%d %H:%M:%S'))
    gps_latitude_list = []
    gps_longitude_list = []
    if (create_time > min_gps_1) & (create_time < max_gps_1):
        trip_interval = pick_up_time - create_time
        print("gps_1-",trip_interval)
        for index, row in gps_1.iterrows():
            gps_time = time.mktime(time.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S'))
            time_interval = abs(gps_time - create_time)
            if (row['driver_id'] == driver_id) & (gps_time <= pick_up_time) & (gps_time >= create_time):
            # if (row['driver_id'] == driver_id) & (gps_time > create_time) & (gps_time < pick_up_time):
                gps_longitude_list.append(row['longitude'])
                gps_latitude_list.append(row['latitude'])
    elif (create_time > min_gps_2) & (create_time < max_gps_2):
        print("gps_2")
        for index, row in gps_2.iterrows():
            time_interval = abs(create_time - time.mktime(time.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S')))
            if (row['driver_id'] == driver_id) & (time_interval < 1800):
                gps_longitude_list.append(row['longitude'])
                gps_latitude_list.append(row['latitude'])
    elif (create_time > min_gps_3) & (create_time < max_gps_3):
        print("gps_3")
        for index, row in gps_3.iterrows():
            time_interval = abs(create_time - time.mktime(time.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S')))
            if (row['driver_id'] == driver_id) & (time_interval < 1800):
                gps_longitude_list.append(row['longitude'])
                gps_latitude_list.append(row['latitude'])
    elif (create_time > min_gps_4) & (create_time < max_gps_4):
        print("gps_4")
        for index, row in gps_4.iterrows():
            time_interval = abs(create_time - time.mktime(time.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S')))
            if (row['driver_id'] == driver_id) & (time_interval < 1800):
                gps_longitude_list.append(row['longitude'])
                gps_latitude_list.append(row['latitude'])
    else:
        return 0.0
    gps_distance = 0
    for i in range(len(gps_latitude_list)):
        if i-1 < 0:
            vec1 = np.array([gps_latitude_list[1],gps_longitude_list[1]])
            vec2 = np.array([gps_latitude_list[0],gps_longitude_list[0]])
        else:
            vec1 = np.array([gps_latitude_list[i],gps_longitude_list[i]])
            vec2 = np.array([gps_latitude_list[i-1],gps_longitude_list[i-1]])
        gps_distance += distance(vec1,vec2)

        #print(gps_latitude_list[i],gps_longitude_list[i],gps_distance)
        #print(vec1-vec2)
    print(gps_distance,pickup_distance)
    distance_error = abs(gps_distance*1000 - pickup_distance)/pickup_distance
    # gps_dist_list.append(gps_distance*1000)
    # order_list.append(distance)
    # x_list = range(len(gps_dist_list))
    # plt.plot(x_list, gps_dist_list, label='gps_distance', color='red')
    # plt.plot(x_list, order_list, label='order_distance', color='blue')
    # plt.legend(loc="best")
    # plt.show()
    return distance_error
def test():
    gps_latitude_list = [115.1850009, 114.1851408]
    gps_longitude_list = [22.2636314, 22.2638548,]
    vec1 = np.array([gps_latitude_list[1], gps_longitude_list[1]])
    vec2 = np.array([gps_latitude_list[0], gps_longitude_list[0]])
    gps_distance = np.linalg.norm(vec1 - vec2)
    print(gps_distance)
def load_extra():
    file_1 = pd.read_csv('/home/shenzijian/etaxi/extra.csv')
    file_2 = pd.read_csv('/home/shenzijian/etaxi/extra_1_360059998_final.csv')
    file_3 = pd.read_csv('/home/shenzijian/etaxi/extra_2_377_059_998_final.csv')
    file_4 = pd.read_csv('/home/shenzijian/etaxi/extra_3_380000000_final.csv')

    # file_1 = file_1.append(file_2)
    # file_1 = file_1.append(file_3)
    # file_1 = file_1.append(file_4)

    for i in range(len(file_1)):
        file_1['created_date'] = str(file_1['created_at'][1]).split(' ', 1 )[0]
        file_1['created_time'] = str(file_1['created_at'][1]).split(' ', 1 )[1]
    file_1 = file_1.drop(['created_at'])
    print(file_1)
def load_json(file_name):
    # Use a breakpoint in the code line below to debug your script.
    with open(file_name, 'r') as load_f:
        load_list = json.load(load_f)
    info_list = list(load_list[0].keys())
    return load_list,info_list


def data_cleaning():
    trip_info, trip_title = load_json("/home/shenzijian/etaxi/trip.json")
    trip_info = pd.DataFrame(trip_info, columns=trip_title)
    trip_gps = pd.read_csv("/home/shenzijian/etaxi/extra.csv")
    #print(trip_gps[])
    print(trip_gps.loc[trip_gps['driver_id'] == 10566],('driver_id','created_at'))
    print(trip_title)
    #print(trip_info['driver_id'])
    #print(trip_info['route_distance_to_passenger'])

    # print(trip_info.loc[(trip_info['route_distance_to_passenger'] == trip_info['route_distance_to_passenger'])&(trip_info['driver_id']==11018 ),('driver_id','route_distance_to_passenger','created_at')])
    driver_with_distance = trip_info.loc[(trip_info['route_distance_to_passenger'] == trip_info['route_distance_to_passenger'])&(trip_info['driver_id'] == trip_info['driver_id'])&(trip_info['created_at'] == trip_info['created_at'])&(trip_info['picked_up_at'] == trip_info['picked_up_at']),('driver_id','route_distance_to_passenger','created_at','pick_up_time','picked_up_at')]
    dist_list_distribution = []
    driver_id_list = []
    for index,row in driver_with_distance.iterrows():
        print(row['created_at'],row['picked_up_at'],row['pick_up_time'])
        if row['driver_id'] not in driver_id_list:
            driver_id_list.append(row['driver_id'])
        dist_list_distribution.append(row['route_distance_to_passenger'])
    driver_gps_dict = cal_gps_to_driver(driver_id_list,driver_with_distance)
    # print(driver_gps_dict,driver_with_distance)
    # sns.set()  # 切换到sns的默认运行配置
    #
    # sns.distplot(dist_list_distribution)
    # plt.show()


    print(len(driver_id_list))
    for index,row in driver_with_distance.iterrows():
        if row['driver_id'] in driver_id_list:

            error = cal_dist_error(row['driver_id'],row['created_at'],row['route_distance_to_passenger'],row['picked_up_at'])
            print("error =",error)
    print("END")

    for item in driver_id_list:
        print(item)
        for index,row in trip_gps.iterrows():

            if row['driver_id'] == 5826:
                print(row['driver_id'],row['created_at'])
            #     time_interval = time.mktime(time.strptime(trip_gps['created_at'][index], '%Y-%m-%d %H:%M:%S')) - time.mktime(time.strptime(trip_gps['created_at'][index+1], '%Y-%m-%d %H:%M:%S'))
            # print(time_interval)
        #print(time.mktime(time.strptime(a, '%Y-%m-%d %H:%M:%S')))
    train_set = pd.DataFrame()
    count = 0
    for i in range(len(trip_info)):
        creat_time = time.mktime(time.strptime(trip_info['created_at'][i], '%Y-%m-%d %H:%M:%S'))
        pick_up_time = time.mktime(time.strptime(trip_info['pick_up_time'][i], '%Y-%m-%d %H:%M:%S'))
        # print(pick_up_time,'||',creat_time,'||',trip_info['pick_up_time'][i],'||',trip_info['created_at'][i])
        time_gap = pick_up_time - creat_time
        if (time_gap <= 0) & (time_gap < 3600):
            count += 1
    print(count)
    # train_set['pickup_time'] = trip_info['route_price'] - trip_info['route_price']
    train_set['reward'] = trip_info['route_price']
    # print(load_cancel("/home/shenzijian/etaxi/driver_cancel.csv"))
    # load_extra()
    # print(trip_info)
    # print(load_dict)
    # for keys in load_dict:
    #     print(keys.keys())
    # load_dict['smallberg'] = [8200, {1: [['Python', 81], ['shirt', 300]]}]
    # print(load_dict)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()
    data_cleaning()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

def MLP_nn():
    data = pd.read_csv("train.csv")
    data = data.dropna()

    output = data['reward']
    input = data.drop(['reward','Unnamed: 0'], axis=1)
    print(input)
    x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=0)
    X = x_train
    y = y_train
    scaler = StandardScaler()  # 标准化转换
    scaler.fit(X)  # 训练标准化对象
    X = scaler.transform(X)  # 转换数据集
    clf = MLPRegressor(solver='sgd', alpha=1e-5, hidden_layer_sizes=(2, 1), random_state=1)
    clf.fit(X, y)
    print('预测结果：', clf.predict([[2.601150, 5, 36005, 40.7679674, -73.9682154, 27.7, 8]]))  # 预测某个输入对象
    cengindex = 0
    for wi in clf.coefs_:
        cengindex += 1  # 表示底第几层神经网络。
        print('第%d层网络层:' % cengindex)
        print('权重矩阵维度:', wi.shape)
        print('系数矩阵：\n', wi)
    return clf

