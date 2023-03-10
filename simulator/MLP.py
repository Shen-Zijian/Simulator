from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pylab as plt
import pickle
import numpy as np
import json
import pandas as pd
import time
import seaborn as sns
from sklearn.metrics import classification_report

pd.options.mode.chained_assignment = None
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
gps_1 = pd.read_csv('/home/shenzijian/etaxi/extra.csv')
gps_2 = pd.read_csv('/home/shenzijian/etaxi/extra_1_360059998_final.csv')
gps_3 = pd.read_csv('/home/shenzijian/etaxi/extra_2_377_059_998_final.csv')
gps_4 = pd.read_csv('/home/shenzijian/etaxi/extra_3_380000000_final.csv')
cancel_order = pd.read_csv('/home/shenzijian/etaxi/driver_cancel.csv')

gps_dist_list = []
order_list = []


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


def save_obj(obj, name):
    with open('/home/shenzijian/??????/Transpotation_Simulator-1/simulator/output3/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    obj = pickle.load(open('/home/shenzijian/??????/Transpotation_Simulator-1/simulator/output3/' + name + '.pkl', 'rb'))
    return obj


def cancel_order_identify(driver_with_dist):
    cancel_pair_list = []
    cancel_np = np.empty(shape=(0, 3))
    for index, row in cancel_order.iterrows():
        cancel_pair_list.append([float(row['passenger_id']), row['driver_id']])

    print('===================')

    for index, row in driver_with_dist.iterrows():
        if (row['cancelled_at'] is not None) & (row['final_price'] == row['final_price']):
            temp = np.array([row['route_distance_to_passenger'], row['final_price'], 0.0])
            # print('distance = ',row['route_distance_to_passenger'],'|','price =',row['ride_price'],'|','cancel = ',row['cancelled_at'])

            cancel_np = np.insert(cancel_np, 0, values=temp, axis=0)
            # print('distance = ',row['route_distance_to_passenger'],'|','price =',row['ride_price'],'|','cancel = ',row['cancelled_at'])
    for index, row in driver_with_dist.iterrows():
        if (row['cancelled_at'] is None) & (row['final_price'] == row['final_price']) & (
                row['route_distance_to_passenger'] == row['route_distance_to_passenger']) & (row['final_price'] != 0.0):
            temp = np.array([row['route_distance_to_passenger'], row['final_price'], 1.0])
            # print('distance = ',row['route_distance_to_passenger'],'|','price =',row['ride_price'],'|','cancel = ',row['cancelled_at'])
            cancel_np = np.insert(cancel_np, 0, values=temp, axis=0)
    result = pd.DataFrame(cancel_np, columns=['Pick_up_distance', 'Price', 'Label'])
    result.to_csv('./output3/behavior_data.csv')
    print(result)


def cal_gps_to_driver(driver_id_list, order_list):
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
            if \
                    (time.mktime(time.strptime(item[2], '%Y-%m-%d %H:%M:%S')) > time.mktime(
                        time.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S'))) & (
                            time.mktime(time.strptime(item[2], '%Y-%m-%d %H:%M:%S')) < time.mktime(
                        time.strptime(row['pick_up_time'], '%Y-%m-%d %H:%M:%S'))):
                temp_gps_list.append([item[0], item[1]])
        gps_dist = 0.0
        if len(temp_gps_list) > 1:
            for i in range(len(temp_gps_list)):
                if i - 1 < 0:
                    vec1 = np.array([temp_gps_list[1], temp_gps_list[1]])
                    vec2 = np.array([temp_gps_list[0], temp_gps_list[0]])
                else:
                    vec1 = np.array([temp_gps_list[i], temp_gps_list[i]])
                    vec2 = np.array([temp_gps_list[i - 1], temp_gps_list[i - 1]])
                gps_dist += distance(vec1, vec2)
            manhatten_dis_list.append(gps_dist)
    save_obj(manhatten_dis_list, 'manhatten_dis_list')
    sns.set()  # ?????????sns?????????????????????
    sns.distplot(manhatten_dis_list)
    plt.show()
    return driver_gps_dict


def cal_dist_error(driver_id, date, pickup_distance, pick_up_at):
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
        print("gps_1-", trip_interval)
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
        if i - 1 < 0:
            vec1 = np.array([gps_latitude_list[1], gps_longitude_list[1]])
            vec2 = np.array([gps_latitude_list[0], gps_longitude_list[0]])
        else:
            vec1 = np.array([gps_latitude_list[i], gps_longitude_list[i]])
            vec2 = np.array([gps_latitude_list[i - 1], gps_longitude_list[i - 1]])
        gps_distance += distance(vec1, vec2)

        # print(gps_latitude_list[i],gps_longitude_list[i],gps_distance)
        # print(vec1-vec2)
    print(gps_distance, pickup_distance)
    distance_error = abs(gps_distance * 1000 - pickup_distance) / pickup_distance
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
    gps_longitude_list = [22.2636314, 22.2638548, ]
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
        file_1['created_date'] = str(file_1['created_at'][1]).split(' ', 1)[0]
        file_1['created_time'] = str(file_1['created_at'][1]).split(' ', 1)[1]
    file_1 = file_1.drop(['created_at'])
    print(file_1)


def load_json(file_name):
    # Use a breakpoint in the code line below to debug your script.
    with open(file_name, 'r') as load_f:
        load_list = json.load(load_f)
    info_list = list(load_list[0].keys())
    return load_list, info_list


def data_cleaning():
    trip_info, trip_title = load_json("/home/shenzijian/etaxi/trip.json")
    trip_info = pd.DataFrame(trip_info, columns=trip_title)
    trip_gps = pd.read_csv("/home/shenzijian/etaxi/extra.csv")
    # print(trip_gps[])
    print(trip_gps.loc[trip_gps['driver_id'] == 10566], ('driver_id', 'created_at'))
    print(trip_title)
    # print(trip_info['driver_id'])
    # print(trip_info['route_distance_to_passenger'])

    # print(trip_info.loc[(trip_info['route_distance_to_passenger'] == trip_info['route_distance_to_passenger'])&(trip_info['driver_id']==11018 ),('driver_id','route_distance_to_passenger','created_at')])
    driver_with_distance = trip_info.loc[
        (trip_info['route_distance_to_passenger'] == trip_info['route_distance_to_passenger']) & (
                trip_info['driver_id'] == trip_info['driver_id']) & (
                trip_info['created_at'] == trip_info['created_at']) & (
                trip_info['picked_up_at'] == trip_info['picked_up_at']), (
            'driver_id', 'route_distance_to_passenger', 'created_at', 'pick_up_time', 'picked_up_at', 'passenger_id',
            'cancelled_at', 'ride_price', 'total_price', 'final_price')]
    dist_list_distribution = []
    driver_id_list = []
    cancel_order_identify(driver_with_distance)
    for index, row in driver_with_distance.iterrows():
        if row['driver_id'] not in driver_id_list:
            driver_id_list.append(row['driver_id'])
        dist_list_distribution.append(row['route_distance_to_passenger'])
    driver_gps_dict = cal_gps_to_driver(driver_id_list, driver_with_distance)
    # print(driver_gps_dict,driver_with_distance)
    # sns.set()  # ?????????sns?????????????????????
    #
    # sns.distplot(dist_list_distribution)
    # plt.show()

    print(len(driver_id_list))
    for index, row in driver_with_distance.iterrows():
        if row['driver_id'] in driver_id_list:
            error = cal_dist_error(row['driver_id'], row['created_at'], row['route_distance_to_passenger'],
                                   row['picked_up_at'])
            print("error =", error)
    print("END")

    for item in driver_id_list:
        print(item)
        for index, row in trip_gps.iterrows():

            if row['driver_id'] == 5826:
                print(row['driver_id'], row['created_at'])
            #     time_interval = time.mktime(time.strptime(trip_gps['created_at'][index], '%Y-%m-%d %H:%M:%S')) - time.mktime(time.strptime(trip_gps['created_at'][index+1], '%Y-%m-%d %H:%M:%S'))
            # print(time_interval)
        # print(time.mktime(time.strptime(a, '%Y-%m-%d %H:%M:%S')))
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

def label_creation(data_miss_label):
    # label_dict = {'0-morning':,'0-evening':,'0-midnight':,'0-other':,
    # '0-morning':,'0-evening':,'0-midnight':,'0-other':,
    # '0-morning':,'0-evening':,'0-midnight':,'0-other':,
    # '0-morning':,'0-evening':,'0-midnight':,'0-other':,
    # '0-morning':,'0-evening':,'0-midnight':,'0-other':,
    # }

    data_eval = pd.read_csv('evaluation_whole_day.csv')
    grid_id_list = data_eval['grid_id'].unique()
    time_list = data_eval['time'].unique()
    group_data = data_eval[['grid_id','time','total price']].groupby(by=['grid_id','time'],as_index=False).max()
    merge_data = pd.merge(group_data,data_eval[['grid_id','time','total price','radius']],on=['grid_id','time','total price'],how='left')
    print(merge_data)
    label_list = []
    for index,item in data_miss_label.iterrows():
        # print(float(item['order_grid_id']),item['time_period'])

        best_radius = merge_data.loc[(merge_data['grid_id']==float(item['order_grid_id']))&(merge_data['time']==item['time_period'])]
        best_value = best_radius['radius'].values[0]
        label_list.append(best_value)
    res_dict = {}
    for item in label_list:
        res_dict[item] = res_dict.get(item,0)+1
    print(res_dict)
    label_col = pd.DataFrame(label_list,columns=['best_radius'])
    result = pd.concat((data_miss_label, label_col), axis=1)
    return result

def MLP_nn():
    data = pd.read_csv("train_random_driver_whole_day.csv")
    data = data.dropna(how='any')
    # print(data.columns)

    input_data = data.drop(
        ['wait_time', 'trip_time', 'trip_distance', 'reward', 'Unnamed: 0', 'Unnamed: 0.1', 'order_lng', 'order_lat','radius'],
        axis=1)
    input_data = input_data.reset_index(drop=True)
    input_data = label_creation(input_data)

    # print(input_data.columns)
    enc = OneHotEncoder()
    switch_feature = np.array(input_data['time_period'].values)
    # switch_feature = np.array(['1','1','2','3','5'])
    switch_feature = switch_feature.reshape(-1, 1)

    enc.fit_transform(switch_feature).toarray()
    encoding_result = enc.fit_transform(switch_feature).toarray()
    encoding_result =pd.DataFrame(encoding_result, columns=['evening', 'midnight', 'morning', 'other'])
    input_data = input_data.drop(['time_period'],axis=1)
    input_data = pd.concat((input_data, encoding_result), axis=1)
    input = input_data.drop(['best_radius'], axis=1)
    output = input_data['best_radius']
    # print(input_data)
    x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=0)
    X = x_train
    y = y_train
    cols = X.columns
    scaler = StandardScaler()  # ???????????????
    scaler.fit(X)  # ?????????????????????
    X = scaler.transform(X)  # ???????????????
    X = pd.DataFrame(X, columns=cols)
    clf = MLPRegressor(solver='sgd', alpha=1e-5, hidden_layer_sizes=(10, 1), random_state=1)

    clf.fit(X, y)
    y_true = y_test
    y_pred = clf.predict(x_test)
    # print(classification_report(y_true, y_pred))
    # print()
    print(y_true.values)
    result_pred = pd.DataFrame(y_true.values,columns=['Best_radius'])
    result_pred['Predict_radius'] = y_pred
    result_pred['Best_radius'] = y_true.values
    result_pred.to_csv('./experiment/predict_MLP.csv')
    print(clf.score(x_test, y_test))
    print('???????????????', clf.predict([[5,20,1.203,0,0,0,1]]))  # ????????????????????????
    # print('???????????????', clf.predict([[2.601150, 10, 40.7679674, -73.9682154, 5, 27.7, 8]]))  # ????????????????????????
    # print('???????????????', clf.predict([[2.601150, 15, 40.7679674, -73.9682154, 5, 27.7, 8]]))  # ????????????????????????
    # print('???????????????', clf.predict([[2.601150, 20, 40.7679674, -73.9682154, 5, 27.7, 8]]))  # ????????????????????????
    # print('???????????????', clf.predict([[2.601150, 25, 40.7679674, -73.9682154, 5, 27.7, 8]]))  # ????????????????????????
    # print('???????????????', clf.predict([[2.601150, 30, 40.7679674, -73.9682154, 5, 27.7, 8]]))  # ????????????????????????
    cengindex = 0
    for wi in clf.coefs_:
        cengindex += 1  # ?????????????????????????????????
        print('???%d????????????:' % cengindex)
        print('??????????????????:', wi.shape)
        print('???????????????\n', wi)
    return clf

def graph():
    data = pd.read_csv('./experiment/predict_MLP.csv')

    y_1 = data['Best_radius'].tolist()
    y_2 = data['Predict_radius'].tolist()
    x = []
    for i in range(len(y_1)):
        x.append(i)
    print(x)
    # x = [1,2,3]
    # y = [4,5,6]
    # print(y_1)
    # print(y_2)

    plt.scatter(x, y_2, color='g', label="Predict radius")  # o-:??????
    plt.scatter(x, y_1, color='r', label="Best radius")  # s-:??????
    plt.xlabel("Index")  # ???????????????
    plt.ylabel("Radius")  # ???????????????
    plt.legend(loc="best")  # ??????
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # test()
    # data_cleaning()
    l1 = [0,0,1,0]
    l2 = [1.5,2,5]
    print(l1+l2)
    MLP_nn()
    # graph()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
