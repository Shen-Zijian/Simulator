from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import pandas as pd
import time
pd.options.mode.chained_assignment = None
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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

    print(trip_info.loc[(trip_info['route_distance_to_passenger'] <= 999999)&(trip_info['driver_id']==11018 ),('driver_id','route_distance_to_passenger','created_at')])
    driver_with_distance = trip_info.loc[(trip_info['route_distance_to_passenger'] == trip_info['route_distance_to_passenger'])&(trip_info['driver_id'] == trip_info['driver_id']),('driver_id','route_distance_to_passenger','pick_up_time')]
    driver_id_list = []
    for index,row in driver_with_distance.iterrows():
        if row['driver_id'] not in driver_id_list:
            driver_id_list.append(row['driver_id'])
    print(driver_id_list)

    a = "2011-09-28 10:00:00"
    for index,row in trip_gps.iterrows():
        if row['driver_id'] in driver_id_list:
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

