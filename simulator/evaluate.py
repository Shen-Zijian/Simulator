import pandas as pd
import pickle
import numpy as np
import json
import time
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from config import env_params
import os
import pickle
def eval_data():
    data = pd.read_csv("./experiment/train_random_driver_MLP.csv")
    print(data)
    columns_name = ['radius', 'time', 'grid_id', 'num of matched order', 'total wait time', 'total price',
                    'driver utilization rate', 'driver delivery rate', 'driver pickup rate', 'total pick up distance',
                    'total reward', 'ratio']
    result = pd.DataFrame(columns=columns_name)

    # data = data.dropna()
    input_data = data.drop(['trip_time'], axis=1)
    input_data = input_data.dropna()

    # radius_list = data['radius'].unique()
    # time_list = ['morning','evening','midnight','other']
    time_list = data['time_period'].unique()
    print(time_list)
    for temp_time in time_list:
        # for temp_rad in radius_list:
        grid_data = data.loc[(data['time_period'] == temp_time), ('order_grid_id', 'price')]
        print(grid_data)
        grid_id_list = grid_data['order_grid_id'].unique()
        for temp_grid_id in grid_id_list:
            # data_radius_5 = data.loc[(data['radius'] == temp_rad),('wait_time', 'price','pickup_distance')]
            data_radius_5 = data.loc[(data['time_period'] == temp_time) & (data['order_grid_id'] == temp_grid_id), (
            'wait_time', 'price', 'pickup_distance', 'trip_distance', 'reward', 'order_grid_id')]
            data_radius_5 = data_radius_5.dropna()
            print(data_radius_5)

            # print(data_radius_5['wait_time'])
            # data_radius_5 = input_data
            if len(data_radius_5) != 0:
                total_wait_time = sum(np.array(data_radius_5['wait_time'], dtype=float))
                total_pickup_time = sum(np.array(data_radius_5['pickup_distance'], dtype=float))
                total_price = sum(np.array(data_radius_5['price'], dtype=float))
                sum_order = data_radius_5.shape
                driver_utilization_rate = (sum(np.array(data_radius_5['trip_distance'], dtype=float)) + sum(
                    np.array(data_radius_5['pickup_distance'], dtype=float))) / (22.788 * 500 * 12)
                driver_utilization_rate = min(driver_utilization_rate, 1.0)
                driver_delivery_utilization_rate = sum(np.array(data_radius_5['trip_distance'], dtype=float)) / (
                            22.788 * 500 * 12)
                driver_pickup_utilization_rate = sum(np.array(data_radius_5['pickup_distance'], dtype=float)) / (
                            22.788 * 500 * 12)
                total_reward = sum(np.array(data_radius_5['reward'], dtype=float))
                print('radius:', 'MLP', 'time:', temp_time, 'grid_id:', temp_grid_id)
                print('num of matched order', sum_order[0])
                print('total wait time:', total_wait_time)
                print('total price:', total_price)
                print('driver utilization rate:', driver_utilization_rate)
                print('driver delivery rate:', driver_delivery_utilization_rate)
                print('driver pickup rate:', driver_pickup_utilization_rate)
                print('total pick up distance:', total_pickup_time)
                print('total reward:', total_reward)
                print('ratio:', total_price / (total_pickup_time))
                tempres = ['MLP', temp_time, temp_grid_id, sum_order[0], total_wait_time, total_price,
                           driver_utilization_rate, driver_delivery_utilization_rate, driver_pickup_utilization_rate,
                           total_pickup_time, total_reward, total_price / (total_pickup_time)]
                test_list = np.array(tempres).reshape(1, 12)
                test_df = pd.DataFrame(test_list, columns=columns_name)
                print(test_df)
                result = pd.concat([result, test_df], ignore_index=True)
                print("==============")
    result.to_csv('./experiment/evaluation_MLP.csv')

from simulator_env import Simulator
def joint_csv():
    data_1 = pd.read_csv("train_random_driver.csv")
    data_2 = pd.read_csv("train_random_driver_morning.csv")
    data_3 = pd.read_csv("train_random_driver_evening_2.csv")
    data_4 = pd.read_csv("train_random_driver_midnight.csv")
    data_final = pd.concat([data_1, data_2, data_3, data_4], ignore_index=True)
    data_final.to_csv("train_random_driver_whole_day.csv")

def fix_data_correction(fix_data,generate_data):
    for index, row in fix_data.iterrows():
        timestamp = row['time_stamp']  # 获得timestamp列的值
        grid_id = row['grid_id']  # 获得gridid列的值
        key = f'{grid_id}_{timestamp+30}'  # 拼接为key
        if key in generate_data:  # 如果key已经在stats中,计数加1
            value = generate_data[key]
        else:  # 否则初始化为1
            value = 0
        fix_data.at[index,'num_order'] = fix_data.at[index,'num_order'] + fix_data.at[index,'num_matched_order'] - value
    return fix_data
def cal_matched_order(df):
    data = df.loc[df['radius']=='policy']
    return np.sum(data['num_matched_order'].to_numpy())
def eval_grid_model_data():
    # model = env_params['model_name']
    model = 'policy'

    # label = env_params['label_name']
    # label = 'matched_ratio'
    # with open(f'./experiment_policy/record/result.pkl', 'rb') as f:
    #     data = f.read()
    # num_order_generation = pickle.loads(data)

    # label_list = ['am','amw','es','esw','fixed','matched_ratio']#, 'matched_ratio','avg_matched_price','avg_matched_pickup_distance','total_matched_price','total_matched_pickup_distance']
    label_list = ['am']
    # label_list = ['am_30s','am-weighted_30s','es_30s','es-weighted_30s','fixed_30s']#, 'matched_ratio','avg_matched_price','avg_matched_pickup_distance','total_matched_price','total_matched_pickup_distance']
    # label_list = ['fixed_data_30']
    cal_total_day = True
    fix_dataset = pd.read_csv('D:/Files/Smart Mobility Lab/Transpotation_Simulator-1/simulator/experiment_fixed/record/fixed_data_30.csv'
)

    # fix_dataset = fix_data_correction(fix_dataset,num_order_generation)
    # fix_dataset.loc[fix_dataset['num_matched_order']==0,'avg_pickup_distance'] = fix_dataset.loc[fix_dataset['num_matched_order']==0,'avg_pickup_distance'] - 100*(fix_dataset.loc[fix_dataset['num_matched_order']==0,'num_order'])
    for label in label_list:
        print(label)
        dataset = pd.read_csv(f'./experiment_{model}/record/matched_record_{label}.csv')
        # dataset = fix_data_correction(dataset,num_order_generation)
        # print(label,cal_matched_order(dataset), np.sum(dataset.loc[dataset['radius']=='lstm',('num_order')].to_numpy()))
        grid_list = dataset['grid_id'].unique()
        radius_list = fix_dataset['radius'].unique()
        # print(grid_list)
        time_period_dict = {2: 'morning', 3: 'other', 0: 'evening', 1: 'midnight'}
        grid_eval_result = pd.DataFrame(columns=['time_period','radius','avg_pickup_distance','total_price','match_rate','avg_driver_util_rate'])

        for time_period in list(time_period_dict.keys()):
            cur_grid_data = dataset.loc[(dataset['grid_id'] == id) & (dataset['time_period'] == time_period)]
            # print(cur_grid_data)
            if np.sum(cur_grid_data['num_available_driver']) != 0:
                avg_driver_util_rate = np.sum(cur_grid_data['driver_utilization_rate'].values * cur_grid_data[
                    'num_available_driver']) / np.sum(cur_grid_data['num_available_driver'])
            else:
                avg_driver_util_rate = 0
            total_price = np.sum(cur_grid_data['num_matched_order'].values * cur_grid_data['avg_matched_price'].values)
            total_pickup_distance = np.sum(cur_grid_data['num_matched_order'].values * cur_grid_data['avg_matched_pickup_distance'].values)
            if np.sum(cur_grid_data['num_matched_order'].values) == 0:
                avg_pickup_distance = 0
            else:
                avg_pickup_distance = total_pickup_distance / np.sum(cur_grid_data['num_matched_order'].values)
            num_order = np.sum(cur_grid_data['num_order'].values)
            if num_order != 0:
                match_rate = np.sum(cur_grid_data['num_matched_order'].values) / np.sum(cur_grid_data['num_order'].values)
            else:
                match_rate = 0

            cur_period = time_period_dict[time_period]
            # cur_grid = id
            grid_eval_result.loc[len(grid_eval_result.index)] = [cur_period,label,avg_pickup_distance,total_price,match_rate,avg_driver_util_rate]


        if cal_total_day == True:
            cur_grid_data = dataset
            # print(cur_grid_data)
            if np.sum(cur_grid_data['num_available_driver']) != 0:
                avg_driver_util_rate = np.sum(cur_grid_data['driver_utilization_rate'].values * cur_grid_data[
                    'num_available_driver']) / np.sum(cur_grid_data['num_available_driver'])
            else:
                avg_driver_util_rate = 0
            total_price = np.sum(
                cur_grid_data['num_matched_order'].values * cur_grid_data['avg_matched_price'].values)
            total_pickup_distance = np.sum(
                cur_grid_data['num_matched_order'].values * cur_grid_data['avg_matched_pickup_distance'].values)
            if np.sum(cur_grid_data['num_matched_order'].values) == 0:
                avg_pickup_distance = 0
            else:
                avg_pickup_distance = total_pickup_distance/np.sum(cur_grid_data['num_matched_order'].values)
            num_order = np.sum(cur_grid_data['num_order'].values)
            if num_order != 0:
                match_rate = np.sum(cur_grid_data['num_matched_order'].values) / np.sum(
                    cur_grid_data['num_order'].values)
            else:
                match_rate = 0
            grid_eval_result.loc[len(grid_eval_result.index)] = [ 'whole_day', label,
                                                                 avg_pickup_distance, total_price, match_rate,
                                                                 avg_driver_util_rate]

        for time_period in list(time_period_dict.keys()):
            for radius_ in radius_list:
                cur_grid_data = fix_dataset.loc[(fix_dataset['time_period'] == time_period)&(fix_dataset['radius']==radius_)]
                if np.sum(cur_grid_data['num_available_driver']) !=0:
                    avg_driver_util_rate = np.sum(cur_grid_data['driver_utilization_rate'].values*cur_grid_data['num_available_driver'])/np.sum(cur_grid_data['num_available_driver'])
                else:
                    avg_driver_util_rate = 0
                total_price = np.sum(cur_grid_data['num_matched_order'].values * cur_grid_data['avg_matched_price'].values)
                total_pickup_distance = np.sum(
                    cur_grid_data['num_matched_order'].values * cur_grid_data['avg_matched_pickup_distance'].values)
                if np.sum(cur_grid_data['num_matched_order'].values) == 0:
                    avg_pickup_distance = 0
                else:
                    avg_pickup_distance = total_pickup_distance / np.sum(cur_grid_data['num_matched_order'].values)
                num_order = np.sum(cur_grid_data['num_order'].values)
                if num_order != 0:
                    match_rate = np.sum(cur_grid_data['num_matched_order'].values) / np.sum(
                        cur_grid_data['num_order'].values)
                else:
                    match_rate = 0
                cur_period = time_period_dict[time_period]
                # cur_grid = id

                grid_eval_result.loc[len(grid_eval_result.index)] = [cur_period, f'{float(radius_)}',
                                                                     avg_pickup_distance, total_price, match_rate,
                                                                     avg_driver_util_rate]

        for radius_ in radius_list:
            cur_grid_data = fix_dataset.loc[fix_dataset['radius'] == radius_]
            if np.sum(cur_grid_data['num_available_driver']) != 0:
                avg_driver_util_rate = np.sum(cur_grid_data['driver_utilization_rate'].values * cur_grid_data[
                    'num_available_driver']) / np.sum(cur_grid_data['num_available_driver'])
            else:
                avg_driver_util_rate = 0
            total_price = np.sum(
                cur_grid_data['num_matched_order'].values * cur_grid_data['avg_matched_price'].values)
            total_pickup_distance = np.sum(
                cur_grid_data['num_matched_order'].values * cur_grid_data[
                    'avg_matched_pickup_distance'].values)
            if np.sum(cur_grid_data['num_matched_order'].values) == 0:
                avg_pickup_distance = 0
            else:
                avg_pickup_distance = total_pickup_distance / np.sum(cur_grid_data['num_matched_order'].values)
            num_order = np.sum(cur_grid_data['num_order'].values)
            if num_order != 0:
                match_rate = np.sum(cur_grid_data['num_matched_order'].values) / np.sum(
                    cur_grid_data['num_order'].values)
            else:
                match_rate = 0
            # cur_grid = id
            grid_eval_result.loc[len(grid_eval_result.index)] = ['whole_day', f'{float(radius_)}',
                                                                 avg_pickup_distance, total_price,
                                                                 match_rate,
                                                                 avg_driver_util_rate]




        folder_path = f"./experiment_{model}/eval/"
        file_name = f"grid_model_{label}_eval.csv"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, file_name)
        grid_eval_result.to_csv(file_path,index=False)
# label_list = ['avg_matched_pickup_distance','total_matched_pickup_distance','total_matched_price']
def graphic():
    # model = env_params['model_name']
    # label = env_params['label_name']
    model = 'policy'
    label = 'am'
    # print(label)
    dataset = pd.read_csv(f'./experiment_{model}/eval/grid_model_{label}_eval.csv')
    # dataset = pd.read_csv('./experiment_combine/data_lr_lstm_matched_ratio.csv')
    dataset['avg_pickup_distance_norm'] = (dataset['avg_pickup_distance'] - dataset[
        'avg_pickup_distance'].min()) / (dataset['avg_pickup_distance'].max() - dataset[
        'avg_pickup_distance'].min())
    dataset['total_price_norm'] = (dataset['total_price'] - dataset['total_price'].min()) / (
                dataset['total_price'].max() - dataset['total_price'].min())
    dataset['match_rate_norm'] = (dataset['match_rate'] - dataset['match_rate'].min()) / (
                dataset['match_rate'].max() - dataset['match_rate'].min())
    dataset['avg_driver_util_rate_norm'] = (dataset['avg_driver_util_rate'] - dataset['avg_driver_util_rate'].min()) / (
                dataset['avg_driver_util_rate'].max() - dataset['avg_driver_util_rate'].min())
    # dataset['avg_driver_util_rate_norm'] = 0
    dataset['score'] = -0.4 * dataset['avg_pickup_distance_norm'] + 0.05* dataset['total_price_norm'] + 0.45 * dataset[
        'match_rate_norm'] + 0.01 * dataset['avg_driver_util_rate_norm']
    # print(dataset)
    label_list = ['matched_ratio']# 'matched_ratio','avg_matched_price','avg_matched_pickup_distance','total_matched_price','total_matched_pickup_distance']
    for label in label_list:
        # dataset = pd.read_csv(f'./experiment_{model}/eval/grid_model_{label}_eval.csv')
        # dataset = pd.read_csv('./experiment_lstm/eval/lstm_combine_eval.csv')
        time_list = ['morning', 'evening', 'midnight', 'other','whole_day']
        value_list = ['avg_pickup_distance', 'total_price', 'match_rate', 'avg_driver_util_rate','score']
        fig = plt.figure(figsize=(16, 16))
        fig.suptitle(f'Grid Model {label} Evaluation', fontsize=20)
        #
        # for
        # # 第一幅图的下标从1开始，设置6张子图
        # for plt_index in range(1, 13):
        #     # 往画布上添加子图：按三行二列，添加到下标为plt_index的位置
        #
        #     # 绘制对应的子图
        print(dataset['radius'].unique())
        # x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']  # 点的横坐标
        x = ['1.0', '2.0', '3.0', '4.0', '5.0']  # 点的横坐标
        # x = ['1','2', '3','lr', 'fix', 'am','amw','es','esw']  # 点的横坐标
        colors= ['darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','r','r','r']
        # colors = ['darkblue', 'darkblue', 'darkblue', 'r','r', 'r','r', 'r','r']
        print(colors)
        index = 1
        for time in time_list:
            for value in value_list:
                k1 = []
                ax = fig.add_subplot(5, 5, index)
                index+=1
                for item in x:
                    temp_data = dataset.loc[(dataset['radius']==item)&(dataset['time_period']==time),value]
                    k1.append(np.average(temp_data))
                # plt.bar(x, k1,color=colors[np.random.randint(1,9)])  # o-:圆形
                ax.plot(x, k1, color='g', linewidth=2, linestyle="-")
                plt.bar(x, k1, color=colors)
                ax.set_title(value+' | '+time, fontsize=10)  # 为子图添加标题，设置标题的字体，字体的大小，字体的颜色
                # ax.set_xlabel(value)  # 为x轴添加标签
        folder_path = f"./experiment_{model}/picture/"
        file_name = f"grid_model_{label}_evaluation_1.png"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, file_name)
        # print(file_path)
        # plt.show()
        plt.savefig(file_path)
        plt.clf()

# def static_order_num:


if __name__ == '__main__':
    # eval_data()
    # joint_csv()
    eval_grid_model_data()
    graphic()
    # a = np.random.random((100, 100))
    # b = np.random.random((100, 100))
    # c = (a<b)+0
    # print(c)

