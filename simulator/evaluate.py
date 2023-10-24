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
def plot_data():
    # Assuming df is your DataFrame
    df = pd.read_csv("./report_data.csv")
    # mean_values = df[df['model'].isin(['0.5','1','1.5','2','2.5','3','3.5','4','4.5','5','6'])].mean()

    # Add a new row 'baseline' with the calculated mean values
    # mean_values['model'] = 'bl'
    # print(mean_values)
    # df = df.append(mean_values, ignore_index=True)
    fig, axs = plt.subplots(3, 2, figsize=(20, 25))  # Change layout to 3x2
    print(df[df['model'].isin(['esw','bl'])].to_string())
    # Create a bar plot for each column, adjust indices for new layout
    columns = ['num_matched_order', 'total_price', 'avg_wait_time', 'avg_pickup_time', 'driver_utilization_rate','avg_pickup_dis']
    for i, col in enumerate(columns):
        row = i // 2
        col = i % 2
        bars = axs[row, col].bar(df['model'], df[columns[i]])
        for j, model in enumerate(df['model']):
            if model in ['lr','lstm', 'bl']:
                bars[j].set_color('r')
        axs[row, col].set_xlabel('Model')
        axs[row, col].set_ylabel(columns[i])
        axs[row, col].set_title(f'{columns[i]} by Model')
    plt.subplots_adjust(hspace=1, wspace=0.5)
    # Remove unused subplot
    # fig.delaxes(axs[2, 1])
    plt.show()

def eval_data(model_name,label_name):
    if model_name == 'fixed':
        data = pd.read_csv(f"./experiment_{model_name}/record/matched_record_radius_{label_name}.csv")
    else:
        data = pd.read_csv(f"./experiment_{model_name}/record/matched_record_{label_name}.csv")

    result_data = pd.read_csv('./report_data.csv')
    # print(data.columns)
    num_matched_order = sum(np.array(data['num_matched_order'].values, dtype=float))
    total_wait_time = np.sum(data['wait_time'].values)
    total_pickup_time = sum(np.array(data['pickup_time'].values*data['num_matched_order'].values, dtype=float))
    avg_pickup_dis = np.sum(data['avg_matched_pickup_distance'].values*data['num_matched_order'].values)/num_matched_order
    total_price = sum(np.array(data['total_matched_price'].values, dtype=float))
    num_order = sum(np.array(data['num_order'].values, dtype=float))

    matched_rate = num_matched_order/42045
    driver_utilization_rate = sum(np.array(data['num_driver'], dtype=float)*np.array(data['driver_utilization_rate'], dtype=float)) / sum(np.array(data['num_driver'], dtype=float))
    occupancy_rate = 1- sum(np.array(data['num_available_driver'], dtype=float)) / sum(np.array(data['num_driver'], dtype=float))

    print('='*20)
    print("model:",label_name)
    print('num of matched order', num_matched_order)
    print('matched rate:', matched_rate)
    print('avg wait time:', total_wait_time/num_order)
    print('avg pickup time:', avg_pickup_dis*3600/20.6)
    print('total price:', total_price)
    print('avg pickup distance',avg_pickup_dis)
    print('driver utilization rate:', driver_utilization_rate)
    print('occupancy rate:',occupancy_rate)
    if not result_data.loc[result_data['model']==label_name].empty:
        result_data.loc[result_data['model']==label_name] = [label_name, matched_rate, total_price,total_wait_time/num_order, avg_pickup_dis*3600/20.6,driver_utilization_rate,avg_pickup_dis]
    else:
        result_data.loc[len(result_data.index)] = [label_name, matched_rate, total_price,total_wait_time/num_order, avg_pickup_dis*3600/20.6,driver_utilization_rate,avg_pickup_dis]
    # tempres = [label_name, num_matched_order, total_price,total_wait_time/num_matched_order, total_pickup_time/num_matched_order,driver_utilization_rate,]
    # result_data = pd.DataFrame(columns=['model','num_matched_order', 'total_price','avg_wait_time', 'avg_pickup_time','driver_utilization_rate'])
    result_data.to_csv('./report_data.csv',index=False)

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

    label_list = ['am','amw','es','esw','fix','lr']#, 'matched_ratio','avg_matched_price','avg_matched_pickup_distance','total_matched_price','total_matched_pickup_distance']
    # label_list = ['esw']
    # label_list = ['am_30s','am-weighted_30s','es_30s','es-weighted_30s','fixed_30s']#, 'matched_ratio','avg_matched_price','avg_matched_pickup_distance','total_matched_price','total_matched_pickup_distance']
    # label_list = ['fixed_data_30']
    cal_total_day = True
    # fix_dataset = pd.read_csv('./experiment_fixed/record_original_driver_distribute/fixed_data_30.csv')
    fix_dataset = pd.read_csv('./experiment_fixed/record/fixed_data_30.csv')

    # fix_dataset = fix_data_correction(fix_dataset,num_order_generation)
    # fix_dataset.loc[fix_dataset['num_matched_order']==0,'avg_pickup_distance'] = fix_dataset.loc[fix_dataset['num_matched_order']==0,'avg_pickup_distance'] - 100*(fix_dataset.loc[fix_dataset['num_matched_order']==0,'num_order'])
    for label in label_list:
        print(label)

        dataset = pd.read_csv(f'./experiment_{model}/record/7_13_pm_4obj/matched_record_{label}.csv')
        print(dataset.columns)
        # dataset = fix_data_correction(dataset,num_order_generation)
        # print(label,cal_matched_order(dataset), np.sum(dataset.loc[dataset['radius']=='lstm',('num_order')].to_numpy()))

        radius_list = fix_dataset['radius'].unique()
        # print(grid_list)
        time_period_dict = {2: 'morning', 3: 'other', 0: 'evening', 1: 'midnight'}
        grid_eval_result = pd.DataFrame(columns=['time_period','radius','avg_pickup_distance','total_price','match_rate','avg_driver_util_rate','DOAR'])

        for time_period in list(time_period_dict.keys()):
            cur_grid_data = dataset.loc[dataset['time_period'] == time_period]
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

            if np.sum(cur_grid_data['num_attend'].values) == 0:
                doar = 0
            else:
                doar = np.sum(cur_grid_data['num_accepted'].values) / np.sum(cur_grid_data['num_attend'].values)

            if num_order != 0:
                match_rate = np.sum(cur_grid_data['num_matched_order'].values) / np.sum(cur_grid_data['num_order'].values)
            else:
                match_rate = 0

            cur_period = time_period_dict[time_period]
            # cur_grid = id
            grid_eval_result.loc[len(grid_eval_result.index)] = [cur_period,label,avg_pickup_distance,total_price,match_rate,avg_driver_util_rate,doar]


        if cal_total_day == True:
            cur_grid_data = dataset
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

            if np.sum(cur_grid_data['num_attend'].values) == 0:
                doar = 0
            else:
                doar = np.sum(cur_grid_data['num_accepted'].values) / np.sum(cur_grid_data['num_attend'].values)

            num_order = np.sum(cur_grid_data['num_order'].values)
            if num_order != 0:
                match_rate = np.sum(cur_grid_data['num_matched_order'].values) / np.sum(
                    cur_grid_data['num_order'].values)
            else:
                match_rate = 0
            grid_eval_result.loc[len(grid_eval_result.index)] = [ 'whole_day', label,
                                                                 avg_pickup_distance, total_price, match_rate,
                                                                 avg_driver_util_rate,doar]

        for time_period in list(time_period_dict.keys()):
            for radius_ in radius_list:
                cur_grid_data = fix_dataset.loc[(fix_dataset['time_period'] == time_period)&(fix_dataset['radius']==radius_)]
                print(time_period,radius_,np.sum(cur_grid_data['num_matched_order']))
                print(cur_grid_data)
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

                if np.sum(cur_grid_data['num_attend'].values) == 0:
                    doar = 0
                else:
                    doar = np.sum(cur_grid_data['num_accepted'].values) / np.sum(cur_grid_data['num_attend'].values)


                if num_order != 0:
                    match_rate = np.sum(cur_grid_data['num_matched_order'].values) / np.sum(
                        cur_grid_data['num_order'].values)
                else:
                    match_rate = 0
                cur_period = time_period_dict[time_period]
                # cur_grid = id

                grid_eval_result.loc[len(grid_eval_result.index)] = [cur_period, f'{float(radius_)}',
                                                                     avg_pickup_distance, total_price, match_rate,
                                                                     avg_driver_util_rate,doar]

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

            if np.sum(cur_grid_data['num_attend'].values) == 0:
                doar = 0
            else:
                doar = np.sum(cur_grid_data['num_accepted'].values) / np.sum(cur_grid_data['num_attend'].values)

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
                                                                 avg_driver_util_rate,doar]




        folder_path = f"./experiment_{model}/eval/"
        file_name = f"grid_model_{label}_eval.csv"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, file_name)
        grid_eval_result.to_csv(file_path,index=False)
# label_list = ['avg_matched_pickup_distance','total_matched_pickup_distance','total_matched_price']
def generate_combine(label_list):
    combine_df = pd.DataFrame()
    for label in label_list:
        eval_csv = pd.read_csv(f'./experiment_policy/eval/grid_model_{label}_eval.csv')
        combine_df = pd.concat([combine_df,eval_csv],axis=0,ignore_index=True)
    combine_df.drop_duplicates(inplace=True)
    combine_df.to_csv(f'./experiment_policy/eval/grid_model_combine_eval.csv',index=False)
def graphic():
    # model = env_params['model_name']
    # label = env_params['label_name']
    model = 'policy'
    label = 'combine'
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
        value_list = ['avg_pickup_distance', 'total_price', 'match_rate', 'avg_driver_util_rate']
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
        # x = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0','2.0','3.0']  # 点的横坐标
        # x = ['1.0', '2.0', '3.0', '4.0', '5.0']  # 点的横坐标
        x = ['0.1', '0.3','1.0','lr', 'fix', 'am','amw','es','esw']  # 点的横坐标
        # colors= ['darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue']
        colors = ['darkblue', 'darkblue', 'darkblue', 'r','r', 'r','r', 'r','r']
        print(colors)
        index = 1
        for time in time_list:
            for value in value_list:
                k1 = []
                ax = fig.add_subplot(5, 4, index)
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
        file_name = f"grid_model_{label}_evaluation_new.png"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, file_name)
        print(file_path)
        # print(file_path)
        # plt.show()
        plt.savefig(file_path)
        plt.clf()

# def static_order_num:num_matched_order, total_price,total_wait_time/num_matched_order, total_pickup_time/num_matched_order,driver_utilization_rate,


if __name__ == '__main__':
    model_list = ['policy','fixed']
    # fixed_list = ['0.5','1','1.5','2','2.5','3','3.5','4','4.5','5','6']
    # model_list = ['fixed']
    fixed_list = ['2.5','3','4','5','6','7','8','9','10']
    policy_list = ['transformer_fix','transformer_am','transformer_amw','transformer_es','transformer_esw','transformer_esw_HK']
    # policy_list = ['lstm_fix','lstm_am','lstm_amw','lstm_es','lstm_esw']
    for model in model_list:
        if model == 'policy':
            for label in policy_list:
                eval_data(model_name='policy',label_name=label)
        else:
            for label in fixed_list:
                eval_data(model_name='fixed',label_name=label)
    # plot_data()
    # joint_csv()
    # eval_grid_model_data()
    # generate_combine(label_list = ['am','amw','es','esw','fix','lr'])
    # graphic()
    # a = np.random.random((100, 100))
    # b = np.random.random((100, 100))
    # c = (a<b)+0
    # print(c)

