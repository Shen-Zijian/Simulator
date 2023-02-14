import pandas as pd
import pickle
import numpy as np
import json
import time
import seaborn as sns


def eval_data():
    data = pd.read_csv("train_fix_radius.csv")
    columns_name = ['radius','time','grid_id','num of matched order','total wait time','total reward','driver utilization rate','driver delivery rate','driver pickup rate','total pick up distance','total reward','ratio']
    result = pd.DataFrame(columns=columns_name)

    # data = data.dropna()
    input_data = data.drop(['trip_time'], axis=1)
    input_data = input_data.dropna()

    radius_list = [5, 10, 15, 20, 25, 30,35]
    time_list = ['morning','evening','midnight','other']

    for temp_time in time_list:
        for temp_rad in radius_list:
            grid_data = data.loc[(data['radius'] == f'{temp_rad}')&(data['time_period']==temp_time),('order_grid_id','price')]
            print(grid_data)
            grid_id_list = grid_data['order_grid_id'].unique()
            for temp_grid_id in grid_id_list:
                # data_radius_5 = data.loc[(data['radius'] == temp_rad),('wait_time', 'price','pickup_distance')]
                data_radius_5 = data.loc[(data['radius'] == f'{temp_rad}')&(data['time_period']==temp_time)&(data['order_grid_id']==temp_grid_id),('wait_time', 'price','pickup_distance','trip_distance','reward','order_grid_id')]
                data_radius_5 = data_radius_5.dropna()


                # print(data_radius_5['wait_time'])
                # data_radius_5 = input_data
                if len(data_radius_5)!=0:
                    total_wait_time = sum(np.array(data_radius_5['wait_time'],dtype=float))
                    total_pickup_time = sum(np.array(data_radius_5['pickup_distance'],dtype=float))
                    total_price = sum(np.array(data_radius_5['price'],dtype=float))
                    sum_order = data_radius_5.shape
                    driver_utilization_rate = (sum(np.array(data_radius_5['trip_distance'],dtype=float)) + sum(np.array(data_radius_5['pickup_distance'],dtype=float)))/(22.788*500*12)
                    driver_utilization_rate = min(driver_utilization_rate, 1.0)
                    driver_delivery_utilization_rate = sum(np.array(data_radius_5['trip_distance'],dtype=float))/(22.788*500*12)
                    driver_pickup_utilization_rate = sum(np.array(data_radius_5['pickup_distance'],dtype=float)) / (22.788 * 500 * 12)
                    total_reward = sum(np.array(data_radius_5['reward'],dtype=float))
                    print('radius:', temp_rad,'time:', temp_time,'grid_id:', temp_grid_id)
                    print('num of matched order',sum_order[0])
                    print('total wait time:', total_wait_time)
                    print('total reward:', total_price)
                    print('driver utilization rate:',driver_utilization_rate)
                    print('driver delivery rate:', driver_delivery_utilization_rate)
                    print('driver pickup rate:', driver_pickup_utilization_rate)
                    print('total pick up distance:', total_pickup_time)
                    print('total reward:',total_reward)
                    print('ratio:', total_price / (total_pickup_time))
                    tempres =[temp_rad,temp_time,temp_grid_id,sum_order[0],total_wait_time,total_price,driver_utilization_rate,driver_delivery_utilization_rate,driver_pickup_utilization_rate,total_pickup_time,total_reward,total_price / (total_pickup_time)]
                    test_list = np.array(tempres).reshape(1, 12)
                    test_df = pd.DataFrame(test_list, columns=columns_name)
                    print(test_df)
                    result = pd.concat([result,test_df],ignore_index=True)
                    print("==============")
    result.to_csv('evaluation.csv')
if __name__ == '__main__':
    eval_data()