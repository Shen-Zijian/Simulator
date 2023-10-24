import os
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import numpy as np
import torch


def concatenate_csv_files(input_folder):
    # 获取文件夹内的所有CSV文件名
    csv_files = ['matched_record_radius_0.5.csv','matched_record_radius_1.csv','matched_record_radius_1.5.csv','matched_record_radius_2.csv','matched_record_radius_2.5.csv','matched_record_radius_3.csv','matched_record_radius_3.5.csv','matched_record_radius_4.csv','matched_record_radius_4.5.csv','matched_record_radius_5.csv','matched_record_radius_6.csv']
    col_names = pd.read_csv(input_folder + 'matched_record_radius_1.csv').columns.tolist()
    print(col_names)
    # 初始化一个空的DataFrame
    combined_csv = pd.DataFrame(columns=col_names)
    # dataframes = []

    # 遍历所有CSV文件并读取为DataFrame对象
    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        current_csv = pd.read_csv(file_path)
        combined_csv = pd.concat([combined_csv, current_csv], ignore_index=True)

    # 使用pandas.concat拼接所有的DataFrame

    # print(combined_csv.columns)
    combined_csv = combined_csv.sort_values(['time_stamp', 'time_period', 'grid_id', 'radius'])
    new_order = ['time_stamp', 'time_period', 'grid_id', 'num_available_driver', 'avg_pickup_distance', 'avg_price',
                 'radius', 'num_order', 'num_matched_order', 'avg_matched_pickup_distance', 'avg_matched_price',
                 'driver_utilization_rate', 'total_matched_price', 'matched_ratio']
    combined_csv['matched_ratio'] = combined_csv['num_matched_order'] / combined_csv['num_order'].replace(0, 1)
    # print(data.loc[data[]])

    time_dict = {2: [25200, 61200], 0: [61200, 68400], 1: [0, 25200], 3: [68400, 86400]}

    def get_time_label(time_seconds):
        for label, (start, end) in time_dict.items():
            if start <= time_seconds < end:
                return label
        return 4

    combined_csv['time_period'] = combined_csv['time_stamp'].apply(get_time_label)

    data_sorted_2 = combined_csv[new_order]
    data_sorted_2 = data_sorted_2.sort_values(['grid_id', 'time_stamp', 'radius'])
    data_sorted_2 = data_sorted_2.astype(float)
    data_sorted_2.to_csv(input_folder + 'HongKong_whole.csv', index=False)


    combined_csv.to_csv(input_folder+'HongKong_whole.csv',index=False)




    # 保存合并后的CSV文件
    # combined_csv.to_csv(output_file, index=False)
    # train_data, test_data = train_test_split(combined_csv, test_size=0.2, random_state=42)
    # train_data = train_data.sort_values(['time_stamp', 'time_period', 'grid_id', 'radius'])
    # test_data = test_data.sort_values(['time_stamp', 'time_period', 'grid_id', 'radius'])
    # train_data.to_csv(input_folder + 'train_set_30s.csv', index=False)
    # test_data.to_csv(input_folder + 'test_set_30s.csv', index=False)


def data_transform(input_file):
    data = pd.read_csv(input_file+'train_set_30s.csv')
    data_1 = pd.read_csv(input_file+'test_set_30s.csv')
    data_2 = pd.read_csv(input_file+'HongKong_whole.csv')
    # data = input_file
    # print(data)
    new_order = ['time_stamp', 'time_period', 'grid_id', 'num_available_driver', 'avg_pickup_distance', 'avg_price',
                 'radius', 'num_order', 'num_matched_order', 'avg_matched_pickup_distance', 'avg_matched_price',
                 'driver_utilization_rate', 'total_matched_price', 'matched_ratio']
    data['matched_ratio'] = data['num_matched_order'] / data['num_order'].replace(0, 1)
    data_1['matched_ratio'] = data_1['num_matched_order'] / data_1['num_order'].replace(0, 1)
    data_2['matched_ratio'] = data_2['num_matched_order'] / data_2['num_order'].replace(0, 1)
    # print(data.loc[data[]])
    data_sorted = data[new_order]
    data_sorted = data_sorted.sort_values(['grid_id','time_stamp','radius'])
    data_sorted = data_sorted.astype(float)
    data_sorted.to_csv(input_file+'trainset_change_30s.csv',index=False)
    data_sorted_1 = data_1[new_order]
    data_sorted_1 = data_sorted_1.sort_values(['grid_id', 'time_stamp', 'radius'])
    data_sorted_1 = data_sorted_1.astype(float)
    data_sorted_1.to_csv(input_file + 'testset_change_30s.csv', index=False)
    data_sorted_2 = data_1[new_order]
    data_sorted_2 = data_sorted_2.sort_values(['grid_id', 'time_stamp', 'radius'])
    data_sorted_2 = data_sorted_2.astype(float)
    data_sorted_2.to_csv(input_file + 'HongKong_whole.csv', index=False)

def change_penalty(input_file):
    dataset = pd.read_csv(input_file + 'dataset.csv')
    print(dataset.columns)
    # dataset = pd.read_csv(input_file)
    dataset.loc[(dataset['num_order']==0)&(dataset['num_matched_order']==0),'avg_pickup_distance'] += 100
    print(dataset.loc[(dataset['num_order']==0),['avg_pickup_distance','num_order','num_matched_order']][:500].to_string())
    dataset.to_csv(input_file+'dataset_change.csv',index=False)
side = 20
radius =5
center = (10,20)
interval = 2 * radius / side


def get_zone(lat, lng):
    """
    :param lat: the latitude of coordinate
    :type : float
    :param lng: the longitude of coordinate
    :type lng: float
    :return: the id of zone that the point belongs to
    :rtype: float
    """
    if lat < center[1]:
        i = math.floor(side / 2) - math.ceil((center[1] - lat) / interval) + side % 2
    else:
        i = math.floor(side / 2) + math.ceil((lat - center[1]) / interval) - 1

    if lng < center[0]:
        j = math.floor(side / 2) - math.ceil((center[0] - lng) / interval) + side % 2
    else:
        j = math.floor(side / 2) + math.ceil((lng - center[0]) / interval) - 1
    return i * side + j

def apply_get_zone(row):
    return get_zone(row['lat'], row['lng'])
# 用法示例
# input_folder = 'D:/Shen Zijian/Simulator_Broadcasting/Transpotation_Simulator_Manhattan/Transpotation_Simulator-Manhattan/simulator/experiment_fixed/record/'
# output_file = 'D:/Shen Zijian/Simulator_Broadcasting/Transpotation_Simulator_Manhattan/Transpotation_Simulator-Manhattan/simulator/experiment_fixed/record/train_set.csv'
# # concatenate_csv_files(input_folder, output_file)
# # data_transform(input_folder)
# data = np.random.random(size=(3,3))
# print(data)
# test_df = pd.DataFrame(data,columns=['lng','lat','grid_id'])
#
# print(test_df)
# input_data = 'D:/Shen Zijian/lstm_regressor/regression-model-main/regression_model/'
# change_penalty(input_data)
def test_model(file_path):
    device = 'cuda'
    pth_model = torch.jit.load(file_path).to(device)
    test_tensor = torch.rand(size=(1, 8)).to(device)
    print(pth_model(test_tensor))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_model(model,name, input=torch.tensor(torch.rand(size=(1,8)))):
    input = input.to(device)
    model = torch.jit.trace(model, input).to(device)
    print(device)
    torch.jit.save(model,'./weights/tjm_weights/'+ name+'.tjm')
    test_tensor = torch.rand(size=(1, 8)).to(device)
    model_test = torch.jit.load('./weights/tjm_weights/'+ name+'.tjm').to(device)
    print(model_test(test_tensor))


if __name__ == "__main__":
    # test_model('D:/Shen Zijian/lstm_regressor/regression-model-main/regression_model/weights/policy_am_2s.tjm')
    #
    # pickle5.load(open('D:/Shen Zijian/Simulator_Broadcasting/Transpotation_Simulator_Manhattan/Transpotation_Simulator-Manhattan/simulator/input/hongkong_driver_info.pickle','rb'))
    mode = 'concat_data'
    if mode == 'concat_data':
        # concatenate_csv_files('./experiment_fixed/record/')
        combined_data = pd.read_csv("./experiment_fixed/record/HongKong_whole.csv")
        time_dict = {2: [25200, 61200], 0: [61200, 68400], 1: [0, 25200], 3: [68400, 86400]}


        def get_time_label(time_seconds):
            for label, (start, end) in time_dict.items():
                if start <= time_seconds < end:
                    return label
            return 4


        combined_data['time_period'] = combined_data['time_stamp'].apply(get_time_label)
        combined_data.to_csv("./experiment_fixed/record/HongKong_whole.csv",index=False)
        # data_transform('./experiment_fixed/record/')
    elif mode == 'change_weight':
        folder_path = './input/policy_weights/'

        # 遍历文件夹内的所有文件
        for filename in os.listdir(folder_path):
            name = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            print(file_path)
            # 检查是否是文件
            if os.path.isfile(file_path):
                # 调用weight_transform()函数处理文件
                pth_model = torch.load(file_path).to(device)
                outputs = pth_model(torch.rand(size=(1, 8)).to(device))
                print(outputs)
                convert_model(pth_model, name)
    # df = pd.read_csv('./broadcasting_statistics_threshold=0.55.csv')
    # df['n/N'] = df['n']/df['N'].replace(0,1)
    # df.to_csv('./broadcasting_statistics_threshold=0.55.csv')