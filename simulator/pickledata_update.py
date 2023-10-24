import pandas as pd
import pickle

# 旧版本的 pandas
load_path = './input/'  # 请替换为你的文件路径
driver_file_name = 'HongKong_island_driver'  # 请替换为你的文件名（不包括扩展名）

# 加载 pickle 文件
with open(load_path + driver_file_name + '.pickle', 'rb') as f:
    driver_info = pickle.load(f)

# 将 DataFrame 保存为 CSV 文件
driver_info.to_csv(load_path + driver_file_name + '.csv', index=False)