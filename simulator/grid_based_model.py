import torch
import numpy as np
import joblib
import sys

if __name__ == "__main__":
    sys.path.insert(0, '/home/shenzijian/下载/regression-model/regression_model')
    grid_model = torch.load('./input/model_14079.pth')
    stand_scaler = joblib.load('./input/stand_scaler.bin')
    # time_stamp
    # time_period
    # grid_id
    # num_available_driver
    # avg_pickup_distance
    # avg_price
    # radius
    # num_order

    x = np.array([15200, 1, 5, 25, 2.6, 23, 2.5,50],dtype='float32')
    # x = x.reshape(1, -1)

    x = stand_scaler.transform(x)
    x = torch.from_numpy(x)
    outputs = grid_model(x).item()
    print(outputs)
