PRICE_MEAN = float('inf')
PRICE_STD = 0
PICKUP_TIME_MEAN = 300
PICKUP_TIME_STD = 0
CRUISE = True
CRUISE_MODE = 'nearby'
REPOSITION = False

# morning :25200-32400, evening:61200-68400 midnight:0-18000 other:68400-86400
env_params = {
    't_initial': 0,
    't_end': 86400,
    'time_period': 'midnight',
    'delta_t': 5,  # s
    'vehicle_speed': 20.6,  # km / h
    'repo_speed': 20.6,  # 目前的设定需要与vehicl speed保持一致
    'order_sample_ratio': 1,
    'order_generation_mode': 'sample_from_base',
    'driver_sample_ratio': 1,
    'maximum_wait_time_mean': 500,
    'maximum_wait_time_std': 0,
    'radius_list': [0.25,0.5,1,1.25,1.5,1.75,2,2.25,2.5,3,3.5,4,5,6,7,8,9,10],
    'grid_radius_dict': {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25, 5: 0.25, 6: 0.25, 7: 0.25, 8: 0.25, 9: 0.25,
                         10: 0.25,
                         11: 0.25, 12: 0.25, 13: 0.25, 14: 0.25, 15: 0.25},
    'model_name': 'policy',  # lstm,fixed,mlp,lr,mtl
    'online_weight_update': False,  # for policy choose one label with minimum loss or sum all 3 labels
    'mtl_weights': [1, -1, 1, 1],  # [0, -1, 0, 1, 1, 1],[-1, 1, 1, 1]
    'lstm_dict': {'len_seq': 100},
    'record_time_interval': 5,  # saving grida data interval
    'label_name': 'esw',
    # "maximum_pickup_time_passenger_can_tolerate_mean":200,  # s
    "maximum_pickup_time_passenger_can_tolerate_mean": float('inf'),
    "maximum_pickup_time_passenger_can_tolerate_std": 0,  # s
    "maximum_price_passenger_can_tolerate_mean": float('inf'),  # ￥
    "maximum_price_passenger_can_tolerate_std": 0,  # ￥
    'maximal_pickup_distance': 20,  # km
    'broadcasting_scale': 0.75,
    'request_interval': 2,  #
    'cruise_flag': CRUISE,
    'delivery_mode': 'rg',
    'pickup_mode': 'rg',
    'max_idle_time': 1,
    'cruise_mode': CRUISE_MODE,
    'reposition_flag': REPOSITION,
    'eligible_time_for_reposition': 10,  # s
    'reposition_mode': '',
    'track_recording_flag': True,
    'driver_far_matching_cancel_prob_file': 'driver_far_matching_cancel_prob',
    'input_file_path': 'input/dataset.csv',
    'request_file_name': 'input/hongkong_processed_order_new_road_network_60000',#'input/HongKong_island_order', #'input/hongkong_processed_order_new_road_network_60000',  # 'toy_requests',
    'driver_file_name': 'input/hongkong_driver_info_allday',#'input/HongKong_island_driver',#'input/hongkong_driver_info_allday',
    'road_network_file_name': 'road_network_information.pickle',
    'dispatch_method': 'LD',  # LD: lagarange decomposition method designed by Peibo Duan
    'method': 'instant_reward_no_subway',
    'simulator_mode': 'toy_mode',
    'experiment_mode': 'test',
    'driver_num': 1000,
    'side': 10,
    'change_grid_id': True,
    'price_per_km': 5,  # ￥ / km
    'road_information_mode': 'load',
    'north_lat': 22.55,  # 40.8845
    'south_lat': 22.13,  # 40.6968
    'east_lng': 114.42,  # -74.0831
    'west_lng': 113.81  # -73.8414
}
