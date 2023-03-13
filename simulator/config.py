PRICE_MEAN = float('inf')
PRICE_STD = 0
PICKUP_TIME_MEAN = float('inf')
PICKUP_TIME_STD = 0
CRUISE = True
CRUISE_MODE = 'random'
REPOSITION = False

# morning :25200-32400, evening:61200-68400 midnight:0-18000 other:68400-86400
env_params = {
    't_initial': 0,
    't_end': 18000,
    'time_period': 'midnight',
    'delta_t': 5,  # s
    'vehicle_speed': 22.788,  # km / h
    'repo_speed': 1,  # 目前的设定需要与vehicl speed保持一致
    'order_sample_ratio': 0.5,
    'order_generation_mode': 'sample_from_base',
    'driver_sample_ratio': 1,
    'maximum_wait_time_mean': 300,
    'maximum_wait_time_std': 0,
    'radius_list': [1, 2, 3, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10],
    'grid_radius_dict': {1: 2.5, 2: 2.5, 3: 2.5, 4: 2.5, 5: 2.5, 6: 2.5, 7: 2.5, 8: 2.5, 9: 2.5,
                         10: 2.5,
                         11: 2.5, 12: 2.5, 13: 2.5, 14: 2.5, 15: 2.5, 16: 2.5},
    # "maximum_pickup_time_passenger_can_tolerate_mean":200,  # s
    "maximum_pickup_time_passenger_can_tolerate_mean": PICKUP_TIME_MEAN,
    "maximum_pickup_time_passenger_can_tolerate_std": PICKUP_TIME_STD,  # s
    "maximum_price_passenger_can_tolerate_mean": PRICE_MEAN,  # ￥
    "maximum_price_passenger_can_tolerate_std": PRICE_STD,  # ￥
    'maximal_pickup_distance': 20,  # km
    'broadcasting_scale': 2.5,
    'request_interval': 5,  #
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
    'request_file_name': 'input/order',  # 'toy_requests',
    'driver_file_name': 'input/driver_info',
    'road_network_file_name': 'road_network_information.pickle',
    'dispatch_method': 'LD',  # LD: lagarange decomposition method designed by Peibo Duan
    'method': 'instant_reward_no_subway',
    'simulator_mode': 'toy_mode',
    'experiment_mode': 'test',
    'driver_num': 500,
    'side': 4,
    'price_per_km': 5,  # ￥ / km
    'road_information_mode': 'load',
    'north_lat': 40.8845,
    'south_lat': 40.6968,
    'east_lng': -74.0831,
    'west_lng': -73.8414
}
