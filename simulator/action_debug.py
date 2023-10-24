import json

def driver_exists(driver_id, data):
    for item in data:
        if item['driverid'] == driver_id:
            return True
    return False

def find_incorrect_cycle_driver(data, correct_cycle):
    incorrect_drivers = []
    for item in data:
        actions = item['actions']
        if len(actions) % len(correct_cycle) != 0:  # If the number of actions is not a multiple of the cycle length
            incorrect_drivers.append(item['driver_id'])
            continue
        for i in range(0, len(actions), len(correct_cycle)):
            if actions[i:i+len(correct_cycle)] != correct_cycle:  # If any segment of the actions does not match the cycle
                incorrect_drivers.append(item['driver_id'])
                break
    return incorrect_drivers

if __name__ == "__main__":
    action_data = json.load(open("./actions.json",'r'))
    driver_action = []
    for time,actions in action_data.items():
        if len(actions) > 0:
            for item in actions:
                if item['actionType'] == 'orderReceivedAction' or item['actionType'] == 'pickUpAction' or item['actionType'] == 'dropOffAction' :
                   cur_action_data = item['data']
                   for single_dict in cur_action_data:
                       # print(single_dict,item['actionType'])
                       for driver_data in driver_action:
                            if single_dict['driverid'] == driver_data['driver_id']:
                               # 如果找到了，就添加新的经纬度数据
                               driver_data['actions'].append(item['actionType'])
                       else:
                       # 如果没有找到，就添加一个新的司机ID，并设置其路线数据
                           driver_action.append({
                               'driver_id': single_dict['driverid'],
                               'actions': [item['actionType']]})

    correct_cycle = ['orderReceivedAction', 'pickUpAction', 'dropOffAction']
    for item in driver_action:
        if len(item['actions'])%3 !=0:
            print(item)

    wrong_data = find_incorrect_cycle_driver(driver_action, correct_cycle)
    print(len(wrong_data))