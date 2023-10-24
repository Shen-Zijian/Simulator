import json
from math import radians, cos, sin, asin, sqrt
import numpy as np
import matplotlib.pyplot as plt
import requests
import polyline


# start = [114.2119, 22.2828]
# end = [114.1467, 22.2436]

def interpolate(start, end, num_sample):
    # Your Mapbox access token
    # (Assuming url is defined somewhere else in your code)
    mapbox_access_token = "pk.eyJ1IjoibWF0dGp3YW5nIiwiYSI6ImNsaXB5NDN1cTAzMnAza28xaG54ZWRrMzgifQ.cUju1vqjuW7XmAuO2iEZmg";

    url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{start[0]},{start[1]};{end[0]},{end[1]}?access_token={mapbox_access_token}"
    # a=1==4==============b
    response = requests.get(url).json()
    # print(response)
    for route in response['routes']:
        coordinates = polyline.decode(route['geometry'])
        coordinates = [[coord[1], coord[0]] for coord in coordinates]

        # If there are more coordinates than needed
        if len(coordinates) > num_sample:
            indices = np.round(np.linspace(0, len(coordinates) - 1, num_sample)).astype(int)
            coordinates = [coordinates[i] for i in indices]
        # If there are fewer coordinates than needed
        elif len(coordinates) < num_sample:
            lat = np.interp(np.linspace(0, len(coordinates) - 1, num_sample), np.arange(len(coordinates)), [coord[0] for coord in coordinates])
            lon = np.interp(np.linspace(0, len(coordinates) - 1, num_sample), np.arange(len(coordinates)), [coord[1] for coord in coordinates])
            coordinates = list(zip(lat, lon))
    coordinates = [list(coord) for coord in coordinates]
    coordinates[0] = start
    coordinates[-1] = end
    return coordinates
def plot_coords(coords,list_cood=None):
    # 转置坐标列表，使得经度和纬度在单独的列表中
    lon, lat = zip(*coords)

    # 创建一个新的图形
    plt.figure()

    # 绘制坐标点
    plt.scatter(lon, lat)

    # 绘制连接坐标点的线
    plt.plot(lon, lat, 'r-')
    if list_cood != None:
        plt.scatter(list_cood[0][0],list_cood[0][1],color='yellow')
        plt.scatter(list_cood[1][0], list_cood[1][1], color='yellow')
    # 添加标题和标签
    plt.title('Coordinates')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # 显示图形
    plt.show()

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in meters between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # 地球平均半径，单位是公里
    return c * r * 1000
def test_dist(coords):
    i=0
    num_same = 0
    num_exceed = 0
    while i < len(coords) - 1:
        lat1, lon1 = coords[i][1], coords[i][0]
        lat2, lon2 = coords[i+1][1], coords[i+1][0]
        distance = haversine(lon1, lat1, lon2, lat2)
        i+=1
        print(lat1, lon1)
        plot_coords(coords, [[lon1, lat1], [lon2, lat2]])
        if distance > 100:

            print("ERROR! ",distance)
            num_exceed +=1
        if lat1 == lat2 and lon1 == lon2:
            num_same += 1
        else:
            num_same = 0
    # print("num_exeed",num_exceed)
def process_coords(coords):
    i = 0
    num_same = 0
    num_exeed = 0
    while i < len(coords) - 1:
        lat1, lon1 = coords[i][1], coords[i][0]
        lat2, lon2 = coords[i+1][1], coords[i+1][0]
        distance = haversine(lon1, lat1, lon2, lat2)

        if distance > 100:
            new_coords = interpolate([lon1, lat1], [lon2, lat2], num_same+1)
            # new_coords = interpolate_linear(lat1,lon1, lat2,lon2, num_same+1)

            coords = coords[:i-num_same] + new_coords + coords[i+1:]
            # plot_coords(coords)

            # coords[i-num_same-1:i-1] = new_coords
            # plot_coords(coords)
            # print(coords)
            # print("=" * 20)
            # print(new_coords)
            # print("="*20)
            # print(coords[i-num_same],new_coords[0],new_coords[-1],coords[i+1])
            print(len(coords),distance/num_same)
            # plot_coords(coords)
            num_exeed += 1
        i += 1

        if lat1 == lat2 and lon1 == lon2:
            num_same += 1
        else:
            num_same = 0
    return coords

def interpolate_linear(lat1, lon1, lat2, lon2, num):
    lat_linspace = np.linspace(lat1, lat2, num)

    print("start point:",lon1, lat1)

    print("end point:",lon2, lat2)

    lon_linspace = np.linspace(lon1, lon2, num)
    index = 0
    # while index <len(list(zip(lon_linspace, lat_linspace))) -1:
    #     item = list(zip(lon_linspace, lat_linspace))[index]
    #     next_item =  list(zip(lon_linspace, lat_linspace))[index+1]
    #     distance = haversine(item[0],item[1],next_item[0],next_item[1])
    #     print(distance)
    return list(zip(lon_linspace, lat_linspace))

if __name__ == "__main__":
    test_flag = True
    data = json.load(open("./driver_route.json"))
    index = 0
    for item in data['drivers']:
        driver_route_dict = item
        processed_coord = process_coords(item['route'])
        data['drivers'][index]['route'] = processed_coord
        index += 1
    with open("./driver_route_interploation.json", 'w') as f:
        json.dump(data, f, indent=2)
    if test_flag == True:
        data = json.load(open("./driver_route_interploation.json"))
        for item in data['drivers']:
            test_dist(item['route'])
            # print(item['route'])