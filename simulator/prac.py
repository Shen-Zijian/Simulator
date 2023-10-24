import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import random
def distance(coord_1, coord_2):
    """
    :param coord_1: the coordinate of one point
    :type coord_1: tuple -- (latitude,longitude)
    :param coord_2: the coordinate of another point
    :type coord_2: tuple -- (latitude,longitude)
    :return: the manhattan distance between these two points
    :rtype: float
    """
    lat1, lon1 = coord_1
    lat2, lon2 = coord_2
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = abs(lon2 - lon1)
    dlat = abs(lat2 - lat1)
    r = 6371

    alat = np.sin(dlat / 2) ** 2
    clat = 2 * np.arctan2(alat ** 0.5, (1 - alat) ** 0.5)
    lat_dis = clat * r

    alon = np.sin(dlon / 2) ** 2
    clon = 2 * np.arctan2(alon ** 0.5, (1 - alon) ** 0.5)
    lon_dis = clon * r

    manhattan_dis = abs(lat_dis) + abs(lon_dis)

    return manhattan_dis

def save_obj(obj, name ):
    with open('/home/shenzijian/下载/Transpotation_Simulator-1/simulator/output3/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    obj = pickle.load(open('/home/shenzijian/下载/Transpotation_Simulator-1/simulator/output3/' + name + '.pkl', 'rb'))
    return obj

def test():
    gps_latitude_list = [114.1632465, 114.1632879, 114.163306, 114.1633269, 114.1633545, 114.1634062, 114.1635217, 114.1636139, 114.1638712, 114.16405, 114.1641748, 114.1642699, 114.1640789, 114.1640506, 114.1640551, 114.1640545, 114.1640715, 114.164097, 114.1640426, 114.1640151, 114.1640259, 114.1640761, 114.1641276, 114.164171, 114.1641768, 114.1641507, 114.1641348, 114.1641474, 114.1642285, 114.1642935, 114.164326, 114.1643329, 114.164335, 114.1643348, 114.1643212, 114.1643046, 114.164268, 114.1642126, 114.1641487, 114.1641596, 114.1641708, 114.1642089, 114.164407, 114.1644986, 114.1645051, 114.1644909, 114.1646414, 114.1650111, 114.1655102, 114.1659984, 114.1663906, 114.1665521, 114.1664299, 114.1657973, 114.1656575, 114.1655293, 114.1653738, 114.1651501, 114.164767, 114.1644459, 114.1642334, 114.164061, 114.1639731, 114.163945, 114.1639741, 114.164049, 114.1640797, 114.1641014, 114.1640967, 114.1639472, 114.1636383, 114.1633825, 114.1631809, 114.1630601, 114.1629984, 114.1629622, 114.1629043, 114.1626498, 114.1621315, 114.1613609, 114.1607536, 114.1594001, 114.1583482, 114.1576227, 114.1569505, 114.1565138, 114.1566873, 114.1567242, 114.1568077, 114.1570537, 114.1578153, 114.1586015, 114.159337, 114.1596335, 114.1597364, 114.1594978, 114.1588063, 114.1583394, 114.157924, 114.1573428, 114.1563828, 114.1557106, 114.1550236, 114.1541456, 114.1534327, 114.152811, 114.1520681, 114.1513291, 114.1505633, 114.1496455, 114.1487533, 114.1479389, 114.1471301, 114.1462918, 114.1454541, 114.1446703, 114.1438724, 114.1431671, 114.1427005, 114.1442804, 114.1449084, 114.145142, 114.1445804, 114.1445783, 114.1445335, 114.1444395, 114.1437921, 114.1433598, 114.1432733, 114.1435404, 114.1440048, 114.1446279, 114.1453751, 114.1461454, 114.1469135, 114.1476927, 114.1488749, 114.1487916, 114.1484597, 114.1478137, 114.1475807, 114.145979, 114.145979, 114.145979, 114.145979, 114.145979, 114.145979, 114.1597272, 114.1597272, 114.1597272, 114.1597272, 114.1615851, 114.1615851, 114.1615851, 114.1625729, 114.1629002, 114.1630369, 114.1630945, 114.1630623, 114.1632021, 114.1633823, 114.1635826, 114.1638722, 114.1640984, 114.1644152, 114.1647585, 114.1650097, 114.1651426, 114.1712457, 114.1712457, 114.1720146, 114.1710949, 114.1717406, 114.1724016, 114.1740975, 114.1712457, 114.1748981, 114.1749371, 114.1752707, 114.1755419, 114.1758188, 114.176232, 114.1769961, 114.1778867]
    gps_longitude_list = [22.3174409, 22.3174273, 22.3173683, 22.3172983, 22.3172545, 22.3171757, 22.3170807, 22.3169175, 22.3165681, 22.3160988, 22.3155778, 22.3153795, 22.3154677, 22.3155374, 22.3155558, 22.3155518, 22.3154966, 22.3148134, 22.3143653, 22.3142131, 22.3142904, 22.3143459, 22.3144116, 22.3144795, 22.314531, 22.3145564, 22.3145799, 22.3146117]
    gps_distance = 0

    for i in range(len(gps_latitude_list)):
        if i - 1 < 0:
            vec1 = np.array([gps_latitude_list[1], gps_longitude_list[1]])
            vec2 = np.array([gps_latitude_list[0], gps_longitude_list[0]])
        else:
            vec1 = np.array([gps_latitude_list[i], gps_longitude_list[i]])
            vec2 = np.array([gps_latitude_list[i - 1], gps_longitude_list[i - 1]])
        gps_distance += distance(vec1,vec2)
        print(gps_distance)
def generate_random_num(length):
    if length < 1:
        res = 0
    else:
        res = random.randint(0, length)
    return res

index = 0
# driver_pick_flag = np.zeros([6, 20], dtype=int)
# driver_pick_flag[:5,5] = 1
# driver_pick_flag[:5,6] = 1
# driver_pick_flag[:5,8] = 1
# driver_pick_flag[0,10] = 1
# print(driver_pick_flag)
# for row in driver_pick_flag:
#     temp_line = np.argwhere(row == 1)
#     if len(temp_line) >= 1:
#         temp_num = (len(temp_line) - 1)
#         # temp_num = 1
#         driver_pick_flag[:, temp_line[temp_num, 0]] = 0
#         row[:] = 0
#         row[temp_line[temp_num, 0]] = 1
#         driver_pick_flag[index, :] = row
#         driver_pick_flag[index + 1:, temp_line[temp_num, 0]] = 0
#         print(f"=========={index}=========")
#         print(driver_pick_flag)
#
#
#     index += 1
def driver_relocation():

    pass

if __name__ == '__main__':
    driver_relocation()
# list2 = pickle.load(
#                             open('/home/shenzijian/下载/Transpotation_Simulator-1/simulator/input/driver_distribution_3am-4am.pickle',
#                                  'rb'))
# id_list = list2['grid_id'].unique()
# for item in id_list:
#     temp_col = list2.loc[(list2['grid_id']==item),('grid_id')]
#     print(item,':',len(temp_col))
# print(list2['grid_id'])
# #test()
# gps_dist_list=[145.997,70.5344,20.11936,21.749949,16.4,20.11,102.33,79.6]
# order_list = [24747.0,1060,2805,2244]
# error_list = [0.992,0.93,0.991,0.98,0.89,0.99,0.92,0.96]
# dist_ = {'a':11,'b':12}
# save_obj(dist_,'error')
# error = load_obj('error')
# print(error)
# x_list = range(len(gps_dist_list))
# print(x_list)
# figure1 = plt.plot(x_list, gps_dist_list, label='gps_distance', color='red',linewidth=3)
# plt.show()
# figure2 = plt.plot(x_list, order_list, label='order_distance', color='blue',linewidth=3)
# plt.show()
# figure3 = plt.plot(x_list, error_list, label='Error', color='green',linewidth=3)
# plt.show()
# plt.legend(loc="best")
#
# sns.set()  #切换到sns的默认运行配置
#
# x=np.random.randn(100)
# sns.distplot(x)
# plt.show()


'''
gps_1
21.019677360786943 24747.0
error = 0.15061715113803922
gps_1
9.570836443115278 1060.0
error = 8.029090984071017
gps_1
2.6802091533679846 2805.0
error = 0.04448871537683261
gps_1
2.8696964278596897 2244.0
error = 0.2788308502048528
======
ps_1- 903.0
2.896038505978358 24747.0
error = 0.8829741582422775
gps_1- 840.0
0.01899209347298969 16112.0
error = 0.9988212454398592
gps_1- 606.0
0.1269734867369734 33016.0
error = 0.996154183222166
'''