import os, json, time, pickle
import multiprocessing as mp
import warnings, logging
warnings.filterwarnings('ignore')
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import pyart
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import distinctipy

def check_inside_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[(i + 1) % n]
        if yi == yj:
            continue
        if min(yi, yj) < y <= max(yi, yj) and x <= max(xi, xj):
            x_intersect = (y - yi) * (xj - xi) / (yj - yi) + xi
            if x <= x_intersect:
                inside = not inside

    return inside

def list_all_provinces():
    province_list = []
    provinces = [file[:-5] for file in os.listdir("coordinate/provinces") if file.endswith('.json')]

    for province in provinces:  
        with open(f'coordinate/provinces/{province}.json', 'r') as file:
            data = json.load(file)

        province_border = data['coordinates'][0][0]
        province_list += [(province, province_border)]
    
    return province_list

def check_inside_province(coordinate_list):
    province_list = list_all_provinces()
    result = []
    for coordinate in coordinate_list:
        for i in range(len(province_list)):
            if check_inside_polygon(coordinate, province_list[i][1]):
                result += [province_list[i][0]]
                if i != 0:
                    most_likely_province = province_list.pop(i)
                    province_list.insert(0, most_likely_province)
                break
        else:
            result += ["NotAvailable"]
    return result

def map_str_to_num(data):
    provinces = [file[:-5] for file in os.listdir("coordinate/provinces") if file.endswith('.json')]
    provinces = {text: index for index, text in enumerate(provinces)}
    provinces['NotAvailable'] = -1

    print(provinces)

    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = provinces[data[i][j]]
            
    return data

def plot_map(save_name):
    with open(f'coordinate/{save_name}.pkl', 'rb') as file:
        data = pickle.load(file)
        data = data.astype(int)
        data = data[::-1]

    colors = distinctipy.get_colors(100)
    cmap = ListedColormap(colors)
    
    plt.imshow(data, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.savefig(f"coordinate/{save_name}.png")

if __name__ == '__main__':
    radar_range = 300000 # 120000 | 300000
    
    data = pyart.io.read_sigmet("sample_data/2022-09-23 13-23-39")

    grid_data = pyart.map.grid_from_radars(data, 
                                           grid_shape=(1, 500, 500),
                                           grid_limits=((0, 1), (-radar_range, radar_range), 
                                                                (-radar_range, radar_range)))

    longitude_list = np.array(grid_data.point_longitude['data'][0]).flatten()
    latitude_list = np.array(grid_data.point_latitude['data'][0]).flatten()
    coordinate_list = np.array([[longitude_list[i], latitude_list[i]] for i in range(len(longitude_list))])
    
    num_processes = 16
    try:
        with mp.Pool(processes=num_processes) as pool:
            start_time = time.time()
            results = pool.map(check_inside_province, np.array_split(coordinate_list, num_processes))
            results = np.array([ele for sublist in results for ele in sublist]).reshape(500, 500)
            results = map_str_to_num(results)
            end_time = time.time() - start_time
    except Exception as e:
        print(e)
        logging.error(e, exc_info=True)
    
    print(f"Time: {end_time}")
    
    if radar_range == 120000:
        save_name = "provinces_120km"
    elif radar_range == 300000:
        save_name = "provinces_300km"
    
    with open(f"coordinate/{save_name}.pkl", "wb") as f:
        pickle.dump(results, f)
        
    plot_map(save_name)
    
# {'an_giang': 0, 'bac_lieu': 1, 'ben_tre': 2, 'binh_duong': 3, 'binh_phuoc': 4, 'binh_thuan': 5, 'brvt': 6, 
# 'can_tho': 7, 'ca_mau': 8, 'dak_lak': 9, 'dak_nong': 10, 'dong_nai': 11, 'dong_thap': 12, 'gia_lai': 13, 
# 'hau_giang': 14, 'ho_chi_minh': 15, 'kampong_cham': 16, 'kampong_chhnang': 17, 'kampong_speu': 18, 
# 'kampong_thom': 19, 'kampot': 20, 'kandal': 21, 'kep': 22, 'khanh_hoa': 23, 'kien_giang': 24, 'koh_kong': 25, 
# 'kratie': 26, 'lam_dong': 27, 'long_an': 28, 'mondulkiri': 29, 'ninh_thuan': 30, 'phu_yen': 31, 
# 'preah_sihanouk': 32, 'preah_vihear': 33, 'prey_veng': 34, 'pursat': 35, 'ratanakiri': 36, 'siem_reap': 37, 
# 'soc_trang': 38, 'stung_treng': 39, 'svay_rieng': 40, 'takeo': 41, 'tay_ninh': 42, 'tbong_khmum': 43, 
# 'tien_giang': 44, 'tra_vinh': 45, 'vinh_long': 46, 'NotAvailable': -1}



        
        
    