import os, json, time, pickle, random
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

PROVINCE_MAPPING = {'an_giang':    0,  'bac_lieu':     1,  'ben_tre':         2,  'binh_duong':   3,  'binh_phuoc':   4, 
                    'binh_thuan':  5,  'brvt':         6,  'can_tho':         7,  'ca_mau':       8,  'dak_lak':      9, 
                    'dak_nong':    10, 'dong_nai':     11, 'dong_thap':       12, 'gia_lai':      13, 'hau_giang':    14, 
                    'ho_chi_minh': 15, 'kampong_cham': 16, 'kampong_chhnang': 17, 'kampong_speu': 18, 'kampong_thom': 19, 
                    'kampot':      20, 'kandal':       21, 'kep':             22, 'khanh_hoa':    23, 'kien_giang':   24, 
                    'koh_kong':    25, 'kratie':       26, 'lam_dong':        27, 'long_an':      28, 'mondulkiri':   29, 
                    'ninh_thuan':  30, 'phu_yen':      31, 'preah_sihanouk':  32, 'preah_vihear': 33, 'prey_veng':    34, 
                    'pursat':      35, 'ratanakiri':   36, 'siem_reap':       37, 'soc_trang':    38, 'stung_treng':  39,
                    'svay_rieng':  40, 'takeo':        41, 'tay_ninh':        42, 'tbong_khmum':  43, 'tien_giang':   44,
                    'tra_vinh':    45, 'vinh_long':    46, 
                    'other':       -1
                    }

def list_all_provinces():
    province_list = []
    provinces = sorted([file[:-5] for file in os.listdir("coordinate/provinces") if file.endswith('.json')])
    print({name: index for index, name in enumerate(provinces)})

    for province in provinces:  
        with open(f'coordinate/provinces/{province}.json', 'r') as file:
            data = json.load(file)

        province_border = data['coordinates'][0][0]
        province_list += [{'name': province, 'border': province_border}]
    
    return province_list

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

def check_inside_province(coordinate_list):
    province_list = list_all_provinces()
    result = []
    
    for coordinate in coordinate_list:
        for i in range(len(province_list)):
            province_border = province_list[i]['border']
            if check_inside_polygon(coordinate, province_border):
                province_name = province_list[i]['name']
                result += [PROVINCE_MAPPING[province_name]]
                if i != 0:
                    most_likely_province = province_list.pop(i)
                    province_list.insert(0, most_likely_province)
                break
        else:
            province_name = "other"
            result += [PROVINCE_MAPPING[province_name]]
            
    return result

def plot_province_map():
    province_list = list_all_provinces()
    name_list = [ele['name'] for ele in province_list]

    num_provinces = len(name_list)
    num_colors = num_provinces + 1
    colors = distinctipy.get_colors(num_colors, colorblind_type="Deuteranomaly")
    random.shuffle(colors)
    cmap = ListedColormap(colors)
    
    for save_name in ["120km", "300km"]:
        if os.path.exists(f'coordinate/provinces_{save_name}.pkl'):
            with open(f'coordinate/provinces_{save_name}.pkl', 'rb') as file:
                data = pickle.load(file)
                data = data.astype(int)
                data = data[::-1]
            
            fig, ax = plt.subplots(figsize=(15, 13))
            plt.imshow(data, cmap=cmap, interpolation='nearest')
            plt.title(f'Province map for Vietnam and Cambodia - {save_name}', fontsize=25)
            plt.axis('off')
            
            # Make tick at the center of color
            ticks = np.linspace(-1, num_provinces - 1, 2 * num_colors + 1)[1::2]
            labels = ["-1 - other"] + [f"{i} - {name_list[i]}" for i in range(num_provinces)]
            cbar = plt.colorbar(ticks=ticks)
            cbar.ax.set_yticklabels(labels, fontsize=11)
                
            plt.savefig(f"coordinate/provinces_{save_name}.png")
            
            plt.close()

if __name__ == '__main__':
    radar_range = 120000 # 120000 | 300000
    
    if radar_range == 120000:
        file_name = "2022-09-23 13-23-39"
        save_name = "120km"
    elif radar_range == 300000:
        file_name = "2022-09-22 09-00-48"
        save_name = "300km"
    
    data = pyart.io.read_sigmet(f"sample_data/{file_name}")
    grid_data = pyart.map.grid_from_radars(data, 
                                           grid_shape=(1, 500, 500),
                                           grid_limits=((0, 1), (-radar_range, radar_range),
                                                                (-radar_range, radar_range)))

    longitude_list = np.array(grid_data.point_longitude['data'][0]).flatten()
    latitude_list = np.array(grid_data.point_latitude['data'][0]).flatten()
    coordinate_list = np.column_stack((longitude_list, latitude_list))
    print(len(coordinate_list))
    # print(coordinate_list.reshape(500, 500))
    print(f"Matrix size: {coordinate_list.size}")
    # num_processes = 16
    # try:
    #     with mp.Pool(processes=num_processes) as pool:
    #         start_time = time.time()
            
    #         results = pool.map(check_inside_province, np.array_split(coordinate_list, num_processes))
    #         results = np.array([ele for sublist in results for ele in sublist]).reshape(500, 500)
    #         with open(f"coordinate/provinces_{save_name}.pkl", "wb") as f:
    #             pickle.dump(results, f)
                
    #         end_time = time.time() - start_time
    #         print(f"Time: {end_time}")
    # except Exception as e:
    #     print(e)
    #     logging.error(e, exc_info=True)
            
    # plot_province_map()



        
        
    