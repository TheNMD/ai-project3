import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import distinctipy

# Generate 50 distinct colors
colors = distinctipy.get_colors(100)

# Create the colormap
cmap = ListedColormap(colors)

# Create some example data
data = np.random.rand(10, 10)

# Plot with the custom colormap
plt.imshow(data, cmap=cmap)
plt.colorbar()
plt.title('Custom Colormap with 50 Colors')

plt.savefig("test.jpg")

# {'an_giang': 0, 'bac_lieu': 1, 'ben_tre': 2, 'binh_duong': 3, 'binh_phuoc': 4, 'binh_thuan': 5, 'brvt': 6, 
# 'can_tho': 7, 'ca_mau': 8, 'dak_lak': 9, 'dak_nong': 10, 'dong_nai': 11, 'dong_thap': 12, 'gia_lai': 13, 
# 'hau_giang': 14, 'ho_chi_minh': 15, 'kampong_cham': 16, 'kampong_chhnang': 17, 'kampong_speu': 18, 
# 'kampong_thom': 19, 'kampot': 20, 'kandal': 21, 'kep': 22, 'khanh_hoa': 23, 'kien_giang': 24, 'koh_kong': 25, 
# 'kratie': 26, 'lam_dong': 27, 'long_an': 28, 'mondulkiri': 29, 'ninh_thuan': 30, 'phu_yen': 31, 
# 'preah_sihanouk': 32, 'preah_vihear': 33, 'prey_veng': 34, 'pursat': 35, 'ratanakiri': 36, 'siem_reap': 37, 
# 'soc_trang': 38, 'stung_treng': 39, 'svay_rieng': 40, 'takeo': 41, 'tay_ninh': 42, 'tbong_khmum': 43, 
# 'tien_giang': 44, 'tra_vinh': 45, 'vinh_long': 46, 'NotAvailable': -1}
