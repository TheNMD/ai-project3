import pyart
import numpy as np
import aiohttp
import asyncio
from loguru import logger
from typing import Dict, List, Tuple
from functools import lru_cache


async def calculate_avg_reflectivity(reflectivity: np.ndarray) -> Tuple[float, str]:
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    hist, _ = np.histogram(reflectivity, bins=bins + [np.inf])
    percentages = hist / len(reflectivity)

    weight_set = [10 ** (100 * (1 - p)) for p in percentages]

    @np.vectorize
    def get_weight(value):
        return weight_set[np.searchsorted(bins, value, side='right') - 1]

    weights = get_weight(reflectivity)
    avg_reflectivity = np.average(reflectivity, weights=weights)

    if avg_reflectivity < 30:
        label = "clear"
    elif avg_reflectivity < 40:
        label = "light_rain"
    elif avg_reflectivity < 47.5:
        label = "moderate_rain"
    elif avg_reflectivity < 55:
        label = "heavy_rain"
    else:
        label = "very_heavy_rain"

    return avg_reflectivity, label


@lru_cache(maxsize=1)
def get_lat_lon_values() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = pyart.io.read_sigmet(
        "./DatailedLabelSample/NHB230601000008.RAWLGZV")
    data.fields['reflectivity']['data'] = data.fields['reflectivity']['data'].astype(
        np.float16)

    grid_data = pyart.map.grid_from_radars(
        data,
        grid_shape=(1, 500, 500),
        grid_limits=((0, 1), (-250000, 250000), (-250000, 250000)),
    )

    reflectivity_array = np.array(grid_data.fields['reflectivity']['data'])[0]
    longitude_array = np.array(grid_data.point_longitude['data'])[0]
    latitude_array = np.array(grid_data.point_latitude['data'])[0]

    return reflectivity_array, longitude_array, latitude_array


async def get_province(session: aiohttp.ClientSession, latitude: float, longitude: float) -> str:
    url = 'https://api-bdc.net/data/reverse-geocode-client'
    params = {"latitude": latitude, "longitude": longitude}

    async with session.get(url, params=params) as response:
        if response.status != 200:
            raise Exception(f"Error: {response.status}, {await response.text()}")
        data = await response.json()
        return data['principalSubdivision']


async def process_grid_point(session: aiohttp.ClientSession, i: int, j: int,
                             reflectivity_array: np.ndarray,
                             latitude_array: np.ndarray,
                             longitude_array: np.ndarray) -> Tuple[str, float]:
    province = await get_province(session, latitude_array[i, j], longitude_array[i, j])
    return province, reflectivity_array[i, j]


async def generate_province_detail_label() -> Dict[str, List[float]]:
    reflectivity_array, longitude_array, latitude_array = get_lat_lon_values()
    provinces_dict: Dict[str, List[float]] = {}

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(0, 500, 10):  # Step by 10 to reduce the number of requests
            for j in range(0, 500, 10):
                task = process_grid_point(
                    session, i, j, reflectivity_array, latitude_array, longitude_array)
                tasks.append(task)

        results = await asyncio.gather(*tasks)

    for province, reflectivity in results:
        if province in provinces_dict:
            provinces_dict[province].append(reflectivity)
        else:
            provinces_dict[province] = [reflectivity]

    return provinces_dict


async def main():
    try:
        province_dict = await generate_province_detail_label()
        for province, reflectivity_list in province_dict.items():
            avg_reflectivity, label = await calculate_avg_reflectivity(np.array(reflectivity_list))
            logger.info(
                f"{province}: Avg Reflectivity = {avg_reflectivity:.2f}, Label = {label}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == '__main__':
    asyncio.run(main())
