"""
Module for managing weather station data.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, Optional

def get_station_info(stationID: str, dataType: str) -> Dict[str, Any]:
    """
    Get station information.
    
    Parameters
    ----------
    stationID : str
        Station ID.
    dataType : str
        Data type ('BOM' or 'NOAA').
    
    Returns
    -------
    Dict[str, Any]
        Station information.
    """
    # This is a simplified version for the example
    # In a real implementation, this would look up the station in a database
    
    if dataType == 'BOM':
        if stationID == '001006':
            return {
                'Station Name': 'WYNDHAM AERO',
                'Country': 'Australia',
                'State': 'WA',
                'Latitude': -15.51,
                'Longitude': 128.1503,
                'Altitude': 3.8,
                'Years Active': '1951 - 2023',
                'Timezone': 'Australia/Perth',
                'Timezone Ofset': 'UTC +08:00',
                'Source': 'BOM',
                'Data Available': {
                    'Wind Direction': 'True',
                    'Wind Speed': 'True',
                    'Wind Gust': 'True',
                    'Sea Level Pressure': 'True',
                    'Dry Bulb Temperature': 'True',
                    'Wet Bulb Temperature': 'True',
                    'Relative Humidity': 'True',
                    'Rain': 'True',
                    'Rain Intensity': 'True',
                    'Cloud Oktas': 'False'
                }
            }
        elif stationID == '066037':
            return {
                'Station Name': 'SYDNEY AIRPORT AMO',
                'Country': 'Australia',
                'State': 'NSW',
                'Latitude': -33.96095,
                'Longitude': 151.1818167,
                'Altitude': 6.0,
                'Years Active': '1929 - 2023',
                'Timezone': 'Australia/Sydney',
                'Timezone Ofset': 'UTC +11:00',
                'Source': 'BOM',
                'M Correction Factors': '0.96158, 0.97451, 0.973795, 0.941329, 0.932394, 0.872557, 0.825254, 0.829221, 0.862977, 0.837179, 0.87703, 1.007511, 1.027576, 1.032383, 0.954836, 0.908574',
                'Data Available': {
                    'Wind Direction': 'True',
                    'Wind Speed': 'True',
                    'Wind Gust': 'True',
                    'Sea Level Pressure': 'True',
                    'Dry Bulb Temperature': 'True',
                    'Wet Bulb Temperature': 'True',
                    'Relative Humidity': 'True',
                    'Rain': 'True',
                    'Rain Intensity': 'True',
                    'Cloud Oktas': 'True'
                }
            }
        else:
            return {
                'Station Name': f'BOM Station {stationID}',
                'Country': 'Australia',
                'State': 'Unknown',
                'Latitude': -33.0,
                'Longitude': 151.0,
                'Altitude': 0.0,
                'Years Active': '2000 - 2023',
                'Timezone': 'Australia/Sydney',
                'Timezone Ofset': 'UTC +10:00',
                'Source': 'BOM',
                'Data Available': {
                    'Wind Direction': 'True',
                    'Wind Speed': 'True',
                    'Wind Gust': 'True',
                    'Sea Level Pressure': 'True',
                    'Dry Bulb Temperature': 'True',
                    'Wet Bulb Temperature': 'True',
                    'Relative Humidity': 'True',
                    'Rain': 'True',
                    'Rain Intensity': 'True',
                    'Cloud Oktas': 'False'
                }
            }
    elif dataType == 'NOAA':
        if stationID == '72509014739':
            return {
                'Station Name': 'NEW YORK CENTRAL PARK',
                'Country': 'United States',
                'State': 'NY',
                'Latitude': 40.779,
                'Longitude': -73.88,
                'Altitude': 3.0,
                'Years Active': '1973 - 2023',
                'Timezone': 'America/New_York',
                'Timezone Ofset': 'UTC -05:00',
                'Source': 'NOAA',
                'Data Available': {
                    'Wind Direction': 'True',
                    'Wind Speed': 'True',
                    'Wind Gust': 'True',
                    'Sea Level Pressure': 'True',
                    'Dry Bulb Temperature': 'True',
                    'Wet Bulb Temperature': 'True',
                    'Relative Humidity': 'True',
                    'Rain': 'True',
                    'Rain Intensity': 'True',
                    'Cloud Oktas': 'True'
                }
            }
        else:
            return {
                'Station Name': f'NOAA Station {stationID}',
                'Country': 'United States',
                'State': 'NY',
                'Latitude': 40.779,
                'Longitude': -73.88,
                'Altitude': 3.0,
                'Years Active': '1973 - 2023',
                'Timezone': 'America/New_York',
                'Timezone Ofset': 'UTC -05:00',
                'Source': 'NOAA',
                'Data Available': {
                    'Wind Direction': 'True',
                    'Wind Speed': 'True',
                    'Wind Gust': 'True',
                    'Sea Level Pressure': 'True',
                    'Dry Bulb Temperature': 'True',
                    'Wet Bulb Temperature': 'True',
                    'Relative Humidity': 'True',
                    'Rain': 'True',
                    'Rain Intensity': 'True',
                    'Cloud Oktas': 'True'
                }
            }
    else:
        return {} 