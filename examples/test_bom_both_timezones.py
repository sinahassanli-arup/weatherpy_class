import sys
import os
import pandas as pd
import numpy as np
import logging
import traceback
from datetime import datetime
import pytz

# Add the weatherpy_class directory to the path
sys.path.insert(0, r'C:\Users\Administrator\Documents\weatherpy_class')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the OOP-based importer
from weatherpy.data.wd_importer import BOMWeatherDataImporter
from weatherpy.data.wd_stations import WeatherStationDatabase

def test_bom_import(station_id, year_start, year_end, time_zone, interval):
    """Test the BOMWeatherDataImporter class."""
    print(f"\nTesting BOMWeatherDataImporter for station {station_id} with timezone {time_zone}...")
    try:
        # Create the importer
        importer = BOMWeatherDataImporter(
            station_id=station_id,
            data_type='BOM',
            time_zone=time_zone,
            year_start=year_start,
            year_end=year_end,
            interval=interval
        )
        
        # Import data
        weather_data = importer.import_data()
        
        print(f"Import successful. Data shape: {weather_data.data.shape}")
        print(f"Data index type: {type(weather_data.data.index)}")
        print(f"Data index timezone: {weather_data.data.index.tz if hasattr(weather_data.data.index, 'tz') else 'None'}")
        print(f"Data columns: {weather_data.data.columns.tolist()}")
        
        return weather_data.data
    except Exception as e:
        print(f"Error in BOM import: {e}")
        traceback.print_exc()
        return None

# Test parameters
station_id = '066037'
year_start = 2010
year_end = 2020
interval = 60

# Test the station database
print("Testing WeatherStationDatabase...")
try:
    bom_db = WeatherStationDatabase('BOM')
    print(f"BOM database loaded with {len(bom_db._data)} stations")
    
    # Test getting a station
    station_info = bom_db.get_station_info(station_id)
    print(f"Station info for {station_id}: {station_info}")
except Exception as e:
    print(f"Error testing WeatherStationDatabase: {e}")
    traceback.print_exc()

# Test the BOM import with UTC timezone
utc_data = test_bom_import(station_id, year_start, year_end, 'UTC', interval)

if utc_data is not None:
    print("\nUTC Import successful!")
    print(f"Data shape: {utc_data.shape}")
    print(f"Data columns: {utc_data.columns.tolist()}")
    print(f"Data index: {utc_data.index.name}")
    print(f"Data index timezone: {utc_data.index.tz if hasattr(utc_data.index, 'tz') else 'None'}")
    print(f"First few rows:")
    print(utc_data.head())

# Test the BOM import with LocalTime timezone
local_data = test_bom_import(station_id, year_start, year_end, 'LocalTime', interval)

if local_data is not None:
    print("\nLocalTime Import successful!")
    print(f"Data shape: {local_data.shape}")
    print(f"Data columns: {local_data.columns.tolist()}")
    print(f"Data index: {local_data.index.name}")
    print(f"Data index timezone: {local_data.index.tz if hasattr(local_data.index, 'tz') else 'None'}")
    print(f"First few rows:")
    print(local_data.head()) 