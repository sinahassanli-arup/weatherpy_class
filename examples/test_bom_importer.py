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

def test_oop_import(station_id, year_start, year_end, time_zone, interval):
    """Test the OOP-based import function."""
    print(f"\nTesting OOP import for BOM station {station_id}...")
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
        
        print(f"OOP import successful. Data shape: {weather_data.data.shape}")
        print(f"Data index type: {type(weather_data.data.index)}")
        print(f"Data index timezone: {weather_data.data.index.tz if hasattr(weather_data.data.index, 'tz') else 'None'}")
        print(f"Data columns: {weather_data.data.columns.tolist()}")
        
        return weather_data.data
    except Exception as e:
        print(f"Error in OOP import: {e}")
        traceback.print_exc()
        return None

# Test parameters
station_id = '066037'
year_start = 2010
year_end = 2020
time_zone = 'UTC'
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

# Test the OOP import
oop_data = test_oop_import(station_id, year_start, year_end, time_zone, interval)

if oop_data is not None:
    print("\nImport successful!")
    print(f"Data shape: {oop_data.shape}")
    print(f"Data columns: {oop_data.columns.tolist()}")
    print(f"Data index: {oop_data.index.name}")
    print(f"Data index timezone: {oop_data.index.tz if hasattr(oop_data.index, 'tz') else 'None'}")
    print(f"First few rows:")
    print(oop_data.head()) 